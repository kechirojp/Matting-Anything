"""非同期ジョブ基盤（ERR058）。

Gradio の 1 リクエストで数分の重い処理を同期実行すると、無料 ``gradio.live`` FRP トンネル
の長時間接続上限で SSE が切れて全出力が「Error」になる（ERR048-057 の系統）。本モジュールは
重い処理をバックグラウンドのデーモンスレッドへ逃がし、HTTP リクエストを短く保つための
スレッドセーフなジョブレジストリを提供する。UI 側は ``gr.Timer`` で ``snapshot`` を
ポーリングし、進捗テキスト更新・完了出力・エラー通知を行う。

設計方針:
    - 例外は **握り潰さない**。`work` が投げた例外は ``JobState.error`` に保持し、
      呼び出し側（poll ハンドラ）が ``gr.Error`` 等で必ず通知する。
    - GPU / モデルに依存しない純 Python なので単体テストで挙動を固定できる。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

#: 進捗報告コールバックの型。``report(fraction, description)`` の形で呼ばれる。
ProgressReport = Callable[[float, str], None]

#: ``work`` の型。進捗報告コールバックを受け取り、任意の結果を返す。
JobWork = Callable[[ProgressReport], Any]


@dataclass
class JobState:
    """1 ジョブの状態スナップショット。

    Attributes:
        job_id: ジョブ識別子。
        status: ``"running"`` / ``"done"`` / ``"error"`` のいずれか。
        fraction: 進捗率（0.0〜1.0）。
        description: 進捗の説明テキスト。
        result: ``work`` の戻り値（成功時のみ）。
        error: 例外メッセージ（失敗時のみ）。握り潰さず保持する。
        created_at: 生成時刻（``time.monotonic``）。
        updated_at: 最終更新時刻（``time.monotonic``）。
    """

    job_id: str
    status: str = "running"
    fraction: float = 0.0
    description: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)


class JobManager:
    """スレッドセーフなインプロセス・ジョブレジストリ。

    Args:
        max_jobs: 保持する最大ジョブ数。超過時は完了/エラー済みの古いものから退避する。
    """

    def __init__(self, *, max_jobs: int = 64) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._max_jobs = max(1, int(max_jobs))

    def submit(self, work: JobWork) -> str:
        """``work`` をデーモンスレッドで実行し、即座に ``job_id`` を返す。

        ``work`` には ``report(fraction, description)`` コールバックが渡される。例外は
        握り潰さず ``JobState.error`` に保持し、成功時は戻り値を ``JobState.result`` に格納する。

        Args:
            work: 進捗報告コールバックを受け取り結果を返す処理本体。

        Returns:
            生成したジョブの識別子。
        """
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobState(job_id=job_id)
            self._evict_if_needed_locked()

        def _report(fraction: float, description: str) -> None:
            with self._lock:
                state = self._jobs.get(job_id)
                if state is None or state.status != "running":
                    return
                state.fraction = float(fraction)
                state.description = str(description)
                state.updated_at = time.monotonic()

        def _runner() -> None:
            try:
                result = work(_report)
            except Exception as exc:  # noqa: BLE001 - 握り潰さず error に保持し poll で通知
                with self._lock:
                    state = self._jobs.get(job_id)
                    if state is not None:
                        state.status = "error"
                        state.error = f"{type(exc).__name__}: {exc}"
                        state.updated_at = time.monotonic()
                return
            with self._lock:
                state = self._jobs.get(job_id)
                if state is not None:
                    state.status = "done"
                    state.result = result
                    state.fraction = 1.0
                    state.updated_at = time.monotonic()

        thread = threading.Thread(target=_runner, name=f"job-{job_id}", daemon=True)
        thread.start()
        return job_id

    def snapshot(self, job_id: str) -> JobState | None:
        """``job_id`` の状態の独立コピーを返す。未知の場合は ``None``。

        返すのは ``dataclasses.replace`` による浅いコピーなので、呼び出し側が値を
        書き換えても内部状態には影響しない。
        """
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            return replace(state)

    def _evict_if_needed_locked(self) -> None:
        """容量超過時、完了/エラー済みの古いジョブから削除する（ロック保持前提）。"""
        if len(self._jobs) <= self._max_jobs:
            return
        finished = sorted(
            (s for s in self._jobs.values() if s.status != "running"),
            key=lambda s: s.updated_at,
        )
        for state in finished:
            if len(self._jobs) <= self._max_jobs:
                break
            self._jobs.pop(state.job_id, None)
