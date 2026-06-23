"""`_ProgressKeepAlive`（SSE idle 切断対策・ERR048）の単体テスト。

frame 数ベースの間引きでは、低速 frame で通知間隔が数十秒に広がり Colab/gradio.live の
共有トンネルが event SSE を idle 切断してしまう。本テストは、時間ベース keep-alive が
最初/最後の frame に加えて経過時間で必ず進捗を流すことを検証する。
"""

from __future__ import annotations

from pipelines.components.video_model_components import _ProgressKeepAlive


class _FakeClock:
    """注入可能な単調増加クロック。`advance` で任意秒だけ進める。"""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _record_callback(records: list[tuple[str, float, str]]):
    def _callback(stage: str, fraction: float, description: str) -> None:
        records.append((stage, fraction, description))

    return _callback


def test_keepalive_emits_on_first_and_last_frame() -> None:
    """最初と最後の frame は経過時間に関わらず必ず流す。"""
    records: list[tuple[str, float, str]] = []
    clock = _FakeClock()
    throttle = _ProgressKeepAlive(_record_callback(records), "sam2_video", min_interval_sec=2.0, clock=clock)

    throttle.maybe(0, 5, 0.0, "first")  # 最初の frame
    throttle.maybe(4, 5, 1.0, "last")  # 最後の frame（index+1 >= total）

    assert [rec[2] for rec in records] == ["first", "last"]


def test_keepalive_emits_after_interval_even_for_non_boundary_frame() -> None:
    """境界でない frame でも min_interval を超えれば流す（frame 数ベースでは落ちる回帰）。

    旧実装（`index % 10 == 0` 等）は index=3 では通知しないが、10s/frame の低速ループでは
    通知間隔が広がり SSE が切れる。時間ベースなら index=3 でも >2s 経過で流れる。
    """
    records: list[tuple[str, float, str]] = []
    clock = _FakeClock()
    throttle = _ProgressKeepAlive(_record_callback(records), "sam2_video", min_interval_sec=2.0, clock=clock)

    # index=1,2,3 を 10s 間隔で処理（いずれも境界でも 10 の倍数でもない）。
    for idx in (1, 2, 3):
        clock.advance(10.0)
        throttle.maybe(idx, 100, idx / 100, f"frame {idx}")

    # 旧 frame 数ベースなら 0 件。時間ベースなら毎回（>2s 経過）流れる。
    assert [rec[2] for rec in records] == ["frame 1", "frame 2", "frame 3"]


def test_keepalive_suppresses_emit_within_interval() -> None:
    """min_interval 未満の連続 frame は境界以外で抑制し、過剰な SSE を避ける。"""
    records: list[tuple[str, float, str]] = []
    clock = _FakeClock()
    throttle = _ProgressKeepAlive(_record_callback(records), "transparent_bg", min_interval_sec=2.0, clock=clock)

    throttle.maybe(0, 100, 0.0, "boundary-first")  # 境界: 流れる
    clock.advance(0.5)
    throttle.maybe(1, 100, 0.01, "too-soon-1")  # 0.5s: 抑制
    clock.advance(0.5)
    throttle.maybe(2, 100, 0.02, "too-soon-2")  # 累計1.0s: 抑制

    assert [rec[2] for rec in records] == ["boundary-first"]


def test_keepalive_force_always_emits() -> None:
    """force=True は間隔に関わらず必ず流す。"""
    records: list[tuple[str, float, str]] = []
    clock = _FakeClock()
    throttle = _ProgressKeepAlive(_record_callback(records), "sam2_video", min_interval_sec=2.0, clock=clock)

    throttle.maybe(0, 100, 0.0, "first")  # 境界
    clock.advance(0.1)
    throttle.maybe(1, 100, 0.01, "forced", force=True)

    assert [rec[2] for rec in records] == ["first", "forced"]


def test_keepalive_noop_without_callback() -> None:
    """progress_callback が None でも例外を出さず no-op になる。"""
    throttle = _ProgressKeepAlive(None, "sam2_video", min_interval_sec=2.0, clock=_FakeClock())
    throttle.maybe(0, 5, 0.0, "noop")  # 例外が出ないことを確認
