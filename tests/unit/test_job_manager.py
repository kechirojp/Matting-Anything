"""``pipelines.job_manager.JobManager`` の単体テスト（ERR058）。

非同期ジョブ基盤は GPU / モデルに依存しない純 Python なので、ここで挙動を固定する。
"""

from __future__ import annotations

import threading
import time

import pytest

from pipelines.job_manager import JobManager, JobState


def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.005) -> bool:
    """``predicate()`` が真を返すまで最大 ``timeout`` 秒ポーリングする。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_submit_runs_work_and_reports_progress() -> None:
    manager = JobManager()
    seen: list[float] = []

    def work(report):
        report(0.5, "途中")
        seen.append(0.5)
        return ("done-value",)

    job_id = manager.submit(work)
    assert isinstance(job_id, str) and job_id

    assert _wait_until(lambda: manager.snapshot(job_id).status == "done")
    state = manager.snapshot(job_id)
    assert state.status == "done"
    assert state.result == ("done-value",)
    assert state.fraction == pytest.approx(1.0)
    assert seen == [0.5]


def test_progress_report_updates_running_state() -> None:
    manager = JobManager()
    gate = threading.Event()

    def work(report):
        report(0.25, "running stage")
        gate.wait(timeout=2.0)
        return "ok"

    job_id = manager.submit(work)
    assert _wait_until(lambda: manager.snapshot(job_id).fraction == pytest.approx(0.25))
    running = manager.snapshot(job_id)
    assert running.status == "running"
    assert running.description == "running stage"
    gate.set()
    assert _wait_until(lambda: manager.snapshot(job_id).status == "done")


def test_exception_is_captured_not_swallowed() -> None:
    manager = JobManager()

    def work(report):
        raise ValueError("boom")

    job_id = manager.submit(work)
    assert _wait_until(lambda: manager.snapshot(job_id).status == "error")
    state = manager.snapshot(job_id)
    assert state.status == "error"
    assert state.error is not None
    assert "boom" in state.error
    assert "ValueError" in state.error
    assert state.result is None


def test_snapshot_is_isolated_copy() -> None:
    manager = JobManager()
    done = threading.Event()

    def work(report):
        report(0.5, "x")
        done.wait(timeout=2.0)
        return "v"

    job_id = manager.submit(work)
    assert _wait_until(lambda: manager.snapshot(job_id) is not None)
    snap = manager.snapshot(job_id)
    snap.fraction = 0.999
    snap.description = "mutated"
    assert manager.snapshot(job_id).fraction == pytest.approx(0.5)
    assert manager.snapshot(job_id).description == "x"
    done.set()


def test_snapshot_unknown_job_returns_none() -> None:
    manager = JobManager()
    assert manager.snapshot("does-not-exist") is None


def test_job_state_defaults_to_running() -> None:
    state = JobState(job_id="abc")
    assert state.status == "running"
    assert state.fraction == 0.0
    assert state.error is None
    assert state.result is None


def test_finished_jobs_are_evicted_when_over_capacity() -> None:
    manager = JobManager(max_jobs=2)
    ids = []
    for _ in range(5):
        ids.append(manager.submit(lambda report: "ok"))
    assert _wait_until(
        lambda: all(
            manager.snapshot(j) is None or manager.snapshot(j).status == "done" for j in ids
        )
    )
    alive = [j for j in ids if manager.snapshot(j) is not None]
    assert len(alive) <= 2
