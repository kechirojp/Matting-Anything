"""RouteA 動画アプリの非同期ジョブ配線テスト（ERR058）。

重いパイプライン関数は monkeypatch でスタブ化し、``start_*_job`` / ``poll_*_job`` の
契約（job_id 返却・進捗テキスト・完了出力・エラー通知）のみ検証する。GPU/torch は
``--help`` smoke と同様にモジュール import 経由で必要だが、推論自体は走らせない。
"""
from __future__ import annotations

import importlib
import time

import gradio as gr
import pytest

app = importlib.import_module("gradio_app_sam2_ben2_route_a_for_Movie")


def _wait_status(job_id: str, status: str, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app._JOB_MANAGER.snapshot(job_id).status == status:
            return True
        time.sleep(0.005)
    return False


def _valid_prompt_state() -> dict:
    state = app.empty_prompt_state()
    state["points"] = [[10, 20]]
    state["labels"] = [1]
    return state


_ROUTE_A_ARGS = (
    "dummy.mp4",            # video_path
    None,                   # prompt_state (overridden per test)
    0,                      # prompt_frame_idx
    False,                  # bidirectional
    30,                     # max_frames
    1,                      # frame_step
    "動画 (video)",          # output_mode_label
    "webm_vp9",             # rgba_codec
    "sam2_hiera_l",         # tracker_model
    "union",                # matte_mode_label
    24, 41, 0.0, 12,        # dilation_px, blur_kernel, blur_sigma, feather_px
    False, False,           # refine_foreground, gate_alpha
    "none",                 # mask_floor_mode_label
    "rgba",                 # output_type
    True,                   # overlay_enabled
)


def _route_a_args(prompt_state: dict) -> tuple:
    args = list(_ROUTE_A_ARGS)
    args[1] = prompt_state
    return tuple(args)


def test_start_route_a_job_requires_video() -> None:
    args = _route_a_args(_valid_prompt_state())
    args = (None,) + args[1:]
    with pytest.raises(gr.Error):
        app.start_route_a_job(*args)


def test_start_route_a_job_requires_prompt() -> None:
    args = _route_a_args(app.empty_prompt_state())
    with pytest.raises(gr.Error):
        app.start_route_a_job(*args)


def test_start_and_poll_route_a_completes(monkeypatch) -> None:
    sentinel = ("rgba.mp4", "alpha.mp4", "preview.mp4", "overlay.mp4", ["f.png"], "dir", "done status")

    def _stub(*_args, progress=None):
        if progress is not None:
            progress(0.5, "halfway")
        return sentinel

    monkeypatch.setattr(app, "run_route_a_background_removal", _stub)

    outputs = app.start_route_a_job(*_route_a_args(_valid_prompt_state()))
    job_id = outputs[0]
    assert isinstance(job_id, str) and job_id

    assert _wait_status(job_id, "done")
    polled = app.poll_route_a_job(job_id)
    assert polled[:7] == sentinel


def test_poll_route_a_error_surfaces_then_resets(monkeypatch) -> None:
    def _stub(*_args, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "run_route_a_background_removal", _stub)

    outputs = app.start_route_a_job(*_route_a_args(_valid_prompt_state()))
    job_id = outputs[0]
    assert _wait_status(job_id, "error")

    with pytest.raises(gr.Error):
        app.poll_route_a_job(job_id)

    # 2 回目以降は Timer 停止・ボタン復帰し、例外を再送出しない。
    reset = app.poll_route_a_job(job_id)
    assert "boom" in reset[6]


def test_poll_route_a_running_reports_progress_text(monkeypatch) -> None:
    import threading

    release = threading.Event()

    def _stub(*_args, progress=None):
        if progress is not None:
            progress(0.42, "中間")
        assert release.wait(timeout=2.0)
        return ("a", "b", "c", "d", [], "", "ok")

    monkeypatch.setattr(app, "run_route_a_background_removal", _stub)
    outputs = app.start_route_a_job(*_route_a_args(_valid_prompt_state()))
    job_id = outputs[0]
    try:
        assert _wait_status(job_id, "running")
        # 進捗が報告されるまで待つ
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and app._JOB_MANAGER.snapshot(job_id).fraction == 0.0:
            time.sleep(0.005)
        polled = app.poll_route_a_job(job_id)
        assert "42%" in polled[6]
        assert "中間" in polled[6]
    finally:
        release.set()
    assert _wait_status(job_id, "done")


def test_start_route_a_only_job_requires_video() -> None:
    with pytest.raises(gr.Error):
        app.start_route_a_only_job(None, 30, 1, "動画 (video)", "webm_vp9", False, "rgba")


def test_start_and_poll_route_a_only_completes(monkeypatch) -> None:
    sentinel = ("rgba.mp4", "alpha.mp4", "preview.mp4", ["f.png"], "dir", "done status")

    def _stub(*_args, progress=None):
        if progress is not None:
            progress(1.0, "done")
        return sentinel

    monkeypatch.setattr(app, "run_route_a_only_background_removal", _stub)
    outputs = app.start_route_a_only_job("dummy.mp4", 30, 1, "動画 (video)", "webm_vp9", False, "rgba")
    job_id = outputs[0]
    assert isinstance(job_id, str) and job_id
    assert _wait_status(job_id, "done")
    polled = app.poll_route_a_only_job(job_id)
    assert polled[:6] == sentinel
