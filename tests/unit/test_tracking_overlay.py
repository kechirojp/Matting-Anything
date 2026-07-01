"""トラッキング可視化（render_tracking_overlay_frame / TrackingOverlayWriter）の unit test。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _make_masks(mask: np.ndarray, frame_count: int, metadata: dict | None = None) -> dict:
    """FrameMaskSequence 契約 dict を組み立てるテストヘルパー。"""
    frame_masks = {index: mask for index in range(frame_count)}
    return {
        "frame_masks": frame_masks,
        "object_ids": [1],
        "frame_indices": list(range(frame_count)),
        "source": "sam2_video",
        "metadata": dict(metadata or {}),
    }


def test_render_tracking_overlay_frame_highlights_mask_region() -> None:
    """mask 内には色が乗り、mask 外と元 frame は破壊されない。"""
    from pipelines.components.common import render_tracking_overlay_frame

    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:20] = True

    overlay = render_tracking_overlay_frame(frame, mask, color=(255, 0, 0), fill_alpha=0.5)

    assert overlay.shape == frame.shape
    assert overlay.dtype == np.uint8
    assert overlay[10, 14].sum() > 0  # mask 内は着色される
    assert overlay[0, 0].sum() == 0  # mask 外は変化しない
    assert frame.sum() == 0  # 入力 frame を破壊しない


def test_render_tracking_overlay_frame_resizes_mask_mismatch() -> None:
    """mask 解像度が frame と異なっても frame サイズの overlay を返す。"""
    from pipelines.components.common import render_tracking_overlay_frame

    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.ones((10, 15), dtype=bool)

    overlay = render_tracking_overlay_frame(frame, mask, color=(0, 255, 0), fill_alpha=0.5)

    assert overlay.shape == frame.shape


def test_tracking_overlay_writer_streams_without_retaining(tmp_path, monkeypatch) -> None:
    """overlay 動画を逐次書き出し、全 frame list を RAM に保持しない（ERR030）。"""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from pipelines.components.video_model_components import TrackingOverlayWriter

    writer = TrackingOverlayWriter(output_dir="outputs")
    frames = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(3)]
    mask = np.zeros((16, 24), dtype=bool)
    mask[4:12, 6:18] = True
    masks = _make_masks(mask, 3, {"tracker_config": "configs/samurai/sam2.1_hiera_l.yaml", "samurai_mode": True})

    events: list[tuple[str, float, str]] = []
    result = writer.run(
        frames=frames,
        masks=masks,
        metadata={},
        enabled=True,
        progress_callback=lambda stage, fraction, desc: events.append((stage, fraction, desc)),
    )

    overlay = result["overlay"]
    assert overlay["overlay_video_path"] is not None
    assert Path(overlay["overlay_video_path"]).exists()
    assert "frames" not in overlay  # 全 frame を返さない
    assert overlay["frame_count"] == 3
    assert any(stage == "tracking_overlay" for stage, _, _ in events)


def test_tracking_overlay_writer_disabled_returns_none(tmp_path, monkeypatch) -> None:
    """チェックボックス OFF 時は overlay を生成せず None を返す。"""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from pipelines.components.video_model_components import TrackingOverlayWriter

    writer = TrackingOverlayWriter(output_dir="outputs")
    frames = [np.zeros((16, 24, 3), dtype=np.uint8)]
    masks = _make_masks(np.zeros((16, 24), dtype=bool), 1)

    result = writer.run(frames=frames, masks=masks, enabled=False)

    assert result["overlay"]["overlay_video_path"] is None


def test_tracking_overlay_writer_has_progress_callback() -> None:
    """長時間処理に備え overlay writer も progress_callback を受ける。"""
    import inspect

    from pipelines.components.video_model_components import TrackingOverlayWriter

    assert "progress_callback" in inspect.signature(TrackingOverlayWriter.run).parameters


def test_resolve_run_timestamp_prefers_metadata() -> None:
    """metadata に run_timestamp があれば共有値を返し、無ければ現在時刻を生成する。"""
    from pipelines.components.video_common import resolve_run_timestamp

    shared = resolve_run_timestamp({"metadata": {"run_timestamp": "20991231_235959"}})
    assert shared == "20991231_235959"

    generated = resolve_run_timestamp({})
    assert len(generated) == len("YYYYmmdd_HHMMSS")
    assert generated[8] == "_"


def test_tracking_overlay_uses_shared_run_timestamp(tmp_path, monkeypatch) -> None:
    """overlay は metadata の run_timestamp と同じ outputs/<ts>/ 配下へ書き出す（フォルダ分裂防止）。"""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from pipelines.components.video_model_components import TrackingOverlayWriter

    writer = TrackingOverlayWriter(output_dir="outputs")
    frames = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(2)]
    masks = _make_masks(np.zeros((16, 24), dtype=bool), 2)
    metadata = {"metadata": {"run_timestamp": "20260701_190710", "sampled_frame_indices": [0, 1]}}

    result = writer.run(frames=frames, masks=masks, metadata=metadata, enabled=True, output_mode="動画 (video)")

    overlay_video_path = result["overlay"]["overlay_video_path"]
    assert overlay_video_path is not None
    assert "20260701_190710" in Path(overlay_video_path).as_posix()


def test_tracking_overlay_respects_output_mode_video(tmp_path, monkeypatch) -> None:
    """動画モードでは overlay 動画のみ生成し、連番 PNG は書かない。"""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from pipelines.components.video_model_components import TrackingOverlayWriter

    writer = TrackingOverlayWriter(output_dir="outputs")
    frames = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(2)]
    masks = _make_masks(np.zeros((16, 24), dtype=bool), 2)
    metadata = {"metadata": {"run_timestamp": "20260701_000000", "sampled_frame_indices": [0, 1]}}

    overlay = writer.run(frames=frames, masks=masks, metadata=metadata, enabled=True, output_mode="video")["overlay"]

    assert overlay["overlay_video_path"] is not None
    assert Path(overlay["overlay_video_path"]).exists()
    assert overlay["overlay_sequence_dir"] is None


def test_tracking_overlay_respects_output_mode_sequence(tmp_path, monkeypatch) -> None:
    """連番モードでは overlay PNG 連番のみ生成し、overlay 動画は書かない。"""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from pipelines.components.video_model_components import TrackingOverlayWriter

    writer = TrackingOverlayWriter(output_dir="outputs")
    frames = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(2)]
    masks = _make_masks(np.zeros((16, 24), dtype=bool), 2)
    metadata = {"metadata": {"run_timestamp": "20260701_111111", "sampled_frame_indices": [0, 1]}}

    overlay = writer.run(frames=frames, masks=masks, metadata=metadata, enabled=True, output_mode="連番静止画 (sequence)")["overlay"]

    assert overlay["overlay_video_path"] is None
    assert overlay["overlay_sequence_dir"] is not None
    assert Path(overlay["overlay_sequence_dir"], "frame_000000.png").exists()


def test_tracking_overlay_respects_output_mode_both(tmp_path, monkeypatch) -> None:
    """両方モードでは overlay 動画と PNG 連番の双方を同一 <ts>/ へ生成する。"""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from pipelines.components.video_model_components import TrackingOverlayWriter

    writer = TrackingOverlayWriter(output_dir="outputs")
    frames = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(2)]
    masks = _make_masks(np.zeros((16, 24), dtype=bool), 2)
    metadata = {"metadata": {"run_timestamp": "20260701_222222", "sampled_frame_indices": [0, 1]}}

    overlay = writer.run(frames=frames, masks=masks, metadata=metadata, enabled=True, output_mode="both")["overlay"]

    assert overlay["overlay_video_path"] is not None
    assert overlay["overlay_sequence_dir"] is not None
    video_root = Path(overlay["overlay_video_path"]).parents[1]
    sequence_root = Path(overlay["overlay_sequence_dir"]).parents[1]
    assert video_root == sequence_root  # 同一 <ts>/ 配下

