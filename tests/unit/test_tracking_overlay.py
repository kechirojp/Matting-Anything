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
