from __future__ import annotations

import numpy as np

from pipelines.components.video_common import (
    FrameSampler,
    build_frame_mask_sequence,
    build_video_source,
    frame_cache_bytes,
    normalize_output_mode,
    sample_frame_indices,
)


def test_normalize_output_mode_accepts_ui_labels() -> None:
    assert normalize_output_mode("動画 (video)") == "video"
    assert normalize_output_mode("連番静止画 (sequence)") == "sequence"
    assert normalize_output_mode("両方 (both)") == "both"


def test_sample_frame_indices_respects_step_and_max_frames() -> None:
    assert sample_frame_indices(frame_count=10, max_frames=3, frame_step=2) == [0, 2, 4]


def test_frame_sampler_component_returns_indices() -> None:
    result = FrameSampler().run(frame_count=8, max_frames=4, frame_step=2)

    assert result["frame_indices"] == [0, 2, 4, 6]


def test_build_video_source_contains_required_keys() -> None:
    source = build_video_source("input.mp4", fps=24.0, width=640, height=360, frame_count=12, codec="mp4v")

    assert set(source) == {"path", "fps", "width", "height", "frame_count", "codec", "metadata"}
    assert source["fps"] == 24.0


def test_frame_cache_bytes_uses_uint8_channel_count() -> None:
    assert frame_cache_bytes(10, 64, 32, 3) == 10 * 64 * 32 * 3


def test_build_frame_mask_sequence_uses_frame_masks_key() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    sequence = build_frame_mask_sequence({3: mask}, object_ids=[1])

    assert sequence["frame_indices"] == [3]
    assert sequence["object_ids"] == [1]
    assert sequence["frame_masks"][3].dtype == bool
