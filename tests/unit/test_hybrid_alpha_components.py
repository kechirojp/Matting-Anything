from __future__ import annotations

import numpy as np
import pytest

from pipelines.components.hybrid_alpha_components import (
    BEN2TransparentHybridVideoExtractor,
    build_person_region,
    compose_hybrid_alpha,
    normalize_alpha_float,
)


def test_compose_hybrid_alpha_lighten_uses_max_inside_and_ben2_outside() -> None:
    ben2 = np.array([[200, 200], [200, 200]], dtype=np.uint8)
    tb = np.array([[50, 50], [50, 50]], dtype=np.uint8)
    region = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    out = compose_hybrid_alpha(ben2, tb, region, mode="lighten")

    np.testing.assert_array_equal(out, np.array([[200, 200], [200, 200]], dtype=np.uint8))


def test_compose_hybrid_alpha_person_over_ben2_keeps_feather_overlap() -> None:
    ben2 = np.full((1, 1), 128, dtype=np.uint8)
    tb = np.full((1, 1), 128, dtype=np.uint8)
    region = np.full((1, 1), 0.5, dtype=np.float32)

    out = compose_hybrid_alpha(ben2, tb, region, mode="person_over_ben2")

    # Spatial blend between BEN2 (0.5) and person-over-BEN2 composition (~0.75).
    assert out[0, 0] in {159, 160}


def test_compose_hybrid_alpha_does_not_create_transparent_halo_when_both_sources_opaque() -> None:
    alpha = np.full((1, 1), 255, dtype=np.uint8)
    region = np.full((1, 1), 0.5, dtype=np.float32)

    out = compose_hybrid_alpha(alpha, alpha, region, mode="lighten")

    assert out[0, 0] == 255


def test_compose_hybrid_alpha_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        compose_hybrid_alpha(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float32),
            mode="multiply",
        )


def test_build_person_region_resizes_and_feathers_mask() -> None:
    mask = np.zeros((2, 2), dtype=np.float32)
    mask[0, 0] = 1.0

    region = build_person_region(mask, (6, 6), dilate_px=1, feather_px=2)

    assert region.shape == (6, 6)
    assert region.dtype == np.float32
    assert 0.0 <= float(region.min()) <= float(region.max()) <= 1.0
    assert region.max() > 0.0


def test_normalize_alpha_float_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        normalize_alpha_float(np.zeros((2, 2, 1), dtype=np.uint8))


def test_hybrid_extractor_skips_tb_when_person_region_is_empty(tmp_path) -> None:
    class FakeBEN2:
        def warm_up(self) -> None:
            return None

        def infer_alpha(self, image_rgb, refine_foreground=False):
            return np.full(image_rgb.shape[:2], 200, dtype=np.uint8)

    class FakeTB:
        calls = 0

        def run(self, **kwargs):
            self.calls += 1
            return {
                "alpha": np.full(kwargs["image"].shape[:2], 50, dtype=np.uint8),
                "rgba": np.dstack([kwargs["image"], np.full(kwargs["image"].shape[:2], 50, dtype=np.uint8)]),
                "preview": kwargs["image"],
            }

    tb = FakeTB()
    extractor = BEN2TransparentHybridVideoExtractor(
        ben2_extractor=FakeBEN2(),
        tb_extractor=tb,
        output_dir=str(tmp_path),
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    masks = {"frame_masks": {0: np.zeros((4, 4), dtype=np.float32)}}

    result = extractor.run(frames=[frame], masks=masks, metadata={}, output_mode="sequence")

    assert tb.calls == 0
    assert result["matte"]["frame_count"] == 1


class _FakeBEN2:
    def warm_up(self) -> None:
        return None

    def infer_alpha(self, image_rgb, refine_foreground=False):
        return np.full(image_rgb.shape[:2], 200, dtype=np.uint8)


class _FakeTB:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        alpha = np.full(kwargs["image"].shape[:2], 50, dtype=np.uint8)
        return {"alpha": alpha, "rgba": np.dstack([kwargs["image"], alpha]), "preview": kwargs["image"]}


def test_hybrid_extractor_warns_on_source_index_mismatch(tmp_path, recwarn) -> None:
    """frame_masks はあるが source_index が一致せず 1 度も引けない場合、日本語警告を出す。"""
    tb = _FakeTB()
    extractor = BEN2TransparentHybridVideoExtractor(
        ben2_extractor=_FakeBEN2(), tb_extractor=tb, output_dir=str(tmp_path)
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    # local_index 0 は source_index 100 を引くが、frame_masks のキーは 999 なのでヒットしない。
    masks = {"frame_masks": {999: np.ones((4, 4), dtype=np.float32)}}
    metadata = {"metadata": {"sampled_frame_indices": [100]}}

    result = extractor.run(frames=[frame], masks=masks, metadata=metadata, output_mode="sequence")

    meta = result["matte"]["metadata"]
    assert tb.calls == 0
    assert meta["person_mask_hit_count"] == 0
    assert meta["person_mask_available_frames"] == 1
    assert meta["person_mask_fallback_warning"] is not None
    assert "写像ズレ" in meta["person_mask_fallback_warning"]
    assert any("写像ズレ" in str(w.message) for w in recwarn.list)


def test_hybrid_extractor_warns_when_no_person_mask_available(tmp_path, recwarn) -> None:
    """frame_masks が空（検出0件）の場合、BEN2 単独フォールバックの日本語警告を出す。"""
    tb = _FakeTB()
    extractor = BEN2TransparentHybridVideoExtractor(
        ben2_extractor=_FakeBEN2(), tb_extractor=tb, output_dir=str(tmp_path)
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    result = extractor.run(frames=[frame], masks={"frame_masks": {}}, metadata={}, output_mode="sequence")

    meta = result["matte"]["metadata"]
    assert tb.calls == 0
    assert meta["person_mask_hit_count"] == 0
    assert meta["person_mask_available_frames"] == 0
    assert meta["person_mask_fallback_warning"] is not None
    assert "1 件も得られませんでした" in meta["person_mask_fallback_warning"]


def test_hybrid_extractor_no_warning_when_person_mask_hits(tmp_path, recwarn) -> None:
    """source_index が一致して人物mask を引けた場合は警告を出さない。"""
    tb = _FakeTB()
    extractor = BEN2TransparentHybridVideoExtractor(
        ben2_extractor=_FakeBEN2(), tb_extractor=tb, output_dir=str(tmp_path)
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    masks = {"frame_masks": {0: np.ones((4, 4), dtype=np.float32)}}

    result = extractor.run(frames=[frame], masks=masks, metadata={}, output_mode="sequence")

    meta = result["matte"]["metadata"]
    assert tb.calls == 1
    assert meta["person_mask_hit_count"] == 1
    assert meta["person_mask_fallback_warning"] is None
    assert not any(isinstance(w.message, RuntimeWarning) for w in recwarn.list)

