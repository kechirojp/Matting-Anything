import numpy as np
import pytest
from PIL import Image

from pipelines.components.common import (
    AlphaCompositor,
    BBoxFromMask,
    ImageNormalizer,
    MaskCandidateSelector,
    MaskDilator,
    MaskPreviewComposer,
    MaskUnion,
    ScribbleParser,
    build_mask_set,
    build_selected_mask,
    compose_alpha,
    compose_mask_preview,
    ensure_rgb_array,
    mask_to_bbox,
    mask_set_to_status,
    normalize_masks,
    select_candidate_masks,
    union_masks,
)


def test_ensure_rgb_array_uses_image_editor_background_and_drops_alpha() -> None:
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 0] = 10
    rgba[..., 3] = 255

    rgb = ensure_rgb_array({"background": rgba})

    assert rgb.shape == (2, 3, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[..., 0] == 10)


def test_image_normalizer_accepts_pil_image() -> None:
    component = ImageNormalizer()
    image = Image.new("RGBA", (4, 3), (1, 2, 3, 4))

    result = component.run(image)

    assert result["image"].shape == (3, 4, 3)


def test_mask_to_bbox_clips_padding() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:8] = True

    bbox = mask_to_bbox(mask, padding=5, image_shape=mask.shape)

    assert bbox == (0, 0, 12, 9)


def test_bbox_component_returns_none_for_empty_mask() -> None:
    component = BBoxFromMask()

    result = component.run(np.zeros((5, 5), dtype=bool))

    assert result["box"] is None


def test_mask_dilator_expands_single_pixel() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    component = MaskDilator()

    result = component.run(mask, kernel_size=3)

    assert result["mask"].sum() == 9


def test_compose_alpha_blends_background() -> None:
    image = np.full((1, 2, 3), 100, dtype=np.uint8)
    alpha = np.array([[1.0, 0.0]], dtype=np.float32)

    composite = compose_alpha(image, alpha, (0, 10, 20))

    assert composite[0, 0].tolist() == [100, 100, 100]
    assert composite[0, 1].tolist() == [0, 10, 20]


def test_alpha_compositor_returns_green_screen() -> None:
    component = AlphaCompositor()
    image = np.full((1, 1, 3), 200, dtype=np.uint8)
    alpha = np.zeros((1, 1), dtype=np.float32)

    result = component.run(image, alpha)

    assert result["green_screen"][0, 0].tolist() == [51, 255, 146]


def test_scribble_parser_raises_on_empty_layers() -> None:
    component = ScribbleParser()
    image = np.zeros((5, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        component.run({"background": image, "layers": []})


def test_scribble_parser_extracts_point_and_box() -> None:
    component = ScribbleParser()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    layer = np.zeros((10, 10, 4), dtype=np.uint8)
    layer[4:6, 5:7, 0] = 255

    result = component.run({"background": image, "layers": [layer]}, mode="point")

    assert result["points"].shape == (1, 2)
    assert result["box"].tolist() == [5.0, 4.0, 6.0, 5.0]
    assert result["mask"].sum() == 4


def test_normalize_masks_converts_2d_and_uint8_to_bool_batch() -> None:
    mask = np.array([[0, 255], [1, 0]], dtype=np.uint8)

    masks = normalize_masks(mask)

    assert masks.shape == (1, 2, 2)
    assert masks.dtype == bool
    assert masks[0, 0, 1]


def test_build_mask_set_fills_scores_labels_boxes_and_metadata() -> None:
    masks = np.zeros((2, 4, 5), dtype=bool)
    boxes = np.array([[0, 0, 2, 2], [1, 1, 4, 3]], dtype=np.float32)

    mask_set = build_mask_set(masks, scores=[0.2, 0.8], boxes=boxes, labels=["person", "drum"], source="sam2")

    assert mask_set["masks"].shape == (2, 4, 5)
    assert mask_set["scores"].tolist() == pytest.approx([0.2, 0.8])
    assert mask_set["labels"] == ["person", "drum"]
    assert mask_set["source"] == "sam2"
    assert mask_set["metadata"]["height"] == 4
    assert mask_set["metadata"]["width"] == 5


def test_select_candidate_masks_filters_by_indices_score_and_top_k() -> None:
    masks = np.zeros((3, 4, 4), dtype=bool)
    mask_set = build_mask_set(masks, scores=[0.4, 0.9, 0.7], labels=["low", "best", "mid"])

    selected = select_candidate_masks(mask_set, indices=[0, 1, 2], score_threshold=0.5, top_k=2)

    assert selected["scores"].tolist() == pytest.approx([0.9, 0.7])
    assert selected["labels"] == ["best", "mid"]
    assert selected["metadata"]["source_indices"] == [1, 2]


def test_union_masks_or_combines_candidates_and_filters_small_regions() -> None:
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 1:4, 1:4] = True
    masks[1, 6, 6] = True

    union = union_masks(masks, min_area=2)

    assert union.sum() == 9
    assert union[2, 2]
    assert not union[6, 6]


def test_build_selected_mask_records_source_indices() -> None:
    mask = np.zeros((4, 4), dtype=bool)

    selected = build_selected_mask(mask, source_indices=[0, 2], label="person+drum")

    assert selected["mask"].shape == (4, 4)
    assert selected["source_indices"] == [0, 2]
    assert selected["label"] == "person+drum"


def test_compose_mask_preview_draws_selected_and_union_overlays() -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 3:5, 3:5] = True
    union = masks.any(axis=0)

    preview = compose_mask_preview(image, masks, selected_indices=[1], union_mask=union)

    assert preview.shape == image.shape
    assert preview[1, 1].sum() > 0
    assert preview[3, 3].sum() > 0


def test_mask_set_to_status_summarizes_empty_and_non_empty_sets() -> None:
    empty_status = mask_set_to_status(build_mask_set(np.zeros((0, 3, 3), dtype=bool)))
    filled_status = mask_set_to_status(build_mask_set(np.zeros((2, 3, 3), dtype=bool), scores=[0.1, 0.9]))

    assert "0 masks" in empty_status
    assert "2 masks" in filled_status
    assert "best=1" in filled_status


def test_mask_candidate_selector_and_union_components_use_contract_dicts() -> None:
    masks = np.zeros((2, 6, 6), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 3:5, 3:5] = True
    mask_set = build_mask_set(masks, scores=[0.2, 0.8], labels=["a", "b"])

    selected = MaskCandidateSelector().run(mask_set, indices=[1])
    union = MaskUnion().run(selected["mask_set"])

    assert selected["selected_indices"] == [1]
    assert union["mask"].sum() == 4
    assert union["selected_mask"]["source_indices"] == [1]


def test_mask_preview_composer_component_accepts_mask_set_and_union() -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 2:4, 2:4] = True
    mask_set = build_mask_set(masks, scores=[0.7])

    result = MaskPreviewComposer().run(image, mask_set=mask_set, union_mask=masks[0])

    assert result["preview"].shape == image.shape