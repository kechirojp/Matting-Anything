from __future__ import annotations

import numpy as np

from pipelines.components.ui_helpers import (
    build_prompt_selection_choices,
    clamp_prompt_point,
    copy_prompt_state,
    draw_prompt_overlay,
    empty_prompt_state,
    extend_box_to_edge,
    normalize_box_from_points,
    remove_selected_boxes,
    remove_selected_points,
    select_sam2_prompt,
)


class DummySelectEvent:
    def __init__(self, index):
        self.index = index


def test_clamp_prompt_point_snaps_to_edges() -> None:
    image_shape = (100, 200, 3)

    assert clamp_prompt_point((3, 4), image_shape) == (0, 0)
    assert clamp_prompt_point((196, 96), image_shape) == (199, 99)


def test_normalize_box_from_points_is_order_independent() -> None:
    image_shape = (100, 200, 3)

    assert normalize_box_from_points((80, 70), (20, 20), image_shape) == [20, 20, 80, 70]


def test_select_sam2_prompt_adds_positive_and_negative_points() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    preview, state, status = select_sam2_prompt(image, "point", "positive", None, DummySelectEvent((18, 18)))
    _, state, _ = select_sam2_prompt(image, "point", "negative", state, DummySelectEvent((20, 22)))

    assert preview.shape == image.shape
    assert state["points"] == [(18, 18), (20, 22)]
    assert state["labels"] == [1, 0]
    assert "positive" in status


def test_select_sam2_prompt_builds_box_after_two_clicks() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    _, state, _ = select_sam2_prompt(image, "box", "positive", None, DummySelectEvent((45, 45)))
    _, state, status = select_sam2_prompt(image, "box", "positive", state, DummySelectEvent((5, 6)))

    assert state["box"] == [0, 0, 45, 45]
    assert state["box_buffer"] == []
    assert "Box selected" in status


def test_extend_box_to_edge_updates_requested_side() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    state = {"points": [], "labels": [], "box": [10, 12, 40, 50], "box_buffer": [], "input_mode": "box"}

    _, state, status = extend_box_to_edge(image, state, "right")

    assert state["box"] == [10, 12, 63, 50]
    assert "right" in status


def test_draw_prompt_overlay_preserves_shape_and_dtype() -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=bool)
    mask[8:16, 10:20] = True
    state = {"points": [(12, 13)], "labels": [1], "box": [5, 6, 25, 26], "box_buffer": [], "input_mode": "point"}

    overlay = draw_prompt_overlay(image, state, mask)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_empty_prompt_state_has_boxes_list() -> None:
    state = empty_prompt_state()

    assert state["boxes"] == []


def test_copy_prompt_state_deep_copies_boxes() -> None:
    original = empty_prompt_state()
    original["boxes"] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    copied = copy_prompt_state(original)
    copied["boxes"].append([9, 9, 9, 9])
    copied["boxes"][0][0] = 99

    assert original["boxes"] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert copied["boxes"] == [[99, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]]


def test_draw_prompt_overlay_draws_multiple_boxes() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    state = empty_prompt_state()
    state["boxes"] = [[2, 3, 20, 25], [30, 5, 55, 35]]

    overlay = draw_prompt_overlay(image, state, None)

    assert overlay.shape == image.shape
    # 各 box の枠線が描画され、元の真っ黒画像から変化していること。
    assert overlay.sum() > 0


def test_build_prompt_selection_choices_includes_points_and_boxes() -> None:
    state = empty_prompt_state()
    state["points"] = [(10, 12), (20, 22)]
    state["labels"] = [1, 0]
    state["box"] = [1, 2, 30, 40]
    state["boxes"] = [[5, 6, 7, 8]]

    choices = build_prompt_selection_choices(state)

    assert choices["point_choices"][0].startswith("pt#1 positive")
    assert choices["point_choices"][1].startswith("pt#2 negative")
    assert choices["box_choices"][0].startswith("box:manual")
    assert choices["box_choices"][1].startswith("box:u#1")


def test_remove_selected_points_removes_only_target_indices() -> None:
    state = empty_prompt_state()
    state["points"] = [(10, 12), (20, 22), (30, 32)]
    state["labels"] = [1, 0, 1]

    updated = remove_selected_points(state, ["pt#2 negative (20,22)"])

    assert updated["points"] == [(10, 12), (30, 32)]
    assert updated["labels"] == [1, 1]


def test_remove_selected_boxes_handles_manual_and_union() -> None:
    state = empty_prompt_state()
    state["box"] = [1, 2, 3, 4]
    state["box_buffer"] = [(1, 1)]
    state["boxes"] = [[10, 10, 20, 20], [30, 30, 40, 40]]

    updated = remove_selected_boxes(state, ["box:manual [1,2,3,4]", "box:u#2 [30,30,40,40]"])

    assert updated["box"] is None
    assert updated["box_buffer"] == []
    assert updated["boxes"] == [[10, 10, 20, 20]]
