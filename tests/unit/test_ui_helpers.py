from __future__ import annotations

import numpy as np

from pipelines.components.ui_helpers import (
    clamp_prompt_point,
    draw_prompt_overlay,
    extend_box_to_edge,
    normalize_box_from_points,
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
