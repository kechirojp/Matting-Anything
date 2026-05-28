"""SAM2 prompt UI で共有する純粋ヘルパー。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import gradio as gr
import numpy as np

from .common import ensure_rgb_array


EDGE_SNAP_PIXELS = 16
EDGE_SIDES = ("left", "right", "top", "bottom")
_CURRENT_OVERLAY_MASK_PROVIDER: Callable[[], np.ndarray | None] | None = None


def empty_prompt_state() -> dict[str, object]:
    """SAM2 prompt の UI state を初期化する。"""
    return {
        "points": [],
        "labels": [],
        "box": None,
        "box_buffer": [],
        "input_mode": "point",
    }


def copy_prompt_state(prompt_state: dict | None) -> dict[str, object]:
    """Gradio State の共有参照を避けるため prompt state をコピーする。"""
    state = empty_prompt_state()
    if prompt_state:
        state.update(prompt_state)
        state["points"] = list(state.get("points") or [])
        state["labels"] = list(state.get("labels") or [])
        state["box_buffer"] = list(state.get("box_buffer") or [])
        state["box"] = list(state["box"]) if state.get("box") is not None else None
    return state


def set_prompt_overlay_mask_provider(provider: Callable[[], np.ndarray | None] | None) -> None:
    """prompt 更新時に重ねる現在の mask provider を登録する。"""
    global _CURRENT_OVERLAY_MASK_PROVIDER
    _CURRENT_OVERLAY_MASK_PROVIDER = provider


def current_prompt_overlay_mask() -> np.ndarray | None:
    """登録済み provider から現在の overlay mask を取得する。"""
    if _CURRENT_OVERLAY_MASK_PROVIDER is None:
        return None
    return _CURRENT_OVERLAY_MASK_PROVIDER()


def clamp_prompt_point(
    point: tuple[int, int] | list[int],
    image_shape: tuple[int, ...],
    edge_snap: int = EDGE_SNAP_PIXELS,
) -> tuple[int, int]:
    """クリック座標を画像内に収め、端付近は画像端へ吸着する。"""
    image_height, image_width = image_shape[:2]
    x_value = int(round(float(point[0])))
    y_value = int(round(float(point[1])))
    x_value = min(max(x_value, 0), image_width - 1)
    y_value = min(max(y_value, 0), image_height - 1)
    if x_value <= edge_snap:
        x_value = 0
    elif x_value >= image_width - 1 - edge_snap:
        x_value = image_width - 1
    if y_value <= edge_snap:
        y_value = 0
    elif y_value >= image_height - 1 - edge_snap:
        y_value = image_height - 1
    return x_value, y_value


def normalize_box_from_points(
    first_point: tuple[int, int] | list[int],
    second_point: tuple[int, int] | list[int],
    image_shape: tuple[int, ...],
) -> list[int]:
    """2クリックから順序に依存しない SAM2 bbox を作る。"""
    x_first, y_first = clamp_prompt_point(first_point, image_shape)
    x_second, y_second = clamp_prompt_point(second_point, image_shape)
    return [min(x_first, x_second), min(y_first, y_second), max(x_first, x_second), max(y_first, y_second)]


def draw_prompt_overlay(input_image: Any, prompt_state: dict | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """SAM2 prompt の点・bbox・マスクを画像に重ねる。"""
    image_rgb = ensure_rgb_array(input_image).copy()
    state = copy_prompt_state(prompt_state)
    if mask is not None:
        mask_bool = mask.astype(bool)
        overlay_color = np.array([30, 144, 255], dtype=np.uint8)
        image_rgb[mask_bool] = (image_rgb[mask_bool] * 0.5 + overlay_color * 0.5).astype(np.uint8)
    if state.get("box") is not None:
        x_min, y_min, x_max, y_max = [int(value) for value in state["box"]]
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 215, 0), 3)
    for point, label in zip(state.get("points") or [], state.get("labels") or []):
        x_value, y_value = [int(value) for value in point]
        color = (0, 255, 0) if int(label) == 1 else (255, 0, 0)
        cv2.circle(image_rgb, (x_value, y_value), 7, color, -1)
        cv2.circle(image_rgb, (x_value, y_value), 9, (255, 255, 255), 2)
    for point in state.get("box_buffer") or []:
        x_value, y_value = [int(value) for value in point]
        cv2.circle(image_rgb, (x_value, y_value), 7, (255, 215, 0), -1)
        cv2.circle(image_rgb, (x_value, y_value), 9, (255, 255, 255), 2)
    return image_rgb


def select_sam2_prompt(input_image: Any, prompt_mode: str, point_label: str | bool, prompt_state: dict | None, evt: gr.SelectData):
    """画像クリックで SAM2 point / bbox prompt を蓄積する。"""
    if input_image is None:
        raise gr.Error("先に画像をアップロードしてください。")
    image_rgb = ensure_rgb_array(input_image)
    state = copy_prompt_state(prompt_state)
    if state.get("input_mode") != prompt_mode:
        state["box_buffer"] = []
    state["input_mode"] = prompt_mode
    clicked_point = clamp_prompt_point(evt.index, image_rgb.shape)

    # point prompt は positive / negative を label 1 / 0 として SAM2 に渡す。
    if isinstance(point_label, str):
        is_positive = point_label.lower() == "positive"
    else:
        is_positive = bool(point_label)

    if prompt_mode == "point":
        state["points"].append(clicked_point)
        state["labels"].append(1 if is_positive else 0)
        status = f"Point selected: {clicked_point}, label={'positive' if is_positive else 'negative'}"
    else:
        state["box_buffer"].append(clicked_point)
        if len(state["box_buffer"]) >= 2:
            state["box"] = normalize_box_from_points(state["box_buffer"][0], state["box_buffer"][1], image_rgb.shape)
            state["box_buffer"] = []
            status = f"Box selected: {state['box']}"
        else:
            status = f"Box first corner selected: {clicked_point}"
    return draw_prompt_overlay(image_rgb, state, current_prompt_overlay_mask()), state, status


def extend_box_to_edge(input_image: Any, prompt_state: dict | None, side: str):
    """確定済み bbox の指定辺を画像端へ延長する。"""
    if input_image is None:
        raise gr.Error("先に画像をアップロードしてください。")
    if side not in EDGE_SIDES:
        raise gr.Error(f"未知のエッジ指定です: {side}")
    image_rgb = ensure_rgb_array(input_image)
    image_height, image_width = image_rgb.shape[:2]
    state = copy_prompt_state(prompt_state)
    if state.get("box") is None:
        raise gr.Error("先に画像を2回クリックして bbox を作成してから端を延長してください。")
    x_min, y_min, x_max, y_max = [int(value) for value in state["box"]]
    if side == "left":
        x_min = 0
    elif side == "right":
        x_max = image_width - 1
    elif side == "top":
        y_min = 0
    elif side == "bottom":
        y_max = image_height - 1
    state["box"] = [x_min, y_min, x_max, y_max]
    preview = draw_prompt_overlay(image_rgb, state, current_prompt_overlay_mask())
    return preview, state, f"Box {side} edge extended -> {state['box']}"
