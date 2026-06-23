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
POINT_CHOICE_PREFIX = "pt#"
BOX_CHOICE_MANUAL = "box:manual"
BOX_CHOICE_UNION_PREFIX = "box:u#"


def empty_prompt_state() -> dict[str, object]:
    """SAM2 prompt の UI state を初期化する。"""
    return {
        "points": [],
        "labels": [],
        "box": None,
        "boxes": [],
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
        # 複合対象 union 用の複数 bbox は要素までディープコピーし共有参照を避ける。
        state["boxes"] = [list(single_box) for single_box in (state.get("boxes") or [])]
    return state


def build_prompt_selection_choices(prompt_state: dict | None) -> dict[str, list[str]]:
    """現在の prompt state から point / bbox 個別削除用の選択肢を作る。"""
    state = copy_prompt_state(prompt_state)
    points = state.get("points") or []
    labels = state.get("labels") or []
    point_choices: list[str] = []
    for index, point in enumerate(points):
        x_value, y_value = int(point[0]), int(point[1])
        label_value = int(labels[index]) if index < len(labels) else 1
        label_text = "positive" if label_value == 1 else "negative"
        point_choices.append(f"{POINT_CHOICE_PREFIX}{index + 1} {label_text} ({x_value},{y_value})")

    box_choices: list[str] = []
    if state.get("box") is not None:
        x1, y1, x2, y2 = [int(value) for value in state["box"]]
        box_choices.append(f"{BOX_CHOICE_MANUAL} [{x1},{y1},{x2},{y2}]")
    for index, single_box in enumerate(state.get("boxes") or []):
        x1, y1, x2, y2 = [int(value) for value in single_box]
        box_choices.append(f"{BOX_CHOICE_UNION_PREFIX}{index + 1} [{x1},{y1},{x2},{y2}]")
    return {"point_choices": point_choices, "box_choices": box_choices}


def remove_selected_points(prompt_state: dict | None, selected_labels: list[str] | None) -> dict[str, object]:
    """選択された point prompt（positive/negative）だけを state から削除する。"""
    state = copy_prompt_state(prompt_state)
    labels = list(selected_labels or [])
    if not labels:
        return state
    remove_indices: set[int] = set()
    for label in labels:
        text = str(label)
        if not text.startswith(POINT_CHOICE_PREFIX):
            continue
        suffix = text[len(POINT_CHOICE_PREFIX):]
        index_text = suffix.split(" ", 1)[0]
        try:
            remove_indices.add(max(int(index_text) - 1, 0))
        except ValueError:
            continue

    if not remove_indices:
        return state
    points = state.get("points") or []
    label_values = state.get("labels") or []
    kept_points: list[Any] = []
    kept_labels: list[int] = []
    for index, point in enumerate(points):
        if index in remove_indices:
            continue
        kept_points.append(point)
        kept_labels.append(int(label_values[index]) if index < len(label_values) else 1)
    state["points"] = kept_points
    state["labels"] = kept_labels
    return state


def remove_selected_boxes(prompt_state: dict | None, selected_labels: list[str] | None) -> dict[str, object]:
    """選択された bbox（manual / union）だけを state から削除する。"""
    state = copy_prompt_state(prompt_state)
    labels = list(selected_labels or [])
    if not labels:
        return state

    remove_manual = any(str(label).startswith(BOX_CHOICE_MANUAL) for label in labels)
    remove_union_indices: set[int] = set()
    for label in labels:
        text = str(label)
        if not text.startswith(BOX_CHOICE_UNION_PREFIX):
            continue
        suffix = text[len(BOX_CHOICE_UNION_PREFIX):]
        index_text = suffix.split(" ", 1)[0]
        try:
            remove_union_indices.add(max(int(index_text) - 1, 0))
        except ValueError:
            continue

    if remove_manual:
        state["box"] = None
        state["box_buffer"] = []

    if remove_union_indices:
        kept_boxes = [
            list(single_box)
            for index, single_box in enumerate(state.get("boxes") or [])
            if index not in remove_union_indices
        ]
        state["boxes"] = kept_boxes
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
    # 複合対象 union 用の複数 bbox を色分け・番号付きで描画する。
    multi_box_colors = ((255, 99, 71), (60, 179, 113), (65, 105, 225), (238, 130, 238), (255, 165, 0))
    for box_index, single_box in enumerate(state.get("boxes") or []):
        x_min, y_min, x_max, y_max = [int(value) for value in single_box]
        color = multi_box_colors[box_index % len(multi_box_colors)]
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image_rgb, str(box_index + 1), (x_min + 2, max(y_min + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
