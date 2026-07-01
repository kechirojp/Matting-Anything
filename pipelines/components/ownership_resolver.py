"""OwnershipResolver: per-pixel soft ownership from per-object logits.

This component accepts a dict masks contract produced by SAM2VideoPropagator
where `masks` contains `frame_masks: {frame_index: logit_or_prob_array}` but
we will expect `per_object_logits` shape semantics when SAM2 emits per-object
logits. The component computes a pixel-wise softmax across objects (and an
optional background) to produce `ownership` soft masks that sum to 1.

Contract (inputs):
 - masks: dict  # FrameMaskSequence carrying `per_object_logits: {frame_idx: (N,H,W)}`
 - temperature: float = 1.0

Outputs:
 - masks: dict  # input carried over, with `frame_masks` replaced by foreground soft
   and `ownership` added (per frame float32 (N+1,H,W); last channel is background).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from haystack import component


def _softmax_across_objects(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    # logits: (N, H, W) -> compute softmax across axis=0 per-pixel
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    arr = np.asarray(logits, dtype=np.float32) / float(temperature)
    # shape (N, H, W) -> move objects axis last for stable softmax
    # compute per-pixel softmax along axis 0
    max_per_pixel = np.max(arr, axis=0, keepdims=True)
    exp = np.exp(arr - max_per_pixel)
    sum_exp = np.sum(exp, axis=0, keepdims=True)
    soft = exp / (sum_exp + 1e-12)
    return soft.astype(np.float32)


@component
class OwnershipResolver:
    """per-object logits から画素単位の soft 所有権を算出する Component。

    背景 logit=0 を明示チャネルとして加えた softmax で、object チャネル + 背景チャネルが
    画素ごとに和 1 になる所有権を得る。前景 soft（1 - 背景）を `frame_masks` に格納し、
    下流の transparent-background へ soft guard として渡す。
    """

    @component.output_types(masks=dict)
    def run(self, masks: dict | None = None, temperature: float = 1.0) -> dict[str, Any]:
        masks = masks or {}
        per_object_logits = masks.get("per_object_logits", {}) or {}

        ownership_map: dict[int, np.ndarray] = {}
        foreground_masks: dict[int, np.ndarray] = {}
        for frame_idx, logits in per_object_logits.items():
            arr = np.asarray(logits, dtype=np.float32)
            if arr.ndim == 2:
                # single object -> (1, H, W)
                arr = arr[None, ...]
            if arr.ndim != 3:
                raise ValueError(f"per_object_logits frame must be (N,H,W) or (H,W), got {arr.shape}")
            # 背景 logit=0 を明示チャネルとして追加し、和 1 になる softmax を計算する。
            bg = np.zeros((1,) + arr.shape[1:], dtype=np.float32)
            arr_with_bg = np.concatenate([arr, bg], axis=0)
            ownership = _softmax_across_objects(arr_with_bg, temperature=temperature).astype(np.float32)
            ownership_map[int(frame_idx)] = ownership
            # 前景 soft = 1 - 背景所有権（最終チャネル）。
            foreground_masks[int(frame_idx)] = np.clip(1.0 - ownership[-1], 0.0, 1.0).astype(np.float32)

        # DEVA tracker が host-RAM 抑制のため per_object_logits を縮小した場合、原寸 (H,W) が
        # ``frame_hw`` に付与される（ERR068）。BEN2 union ゲートは原寸の soft guard を要求するため、
        # foreground（frame_masks）を原寸へアップスケールする。frame_hw が無ければ（基底アプリ等）
        # 従来通り logits 解像度のままとし、振る舞いを変えない。ownership は per_object 経路（frame_hw
        # 未付与＝原寸）でのみ使われるため、ここでは拡大しない。
        frame_hw = masks.get("frame_hw")
        if frame_hw is not None:
            target_h, target_w = int(frame_hw[0]), int(frame_hw[1])
            for fi, fg in foreground_masks.items():
                if fg.shape[:2] != (target_h, target_w):
                    foreground_masks[fi] = cv2.resize(
                        fg, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    ).astype(np.float32)

        # 入力 FrameMaskSequence を引き継ぎつつ frame_masks を前景 soft に差し替える。
        result = dict(masks)
        result["frame_masks"] = foreground_masks
        result["ownership"] = ownership_map
        return {"masks": result}
