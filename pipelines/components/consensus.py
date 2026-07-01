"""DEVA方式 consensus マージの純関数群（GPU 非依存・テスト容易）。

検出（image-level 仮説）と伝播（temporal propagation）を IoU マッチで
突き合わせ、track memory を更新する中核ロジックをここに閉じ込める。

設計方針（計画書 §0 の破壊防止ライン）:
- 検出と伝播の分離を保つため、ここは「マッチと状態遷移」だけを担う純関数。
- モデル呼び出し・I/O は一切持たない（`@component` 側が前段で用意した
  マスク配列だけを受け取る）。
- フィードバック（再シード状態）の保持は呼び出し側コーディネータが行い、
  本モジュールは入力を破壊しない（純関数）。

track（dict）の契約:
    {
        "object_id": int,
        "missed": int,                       # 連続未検出カウント
        "mask": np.ndarray (H, W) bool,      # 直近の確定マスク（再シード手がかり）
        "box": tuple[float, ...] | None,     # 直近の検出 box（xyxy）
        "label": str | None,
        "score": float,
    }

detected（dict, DetectionIsland 出力の 1 フレーム分）の契約:
    {
        "masks": np.ndarray (K, H, W) bool,
        "boxes": np.ndarray (K, 4) float,    # xyxy
        "scores": np.ndarray (K,) float,
        "labels": list[str],
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "compute_mask_iou",
    "match_by_iou",
    "build_detection_masks",
    "merge_consensus",
]


def _binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """bool / soft float マスクを bool 化する。

    Args:
        mask: (H, W) の bool または float マスク。
        threshold: float マスクを 2 値化する閾値。

    Returns:
        (H, W) bool マスク。
    """
    if mask.dtype == bool:
        return mask
    return mask > threshold


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """2 つのマスクの IoU を返す（和集合が空なら 0.0）。

    Args:
        mask_a: (H, W) bool または soft float マスク。
        mask_b: (H, W) bool または soft float マスク。

    Returns:
        IoU 値（0.0..1.0）。
    """
    a = _binarize(mask_a)
    b = _binarize(mask_b)
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return intersection / union


def match_by_iou(
    propagated_masks: list[np.ndarray],
    detected_masks: list[np.ndarray],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """伝播マスクと検出マスクを IoU 貪欲法でマッチングする。

    IoU 降順に走査し、閾値以上かつ両側未割当のペアを確定する。

    Args:
        propagated_masks: 伝播側マスクのリスト（index = track の順序）。
        detected_masks: 検出側マスクのリスト（index = 検出の順序）。
        iou_threshold: マッチ成立に必要な最小 IoU。

    Returns:
        (matches, unmatched_prop, unmatched_det):
            - matches: (prop_index, det_index) のリスト。
            - unmatched_prop: マッチしなかった伝播 index のリスト（昇順）。
            - unmatched_det: マッチしなかった検出 index のリスト（昇順）。
    """
    candidates: list[tuple[float, int, int]] = []
    for p_idx, p_mask in enumerate(propagated_masks):
        for d_idx, d_mask in enumerate(detected_masks):
            iou = compute_mask_iou(p_mask, d_mask)
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, d_idx))

    # IoU 降順、同点は index 安定順で決定的に。
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    matched_prop: set[int] = set()
    matched_det: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _iou, p_idx, d_idx in candidates:
        if p_idx in matched_prop or d_idx in matched_det:
            continue
        matched_prop.add(p_idx)
        matched_det.add(d_idx)
        matches.append((p_idx, d_idx))

    unmatched_prop = [i for i in range(len(propagated_masks)) if i not in matched_prop]
    unmatched_det = [i for i in range(len(detected_masks)) if i not in matched_det]
    return matches, unmatched_prop, unmatched_det


def build_detection_masks(detected: dict[str, Any]) -> list[np.ndarray]:
    """検出 dict から bool マスクのリストを取り出す。

    Args:
        detected: DetectionIsland の 1 フレーム分出力。

    Returns:
        (H, W) bool マスクのリスト（検出が無ければ空）。
    """
    masks = detected.get("masks")
    if masks is None or len(masks) == 0:
        return []
    return [_binarize(np.asarray(m)) for m in masks]


def merge_consensus(
    tracks: list[dict[str, Any]],
    propagated: dict[int, np.ndarray],
    detected: dict[str, Any],
    iou_threshold: float,
    max_missed: int,
    next_object_id: int,
) -> dict[str, Any]:
    """検出と伝播を突き合わせて track memory を更新する（純関数）。

    挙動（計画書 §0 の必須 4 点のうち consensus / memory 掃除）:
        - マッチした track: ``missed=0`` にリセットし、検出の box/label/score で再アンカー。
        - 未マッチ検出（新規対象）: 新しい object_id を採番し track 追加・再シード対象に。
        - 未マッチ track（検出に支持されない伝播）: ``missed += 1``。
          ``missed > max_missed`` なら削除（memory 掃除）。

    入力の ``tracks`` は破壊しない（各 track を複製して更新する）。

    Args:
        tracks: 現在の track リスト（track 契約参照）。
        propagated: ``{object_id: (H, W) マスク}`` 検出フレームでの伝播結果。
        detected: 検出フレームでの DetectionIsland 出力（detected 契約参照）。
        iou_threshold: マッチ成立に必要な最小 IoU。
        max_missed: この回数を超えて未検出が続いた track を削除する。
        next_object_id: 新規採番に使う次の object_id。

    Returns:
        dict:
            - ``tracks``: 更新後の track リスト（削除済みは除外）。
            - ``new_object_ids``: 今回追加した object_id のリスト。
            - ``deleted_object_ids``: 今回削除した object_id のリスト。
            - ``reseed``: 新規 track の再シード情報リスト
              （``{"object_id", "mask", "box", "label", "score"}``）。
            - ``next_object_id``: 採番後の次 object_id。
    """
    ordered_tracks = list(tracks)
    prop_ids = [t["object_id"] for t in ordered_tracks]
    prop_masks = [
        _binarize(np.asarray(propagated[oid]))
        if oid in propagated
        else np.zeros_like(_binarize(np.asarray(ordered_tracks[i]["mask"])))
        for i, oid in enumerate(prop_ids)
    ]
    det_masks = build_detection_masks(detected)
    det_boxes = detected.get("boxes")
    det_scores = detected.get("scores")
    det_labels = detected.get("labels") or []

    matches, unmatched_prop, unmatched_det = match_by_iou(
        prop_masks, det_masks, iou_threshold
    )
    matched_prop_to_det = {p_idx: d_idx for p_idx, d_idx in matches}

    updated_tracks: list[dict[str, Any]] = []
    deleted_object_ids: list[int] = []

    for p_idx, track in enumerate(ordered_tracks):
        new_track = dict(track)
        if p_idx in matched_prop_to_det:
            d_idx = matched_prop_to_det[p_idx]
            new_track["missed"] = 0
            new_track["mask"] = det_masks[d_idx]
            if det_boxes is not None and len(det_boxes) > d_idx:
                new_track["box"] = tuple(float(v) for v in np.asarray(det_boxes[d_idx]))
            if d_idx < len(det_labels):
                new_track["label"] = det_labels[d_idx]
            if det_scores is not None and len(det_scores) > d_idx:
                new_track["score"] = float(det_scores[d_idx])
            updated_tracks.append(new_track)
        else:
            new_track["missed"] = int(track["missed"]) + 1
            if new_track["missed"] > max_missed:
                deleted_object_ids.append(int(track["object_id"]))
                continue
            updated_tracks.append(new_track)

    new_object_ids: list[int] = []
    reseed: list[dict[str, Any]] = []
    for d_idx in unmatched_det:
        object_id = next_object_id
        next_object_id += 1
        box = (
            tuple(float(v) for v in np.asarray(det_boxes[d_idx]))
            if det_boxes is not None and len(det_boxes) > d_idx
            else None
        )
        label = det_labels[d_idx] if d_idx < len(det_labels) else None
        score = (
            float(det_scores[d_idx])
            if det_scores is not None and len(det_scores) > d_idx
            else 0.0
        )
        new_track = {
            "object_id": object_id,
            "missed": 0,
            "mask": det_masks[d_idx],
            "box": box,
            "label": label,
            "score": score,
        }
        updated_tracks.append(new_track)
        new_object_ids.append(object_id)
        reseed.append(
            {
                "object_id": object_id,
                "mask": det_masks[d_idx],
                "box": box,
                "label": label,
                "score": score,
            }
        )

    return {
        "tracks": updated_tracks,
        "new_object_ids": new_object_ids,
        "deleted_object_ids": deleted_object_ids,
        "reseed": reseed,
        "next_object_id": next_object_id,
    }
