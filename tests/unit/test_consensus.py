"""DEVA方式 consensus マージ純関数（consensus）の単体テスト（GPU 非依存）。

検出（image-level）と伝播（temporal）を IoU マッチで突き合わせ、
track memory（object_id 継続・missed カウント・max 超で削除）を更新する
中核ロジックを純関数として検証する。
"""

import numpy as np

from pipelines.components.consensus import (
    build_detection_masks,
    compute_mask_iou,
    match_by_iou,
    merge_consensus,
)


def _box_mask(shape, x0, y0, x1, y1):
    """矩形領域を前景とする bool マスクを作る（テスト用ヘルパ）。"""
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


# --------------------------------------------------------------------------
# compute_mask_iou
# --------------------------------------------------------------------------
def test_compute_mask_iou_identical_is_one():
    mask = _box_mask((10, 10), 2, 2, 6, 6)
    assert compute_mask_iou(mask, mask) == 1.0


def test_compute_mask_iou_disjoint_is_zero():
    a = _box_mask((10, 10), 0, 0, 3, 3)
    b = _box_mask((10, 10), 6, 6, 9, 9)
    assert compute_mask_iou(a, b) == 0.0


def test_compute_mask_iou_both_empty_is_zero():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    # 和集合が空のときは 0.0（ゼロ除算を避ける）。
    assert compute_mask_iou(a, b) == 0.0


def test_compute_mask_iou_partial_overlap():
    a = _box_mask((10, 10), 0, 0, 4, 4)  # 16 px
    b = _box_mask((10, 10), 2, 2, 6, 6)  # 16 px, 交差 4px (2..4 x 2..4)
    # intersection=4, union=16+16-4=28
    iou = compute_mask_iou(a, b)
    assert abs(iou - 4 / 28) < 1e-6


def test_compute_mask_iou_accepts_soft_float_via_threshold():
    a = np.full((6, 6), 0.9, dtype=np.float32)  # >0.5 → 全前景
    b = np.full((6, 6), 0.1, dtype=np.float32)  # <0.5 → 全背景
    assert compute_mask_iou(a, b) == 0.0
    c = np.full((6, 6), 0.8, dtype=np.float32)
    assert compute_mask_iou(a, c) == 1.0


# --------------------------------------------------------------------------
# match_by_iou
# --------------------------------------------------------------------------
def test_match_by_iou_single_pair():
    prop = [_box_mask((10, 10), 0, 0, 4, 4)]
    det = [_box_mask((10, 10), 0, 0, 4, 4)]
    matches, unmatched_prop, unmatched_det = match_by_iou(prop, det, iou_threshold=0.5)
    assert matches == [(0, 0)]
    assert unmatched_prop == []
    assert unmatched_det == []


def test_match_by_iou_below_threshold_is_unmatched():
    prop = [_box_mask((10, 10), 0, 0, 4, 4)]
    det = [_box_mask((10, 10), 3, 3, 7, 7)]  # 小さい重なり → IoU < 0.5
    matches, unmatched_prop, unmatched_det = match_by_iou(prop, det, iou_threshold=0.5)
    assert matches == []
    assert unmatched_prop == [0]
    assert unmatched_det == [0]


def test_match_by_iou_greedy_prefers_highest_iou():
    # prop0 は det0 と完全一致、det1 とは部分重なり。greedy で prop0↔det0 を優先。
    prop = [_box_mask((12, 12), 0, 0, 4, 4), _box_mask((12, 12), 6, 6, 10, 10)]
    det = [_box_mask((12, 12), 0, 0, 4, 4), _box_mask((12, 12), 6, 6, 10, 10)]
    matches, unmatched_prop, unmatched_det = match_by_iou(prop, det, iou_threshold=0.5)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert unmatched_prop == []
    assert unmatched_det == []


def test_match_by_iou_extra_detection_unmatched():
    prop = [_box_mask((12, 12), 0, 0, 4, 4)]
    det = [
        _box_mask((12, 12), 0, 0, 4, 4),
        _box_mask((12, 12), 7, 7, 11, 11),  # 新規検出（伝播側に対応なし）
    ]
    matches, unmatched_prop, unmatched_det = match_by_iou(prop, det, iou_threshold=0.5)
    assert matches == [(0, 0)]
    assert unmatched_prop == []
    assert unmatched_det == [1]


def test_match_by_iou_tiebreak_is_deterministic():
    # 同一 IoU の2ペアでも index 昇順で決定的にマッチする。
    prop = [
        _box_mask((10, 10), 0, 0, 4, 4),
        _box_mask((10, 10), 6, 6, 10, 10),
    ]
    det = [
        _box_mask((10, 10), 0, 0, 4, 4),
        _box_mask((10, 10), 6, 6, 10, 10),
    ]
    matches, unmatched_prop, unmatched_det = match_by_iou(prop, det, iou_threshold=0.5)
    assert matches == [(0, 0), (1, 1)]
    assert unmatched_prop == []
    assert unmatched_det == []


# --------------------------------------------------------------------------
# build_detection_masks
# --------------------------------------------------------------------------
def test_build_detection_masks_extracts_list():
    masks = np.stack(
        [_box_mask((8, 8), 0, 0, 3, 3), _box_mask((8, 8), 4, 4, 7, 7)]
    )
    detected = {
        "masks": masks,
        "boxes": np.array([[0, 0, 3, 3], [4, 4, 7, 7]], dtype=np.float32),
        "scores": np.array([0.9, 0.7], dtype=np.float32),
        "labels": ["person", "person"],
    }
    out = build_detection_masks(detected)
    assert len(out) == 2
    assert out[0].dtype == bool
    np.testing.assert_array_equal(out[0], masks[0])


def test_build_detection_masks_empty():
    detected = {
        "masks": np.zeros((0, 8, 8), dtype=bool),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": [],
    }
    assert build_detection_masks(detected) == []


# --------------------------------------------------------------------------
# merge_consensus（中核の状態遷移）
# --------------------------------------------------------------------------
def _detection(masks_list, labels=None, scores=None):
    if masks_list:
        masks = np.stack(masks_list)
        boxes = np.zeros((len(masks_list), 4), dtype=np.float32)
    else:
        masks = np.zeros((0, 8, 8), dtype=bool)
        boxes = np.zeros((0, 4), dtype=np.float32)
    return {
        "masks": masks,
        "boxes": boxes,
        "scores": np.array(scores if scores is not None else [0.9] * len(masks_list), dtype=np.float32),
        "labels": labels if labels is not None else ["person"] * len(masks_list),
    }


def test_merge_consensus_matched_track_resets_missed():
    track_mask = _box_mask((8, 8), 0, 0, 4, 4)
    tracks = [{"object_id": 1, "missed": 2, "mask": track_mask, "box": None, "label": "person", "score": 0.9}]
    propagated = {1: _box_mask((8, 8), 0, 0, 4, 4)}
    detected = _detection([_box_mask((8, 8), 0, 0, 4, 4)])

    result = merge_consensus(
        tracks=tracks,
        propagated=propagated,
        detected=detected,
        iou_threshold=0.5,
        max_missed=3,
        next_object_id=2,
    )
    updated = {t["object_id"]: t for t in result["tracks"]}
    assert 1 in updated
    assert updated[1]["missed"] == 0  # マッチで missed リセット
    assert result["new_object_ids"] == []
    assert result["deleted_object_ids"] == []


def test_merge_consensus_new_detection_creates_object_and_reseed():
    tracks = []
    propagated = {}
    detected = _detection([_box_mask((8, 8), 5, 5, 8, 8)], labels=["dog"])

    result = merge_consensus(
        tracks=tracks,
        propagated=propagated,
        detected=detected,
        iou_threshold=0.5,
        max_missed=3,
        next_object_id=7,
    )
    assert result["new_object_ids"] == [7]
    assert result["next_object_id"] == 8
    reseed_ids = [r["object_id"] for r in result["reseed"]]
    assert reseed_ids == [7]
    assert result["reseed"][0]["label"] == "dog"
    updated = {t["object_id"]: t for t in result["tracks"]}
    assert updated[7]["missed"] == 0


def test_merge_consensus_unmatched_track_increments_missed():
    track_mask = _box_mask((8, 8), 0, 0, 4, 4)
    tracks = [{"object_id": 1, "missed": 0, "mask": track_mask, "box": None, "label": "person", "score": 0.9}]
    propagated = {1: _box_mask((8, 8), 0, 0, 4, 4)}
    detected = _detection([])  # 検出ゼロ → 伝播 track は未マッチ

    result = merge_consensus(
        tracks=tracks,
        propagated=propagated,
        detected=detected,
        iou_threshold=0.5,
        max_missed=3,
        next_object_id=2,
    )
    updated = {t["object_id"]: t for t in result["tracks"]}
    assert updated[1]["missed"] == 1
    assert result["deleted_object_ids"] == []


def test_merge_consensus_deletes_track_after_max_missed():
    track_mask = _box_mask((8, 8), 0, 0, 4, 4)
    # missed=3 で max_missed=3 → 今回も未マッチなら 4 となり削除。
    tracks = [{"object_id": 1, "missed": 3, "mask": track_mask, "box": None, "label": "person", "score": 0.9}]
    propagated = {1: _box_mask((8, 8), 0, 0, 4, 4)}
    detected = _detection([])

    result = merge_consensus(
        tracks=tracks,
        propagated=propagated,
        detected=detected,
        iou_threshold=0.5,
        max_missed=3,
        next_object_id=2,
    )
    assert result["deleted_object_ids"] == [1]
    assert all(t["object_id"] != 1 for t in result["tracks"])


def test_merge_consensus_keeps_existing_and_adds_new_simultaneously():
    track_mask = _box_mask((12, 12), 0, 0, 4, 4)
    tracks = [{"object_id": 1, "missed": 0, "mask": track_mask, "box": None, "label": "person", "score": 0.9}]
    propagated = {1: _box_mask((12, 12), 0, 0, 4, 4)}
    detected = _detection(
        [
            _box_mask((12, 12), 0, 0, 4, 4),  # 既存にマッチ
            _box_mask((12, 12), 8, 8, 12, 12),  # 新規
        ],
        labels=["person", "person"],
    )

    result = merge_consensus(
        tracks=tracks,
        propagated=propagated,
        detected=detected,
        iou_threshold=0.5,
        max_missed=3,
        next_object_id=2,
    )
    ids = sorted(t["object_id"] for t in result["tracks"])
    assert ids == [1, 2]
    assert result["new_object_ids"] == [2]
    updated = {t["object_id"]: t for t in result["tracks"]}
    assert updated[1]["missed"] == 0


def test_merge_consensus_does_not_mutate_input_tracks():
    track_mask = _box_mask((8, 8), 0, 0, 4, 4)
    tracks = [{"object_id": 1, "missed": 1, "mask": track_mask, "box": None, "label": "person", "score": 0.9}]
    propagated = {1: _box_mask((8, 8), 0, 0, 4, 4)}
    detected = _detection([])

    merge_consensus(
        tracks=tracks,
        propagated=propagated,
        detected=detected,
        iou_threshold=0.5,
        max_missed=3,
        next_object_id=2,
    )
    # 入力 tracks は不変（純関数）。
    assert tracks[0]["missed"] == 1
