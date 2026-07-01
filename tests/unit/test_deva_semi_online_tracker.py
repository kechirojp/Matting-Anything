"""DEVA方式 DevaSemiOnlineTracker（コーディネータ）の単体テスト（GPU 非依存）。

検出島（DetectionIsland）・伝播（SAM2VideoPropagator 互換）・consensus を
クリップ周回で束ね、再シード状態を内部に隔離しつつ BEN2 union 契約の masks を
返す中核コーディネータを、依存注入した fake で検証する。
"""

import numpy as np
import pytest

from pipelines.components.deva_semi_online_tracker import DevaSemiOnlineTracker


# --------------------------------------------------------------------------
# fake 依存
# --------------------------------------------------------------------------
def _empty_entry(h, w):
    return {
        "masks": np.zeros((0, h, w), dtype=bool),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": [],
    }


def _entry(boxes, h, w, labels=None, scores=None):
    """box 群から検出エントリ（masks は box 矩形）を作る。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    k = boxes.shape[0]
    masks = np.zeros((k, h, w), dtype=bool)
    for i in range(k):
        x0, y0, x1, y1 = [int(v) for v in boxes[i]]
        masks[i, y0:y1, x0:x1] = True
    return {
        "masks": masks,
        "boxes": boxes,
        "scores": np.asarray(scores if scores is not None else [0.9] * k, dtype=np.float32),
        "labels": labels if labels is not None else ["person"] * k,
    }


class _FakeDetectionIsland:
    """DetectionIsland 互換の fake。frame_idx ごとの検出エントリを返す。"""

    def __init__(self, per_frame_detections):
        self._det = per_frame_detections
        self.detection_indices_seen = None
        self.text_prompt_seen = None

    def warm_up(self):
        pass

    def run(self, frames, detection_frame_indices, text_prompt,
            box_threshold=0.25, text_threshold=0.25, iou_threshold=0.5, top_k=20):
        self.detection_indices_seen = list(detection_frame_indices)
        self.text_prompt_seen = text_prompt
        h, w = frames[0].shape[:2]
        detections = {}
        for idx in detection_frame_indices:
            detections[idx] = self._det.get(idx, _empty_entry(h, w))
        return {"detections": detections}


class _FakePropagator:
    """SAM2VideoPropagator 互換の fake。box を矩形マスクとしてクリップ全 frame へ定数伝播。"""

    def __init__(self, height, width):
        self.h = height
        self.w = width
        self.calls = []

    def warm_up(self):
        pass

    def run(self, frames, metadata=None, points=None, labels=None, box=None,
            boxes=None, object_id=1, prompt_frame_idx=0, bidirectional=False,
            progress_callback=None):
        box_list = list(boxes) if boxes else ([box] if box is not None else [])
        self.calls.append({
            "num_frames": len(frames),
            "boxes": [list(b) for b in box_list],
            "points": [list(p) for p in points] if points else None,
            "labels": list(labels) if labels else None,
            "prompt_frame_idx": int(prompt_frame_idx),
            "bidirectional": bool(bidirectional),
        })
        n = len(box_list)
        clip_len = len(frames)
        per_object_logits = {}
        frame_masks = {}
        for local in range(clip_len):
            stacked = np.full((n, self.h, self.w), -10.0, dtype=np.float32)
            for k, b in enumerate(box_list):
                x0, y0, x1, y1 = [int(v) for v in b]
                stacked[k, y0:y1, x0:x1] = 10.0
            per_object_logits[local] = stacked
            if n > 0:
                probs = 1.0 / (1.0 + np.exp(-stacked))
                frame_masks[local] = np.max(probs, axis=0).astype(np.float32)
            else:
                frame_masks[local] = np.zeros((self.h, self.w), dtype=np.float32)
        union = {
            "frame_masks": frame_masks,
            "object_ids": list(range(1, n + 1)),
            "frame_indices": sorted(frame_masks.keys()),
            "source": "sam2_video",
            "metadata": {},
            "per_object_logits": per_object_logits,
        }
        return {"masks": union}


def _frames(n, h=16, w=16):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


# --------------------------------------------------------------------------
# BEN2 union 契約
# --------------------------------------------------------------------------
def test_output_matches_ben2_union_contract():
    h = w = 16
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 6, 6]], h, w),
        3: _entry([[0, 0, 6, 6]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(frames=_frames(6, h, w), text_prompt="person", detection_every=3)
    masks = out["masks"]

    # union 契約のキーが揃う。
    for key in ("frame_masks", "object_ids", "frame_indices", "source", "metadata"):
        assert key in masks
    # 全フレーム被覆。
    assert set(masks["frame_masks"].keys()) == set(range(6))
    for fm in masks["frame_masks"].values():
        assert fm.ndim == 2
        assert fm.dtype == np.float32
    assert masks["object_ids"]  # 非空
    assert masks["frame_indices"] == list(range(6))


def test_runs_detection_only_on_detection_frames():
    h = w = 16
    det = _FakeDetectionIsland({0: _entry([[0, 0, 6, 6]], h, w)})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    tracker.run(frames=_frames(6, h, w), text_prompt="person", detection_every=3)
    assert det.detection_indices_seen == [0, 3]
    assert det.text_prompt_seen == "person"


def test_detection_every_controls_clip_count():
    h = w = 16
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 6, 6]], h, w),
        2: _entry([[0, 0, 6, 6]], h, w),
        4: _entry([[0, 0, 6, 6]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    tracker.run(frames=_frames(5, h, w), text_prompt="person", detection_every=2)
    # 検出フレーム [0,2,4] = クリップ 3 本。
    assert det.detection_indices_seen == [0, 2, 4]
    assert len(prop.calls) == 3


# --------------------------------------------------------------------------
# object_id 継続 / 新規 / 削除
# --------------------------------------------------------------------------
def test_object_id_persists_across_clips_on_overlap():
    h = w = 16
    # 0 と 3 で同じ位置の box → 伝播マスクと検出が IoU 一致 → 同一 object_id 継続。
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 8, 8]], h, w),
        3: _entry([[0, 0, 8, 8]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(frames=_frames(6, h, w), text_prompt="person", detection_every=3)
    assert out["masks"]["object_ids"] == [1]


def test_new_object_gets_new_id():
    h = w = 16
    # 3 で離れた位置に追加 box → 伝播と IoU 不一致 → 新規 object_id 採番。
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 6, 6]], h, w),
        3: _entry([[0, 0, 6, 6], [10, 10, 15, 15]], h, w, labels=["person", "dog"]),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(frames=_frames(6, h, w), text_prompt="person . dog", detection_every=3)
    assert 1 in out["masks"]["object_ids"]
    assert 2 in out["masks"]["object_ids"]
    assert out["masks"]["metadata"]["new_object_ids"]  # 新規採番が記録される


def test_lost_object_deleted_after_max_missed():
    h = w = 16
    # 0 で検出、その後 2,4 で未検出 → missed 累積 → max_missed=1 超過で削除。
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 8, 8]], h, w),
        2: _empty_entry(h, w),
        4: _empty_entry(h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(5, h, w),
        text_prompt="person",
        detection_every=2,
        max_missed_detection_count=1,
    )
    assert 1 in out["masks"]["metadata"]["deleted_object_ids"]


# --------------------------------------------------------------------------
# per_object_logits 全フレーム被覆（OwnershipResolver 契約）
# --------------------------------------------------------------------------
def test_per_object_logits_covers_all_frames_including_empty_clips():
    """検出ゼロ（全クリップ対象なし）でも per_object_logits が全 frame を網羅する。

    OwnershipResolver は frame_masks を per_object_logits のキーのみから再構築するため、
    欠落 frame があると下流 BEN2 がマスク無しで処理してしまう。空クリップでも
    (0,H,W) の per_object_logits を出して frame を残すことを保証する。
    """
    h = w = 16
    det = _FakeDetectionIsland({})  # 検出は常に空
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(frames=_frames(6, h, w), text_prompt="person", detection_every=3)
    masks = out["masks"]
    assert "per_object_logits" in masks
    # 全 source frame が per_object_logits に存在する。
    assert set(masks["per_object_logits"].keys()) == set(range(6))
    for arr in masks["per_object_logits"].values():
        assert arr.ndim == 3
        assert arr.shape == (0, h, w)
        assert arr.dtype == np.float32


# --------------------------------------------------------------------------
# per_object_logits メモリ上限（ERR068: 4K×多対象 host-RAM OOM 対策）
# --------------------------------------------------------------------------
def test_default_keeps_full_resolution_per_object_logits():
    """既定（per_object_logits_max_side=0）はフル解像度のまま・frame_hw を付けない（後方互換）。"""
    h = w = 16
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 6, 6]], h, w),
        3: _entry([[0, 0, 6, 6]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(frames=_frames(6, h, w), text_prompt="person", detection_every=3)
    masks = out["masks"]
    for arr in masks["per_object_logits"].values():
        assert arr.shape[1:] == (h, w)
    assert "frame_hw" not in masks


def test_per_object_logits_downsampled_when_max_side_set():
    """per_object_logits_max_side>0 で per_object_logits を縮小し frame_hw に原寸を残す。"""
    h = w = 32
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 12, 12]], h, w),
        3: _entry([[0, 0, 12, 12]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(6, h, w),
        text_prompt="person",
        detection_every=3,
        per_object_logits_max_side=8,
    )
    masks = out["masks"]
    # 原寸は frame_hw に保持される。
    assert masks["frame_hw"] == (h, w)
    # per_object_logits は long side <= 8 へ縮小（object 軸 N は不変、dtype 維持）。
    for arr in masks["per_object_logits"].values():
        assert arr.ndim == 3
        assert max(arr.shape[1], arr.shape[2]) <= 8
        assert arr.dtype == np.float32
    # union 用 frame_masks は原寸を維持（overlay 品質確保）。
    for fm in masks["frame_masks"].values():
        assert fm.shape == (h, w)


def test_empty_clip_per_object_logits_downsampled_when_max_side_set():
    """対象なしクリップの (0,H,W) も縮小形状で全 frame を網羅する。"""
    h = w = 32
    det = _FakeDetectionIsland({})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(6, h, w),
        text_prompt="person",
        detection_every=3,
        per_object_logits_max_side=8,
    )
    masks = out["masks"]
    assert set(masks["per_object_logits"].keys()) == set(range(6))
    for arr in masks["per_object_logits"].values():
        assert arr.shape[0] == 0
        assert max(arr.shape[1], arr.shape[2]) <= 8
    # frame_masks も全 frame 被覆（zero マスク）。
    assert set(masks["frame_masks"].keys()) == set(range(6))


def test_per_object_logits_covers_all_frames_with_detections():
    """検出ありクリップでも per_object_logits が全 frame を網羅する。"""
    h = w = 16
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 8, 8]], h, w),
        3: _entry([[0, 0, 8, 8]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(frames=_frames(6, h, w), text_prompt="person", detection_every=3)
    masks = out["masks"]
    assert set(masks["per_object_logits"].keys()) == set(range(6))
    for arr in masks["per_object_logits"].values():
        assert arr.ndim == 3
        assert arr.shape[0] >= 1
        assert arr.shape[1:] == (h, w)


# --------------------------------------------------------------------------
# バリデーション
# --------------------------------------------------------------------------
def test_empty_frames_raises():
    det = _FakeDetectionIsland({})
    prop = _FakePropagator(16, 16)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)
    with pytest.raises(ValueError):
        tracker.run(frames=[], text_prompt="person", detection_every=3)


def test_invalid_detection_every_raises():
    h = w = 16
    det = _FakeDetectionIsland({0: _entry([[0, 0, 6, 6]], h, w)})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)
    with pytest.raises(ValueError):
        tracker.run(frames=_frames(4, h, w), text_prompt="person", detection_every=0)


# --------------------------------------------------------------------------
# 手動 box / point seed（モードA: 手動のみ / モードB: ハイブリッド）
# --------------------------------------------------------------------------
def test_manual_box_seed_without_text_skips_detection_single_clip():
    """モードA: text 空 + 手動 box → 検出島を呼ばず、全フレームを単一クリップで伝播する。"""
    h = w = 16
    det = _FakeDetectionIsland({})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(6, h, w),
        text_prompt="",
        initial_boxes=[[0, 0, 8, 8]],
        detection_every=3,
    )
    masks = out["masks"]
    # 検出島は呼ばれない（再検出なし）。
    assert det.detection_indices_seen is None
    # 単一クリップ（detection_every は無視）で 1 回だけ伝播する。
    assert len(prop.calls) == 1
    assert prop.calls[0]["num_frames"] == 6
    assert prop.calls[0]["boxes"] == [[0.0, 0.0, 8.0, 8.0]]
    # 全フレーム被覆 + object 採番。
    assert set(masks["frame_masks"].keys()) == set(range(6))
    assert masks["object_ids"] == [1]


def test_manual_points_labels_passed_only_on_first_clip():
    """point/label は第1クリップの propagator にのみ渡る（後続クリップは None）。"""
    h = w = 16
    # モードB: text 検出も走るが、手動 box を pre-populate して seed する。
    det = _FakeDetectionIsland({
        0: _entry([[0, 0, 8, 8]], h, w),
        3: _entry([[0, 0, 8, 8]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    tracker.run(
        frames=_frames(6, h, w),
        text_prompt="person",
        detection_every=3,
        initial_boxes=[[0, 0, 8, 8]],
        initial_points=[[2, 2], [6, 6]],
        initial_labels=[1, 0],  # positive + negative
    )
    # クリップは 2 本（検出フレーム [0,3]）。
    assert len(prop.calls) == 2
    # 第1クリップに point/label が渡る（negative=0 含む）。
    assert prop.calls[0]["points"] == [[2, 2], [6, 6]]
    assert prop.calls[0]["labels"] == [1, 0]
    # 第2クリップ以降は point/label を再投影しない。
    assert prop.calls[1]["points"] is None
    assert prop.calls[1]["labels"] is None


def test_manual_box_pre_populates_track_in_hybrid_mode():
    """モードB: 手動 box が初期 track として被覆し、第1クリップから seed される。"""
    h = w = 16
    det = _FakeDetectionIsland({})  # text 検出は空（手動のみが頼り）
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(6, h, w),
        text_prompt="person",
        detection_every=3,
        initial_boxes=[[1, 1, 9, 9]],
    )
    # 検出が空でも手動 box で seed される（第1クリップ）。
    assert prop.calls[0]["boxes"] == [[1.0, 1.0, 9.0, 9.0]]
    assert out["masks"]["object_ids"] == [1]


def test_text_empty_and_no_manual_seed_raises():
    """text 空 + 手動 box 無し → どちらも無いので ValueError。"""
    h = w = 16
    det = _FakeDetectionIsland({})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)
    with pytest.raises(ValueError):
        tracker.run(frames=_frames(4, h, w), text_prompt="", detection_every=2)


def test_points_without_boxes_raises():
    """手動 point は box が必須（box が無いと割り当て先が無い）。"""
    h = w = 16
    det = _FakeDetectionIsland({0: _entry([[0, 0, 6, 6]], h, w)})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)
    with pytest.raises(ValueError):
        tracker.run(
            frames=_frames(4, h, w),
            text_prompt="person",
            detection_every=2,
            initial_points=[[1, 1]],
        )


def test_labels_length_mismatch_raises():
    """initial_labels の長さは initial_points と一致が必要。"""
    h = w = 16
    det = _FakeDetectionIsland({0: _entry([[0, 0, 6, 6]], h, w)})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)
    with pytest.raises(ValueError):
        tracker.run(
            frames=_frames(4, h, w),
            text_prompt="person",
            detection_every=2,
            initial_boxes=[[0, 0, 6, 6]],
            initial_points=[[1, 1], [2, 2]],
            initial_labels=[1],
        )


# --------------------------------------------------------------------------
# 検出起点フレーム + 双方向逆伝播（被写体が最大に映るフレームで seed）
# --------------------------------------------------------------------------
class _ForwardOnlyFakePropagator(_FakePropagator):
    """forward-only tracker（SAMURAI 相当）の fake。single_object_only=True。"""

    def __init__(self, height, width):
        super().__init__(height, width)
        self.single_object_only = True


def test_detection_start_frame_shifts_detection_indices():
    """text モードで検出/seed が detection_start_frame から始まる。"""
    h = w = 16
    det = _FakeDetectionIsland({
        2: _entry([[0, 0, 6, 6]], h, w),
        5: _entry([[0, 0, 6, 6]], h, w),
    })
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(8, h, w), text_prompt="person",
        detection_every=3, detection_start_frame=2,
    )
    assert det.detection_indices_seen == [2, 5]
    assert out["masks"]["metadata"]["detection_start_frame"] == 2


def test_detection_start_frame_backward_pass_covers_early_frames():
    """検出起点>0 のとき、[0, start) が逆伝播（双方向）でカバーされ全フレーム被覆する。"""
    h = w = 16
    det = _FakeDetectionIsland({3: _entry([[0, 0, 6, 6]], h, w)})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(6, h, w), text_prompt="person",
        detection_every=100, detection_start_frame=3,
    )
    masks = out["masks"]
    # 全 6 フレーム（0..5）が被覆され取りこぼしがない。
    assert set(masks["frame_masks"].keys()) == set(range(6))
    assert set(masks["per_object_logits"].keys()) == set(range(6))
    # 逆伝播クリップが prompt_frame_idx=start, bidirectional=True で 1 回呼ばれる。
    backward_calls = [c for c in prop.calls if c["bidirectional"]]
    assert len(backward_calls) == 1
    assert backward_calls[0]["prompt_frame_idx"] == 3
    assert backward_calls[0]["num_frames"] == 4  # frames[:4] = 0..3


def test_detection_start_frame_zero_has_no_backward_pass():
    """後方互換: detection_start_frame=0 は逆伝播せず従来通り前向きのみ。"""
    h = w = 16
    det = _FakeDetectionIsland({0: _entry([[0, 0, 6, 6]], h, w)})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    tracker.run(frames=_frames(6, h, w), text_prompt="person", detection_every=100)
    assert all(not c["bidirectional"] for c in prop.calls)


def test_detection_start_frame_out_of_range_raises():
    """detection_start_frame >= num_frames は ValueError。"""
    h = w = 16
    det = _FakeDetectionIsland({0: _entry([[0, 0, 6, 6]], h, w)})
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)
    with pytest.raises(ValueError):
        tracker.run(
            frames=_frames(4, h, w), text_prompt="person",
            detection_every=2, detection_start_frame=4,
        )


def test_detection_start_frame_forward_only_tracker_raises():
    """forward-only tracker（SAMURAI）では起点>0（逆伝播）を拒否する。"""
    h = w = 16
    det = _FakeDetectionIsland({2: _entry([[0, 0, 6, 6]], h, w)})
    prop = _ForwardOnlyFakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)
    with pytest.raises(ValueError):
        tracker.run(
            frames=_frames(6, h, w), text_prompt="person",
            detection_every=3, detection_start_frame=2,
        )


def test_manual_box_seed_backward_pass_uses_start_frame_prompt():
    """手動 box + 起点フレーム>0: 逆伝播 seed に手動 box を使い、起点で prompt する。"""
    h = w = 16
    det = _FakeDetectionIsland({})  # モードA（text 空・手動 seed のみ）
    prop = _FakePropagator(h, w)
    tracker = DevaSemiOnlineTracker(detection_island=det, propagator=prop)

    out = tracker.run(
        frames=_frames(6, h, w),
        text_prompt="",
        initial_boxes=[[1, 1, 9, 9]],
        detection_start_frame=2,
    )
    masks = out["masks"]
    assert set(masks["frame_masks"].keys()) == set(range(6))
    backward_calls = [c for c in prop.calls if c["bidirectional"]]
    assert len(backward_calls) == 1
    assert backward_calls[0]["prompt_frame_idx"] == 2
    assert backward_calls[0]["boxes"] == [[1.0, 1.0, 9.0, 9.0]]

