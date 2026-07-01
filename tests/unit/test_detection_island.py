"""DEVA方式 DetectionIsland（image-level 検出島）の単体テスト（GPU 非依存）。

GroundingDINO（text→box）→ SAM2 画像モード（box→mask）を「検出フレームのみ」
実行し、consensus に渡す検出仮説 dict を生成する契約を、依存注入した fake で検証する。
"""

import numpy as np
import pytest

from pipelines.components.detection_island import (
    DetectionIsland,
    normalize_segmenter_masks,
)


# --------------------------------------------------------------------------
# normalize_segmenter_masks（SAM2 出力形状の正規化）
# --------------------------------------------------------------------------
def test_normalize_masks_squeezes_k1hw():
    masks = np.zeros((2, 1, 6, 6), dtype=bool)
    masks[0, 0, 0:3, 0:3] = True
    scores = np.array([[0.9], [0.8]], dtype=np.float32)
    out = normalize_segmenter_masks(masks, scores, num_boxes=2)
    assert out.shape == (2, 6, 6)
    assert out.dtype == bool
    np.testing.assert_array_equal(out[0], masks[0, 0])


def test_normalize_masks_picks_best_of_multimask():
    # (K=1, C=3, H, W) のうち score 最大のチャネルを選ぶ。
    masks = np.zeros((1, 3, 4, 4), dtype=bool)
    masks[0, 2] = True  # チャネル2 が最良
    scores = np.array([[0.1, 0.2, 0.9]], dtype=np.float32)
    out = normalize_segmenter_masks(masks, scores, num_boxes=1)
    assert out.shape == (1, 4, 4)
    np.testing.assert_array_equal(out[0], masks[0, 2])


def test_normalize_masks_picks_best_multimask_multiple_boxes():
    # (K=2, C=3, H, W) で box ごとに score 最大チャネルを選ぶ。
    masks = np.zeros((2, 3, 4, 4), dtype=bool)
    masks[0, 1] = True  # box0 → チャネル1
    masks[1, 2] = True  # box1 → チャネル2
    scores = np.array([[0.1, 0.9, 0.2], [0.3, 0.2, 0.8]], dtype=np.float32)
    out = normalize_segmenter_masks(masks, scores, num_boxes=2)
    assert out.shape == (2, 4, 4)
    np.testing.assert_array_equal(out[0], masks[0, 1])
    np.testing.assert_array_equal(out[1], masks[1, 2])


def test_normalize_masks_accepts_khw_directly():
    masks = np.zeros((3, 5, 5), dtype=bool)
    scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = normalize_segmenter_masks(masks, scores, num_boxes=3)
    assert out.shape == (3, 5, 5)


def test_normalize_masks_single_hw_to_1hw():
    masks = np.zeros((5, 5), dtype=bool)
    scores = np.array([0.7], dtype=np.float32)
    out = normalize_segmenter_masks(masks, scores, num_boxes=1)
    assert out.shape == (1, 5, 5)


# --------------------------------------------------------------------------
# fake 依存
# --------------------------------------------------------------------------
class _FakeDetector:
    """GroundingDINOMultiBoxDetector 互換の fake。"""

    def __init__(self, boxes, phrases, confidences):
        self._boxes = boxes
        self._phrases = phrases
        self._confidences = confidences
        self.calls = []

    def warm_up(self):
        pass

    def run(self, image, text_prompt, box_threshold=0.25, text_threshold=0.25, iou_threshold=0.5, top_k=20):
        self.calls.append(text_prompt)
        if len(self._boxes) == 0:
            raise ValueError("テキストプロンプトに一致する領域が検出されませんでした。")
        return {
            "boxes": np.asarray(self._boxes, dtype=np.float32),
            "phrases": list(self._phrases),
            "confidences": np.asarray(self._confidences, dtype=np.float32),
            "proposals": {},
            "diagnostics": {},
        }


class _FakeSegmenter:
    """SAM2Segmenter 互換の fake（box→mask）。"""

    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.received_boxes = []

    def warm_up(self):
        pass

    def run(self, image, points=None, labels=None, box=None, boxes=None, multimask=True):
        self.received_boxes.append(boxes)
        num = len(boxes) if boxes is not None else 0
        masks = np.zeros((num, 1, self.height, self.width), dtype=bool)
        for k in range(num):
            x0, y0, x1, y1 = [int(v) for v in boxes[k]]
            masks[k, 0, y0:y1, x0:x1] = True
        scores = np.full((num, 1), 0.9, dtype=np.float32)
        return {"masks": masks, "scores": scores, "mask_set": {}, "diagnostics": {}}


def _frames(n, h=8, w=8):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


# --------------------------------------------------------------------------
# DetectionIsland I/O 契約
# --------------------------------------------------------------------------
def test_detection_island_runs_only_on_detection_frames():
    detector = _FakeDetector(
        boxes=[[0, 0, 4, 4]], phrases=["person"], confidences=[0.9]
    )
    segmenter = _FakeSegmenter()
    island = DetectionIsland(detector=detector, segmenter=segmenter)

    out = island.run(
        frames=_frames(5),
        detection_frame_indices=[0, 2, 4],
        text_prompt="person",
    )
    detections = out["detections"]
    assert set(detections.keys()) == {0, 2, 4}
    # 検出フレーム数だけ detector が呼ばれる（毎フレームではない）。
    assert len(detector.calls) == 3


def test_detection_island_output_contract_shapes():
    detector = _FakeDetector(
        boxes=[[0, 0, 4, 4], [4, 4, 8, 8]],
        phrases=["person", "dog"],
        confidences=[0.9, 0.7],
    )
    segmenter = _FakeSegmenter(height=8, width=8)
    island = DetectionIsland(detector=detector, segmenter=segmenter)

    out = island.run(
        frames=_frames(3),
        detection_frame_indices=[1],
        text_prompt="person . dog",
    )
    entry = out["detections"][1]
    assert entry["masks"].shape == (2, 8, 8)
    assert entry["masks"].dtype == bool
    assert entry["boxes"].shape == (2, 4)
    assert entry["scores"].shape == (2,)
    assert entry["labels"] == ["person", "dog"]


def test_detection_island_passes_detector_boxes_to_segmenter():
    boxes = [[1, 1, 5, 5]]
    detector = _FakeDetector(boxes=boxes, phrases=["person"], confidences=[0.9])
    segmenter = _FakeSegmenter()
    island = DetectionIsland(detector=detector, segmenter=segmenter)

    island.run(frames=_frames(2), detection_frame_indices=[0], text_prompt="person")
    assert segmenter.received_boxes  # 呼ばれている
    np.testing.assert_array_equal(
        np.asarray(segmenter.received_boxes[0], dtype=np.float32),
        np.asarray(boxes, dtype=np.float32),
    )


def test_detection_island_empty_detection_frame_yields_empty_entry():
    # 検出ゼロ（対象が一時的に隠れる等）は致命ではなく空エントリにする。
    detector = _FakeDetector(boxes=[], phrases=[], confidences=[])
    segmenter = _FakeSegmenter()
    island = DetectionIsland(detector=detector, segmenter=segmenter)

    out = island.run(frames=_frames(3), detection_frame_indices=[0], text_prompt="person")
    entry = out["detections"][0]
    assert entry["masks"].shape[0] == 0
    assert entry["boxes"].shape == (0, 4)
    assert entry["scores"].shape == (0,)
    assert entry["labels"] == []


def test_detection_island_invalid_frame_index_raises():
    detector = _FakeDetector(boxes=[[0, 0, 4, 4]], phrases=["person"], confidences=[0.9])
    segmenter = _FakeSegmenter()
    island = DetectionIsland(detector=detector, segmenter=segmenter)

    with pytest.raises((IndexError, ValueError)):
        island.run(frames=_frames(2), detection_frame_indices=[5], text_prompt="person")
