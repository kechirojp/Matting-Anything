"""SAM2Segmenter の複数 box（batched）4 次元マスク畳み込みの単体テスト。

SAM2 image predictor は複数 box を渡すと ``(K, C, H, W)``（box × multimask 候補）を返す。
``SAM2Segmenter.run`` はこれを候補軸 C で score 最大に畳み ``(K, H, W)`` に正規化し、
``build_mask_set``（(N,H,W) 契約）へ渡せる形にしなければならない。単一 box / points 経路の
``(C, H, W)`` 3 次元出力には影響しないことも確認する。
"""
from __future__ import annotations

import numpy as np

from pipelines.components.model_components import SAM2Segmenter


class _FakeBatchedPredictor:
    """複数 box で (K, C, H, W) を返す SAM2ImagePredictor 互換 fake。"""

    def __init__(self, height: int = 6, width: int = 5):
        self.height = height
        self.width = width
        self.set_image_calls = 0

    def set_image(self, image):  # noqa: D401 - fake
        self.set_image_calls += 1

    def predict(self, **kwargs):
        box = np.asarray(kwargs.get("box"))
        k = int(box.reshape(-1, 4).shape[0])
        multimask = bool(kwargs.get("multimask_output", True))
        c = 3 if multimask else 1
        masks = np.zeros((k, c, self.height, self.width), dtype=bool)
        scores = np.zeros((k, c), dtype=np.float32)
        for i in range(k):
            # 候補ごとに異なる score を与え、best 候補を一意に決める。
            for j in range(c):
                scores[i, j] = 0.1 + 0.1 * j + 0.01 * i
                masks[i, j, i % self.height, j % self.width] = True
        logits = np.zeros((k, c, self.height, self.width), dtype=np.float32)
        return masks, scores, logits


class _FakeSinglePredictor:
    """単一 box / points で (C, H, W) を返す SAM2ImagePredictor 互換 fake。"""

    def __init__(self, height: int = 6, width: int = 5):
        self.height = height
        self.width = width

    def set_image(self, image):  # noqa: D401 - fake
        pass

    def predict(self, **kwargs):
        c = 3 if bool(kwargs.get("multimask_output", True)) else 1
        masks = np.zeros((c, self.height, self.width), dtype=bool)
        scores = np.asarray([0.2, 0.5, 0.3][:c], dtype=np.float32)
        logits = np.zeros((c, self.height, self.width), dtype=np.float32)
        return masks, scores, logits


def _make_segmenter(predictor) -> SAM2Segmenter:
    seg = SAM2Segmenter(device="cpu")
    seg._predictor = predictor  # warm_up を冪等にスキップさせる（GPU チェック回避）
    return seg


def test_batched_boxes_collapse_to_k_h_w():
    """複数 box の (K, C, H, W) が (K, H, W) に畳まれ、scores も (K,) になる。"""
    fake = _FakeBatchedPredictor()
    seg = _make_segmenter(fake)
    image = np.zeros((fake.height, fake.width, 3), dtype=np.uint8)
    boxes = np.asarray([[0, 0, 4, 4], [1, 1, 3, 3]], dtype=np.float32)

    out = seg.run(image=image, boxes=boxes, multimask=True)

    masks = np.asarray(out["masks"])
    scores = np.asarray(out["scores"])
    assert masks.shape == (2, fake.height, fake.width)
    assert scores.shape == (2,)
    # 各 box で最大 score 候補（j=2）が選ばれる。
    assert np.allclose(scores, [0.1 + 0.2 + 0.0, 0.1 + 0.2 + 0.01])
    # build_mask_set が例外なく構築されている。
    assert out["mask_set"]["metadata"]["count"] == 2


def test_batched_boxes_single_candidate():
    """multimask=False（C=1）でも (K, 1, H, W) → (K, H, W) に畳まれる。"""
    fake = _FakeBatchedPredictor()
    seg = _make_segmenter(fake)
    image = np.zeros((fake.height, fake.width, 3), dtype=np.uint8)
    boxes = np.asarray([[0, 0, 4, 4], [1, 1, 3, 3], [2, 0, 4, 2]], dtype=np.float32)

    out = seg.run(image=image, boxes=boxes, multimask=False)

    masks = np.asarray(out["masks"])
    scores = np.asarray(out["scores"])
    assert masks.shape == (3, fake.height, fake.width)
    assert scores.shape == (3,)


def test_single_box_path_unaffected():
    """単一 box / points 経路の (C, H, W) 3 次元出力はそのまま（畳み込み非対象）。"""
    fake = _FakeSinglePredictor()
    seg = _make_segmenter(fake)
    image = np.zeros((fake.height, fake.width, 3), dtype=np.uint8)

    out = seg.run(image=image, box=[0, 0, 4, 4], multimask=True)

    masks = np.asarray(out["masks"])
    scores = np.asarray(out["scores"])
    assert masks.shape == (3, fake.height, fake.width)
    assert scores.shape == (3,)
    assert out["mask_set"]["metadata"]["count"] == 3
