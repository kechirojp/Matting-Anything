"""Phase2: TransparentBGVideoExtractor の per_object フレーム合成配線の検証（GPU 非依存）。"""

import numpy as np

from pipelines.components.common import stable_sigmoid
from pipelines.components.video_common import composite_alpha_by_ownership
from pipelines.components.video_model_components import TransparentBGVideoExtractor


class _FakeExtractor:
    """soft mask をそのまま alpha(uint8) として返すスタブ（tb 推論を置換）。"""

    def __init__(self):
        self.call_count = 0
        self.received_masks = []

    def run(self, image, mask=None, **kwargs):
        self.call_count += 1
        soft = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        self.received_masks.append(soft.copy())
        alpha_u8 = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
        rgb = np.zeros((*soft.shape, 3), dtype=np.uint8)
        return {"rgba": np.dstack([rgb, alpha_u8]), "alpha": alpha_u8, "preview": np.dstack([rgb, alpha_u8])}


def _build_extractor_with_fake():
    extractor = TransparentBGVideoExtractor.__new__(TransparentBGVideoExtractor)
    extractor.extractor = _FakeExtractor()
    return extractor


def test_per_object_frame_calls_extractor_once_per_object():
    extractor = _build_extractor_with_fake()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    logits = np.array([[[1.0, -1.0], [0.0, 2.0]], [[-2.0, 1.0], [0.5, -0.5]]], dtype=np.float32)  # (2,2,2)
    ownership = np.array(
        [
            [[0.6, 0.2], [0.5, 0.7]],  # obj0
            [[0.3, 0.7], [0.4, 0.2]],  # obj1
            [[0.1, 0.1], [0.1, 0.1]],  # background
        ],
        dtype=np.float32,
    )
    result = extractor._run_per_object_frame(
        frame,
        logits,
        ownership,
        tb_mode="base",
        tb_jit=False,
        tb_threshold=0.0,
        tb_output_type="rgba",
        crop_padding=40,
        mask_guard_feather=0,
    )
    # 対象数 N=2 ぶん tb を呼ぶ（フレームあたり N 回）。
    assert extractor.extractor.call_count == 2


def test_per_object_frame_alpha_matches_ownership_composite():
    extractor = _build_extractor_with_fake()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    logits = np.array([[[1.0, -1.0], [0.0, 2.0]], [[-2.0, 1.0], [0.5, -0.5]]], dtype=np.float32)
    ownership = np.array(
        [
            [[0.6, 0.2], [0.5, 0.7]],
            [[0.3, 0.7], [0.4, 0.2]],
            [[0.1, 0.1], [0.1, 0.1]],
        ],
        dtype=np.float32,
    )
    result = extractor._run_per_object_frame(
        frame,
        logits,
        ownership,
        tb_mode="base",
        tb_jit=False,
        tb_threshold=0.0,
        tb_output_type="rgba",
        crop_padding=40,
        mask_guard_feather=0,
    )
    # FakeExtractor は alpha=soft(=sigmoid(logit)) を返すので、期待値は所有権合成と一致する。
    # extractor は alpha を uint8 で返すため、往復量子化で ±1 の丸め差が出る点を許容する。
    expected_alphas = [stable_sigmoid(logits[0]), stable_sigmoid(logits[1])]
    expected_final = composite_alpha_by_ownership(expected_alphas, ownership)
    expected_u8 = np.clip(expected_final * 255.0, 0, 255).astype(np.uint8)
    assert result["alpha"].shape == (2, 2)
    assert np.all(np.abs(result["alpha"].astype(int) - expected_u8.astype(int)) <= 1)
    # RGBA の alpha チャネルも一致する。
    assert np.array_equal(result["rgba"][..., 3], result["alpha"])
    assert result["rgba"].shape == (2, 2, 4)


class _KwargRecordingExtractor(_FakeExtractor):
    """run の kwargs を記録するスタブ（mask_guard_* の配線検証用）。"""

    def __init__(self):
        super().__init__()
        self.received_kwargs = []

    def run(self, image, mask=None, **kwargs):
        self.received_kwargs.append(dict(kwargs))
        return super().run(image, mask=mask, **kwargs)


def test_per_object_frame_forwards_mask_guard_dilate():
    """改善3: per_object 経路で mask_guard_dilate が tb extractor へ配線される。"""
    extractor = TransparentBGVideoExtractor.__new__(TransparentBGVideoExtractor)
    extractor.extractor = _KwargRecordingExtractor()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    logits = np.array([[[1.0, -1.0], [0.0, 2.0]]], dtype=np.float32)  # (1,2,2)
    ownership = np.array(
        [
            [[0.7, 0.7], [0.7, 0.7]],
            [[0.3, 0.3], [0.3, 0.3]],
        ],
        dtype=np.float32,
    )
    extractor._run_per_object_frame(
        frame,
        logits,
        ownership,
        tb_mode="base",
        tb_jit=False,
        tb_threshold=0.0,
        tb_output_type="rgba",
        crop_padding=40,
        mask_guard_feather=0,
        mask_guard_dilate=7,
    )
    assert extractor.extractor.received_kwargs
    assert all(kw.get("mask_guard_dilate") == 7 for kw in extractor.extractor.received_kwargs)


def test_run_forwards_mask_guard_dilate_to_union_extractor(tmp_path):
    """改善3: union 経路で mask_guard_dilate が tb extractor へ配線される。"""
    extractor = TransparentBGVideoExtractor(output_dir=str(tmp_path))
    received_kwargs: list[dict] = []

    def fake_extract(**kwargs):
        received_kwargs.append(dict(kwargs))
        image = np.asarray(kwargs["image"])
        height, width = image.shape[:2]
        alpha = np.full((height, width), 255, dtype=np.uint8)
        return {"rgba": np.dstack([image, alpha]), "alpha": alpha, "preview": image}

    extractor.extractor.run = fake_extract
    frames = [np.full((4, 5, 3), 10, dtype=np.uint8)]
    extractor.run(frames=frames, output_mode="sequence", mask_guard_dilate=9)
    assert received_kwargs
    assert all(kw.get("mask_guard_dilate") == 9 for kw in received_kwargs)

