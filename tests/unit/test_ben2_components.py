"""ルートA案 BEN2 Component（ben2_components）の単体テスト（GPU/BEN2 非依存）。

BEN2 本体は読み込まず、``BEN2Extractor`` をフェイク注入してルートAの合成配線を検証する。
"""

import numpy as np

from pipelines.components.ben2_components import BEN2RouteAVideoExtractor
from pipelines.components.video_common import build_frame_mask_sequence


class _FakeBEN2Extractor:
    """誘導フレームの輝度から α を作るスタブ（BEN2 推論を置換）。"""

    def __init__(self):
        self.call_count = 0
        self.received_shapes = []

    def warm_up(self):
        return None

    def infer_alpha(self, image_rgb, refine_foreground=False):
        self.call_count += 1
        array = np.asarray(image_rgb)
        self.received_shapes.append(array.shape)
        # シャープに残った（=明るい）画素を前景とみなす単純な擬似 α。
        gray = array.mean(axis=2)
        alpha = np.clip(gray, 0, 255).astype(np.uint8)
        return alpha


def _build_extractor_with_fake(tmp_path):
    extractor = BEN2RouteAVideoExtractor.__new__(BEN2RouteAVideoExtractor)
    extractor.extractor = _FakeBEN2Extractor()
    from pipelines.components.video_model_components import _resolve_output_dir

    extractor.output_dir = _resolve_output_dir(str(tmp_path))
    return extractor


def test_union_frame_calls_ben2_once(tmp_path):
    extractor = _build_extractor_with_fake(tmp_path)
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[3:6, 3:6] = 1.0
    result = extractor._process_union_frame(
        frame,
        mask,
        dilation_px=1,
        blur_kernel=5,
        blur_sigma=0.0,
        feather_px=0,
        refine_foreground=False,
        gate_alpha=False,
        output_type="rgba",
    )
    # union 経路はフレームあたり BEN2 を 1 回だけ呼ぶ。
    assert extractor.extractor.call_count == 1
    assert result["rgba"].shape == (8, 8, 4)
    assert result["alpha"].shape == (8, 8)


def test_union_frame_gate_alpha_zeros_outside_gate(tmp_path):
    extractor = _build_extractor_with_fake(tmp_path)
    frame = np.full((10, 10, 3), 255, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[4:6, 4:6] = 1.0
    result = extractor._process_union_frame(
        frame,
        mask,
        dilation_px=0,
        blur_kernel=3,
        blur_sigma=0.0,
        feather_px=0,
        refine_foreground=False,
        gate_alpha=True,
        output_type="rgba",
    )
    # gate_alpha=True のときゲート外の α は 0 になる。
    assert result["alpha"][0, 0] == 0
    assert result["alpha"][4, 4] > 0


def test_per_object_frame_calls_ben2_once_per_object(tmp_path):
    extractor = _build_extractor_with_fake(tmp_path)
    frame = np.full((6, 6, 3), 180, dtype=np.uint8)
    logits = np.zeros((2, 6, 6), dtype=np.float32)
    logits[0, 0:3, 0:3] = 5.0
    logits[1, 3:6, 3:6] = 5.0
    ownership = np.zeros((3, 6, 6), dtype=np.float32)
    ownership[0, 0:3, 0:3] = 1.0
    ownership[1, 3:6, 3:6] = 1.0
    ownership[2] = 0.1
    result = extractor._process_per_object_frame(
        frame,
        logits,
        ownership,
        dilation_px=1,
        blur_kernel=3,
        blur_sigma=0.0,
        feather_px=0,
        refine_foreground=False,
        gate_alpha=False,
        output_type="rgba",
    )
    # per_object 経路は対象数ぶん BEN2 を呼ぶ（N=2）。
    assert extractor.extractor.call_count == 2
    assert result["rgba"].shape == (6, 6, 4)


def test_run_union_streams_sequence_outputs(tmp_path):
    extractor = _build_extractor_with_fake(tmp_path)
    frames = [np.full((8, 8, 3), 200, dtype=np.uint8) for _ in range(2)]
    union_masks = {0: np.zeros((8, 8), dtype=np.float32), 1: np.zeros((8, 8), dtype=np.float32)}
    for fm in union_masks.values():
        fm[3:6, 3:6] = 1.0
    masks = build_frame_mask_sequence(union_masks, object_ids=[1], metadata={})
    metadata = {"fps": 24.0, "metadata": {"sampled_frame_indices": [0, 1]}}
    output = extractor.run(
        frames=frames,
        masks=masks,
        metadata=metadata,
        output_mode="sequence",
        dilation_px=1,
        blur_kernel=5,
        feather_px=0,
        matte_mode="union",
    )
    matte = output["matte"]
    assert matte["frame_count"] == 2
    assert matte["output_mode"] == "sequence"
    assert matte["metadata"]["route"] == "A_blur_guidance"
    assert matte["metadata"]["matte_mode"] == "union"
    # 連番出力ディレクトリが作られている。
    assert matte["rgba_sequence_dir"] is not None
    rgba_dir = matte["rgba_sequence_dir"]
    import os

    assert os.path.isdir(rgba_dir)
    assert len(os.listdir(rgba_dir)) == 2
