"""ルートA案 BEN2 Component（ben2_components）の単体テスト（GPU/BEN2 非依存）。

BEN2 本体は読み込まず、``BEN2Extractor`` をフェイク注入してルートAの合成配線を検証する。
"""

import sys
import types
from pathlib import Path

import numpy as np

import pipelines.components.ben2_components as ben2_module
from pipelines.components.ben2_components import BEN2Extractor, BEN2RouteAVideoExtractor
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


def test_run_warms_up_ben2_before_first_frame(tmp_path):
    """ERR048 follow-up（RouteA 固有）: BEN2 のモデルロードはループ外で先読みする。

    BEN2 の重いモデルロードが初回フレームの ``infer_alpha`` 内（ループ外の単一
    ブロッキング呼び出し）に埋もれていると、その間 keep-alive 進捗が一切流れず
    gradio.live / Colab の SSE が idle 切断され UI が全出力 "Error" になる。
    ``run()`` は最初の ``infer_alpha`` より前に ``extractor.warm_up`` を呼び、
    SAM2 propagator と同様に通知で挟んでロードギャップを 1 区間に閉じ込めること。
    """
    events: list[str] = []

    class _OrderRecordingExtractor:
        def __init__(self):
            self.call_count = 0

        def warm_up(self):
            events.append("warm_up")

        def infer_alpha(self, image_rgb, refine_foreground=False):
            events.append("infer")
            self.call_count += 1
            return np.asarray(image_rgb).mean(axis=2).astype(np.uint8)

    extractor = BEN2RouteAVideoExtractor.__new__(BEN2RouteAVideoExtractor)
    extractor.extractor = _OrderRecordingExtractor()
    from pipelines.components.video_model_components import _resolve_output_dir

    extractor.output_dir = _resolve_output_dir(str(tmp_path))

    frames = [np.full((8, 8, 3), 200, dtype=np.uint8) for _ in range(2)]
    union_masks = {0: np.zeros((8, 8), dtype=np.float32), 1: np.zeros((8, 8), dtype=np.float32)}
    masks = build_frame_mask_sequence(union_masks, object_ids=[1], metadata={})
    metadata = {"fps": 24.0, "metadata": {"sampled_frame_indices": [0, 1]}}

    extractor.run(
        frames=frames,
        masks=masks,
        metadata=metadata,
        output_mode="sequence",
        dilation_px=1,
        blur_kernel=5,
        feather_px=0,
        matte_mode="union",
    )

    assert "warm_up" in events, "run() が BEN2 モデルを先読み（warm_up）していない"
    assert events.index("warm_up") < events.index("infer"), "warm_up は初回 infer_alpha より前に呼ぶこと"


def test_run_warms_up_ben2_directly_without_keepalive(tmp_path, monkeypatch) -> None:
    """Layer A 撤去: BEN2 の先読みロードは keep-alive ポンプを介さず warm_up を直接呼ぶ。

    Windows local 直結では tunnel/SSE idle 切断が起きず、さらに非同期ジョブ(Layer C)で
    処理全体が background thread 実行されるため、Colab/gradio.live 向けの
    ``run_with_progress_keepalive`` ラップは不要。warm_up は直接呼び出され、進捗通知のみで
    初回ロードを 1 区間に閉じ込めること（共有プリミティブ自体は温存）。
    """
    import pipelines.components.ben2_components as ben2_module

    assert not hasattr(ben2_module, "run_with_progress_keepalive"), (
        "ben2_components は run_with_progress_keepalive を import しない（Layer A 撤去）"
    )

    events: list[str] = []

    class _OrderRecordingExtractor:
        def warm_up(self):
            events.append("warm_up")

        def infer_alpha(self, image_rgb, refine_foreground=False):
            events.append("infer")
            return np.asarray(image_rgb).mean(axis=2).astype(np.uint8)

    recording = _OrderRecordingExtractor()
    extractor = BEN2RouteAVideoExtractor.__new__(BEN2RouteAVideoExtractor)
    extractor.extractor = recording
    from pipelines.components.video_model_components import _resolve_output_dir

    extractor.output_dir = _resolve_output_dir(str(tmp_path))

    frames = [np.full((8, 8, 3), 200, dtype=np.uint8) for _ in range(2)]
    union_masks = {0: np.zeros((8, 8), dtype=np.float32), 1: np.zeros((8, 8), dtype=np.float32)}
    masks = build_frame_mask_sequence(union_masks, object_ids=[1], metadata={})
    metadata = {"fps": 24.0, "metadata": {"sampled_frame_indices": [0, 1]}}

    extractor.run(
        frames=frames,
        masks=masks,
        metadata=metadata,
        output_mode="sequence",
        dilation_px=1,
        blur_kernel=5,
        feather_px=0,
        matte_mode="union",
    )

    assert "warm_up" in events, "run() が BEN2 モデルを先読み（warm_up）していない"
    assert events.index("warm_up") < events.index("infer"), "warm_up は初回 infer_alpha より前"


def test_ben2_extractor_uses_local_checkpoint_without_download(tmp_path, monkeypatch):
    """ローカル checkpoint があれば download せず loadcheckpoints を使う。"""

    calls: dict[str, object] = {}

    class _FakeModel:
        def loadcheckpoints(self, path):
            calls["load_path"] = path

        def to(self, _device):
            calls["to_called"] = True
            return self

        def eval(self):
            calls["eval_called"] = True
            return self

    class _FakeBENBase:
        @staticmethod
        def from_pretrained(_repo_id):
            raise AssertionError("from_pretrained は呼ばれないはず")

        def __new__(cls):
            return _FakeModel()

    fake_ben2 = types.ModuleType("ben2")
    fake_ben2.BEN_Base = _FakeBENBase
    monkeypatch.setitem(sys.modules, "ben2", fake_ben2)
    monkeypatch.setattr(ben2_module, "require_gpu_for_heavy_inference", lambda *_args, **_kwargs: None)

    # ローカルに .pth が存在する状態を作る。
    ckpt_dir = tmp_path / "BEN2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "ckpt_base.pth").write_bytes(b"dummy")

    # download が呼ばれたら失敗させる。
    fake_hf = types.ModuleType("huggingface_hub")

    def _forbidden_snapshot_download(**_kwargs):
        raise AssertionError("snapshot_download は呼ばれないはず")

    fake_hf.snapshot_download = _forbidden_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    extractor = BEN2Extractor(checkpoint_path=str(ckpt_dir), device="cpu")
    extractor.warm_up()

    # loadcheckpoints は torch.load(path) を呼ぶため、ディレクトリではなく
    # 実体の .pth ファイルパスを渡さなければならない（ディレクトリだと Permission denied）。
    assert calls.get("load_path") == str(ckpt_dir / "ckpt_base.pth")
    assert calls.get("to_called") is True
    assert calls.get("eval_called") is True


def test_ben2_extractor_downloads_and_persists_when_local_missing(tmp_path, monkeypatch):
    """ローカルに無い場合は download し、保存先（永続）から loadcheckpoints する。"""

    calls: dict[str, object] = {}

    class _FakeModel:
        def loadcheckpoints(self, path):
            calls["load_path"] = path

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _FakeBENBase:
        @staticmethod
        def from_pretrained(_repo_id):
            raise AssertionError("from_pretrained は使わない実装")

        def __new__(cls):
            return _FakeModel()

    fake_ben2 = types.ModuleType("ben2")
    fake_ben2.BEN_Base = _FakeBENBase
    monkeypatch.setitem(sys.modules, "ben2", fake_ben2)
    monkeypatch.setattr(ben2_module, "require_gpu_for_heavy_inference", lambda *_args, **_kwargs: None)

    target_dir = tmp_path / "BEN2"

    fake_hf = types.ModuleType("huggingface_hub")

    def _snapshot_download(**kwargs):
        calls["download_local_dir"] = kwargs.get("local_dir")
        download_dir = target_dir
        download_dir.mkdir(parents=True, exist_ok=True)
        (download_dir / "ckpt_base.pth").write_bytes(b"downloaded")
        return str(download_dir)

    fake_hf.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    extractor = BEN2Extractor(checkpoint_path=str(target_dir), device="cpu")
    extractor.warm_up()

    assert calls.get("download_local_dir") == str(target_dir)
    assert Path(str(calls.get("load_path"))).exists()
    assert (target_dir / "ckpt_base.pth").exists()


def test_ben2_extractor_warm_up_is_idempotent(tmp_path, monkeypatch):
    """warm_up を複数回呼んでも再 download / 再初期化しない。"""

    calls = {"download": 0, "load": 0}

    class _FakeModel:
        def loadcheckpoints(self, _path):
            calls["load"] += 1

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _FakeBENBase:
        def __new__(cls):
            return _FakeModel()

    fake_ben2 = types.ModuleType("ben2")
    fake_ben2.BEN_Base = _FakeBENBase
    monkeypatch.setitem(sys.modules, "ben2", fake_ben2)
    monkeypatch.setattr(ben2_module, "require_gpu_for_heavy_inference", lambda *_args, **_kwargs: None)

    target_dir = tmp_path / "BEN2"
    fake_hf = types.ModuleType("huggingface_hub")

    def _snapshot_download(**kwargs):
        calls["download"] += 1
        download_dir = Path(str(kwargs["local_dir"]))
        download_dir.mkdir(parents=True, exist_ok=True)
        (download_dir / "ckpt_base.pth").write_bytes(b"downloaded")
        return str(download_dir)

    fake_hf.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    extractor = BEN2Extractor(checkpoint_path=str(target_dir), device="cpu")
    extractor.warm_up()
    extractor.warm_up()

    assert calls["download"] == 1
    assert calls["load"] == 1
