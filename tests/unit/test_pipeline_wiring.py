from pathlib import Path
from types import SimpleNamespace

from pipelines.mam_pipeline import build_mam_pipeline
from pipelines.sam2_tb_pipeline import (
    build_mask_to_matte_pipeline,
    build_mask_union_pipeline,
    build_sam2_maskset_pipeline,
    build_sam2_tb_pipeline,
    build_sam2_union_tb_pipeline,
)


def test_mam_pipeline_builds_without_connect_errors() -> None:
    pipeline = build_mam_pipeline()

    assert pipeline is not None


def test_sam2_tb_pipeline_builds_without_connect_errors() -> None:
    pipeline = build_sam2_tb_pipeline(include_sam2=False)

    assert pipeline is not None


def test_sam2_maskset_pipeline_builds_without_model_warmup() -> None:
    pipeline = build_sam2_maskset_pipeline()

    assert pipeline is not None


def test_mask_union_pipeline_builds_without_connect_errors() -> None:
    pipeline = build_mask_union_pipeline()

    assert pipeline is not None


def test_mask_to_matte_pipeline_builds_without_connect_errors() -> None:
    pipeline = build_mask_to_matte_pipeline()

    assert pipeline is not None


def test_sam2_union_tb_pipeline_builds_without_connect_errors() -> None:
    pipeline = build_sam2_union_tb_pipeline()

    assert pipeline is not None


def test_groundingdino_transformers_bert_compat_patch_is_called_before_model_import() -> None:
    source = Path("pipelines/components/model_components.py").read_text(encoding="utf-8")

    assert "def patch_transformers_bert_for_groundingdino" in source
    assert source.count("patch_transformers_bert_for_groundingdino()") >= 2
    assert "BertModel.get_head_mask = get_head_mask" in source


def test_sam2_segmenter_reads_notebook_environment(monkeypatch) -> None:
    """Notebook launch env should control the SAM2 checkpoint/config used by the Gradio process."""
    from pipelines.components.model_components import SAM2Segmenter

    monkeypatch.setenv("SAM2_CKPT_PATH", "custom/sam2.pt")
    monkeypatch.setenv("SAM2_CONFIG_NAME", "custom/sam2.yaml")

    segmenter = SAM2Segmenter()

    assert segmenter.checkpoint_path == "custom/sam2.pt"
    assert segmenter.config_name == "custom/sam2.yaml"


def test_model_diagnostics_helpers_are_cpu_safe(monkeypatch) -> None:
    """Diagnostics must run in unit tests without CUDA, SAM2, or GroundingDINO imports."""
    from pipelines.components.model_components import (
        build_checkpoint_diagnostics,
        build_runtime_diagnostics,
        cpu_fallback_allowed,
        is_cuda_device,
    )

    monkeypatch.setenv("PROJECT_ROOT", str(Path.cwd()))
    monkeypatch.delenv("MATTING_ANYTHING_ALLOW_CPU", raising=False)

    runtime = build_runtime_diagnostics("cpu")
    checkpoint = build_checkpoint_diagnostics("missing-file.pt")

    assert runtime["selected_device"] == "cpu"
    assert runtime["gpu_required"] is True
    assert runtime["cpu_fallback_allowed"] is False
    assert runtime["process_id"] > 0
    assert is_cuda_device("cpu") is False
    assert cpu_fallback_allowed() is False
    assert checkpoint["exists"] is False
    assert checkpoint["path"].endswith("missing-file.pt")


def test_heavy_inference_requires_gpu_unless_cpu_fallback_is_explicit(monkeypatch) -> None:
    """SAM2/GroundingDINO production paths should not silently run on CPU."""
    import pytest

    from pipelines.components.model_components import cpu_fallback_allowed, require_gpu_for_heavy_inference

    monkeypatch.delenv("MATTING_ANYTHING_ALLOW_CPU", raising=False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        require_gpu_for_heavy_inference("SAM2Segmenter", "cpu")

    monkeypatch.setenv("MATTING_ANYTHING_ALLOW_CPU", "1")
    assert cpu_fallback_allowed() is True
    require_gpu_for_heavy_inference("SAM2Segmenter", "cpu")


def test_is_cuda_device_requires_torch_cuda_available(monkeypatch) -> None:
    """CUDA autocast should only be enabled when torch reports CUDA availability."""
    import sys

    from pipelines.components.model_components import is_cuda_device

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert is_cuda_device("cuda") is False

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    assert is_cuda_device("cuda:0") is True


def test_groundingdino_detector_exposes_custom_op_diagnostics() -> None:
    """GroundingDINO diagnostics should expose resolved device and custom-op status without warm-up."""
    from pipelines.components.model_components import GroundingDINOMultiBoxDetector

    detector = GroundingDINOMultiBoxDetector(device="cpu")
    diagnostics = detector.diagnostics()

    assert diagnostics["device"] == "cpu"
    assert diagnostics["model_cached"] is False
    assert "checkpoint" in diagnostics
    assert "cuda_ops" in diagnostics


def test_legacy_gradio_groundingdino_bert_patch_matches_transformers_behavior() -> None:
    source = Path("gradio_app.py").read_text(encoding="utf-8")

    assert 'if not hasattr(BertModel, "get_head_mask")' in source
    assert "return [None] * num_hidden_layers" in source
    assert "if is_attention_chunked:" in source
    assert "head_mask.to(dtype=self.dtype)" in source
    assert "except ModuleNotFoundError:" in source


def test_legacy_gradio_app_uses_gpu_first_policy() -> None:
    """Legacy GroundingDINO/MAM app should not bypass the GPU-required policy."""
    source = Path("gradio_app.py").read_text(encoding="utf-8")

    assert "default_device" in source
    assert 'require_gpu_for_heavy_inference("gradio_app.py", device)' in source
    assert 'device="cuda"' not in source


def test_groundingdino_autocast_uses_tensor_device_type_for_cpu_fallback() -> None:
    """Emergency CPU fallback should not enter a hardcoded CUDA autocast context."""
    source = Path("GroundingDINO/groundingdino/models/GroundingDINO/transformer.py").read_text(encoding="utf-8")

    assert "device_type = tgt.device.type" in source
    assert "torch.amp.autocast(device_type, enabled=False)" in source
    assert "torch.amp.autocast('cuda', enabled=False)" not in source