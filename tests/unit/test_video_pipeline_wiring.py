from __future__ import annotations

from haystack import Pipeline

from pipelines.sam2_tb_video_pipeline import (
    build_sam2_tb_video_pipeline,
    build_sam2_video_propagation_pipeline,
    build_video_reader_pipeline,
)


def test_video_reader_pipeline_builds() -> None:
    pipeline = build_video_reader_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "video_reader" in pipeline.graph.nodes


def test_sam2_video_propagation_pipeline_builds() -> None:
    pipeline = build_sam2_video_propagation_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "video_reader" in pipeline.graph.nodes
    assert "sam2_video_propagator" in pipeline.graph.nodes


def test_sam2_tb_video_pipeline_builds_with_writers() -> None:
    pipeline = build_sam2_tb_video_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "transparent_bg_video" in pipeline.graph.nodes
    assert "video_writer" in pipeline.graph.nodes
    assert "frame_sequence_writer" in pipeline.graph.nodes


def test_sam2_video_propagator_reads_notebook_environment(monkeypatch) -> None:
    """Movie pipeline should share the same SAM2 checkpoint/config env contract as the static app."""
    from pipelines.components.video_model_components import SAM2VideoPropagator

    monkeypatch.setenv("SAM2_CKPT_PATH", "custom/movie_sam2.pt")
    monkeypatch.setenv("SAM2_CONFIG_NAME", "custom/movie_sam2.yaml")

    propagator = SAM2VideoPropagator()

    assert propagator.checkpoint_path == "custom/movie_sam2.pt"
    assert propagator.config_name == "custom/movie_sam2.yaml"


def test_sam2_video_propagator_warmup_uses_gpu_policy(monkeypatch) -> None:
    """Movie tracking should fail fast on CPU unless emergency fallback is explicit."""
    import pytest

    from pipelines.components.video_model_components import SAM2VideoPropagator

    monkeypatch.delenv("MATTING_ANYTHING_ALLOW_CPU", raising=False)
    propagator = SAM2VideoPropagator(device="cpu")

    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        propagator.warm_up()
