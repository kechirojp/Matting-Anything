from __future__ import annotations

from pathlib import Path

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


def test_video_writer_warm_up_matches_haystack_no_arg_contract() -> None:
    """Haystack Pipeline warm_up calls component warm_up without runtime frame shape inputs."""
    from pipelines.components.video_model_components import VideoWriter

    writer = VideoWriter()

    assert writer.warm_up() is None


def test_video_components_accept_progress_callbacks() -> None:
    """Movie pipeline should report long-running stage progress without duplicating video decoding."""
    import inspect

    from pipelines.components.video_model_components import (
        FrameSequenceWriter,
        SAM2VideoPropagator,
        TransparentBGVideoExtractor,
        VideoReader,
        VideoWriter,
    )

    assert "progress_callback" in inspect.signature(VideoReader.run).parameters
    assert "progress_callback" in inspect.signature(SAM2VideoPropagator.run).parameters
    assert "progress_callback" in inspect.signature(TransparentBGVideoExtractor.run).parameters
    assert "progress_callback" in inspect.signature(VideoWriter.run).parameters
    assert "progress_callback" in inspect.signature(FrameSequenceWriter.run).parameters


def test_transparent_bg_video_streams_sequence_without_retaining_frames(tmp_path) -> None:
    """Movie matte generation should not keep every RGBA/alpha/preview frame in RAM."""
    import numpy as np

    from pipelines.components.video_model_components import TransparentBGVideoExtractor

    extractor = TransparentBGVideoExtractor(output_dir=str(tmp_path))

    def fake_extract(**kwargs):
        image = np.asarray(kwargs["image"])
        height, width = image.shape[:2]
        rgba = np.dstack([image, np.full((height, width), 255, dtype=np.uint8)])
        alpha = np.full((height, width), 255, dtype=np.uint8)
        return {"rgba": rgba, "alpha": alpha, "preview": image}

    extractor.extractor.run = fake_extract
    frames = [
        np.full((4, 5, 3), 10, dtype=np.uint8),
        np.full((4, 5, 3), 20, dtype=np.uint8),
    ]

    matte = extractor.run(frames=frames, output_mode="sequence")["matte"]

    assert matte["metadata"]["streamed_outputs"] is True
    assert matte["rgba_frames"] == []
    assert matte["alpha_frames"] == []
    assert matte["preview_frames"] == []
    assert Path(matte["rgba_sequence_dir"], "frame_000000.png").exists()
    assert Path(matte["alpha_sequence_dir"], "frame_000001.png").exists()


def test_writers_pass_through_streamed_matte_without_frame_lists(tmp_path) -> None:
    """Legacy writer components should accept compact matte dicts produced by streaming extraction."""
    from pipelines.components.video_model_components import FrameSequenceWriter, VideoWriter

    matte = {
        "output_mode": "both",
        "rgba_frames": [],
        "alpha_frames": [],
        "preview_frames": [],
        "rgba_video_path": str(tmp_path / "rgba.webm"),
        "alpha_video_path": str(tmp_path / "alpha.mp4"),
        "preview_video_path": str(tmp_path / "preview.mp4"),
        "rgba_sequence_dir": str(tmp_path / "rgba"),
        "alpha_sequence_dir": str(tmp_path / "alpha"),
        "preview_sequence_dir": str(tmp_path / "preview"),
    }

    assert VideoWriter(output_dir=str(tmp_path)).run(matte=matte)["matte"] is matte
    assert FrameSequenceWriter(output_dir=str(tmp_path)).run(matte=matte)["matte"] is matte


def test_movie_app_exposes_text_prompt_to_box_flow() -> None:
    """Movie UI should preserve semantic text-prompt object selection for compound subjects."""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "Optional: Text Prompt to Box (GroundingDINO)" in source
    assert "GroundingDINOMultiBoxDetector" in source
    assert "Text Prompt から bbox を検出" in source
    assert "person playing drums" in source
    assert "person riding bicycle" in source
    assert "Detected boxes" in source
    assert "STAGE_PROGRESS_RANGES" in source
    assert "build_video_progress_callback" in source
    assert "release_text_detector" in source


def test_text_prompt_detection_copies_top_box_to_video_prompt(monkeypatch) -> None:
    """The text-prompt flow should initialize the first-frame SAM2 video bbox."""
    import numpy as np

    import gradio_app_sam2_transparent_BG_haystack_for_Movie as app

    class DummyDetector:
        def run(self, **kwargs):
            return {
                "boxes": np.asarray([[10.2, 20.6, 100.1, 120.9], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                "phrases": ["person playing drums", "drum"],
                "confidences": np.asarray([0.91, 0.72], dtype=np.float32),
            }

    monkeypatch.setattr(app, "get_text_detector", lambda: DummyDetector())
    frame = np.zeros((160, 200, 3), dtype=np.uint8)

    preview, state, rows, status = app.detect_text_boxes_for_video(
        frame,
        "person playing drums",
        0.25,
        0.25,
        5,
        None,
    )

    assert preview.shape == frame.shape
    assert state["box"] == [10, 21, 100, 121]
    assert rows[0][1] == "person playing drums"
    assert "Top bbox copied" in status


def test_release_text_detector_clears_groundingdino_cache(monkeypatch) -> None:
    """Video execution should be able to free semantic detector memory before heavy matting."""
    import gradio_app_sam2_transparent_BG_haystack_for_Movie as app

    app.TEXT_DETECTOR = object()
    app.release_text_detector()

    assert app.TEXT_DETECTOR is None


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
