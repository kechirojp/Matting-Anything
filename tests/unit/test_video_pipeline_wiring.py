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
    assert "tracking_overlay" in pipeline.graph.nodes


def test_sam2_tb_video_pipeline_accepts_injected_propagator() -> None:
    """tracker 選択を反映できるよう、事前構築した propagator を注入できる（Gap A）。"""
    from pipelines.components.video_model_components import SAM2VideoPropagator

    propagator = SAM2VideoPropagator(
        checkpoint_path="checkpoints/SAM2/sam2.1_hiera_large.pt",
        config_name="configs/samurai/sam2.1_hiera_l.yaml",
    )

    pipeline = build_sam2_tb_video_pipeline(propagator=propagator)

    injected = pipeline.get_component("sam2_video_propagator")
    assert injected is propagator
    assert injected.config_name == "configs/samurai/sam2.1_hiera_l.yaml"


def test_sam2_tb_video_pipeline_defaults_propagator_when_none() -> None:
    """propagator 未指定なら従来通り既定 SAM2 を構築する（後方互換）。"""
    pipeline = build_sam2_tb_video_pipeline()

    assert "sam2_video_propagator" in pipeline.graph.nodes


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

    assert "## 1. フレーム取得系" in source
    assert "## 2. DINO系" in source
    assert "## 3. SAM系" in source
    assert "## 4. 背景透過系" in source
    assert 'elem_id="movie-input-video"' in source
    assert 'elem_id="prompt-frame-idx"' in source
    assert "build_video_seek_sync_js" not in source
    assert 'elem_id="movie-video-fps"' not in source
    assert "Optional: Text Prompt to Box (GroundingDINO)" in source
    assert "GroundingDINOMultiBoxDetector" in source
    assert "Text Prompt から bbox を検出" in source
    assert "person playing drums" in source
    assert "person riding bicycle" in source
    assert "Detected boxes" in source
    assert "STAGE_PROGRESS_RANGES" in source
    assert "build_video_progress_callback" in source
    assert "release_text_detector" in source


def test_extract_first_frame_outputs_resets_prompt_slider(monkeypatch) -> None:
    """Video upload should auto-load the first frame and reset the prompt-frame slider (no fps output)."""
    import numpy as np

    import gradio_app_sam2_transparent_BG_haystack_for_Movie as app

    class DummyReaderPipeline:
        def run(self, *args, **kwargs):
            return {
                "video_reader": {
                    "frames": [np.zeros((8, 10, 3), dtype=np.uint8)],
                    "metadata": {"width": 10, "height": 8, "fps": 24.0},
                }
            }

    monkeypatch.setattr(app, "get_reader_pipeline", lambda: DummyReaderPipeline())

    preview, state, status, prompt_frame_reset = app.extract_first_frame_outputs("movie.mp4")

    assert preview.shape == (8, 10, 3)
    assert state == app.empty_prompt_state()
    assert "第 1 フレームを取得しました" in status
    assert prompt_frame_reset == {"value": 0, "__type__": "update"}


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


def test_sam2_video_propagator_run_supports_multi_box_and_bidirectional() -> None:
    """Phase B: run() should accept boxes/prompt_frame_idx/bidirectional."""
    import inspect

    from pipelines.components.video_model_components import SAM2VideoPropagator

    params = inspect.signature(SAM2VideoPropagator.run).parameters
    assert "boxes" in params
    assert "prompt_frame_idx" in params
    assert "bidirectional" in params


class _FakeMultiObjVideoPredictor:
    """2 obj を左右半分の mask として返す fake video predictor。"""

    def __init__(self) -> None:
        self.added: list[dict] = []
        self.reverse_calls: list[bool] = []

    def init_state(self, video_path):  # noqa: D401 - fake
        return {"video_path": video_path}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None, points=None, labels=None):
        import numpy as np

        self.added.append(
            {
                "frame_idx": int(frame_idx),
                "obj_id": int(obj_id),
                "box": None if box is None else np.asarray(box).tolist(),
            }
        )

    def propagate_in_video(self, state, reverse=False):
        import numpy as np

        self.reverse_calls.append(bool(reverse))
        height, width = 4, 6
        for frame_idx in range(2):
            obj1 = np.full((1, height, width), -1.0, dtype=np.float32)
            obj1[0, :, :3] = 1.0  # 左半分
            obj2 = np.full((1, height, width), -1.0, dtype=np.float32)
            obj2[0, :, 3:] = 1.0  # 右半分
            yield frame_idx, [1, 2], [obj1, obj2]


def test_sam2_video_propagator_unions_objects_and_runs_two_passes() -> None:
    """Phase B: 複数 box は obj_id 1..N 登録 → frame ごと union、bidirectional で 2-pass。"""
    import pytest

    pytest.importorskip("torch")
    import numpy as np

    from pipelines.components.video_model_components import SAM2VideoPropagator

    propagator = SAM2VideoPropagator(device="cpu")
    fake = _FakeMultiObjVideoPredictor()
    propagator._video_predictor = fake  # warm_up を冪等にスキップさせる

    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]
    result = propagator.run(
        frames=frames,
        boxes=[[0, 0, 2, 3], [3, 0, 5, 3]],
        prompt_frame_idx=1,
        bidirectional=True,
    )

    masks = result["masks"]
    # 2 box が obj_id 1,2 として prompt_frame_idx=1 で登録される。
    assert [item["obj_id"] for item in fake.added] == [1, 2]
    assert all(item["frame_idx"] == 1 for item in fake.added)
    # bidirectional → forward + reverse の 2 pass。
    assert fake.reverse_calls == [False, True]
    # 左半分(obj1) ∪ 右半分(obj2) = 全面 True。
    assert masks["object_ids"] == [1, 2]
    for mask in masks["frame_masks"].values():
        assert bool(np.all(mask))


def test_movie_app_exposes_frame_selection_and_union_controls() -> None:
    """Phase D: 動画 UI にフレーム選択・候補選択・双方向伝播の導線があること。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "apply_selected_boxes" in source
    assert "prompt_frame_idx" in source
    assert "bidirectional" in source
    assert "CheckboxGroup" in source


def test_sam2_video_propagator_reports_samurai_tracker_metadata(monkeypatch) -> None:
    """SAMURAI への差し替えを config 駆動で行い、使用 tracker を mask metadata 用に公開する。"""
    from pipelines.components.video_model_components import SAM2VideoPropagator

    monkeypatch.setenv("SAM2_CONFIG_NAME", "configs/samurai/sam2.1_hiera_l.yaml")
    monkeypatch.setenv("SAM2_CKPT_PATH", "checkpoints/SAM2/sam2.1_hiera_large.pt")

    meta = SAM2VideoPropagator().tracker_metadata()

    assert meta["tracker_config"].endswith("sam2.1_hiera_l.yaml")
    assert meta["tracker_checkpoint"].endswith("sam2.1_hiera_large.pt")
    assert meta["samurai_mode"] is True


def test_sam2_video_propagator_reports_plain_sam2_tracker_metadata(monkeypatch) -> None:
    """通常 SAM2 config では samurai_mode を False として記録する。"""
    from pipelines.components.video_model_components import SAM2VideoPropagator

    monkeypatch.setenv("SAM2_CONFIG_NAME", "configs/sam2.1/sam2.1_hiera_l.yaml")

    assert SAM2VideoPropagator().tracker_metadata()["samurai_mode"] is False


def test_movie_app_exposes_tracking_overlay_ui() -> None:
    """追跡が正しく追従しているか確認できる Tracking Overlay UI を提供する。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "Tracking Overlay" in source
    assert "TrackingOverlayWriter" in source or "tracking_overlay" in source
    assert "overlay_enabled" in source


# ---------------------------------------------------------------------------
# Phase R2: 役割別モデル Dropdown の Gradio UI 対応（動画版）
# ---------------------------------------------------------------------------


def test_movie_app_exposes_tracker_dropdown() -> None:
    """動画版 Gradio は tracker role の Dropdown を提供しなければならない。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "tracker_model" in source
    assert "build_dropdown_choices" in source


def test_movie_app_tracker_dropdown_drives_pipeline_config() -> None:
    """tracker Dropdown の選択が SAM2VideoPropagator の config_name / checkpoint_path を変更する。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    # tracker_model が inputs として渡されるか、または pipeline 構築に使われる
    assert source.count("tracker_model") >= 2


def test_movie_app_exposes_background_model_dropdown() -> None:
    """動画版 Gradio は background role の Dropdown を提供しなければならない。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "background_model" in source
