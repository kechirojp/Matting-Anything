from __future__ import annotations

from pathlib import Path

from haystack import Pipeline

from pipelines.sam2_tb_video_pipeline import (
    build_sam2_tb_video_pipeline,
    build_sam2_video_propagation_pipeline,
    build_tb_only_video_pipeline,
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


def test_tb_only_video_pipeline_builds_without_sam2() -> None:
    """tb-only 経路は SAM2/所有権/overlay を含まず video_reader→tb→writers のみ。"""
    pipeline = build_tb_only_video_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "video_reader" in pipeline.graph.nodes
    assert "transparent_bg_video" in pipeline.graph.nodes
    assert "video_writer" in pipeline.graph.nodes
    assert "frame_sequence_writer" in pipeline.graph.nodes
    # SAM2 追跡・所有権解決・tracking overlay は tb-only 経路に存在しない。
    assert "sam2_video_propagator" not in pipeline.graph.nodes
    assert "ownership_resolver" not in pipeline.graph.nodes
    assert "tracking_overlay" not in pipeline.graph.nodes


def test_movie_app_exposes_tb_only_tab_and_callback() -> None:
    """Movie UI は SAM2 経路と tb-only 経路をタブで分離し、専用コールバックを配線する。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "gr.Tabs(" in source
    assert "build_tb_only_video_pipeline" in source
    assert "def run_tb_only_background_removal" in source
    assert "get_tb_only_pipeline" in source


def test_movie_app_exposes_manual_mask_guard_controls() -> None:
    """改善3: mask guard 手動調整の checkbox/feather/dilate を UI と run 配線へ公開する（既定 OFF）。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert 'label="Mask guard を手動調整"' in source
    assert 'label="Mask guard feather"' in source
    assert 'label="Mask guard dilate"' in source
    # 既定 OFF（checkbox value=False）と既定値が後方互換であること。
    assert "mask_guard_enabled = gr.Checkbox(" in source
    # run 配線に 3 コンポーネントが含まれること。
    assert "mask_guard_enabled, mask_guard_feather, mask_guard_dilate" in source
    # pipeline 入力に mask_guard_dilate を渡すこと。
    assert '"mask_guard_dilate": mask_guard_dilate' in source


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

    def init_state(self, video_path, **kwargs):  # noqa: D401 - fake
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
    # 修正2: frame mask は soft 確率[0,1]（float）で二値ではない。
    assert masks["object_ids"] == [1, 2]
    for mask in masks["frame_masks"].values():
        assert np.issubdtype(np.asarray(mask).dtype, np.floating)
        assert np.all((mask >= 0.0) & (mask <= 1.0))
        # 左半分(obj1) ∪ 右半分(obj2) = 閾値 0.5 で全面前景。
        assert bool(np.all(mask >= 0.5))


def test_sam2_video_propagator_assigns_points_to_nearest_box(monkeypatch) -> None:
    """修正1(方針1): boxes + points 併用時、各点を最近傍 box の object prompt に同梱する。"""
    import contextlib
    import sys
    import types

    import numpy as np

    # run() 内の `import torch` を最小 stub で満たす（inference_mode のみ使用）。
    torch_stub = types.ModuleType("torch")
    torch_stub.inference_mode = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    from pipelines.components.video_model_components import SAM2VideoPropagator

    captured: dict[str, list] = {"prompts": []}

    class _FakePointBoxPredictor:
        def init_state(self, video_path, **kwargs):
            return {"video_path": video_path}

        def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None, points=None, labels=None):
            captured["prompts"].append(
                {
                    "obj_id": int(obj_id),
                    "has_box": box is not None,
                    "points": None if points is None else np.asarray(points).tolist(),
                    "labels": None if labels is None else np.asarray(labels).tolist(),
                }
            )

        def propagate_in_video(self, state, reverse=False):
            height, width = 4, 6
            for frame_idx in range(2):
                obj1 = np.full((1, height, width), -1.0, dtype=np.float32)
                obj1[0, :, :3] = 1.0  # 左半分
                obj2 = np.full((1, height, width), -1.0, dtype=np.float32)
                obj2[0, :, 3:] = 1.0  # 右半分
                yield frame_idx, [1, 2], [obj1, obj2]

    propagator = SAM2VideoPropagator(device="cpu")
    propagator._video_predictor = _FakePointBoxPredictor()

    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]
    result = propagator.run(
        frames=frames,
        boxes=[[0, 0, 2, 3], [3, 0, 5, 3]],
        points=[[1, 1], [4, 2]],
        labels=[1, 0],
        prompt_frame_idx=0,
    )

    # 点群を別 obj にせず、最近傍 box の object prompt に同梱する。
    # 点(1,1)→box1(obj1), 点(4,2)→box2(obj2)。追加 obj(3) は作らない。
    obj_ids = sorted({p["obj_id"] for p in captured["prompts"]})
    assert obj_ids == [1, 2]
    obj1_prompts = [p for p in captured["prompts"] if p["obj_id"] == 1]
    obj2_prompts = [p for p in captured["prompts"] if p["obj_id"] == 2]
    # box1 は box + point(1,1, label=1) を同梱。
    assert any(p["has_box"] and p["points"] == [[1, 1]] and p["labels"] == [1] for p in obj1_prompts)
    # box2 は box + point(4,2, label=0=negative) を同梱（SAM2 内部でくり抜き）。
    assert any(p["has_box"] and p["points"] == [[4, 2]] and p["labels"] == [0] for p in obj2_prompts)

    masks = result["masks"]
    # metadata に points/labels が残る。
    assert masks["metadata"]["points"] == [[1, 1], [4, 2]]
    assert masks["metadata"]["labels"] == [1, 0]
    # object_ids は box の 1,2 のみ（点群用の追加 obj は無い）。
    assert masks["object_ids"] == [1, 2]
    # union は soft 確率で、閾値 0.5 で全面前景。
    for mask in masks["frame_masks"].values():
        assert np.issubdtype(np.asarray(mask).dtype, np.floating)
        assert bool(np.all(mask >= 0.5))


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


def _install_fake_sam2(monkeypatch, tmp_path, *, with_samurai_configs: bool):
    """sys.modules['sam2'] を ``__file__`` 付き fake module に差し替える。"""
    import sys
    import types

    pkg_dir = tmp_path / "sam2"
    (pkg_dir / "configs").mkdir(parents=True, exist_ok=True)
    if with_samurai_configs:
        (pkg_dir / "configs" / "samurai").mkdir(parents=True, exist_ok=True)
    fake = types.ModuleType("sam2")
    fake.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "sam2", fake)


def test_require_samurai_capable_sam2_raises_for_facebook_sam2(monkeypatch, tmp_path) -> None:
    """SAMURAI config なのに installed sam2 が fork でない場合は明確に fail-fast する。"""
    import pytest

    from pipelines.components.video_model_components import _require_samurai_capable_sam2

    _install_fake_sam2(monkeypatch, tmp_path, with_samurai_configs=False)

    with pytest.raises(RuntimeError, match="SAMURAI"):
        _require_samurai_capable_sam2("configs/samurai/sam2.1_hiera_l.yaml")


def test_require_samurai_capable_sam2_passes_for_samurai_fork(monkeypatch, tmp_path) -> None:
    """installed sam2 が configs/samurai を含む fork なら no-op で通す。"""
    from pipelines.components.video_model_components import _require_samurai_capable_sam2

    _install_fake_sam2(monkeypatch, tmp_path, with_samurai_configs=True)

    assert _require_samurai_capable_sam2("configs/samurai/sam2.1_hiera_l.yaml") is None


def test_require_samurai_capable_sam2_noop_for_standard_config(monkeypatch, tmp_path) -> None:
    """非 samurai config では sam2 を import せず no-op（facebook sam2 でも通る）。"""
    from pipelines.components.video_model_components import _require_samurai_capable_sam2

    _install_fake_sam2(monkeypatch, tmp_path, with_samurai_configs=False)

    assert _require_samurai_capable_sam2("configs/sam2.1/sam2.1_hiera_l.yaml") is None


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


# ---------------------------------------------------------------------------
# ERR049: Colab T4 で SAMURAI 動画伝搬が GPU メモリ枯渇で stall する対策
#         （config 駆動で SAMURAI tracker のみ CPU offload を有効化）
# ---------------------------------------------------------------------------


def test_sam2_video_propagator_defaults_disable_cpu_offload() -> None:
    """デフォルト（標準 SAM2 経路）では CPU offload を無効にし、既存挙動を維持する。"""
    from pipelines.components.video_model_components import SAM2VideoPropagator

    propagator = SAM2VideoPropagator(device="cpu")

    assert propagator.offload_video_to_cpu is False
    assert propagator.offload_state_to_cpu is False


def test_sam2_video_propagator_passes_offload_flags_to_init_state(monkeypatch) -> None:
    """ERR049: offload を有効化した propagator は init_state に offload kwargs を渡す。"""
    import contextlib
    import sys
    import types

    import numpy as np

    # run() 内の `import torch` を最小 stub で満たす（inference_mode のみ使用）。
    torch_stub = types.ModuleType("torch")
    torch_stub.inference_mode = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    from pipelines.components.video_model_components import SAM2VideoPropagator

    captured: dict[str, object] = {}

    class _OffloadCapturingPredictor:
        def init_state(self, video_path, offload_video_to_cpu=False, offload_state_to_cpu=False, **kwargs):
            captured["offload_video_to_cpu"] = offload_video_to_cpu
            captured["offload_state_to_cpu"] = offload_state_to_cpu
            return {"video_path": video_path}

        def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None, points=None, labels=None):
            pass

        def propagate_in_video(self, state, reverse=False):
            height, width = 4, 6
            mask = np.full((1, height, width), 1.0, dtype=np.float32)
            yield 0, [1], [mask]

    propagator = SAM2VideoPropagator(
        device="cpu",
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    propagator._video_predictor = _OffloadCapturingPredictor()

    frames = [np.zeros((4, 6, 3), dtype=np.uint8)]
    propagator.run(frames=frames, boxes=[[0, 0, 2, 3]], prompt_frame_idx=0)

    assert captured["offload_video_to_cpu"] is True
    assert captured["offload_state_to_cpu"] is True


def test_samurai_registry_entries_enable_cpu_offload() -> None:
    """ERR049: SAMURAI tracker entry のみ CPU offload を config 駆動で有効化する。"""
    from pipelines.components.model_registry import entries_for

    tracker_entries = {entry["id"]: entry for entry in entries_for("tracker")}

    # SAMURAI entry は offload を有効化（GPU メモリ枯渇 stall 対策）。
    for samurai_id in ("samurai_hiera_l", "samurai_hiera_b_plus"):
        assert samurai_id in tracker_entries, f"SAMURAI entry '{samurai_id}' が registry に無い"
        entry = tracker_entries[samurai_id]
        assert entry.get("offload_video_to_cpu") is True
        assert entry.get("offload_state_to_cpu") is True

    # 標準 SAM2 entry は offload を有効化しない（動作実績のある経路を変えない）。
    for sam2_id in ("sam2_hiera_l", "sam2_hiera_b_plus"):
        if sam2_id in tracker_entries:
            entry = tracker_entries[sam2_id]
            assert entry.get("offload_video_to_cpu", False) is False
            assert entry.get("offload_state_to_cpu", False) is False


def test_movie_app_reads_offload_flags_from_tracker_registry() -> None:
    """ERR049: get_video_pipeline が tracker entry の offload 設定を propagator へ伝える。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert 'tracker_entry.get("offload_video_to_cpu"' in source
    assert 'tracker_entry.get("offload_state_to_cpu"' in source


# ── ERR050: SAMURAI forward-only / autocast / 双方向自動 OFF / 推奨設定明記 ──────────


def test_sam2_video_propagator_default_autocast_is_float16() -> None:
    """ERR050: autocast_dtype の既定は SAMURAI 本家と同じ float16。"""
    from pipelines.components.video_model_components import SAM2VideoPropagator

    propagator = SAM2VideoPropagator(device="cpu")
    assert propagator.autocast_dtype == "float16"


def test_autocast_context_is_nullcontext_on_cpu() -> None:
    """ERR050: CPU では autocast を適用せず既存挙動（nullcontext）を保つ。"""
    import contextlib
    import types

    from pipelines.components.video_model_components import SAM2VideoPropagator

    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.autocast = lambda *a, **k: pytest.fail("CPU で autocast を呼んではならない")

    propagator = SAM2VideoPropagator(device="cpu", autocast_dtype="float16")
    ctx = propagator._autocast_context(torch_stub)
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_context_disabled_when_dtype_none_on_cuda() -> None:
    """ERR050: autocast_dtype を 'none' にすると cuda でも autocast を無効化できる。"""
    import contextlib
    import types

    from pipelines.components.video_model_components import SAM2VideoPropagator

    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "float16"
    torch_stub.autocast = lambda *a, **k: pytest.fail("none 指定時は autocast を呼んではならない")

    propagator = SAM2VideoPropagator(device="cuda", autocast_dtype="none")
    ctx = propagator._autocast_context(torch_stub)
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_context_uses_torch_autocast_on_cuda() -> None:
    """ERR050: cuda かつ float16 指定で torch.autocast('cuda', dtype=float16) を返す。"""
    import types

    from pipelines.components.video_model_components import SAM2VideoPropagator

    captured: dict[str, object] = {}

    class _Marker:
        pass

    marker = _Marker()

    def _autocast(device_type, dtype=None):
        captured["device_type"] = device_type
        captured["dtype"] = dtype
        return marker

    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.autocast = _autocast

    propagator = SAM2VideoPropagator(device="cuda", autocast_dtype="float16")
    ctx = propagator._autocast_context(torch_stub)

    assert ctx is marker
    assert captured["device_type"] == "cuda"
    assert captured["dtype"] == "float16"


def test_tracker_registry_declares_autocast_and_bidirectional_flags() -> None:
    """ERR050: registry が autocast_dtype と supports_bidirectional を config 駆動で持つ。"""
    from pipelines.components.model_registry import entries_for

    tracker_entries = {entry["id"]: entry for entry in entries_for("tracker")}

    # SAMURAI は forward-only（双方向非対応）かつ fp16 autocast。
    for samurai_id in ("samurai_hiera_l", "samurai_hiera_b_plus"):
        assert samurai_id in tracker_entries
        entry = tracker_entries[samurai_id]
        assert entry.get("supports_bidirectional") is False
        assert entry.get("autocast_dtype") == "float16"

    # 標準 SAM2 は双方向対応、かつ autocast は無効（実績のある float32 挙動を維持）。
    for sam2_id in ("sam2_hiera_l", "sam2_hiera_b_plus"):
        if sam2_id in tracker_entries:
            entry = tracker_entries[sam2_id]
            assert entry.get("supports_bidirectional") is True
            assert entry.get("autocast_dtype") == "none"


def test_movie_app_wires_autocast_dtype_from_tracker_registry() -> None:
    """ERR050: get_video_pipeline が tracker entry の autocast_dtype を propagator へ伝える。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")
    assert 'autocast_dtype=tracker_entry.get("autocast_dtype"' in source


def test_movie_app_auto_disables_bidirectional_for_samurai() -> None:
    """ERR050: SAMURAI 選択時は双方向伝播を OFF+無効化し、標準 SAM2 では有効化する。"""
    import importlib

    module = importlib.import_module("gradio_app_sam2_transparent_BG_haystack_for_Movie")
    handler = module.update_bidirectional_for_tracker

    samurai_update = handler("samurai_hiera_l")
    # gr.update は dict 互換。value=False かつ interactive=False を確認。
    assert samurai_update["interactive"] is False
    assert samurai_update.get("value") is False

    sam2_update = handler("sam2_hiera_l")
    assert sam2_update["interactive"] is True


def test_movie_app_wires_tracker_change_to_bidirectional() -> None:
    """ERR050: tracker_model.change が双方向 checkbox を出力に持つよう配線されている。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")
    assert "tracker_model.change(update_bidirectional_for_tracker" in source
    assert "outputs=[bidirectional]" in source


def test_samurai_recommended_settings_documented_in_app_and_notebook() -> None:
    """ERR050: SAMURAI 推奨設定が Gradio 冒頭と notebook 正本(.py) に明記されている。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")
    notebook_source = Path("Sam2_Transparent_Background_Haystack_for_Movie.py").read_text(encoding="utf-8")

    for source in (app_source, notebook_source):
        assert "SAMURAI トラッカー推奨設定" in source
        assert "forward-only" in source


def test_sam2_video_propagator_rejects_multi_object_when_single_object_only() -> None:
    """ERR051: single_object_only な tracker(SAMURAI) に複数 box を渡すと伝搬前に明確に raise。

    SAMURAI fork の `_forward_sam_heads` は B=1 前提（`ious[0][best_iou_inds]`）で、複数 obj だと
    'Boolean value of Tensor with more than one value is ambiguous' になる。samurai/ は変更できない
    ため、伝搬前に config 駆動でガードし actionable に止める。
    """
    import numpy as np
    import pytest

    from pipelines.components.video_model_components import SAM2VideoPropagator

    propagator = SAM2VideoPropagator(device="cpu", single_object_only=True)
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(ValueError) as excinfo:
        propagator.run(frames=frames, boxes=[[0, 0, 2, 3], [3, 0, 5, 3]], prompt_frame_idx=0)
    assert "単一オブジェクト" in str(excinfo.value)
    # warm_up（モデル build）に到達せず fail-fast している。
    assert propagator._video_predictor is None


def test_sam2_video_propagator_allows_single_object_when_single_object_only(monkeypatch) -> None:
    """ERR051: single_object_only でも単一 box なら正常に伝搬する（後方互換）。"""
    import contextlib
    import sys
    import types

    import numpy as np

    torch_stub = types.ModuleType("torch")
    torch_stub.inference_mode = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    from pipelines.components.video_model_components import SAM2VideoPropagator

    class _FakeSingleObjPredictor:
        def init_state(self, video_path, **kwargs):
            return {"video_path": video_path}

        def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None, points=None, labels=None):
            pass

        def propagate_in_video(self, state, reverse=False):
            for frame_idx in range(2):
                mask = np.full((1, 4, 6), 1.0, dtype=np.float32)
                yield frame_idx, [1], [mask]

    propagator = SAM2VideoPropagator(device="cpu", single_object_only=True)
    propagator._video_predictor = _FakeSingleObjPredictor()
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]
    result = propagator.run(frames=frames, box=[0, 0, 5, 3], prompt_frame_idx=0)
    assert result["masks"]["object_ids"] == [1]


def test_tracker_registry_declares_single_object_only_for_samurai() -> None:
    """ERR051: SAMURAI は single_object_only=True、標準 SAM2 は False（config 駆動）。"""
    from pipelines.components.model_registry import entries_for

    tracker_entries = {entry["id"]: entry for entry in entries_for("tracker")}

    for samurai_id in ("samurai_hiera_l", "samurai_hiera_b_plus"):
        assert samurai_id in tracker_entries
        assert tracker_entries[samurai_id].get("single_object_only") is True

    for sam2_id in ("sam2_hiera_l", "sam2_hiera_b_plus"):
        if sam2_id in tracker_entries:
            assert tracker_entries[sam2_id].get("single_object_only") is False


def test_movie_app_wires_single_object_only_from_tracker_registry() -> None:
    """ERR051: get_video_pipeline が tracker entry の single_object_only を propagator へ伝える。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")
    assert 'single_object_only=bool(tracker_entry.get("single_object_only"' in source


# ── ERR055: ループの無い単一ブロッキング処理（モデル DL/ロード）中も keep-alive を送る ──


def test_run_with_progress_keepalive_pumps_during_blocking_work() -> None:
    """ブロッキング work の実行中も一定間隔で進捗(keep-alive)を送る。

    BEN2 の初回モデル DL のように「ループの無い 1 回のブロッキング呼び出し」は
    ``_ProgressKeepAlive``（ループ内で maybe を呼ぶ前提）では覆えず、その間 SSE が
    無通信になり idle 切断される（ERR055=ERR048 follow-up）。本ヘルパは work を別
    スレッドで走らせ、呼び出し側から ``min_interval_sec`` ごとに通知を送り接続を保つ。
    """
    import threading

    from pipelines.components.video_model_components import run_with_progress_keepalive

    calls: list[tuple[str, float, str]] = []
    started = threading.Event()
    release = threading.Event()

    def _blocking_work() -> str:
        started.set()
        # 呼び出し側が複数回 keep-alive を送れるよう、解放されるまでブロックする。
        assert release.wait(2.0), "テスト側が work を解放できなかった"
        return "loaded"

    def progress_callback(stage: str, fraction: float, description: str) -> None:
        calls.append((stage, fraction, description))
        # 2 回 keep-alive を観測したら work を解放して完了させる。
        if len(calls) >= 2:
            release.set()

    result = run_with_progress_keepalive(
        _blocking_work,
        progress_callback,
        "ben2_route_a",
        fraction=0.01,
        description="BEN2 モデルを読み込んでいます",
        min_interval_sec=0.05,
    )

    assert result == "loaded"
    assert started.is_set()
    # ブロッキング中に複数回 keep-alive が送られている（無通信ギャップを作らない）。
    assert len(calls) >= 2, f"keep-alive が十分に送られていない: {calls}"
    assert all(stage == "ben2_route_a" for stage, _f, _d in calls)


def test_run_with_progress_keepalive_sends_unique_payload_each_tick() -> None:
    """各 keep-alive は毎回ユニークなペイロードを送る（同一値の SSE coalesce を防ぐ）。

    ERR056（ERR055 follow-up）: Gradio / gradio.live は同一内容の進捗更新を SSE に
    流さず coalesce することがあり、固定 ``(fraction, description)`` を送り続けると
    実際のワイヤ通信が発生せず idle 切断される。本ヘルパは経過秒を説明文に付加し
    fraction を微小に単調増加させて、連続する通知が必ず異なるようにする。
    """
    import threading

    from pipelines.components.video_model_components import run_with_progress_keepalive

    calls: list[tuple[str, float, str]] = []
    release = threading.Event()

    def _blocking_work() -> str:
        assert release.wait(2.0), "テスト側が work を解放できなかった"
        return "loaded"

    def progress_callback(stage: str, fraction: float, description: str) -> None:
        calls.append((stage, fraction, description))
        if len(calls) >= 3:
            release.set()

    result = run_with_progress_keepalive(
        _blocking_work,
        progress_callback,
        "ben2_route_a",
        fraction=0.01,
        description="BEN2 モデルを読み込んでいます",
        min_interval_sec=0.05,
    )

    assert result == "loaded"
    assert len(calls) >= 3, f"keep-alive が十分に送られていない: {calls}"
    payloads = [(fraction, description) for _stage, fraction, description in calls]
    # 連続する通知は必ず異なるペイロード（fraction か description のどちらかが変化）。
    for previous, current in zip(payloads, payloads[1:]):
        assert previous != current, f"連続する keep-alive が同一ペイロード: {previous!r}"
    # 説明文に基準テキストを保ったまま経過情報を付加している。
    assert all("BEN2 モデルを読み込んでいます" in description for _s, _f, description in calls)
    # fraction は基準値以上で、stage 範囲を侵食しない微小 nudge に収まる。
    assert all(0.01 <= fraction <= 0.01 + 9e-4 + 1e-9 for _s, fraction, _d in calls)


def test_run_with_progress_keepalive_reraises_work_error() -> None:
    """work が送出した例外は握り潰さず呼び出し側へ再送出する。"""
    from pipelines.components.video_model_components import run_with_progress_keepalive

    def _failing_work() -> None:
        raise RuntimeError("load failed")

    raised: list[RuntimeError] = []
    try:
        run_with_progress_keepalive(
            _failing_work,
            lambda *_: None,
            "ben2_route_a",
            fraction=0.0,
            description="x",
            min_interval_sec=0.05,
        )
    except RuntimeError as exc:
        raised.append(exc)

    assert raised and str(raised[0]) == "load failed"


def test_run_with_progress_keepalive_no_callback_runs_directly() -> None:
    """progress_callback が None なら work をそのまま実行し戻り値を返す。"""
    from pipelines.components.video_model_components import run_with_progress_keepalive

    result = run_with_progress_keepalive(
        lambda: 42,
        None,
        "ben2_route_a",
        fraction=0.0,
        description="x",
    )

    assert result == 42


