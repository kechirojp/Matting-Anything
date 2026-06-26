from __future__ import annotations

from haystack import Pipeline

from pipelines.route_a_video_pipeline import (
    build_ben2_route_a_only_video_pipeline,
    build_sam2_ben2_route_a_pipeline,
    warm_up_ben2_in_pipelines,
)


def test_route_a_only_pipeline_builds_without_sam2() -> None:
    """BEN2 ルートA only 経路は SAM2/所有権/overlay を含まない。"""
    pipeline = build_ben2_route_a_only_video_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "video_reader" in pipeline.graph.nodes
    assert "ben2_route_a_video" in pipeline.graph.nodes
    assert "video_writer" in pipeline.graph.nodes
    assert "frame_sequence_writer" in pipeline.graph.nodes
    assert "sam2_video_propagator" not in pipeline.graph.nodes
    assert "ownership_resolver" not in pipeline.graph.nodes
    assert "tracking_overlay" not in pipeline.graph.nodes


def test_sam2_ben2_route_a_pipeline_builds_with_writers() -> None:
    """end-to-end ルートA Pipeline が既存 TB と同型の構成で構築される。"""
    pipeline = build_sam2_ben2_route_a_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "video_reader" in pipeline.graph.nodes
    assert "sam2_video_propagator" in pipeline.graph.nodes
    assert "ownership_resolver" in pipeline.graph.nodes
    assert "ben2_route_a_video" in pipeline.graph.nodes
    assert "video_writer" in pipeline.graph.nodes
    assert "frame_sequence_writer" in pipeline.graph.nodes
    assert "tracking_overlay" in pipeline.graph.nodes


def test_route_a_pipeline_accepts_injected_propagator() -> None:
    """tracker 選択を反映できるよう、事前構築した propagator を注入できる。"""
    from pipelines.components.video_model_components import SAM2VideoPropagator

    propagator = SAM2VideoPropagator(
        checkpoint_path="checkpoints/SAM2/sam2.1_hiera_large.pt",
        config_name="configs/samurai/sam2.1_hiera_l.yaml",
    )
    pipeline = build_sam2_ben2_route_a_pipeline(propagator=propagator)

    injected = pipeline.get_component("sam2_video_propagator")
    assert injected is propagator
    assert injected.config_name == "configs/samurai/sam2.1_hiera_l.yaml"


def test_route_a_pipeline_accepts_injected_extractor() -> None:
    """BEN2 repo/checkpoint を反映できるよう、事前構築した extractor を注入できる。"""
    from pipelines.components.ben2_components import BEN2RouteAVideoExtractor

    extractor = BEN2RouteAVideoExtractor(repo_id="PramaLLC/BEN2", output_dir="outputs")
    pipeline = build_sam2_ben2_route_a_pipeline(extractor=extractor)

    injected = pipeline.get_component("ben2_route_a_video")
    assert injected is extractor


class _RecordingExtractor:
    """warm_up 呼び出しを記録するダミー extractor。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.warm_up_calls = 0
        self.fail = fail

    def warm_up(self) -> None:
        self.warm_up_calls += 1
        if self.fail:
            raise RuntimeError("ben2 warm-up failed")


class _FakeBen2Component:
    def __init__(self, extractor: _RecordingExtractor) -> None:
        self.extractor = extractor


class _FakePipeline:
    """``get_component('ben2_route_a_video')`` だけを満たす最小ダミー Pipeline。"""

    def __init__(self, extractor: _RecordingExtractor) -> None:
        self._component = _FakeBen2Component(extractor)

    def get_component(self, name: str) -> _FakeBen2Component:
        assert name == "ben2_route_a_video"
        return self._component


def test_warm_up_ben2_in_pipelines_prewarms_all_extractors() -> None:
    """起動前事前ロードは渡された全 Pipeline の BEN2 extractor を warm_up する。"""
    extractors = [_RecordingExtractor(), _RecordingExtractor()]
    pipelines = [_FakePipeline(e) for e in extractors]
    logs: list[str] = []

    warmed = warm_up_ben2_in_pipelines(pipelines, log=logs.append)

    assert warmed == 2
    assert all(e.warm_up_calls == 1 for e in extractors)
    assert any("完了" in line for line in logs)


def test_warm_up_ben2_in_pipelines_continues_and_logs_on_failure() -> None:
    """1 つの warm_up が失敗しても握り潰さず通知し、残りの事前ロードを続行する。"""
    failing = _RecordingExtractor(fail=True)
    healthy = _RecordingExtractor()
    pipelines = [_FakePipeline(failing), _FakePipeline(healthy)]
    logs: list[str] = []

    warmed = warm_up_ben2_in_pipelines(pipelines, log=logs.append)

    assert warmed == 1  # healthy のみ成功
    assert failing.warm_up_calls == 1
    assert healthy.warm_up_calls == 1  # 失敗後も次を続行
    assert any("失敗" in line for line in logs)
