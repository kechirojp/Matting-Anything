from __future__ import annotations

from haystack import Pipeline

from pipelines.route_a_deva_video_pipeline import build_sam2_ben2_route_a_deva_pipeline


def test_deva_pipeline_builds_with_writers() -> None:
    """DEVA 方式 end-to-end Pipeline がベース RouteA と同型の writer 構成で構築される。"""
    pipeline = build_sam2_ben2_route_a_deva_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "video_reader" in pipeline.graph.nodes
    assert "deva_semi_online_tracker" in pipeline.graph.nodes
    assert "ownership_resolver" in pipeline.graph.nodes
    assert "ben2_route_a_video" in pipeline.graph.nodes
    assert "video_writer" in pipeline.graph.nodes
    assert "frame_sequence_writer" in pipeline.graph.nodes
    assert "tracking_overlay" in pipeline.graph.nodes
    # DEVA はベースの単発 SAM2 propagator をコーディネータに置換する。
    assert "sam2_video_propagator" not in pipeline.graph.nodes


def test_deva_pipeline_wires_tracker_to_ownership_and_overlay() -> None:
    """tracker.masks が ownership_resolver と tracking_overlay の双方へ配線される。"""
    pipeline = build_sam2_ben2_route_a_deva_pipeline()
    edges = {(u, v) for u, v in pipeline.graph.edges()}

    assert ("video_reader", "deva_semi_online_tracker") in edges
    assert ("deva_semi_online_tracker", "ownership_resolver") in edges
    assert ("ownership_resolver", "ben2_route_a_video") in edges
    assert ("deva_semi_online_tracker", "tracking_overlay") in edges
    assert ("ben2_route_a_video", "video_writer") in edges
    assert ("ben2_route_a_video", "frame_sequence_writer") in edges


def test_deva_pipeline_accepts_injected_tracker() -> None:
    """tracker（DevaSemiOnlineTracker）を注入でき、そのインスタンスが配線される。"""
    from pipelines.components.deva_semi_online_tracker import DevaSemiOnlineTracker

    tracker = DevaSemiOnlineTracker()
    pipeline = build_sam2_ben2_route_a_deva_pipeline(tracker=tracker)

    injected = pipeline.get_component("deva_semi_online_tracker")
    assert injected is tracker


def test_deva_pipeline_accepts_injected_extractor() -> None:
    """BEN2 extractor を注入でき、そのインスタンスが配線される。"""
    from pipelines.components.ben2_components import BEN2RouteAVideoExtractor

    extractor = BEN2RouteAVideoExtractor(repo_id="PramaLLC/BEN2", output_dir="outputs")
    pipeline = build_sam2_ben2_route_a_deva_pipeline(extractor=extractor)

    injected = pipeline.get_component("ben2_route_a_video")
    assert injected is extractor
