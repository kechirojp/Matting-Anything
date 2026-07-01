from __future__ import annotations

from haystack import Pipeline

from pipelines.route_a_deva_hybrid_video_pipeline import build_sam2_ben2_tb_deva_hybrid_pipeline


def test_deva_hybrid_pipeline_builds_with_writers() -> None:
    pipeline = build_sam2_ben2_tb_deva_hybrid_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "video_reader" in pipeline.graph.nodes
    assert "deva_semi_online_tracker" in pipeline.graph.nodes
    assert "ben2_tb_hybrid_video" in pipeline.graph.nodes
    assert "video_writer" in pipeline.graph.nodes
    assert "frame_sequence_writer" in pipeline.graph.nodes
    assert "tracking_overlay" in pipeline.graph.nodes
    assert "ownership_resolver" not in pipeline.graph.nodes
    assert "ben2_route_a_video" not in pipeline.graph.nodes


def test_deva_hybrid_pipeline_wires_tracker_masks_to_hybrid_and_overlay() -> None:
    pipeline = build_sam2_ben2_tb_deva_hybrid_pipeline()
    edges = {(u, v) for u, v in pipeline.graph.edges()}

    assert ("video_reader", "deva_semi_online_tracker") in edges
    assert ("deva_semi_online_tracker", "ben2_tb_hybrid_video") in edges
    assert ("deva_semi_online_tracker", "tracking_overlay") in edges
    assert ("ben2_tb_hybrid_video", "video_writer") in edges
    assert ("ben2_tb_hybrid_video", "frame_sequence_writer") in edges


def test_deva_hybrid_pipeline_accepts_injected_tracker_and_extractor() -> None:
    from pipelines.components.deva_semi_online_tracker import DevaSemiOnlineTracker
    from pipelines.components.hybrid_alpha_components import BEN2TransparentHybridVideoExtractor

    tracker = DevaSemiOnlineTracker()
    extractor = BEN2TransparentHybridVideoExtractor(output_dir="outputs")

    pipeline = build_sam2_ben2_tb_deva_hybrid_pipeline(tracker=tracker, extractor=extractor)

    assert pipeline.get_component("deva_semi_online_tracker") is tracker
    assert pipeline.get_component("ben2_tb_hybrid_video") is extractor
