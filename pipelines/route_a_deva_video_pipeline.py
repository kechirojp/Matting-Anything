"""DEVA 方式（周期再検出 + SAM2 伝播 + consensus 統合）動画用 Haystack Pipeline。

ベースの ``route_a_video_pipeline`` と同型の配線・I/O 契約を保ちつつ、単発の
``SAM2VideoPropagator`` を ``DevaSemiOnlineTracker`` に置換する。tracker は内部で
「検出島（GroundingDINO→SAM2）× SAM2 伝播 × consensus 統合 × メモリ整理」を semi-online に
回し、外部 DAG には forward な ``masks``（BEN2 union 契約 + per_object_logits）を出力する。

これにより OwnershipResolver / BEN2RouteAVideoExtractor / VideoWriter / FrameSequenceWriter /
TrackingOverlayWriter をベースとまったく同じ配線で再利用できる（疎結合・差し替え容易性）::

    VideoReader -> DevaSemiOnlineTracker -> OwnershipResolver -> BEN2RouteAVideoExtractor
                -> VideoWriter / FrameSequenceWriter / TrackingOverlayWriter

NOTE: DEVA「ライブラリ」は採用しない。DEVA の「方式」を SAM2.1 / BEN2 / GroundingDINO 上に
再構成したものである（プロジェクト方針）。
"""

from __future__ import annotations

from haystack import Pipeline

from .components.ben2_components import BEN2RouteAVideoExtractor
from .components.deva_semi_online_tracker import DevaSemiOnlineTracker
from .components.video_model_components import (
    FrameSequenceWriter,
    TrackingOverlayWriter,
    VideoReader,
    VideoWriter,
)

DEVA_TRACKER_COMPONENT_NAME = "deva_semi_online_tracker"
BEN2_COMPONENT_NAME = "ben2_route_a_video"


def build_sam2_ben2_route_a_deva_pipeline(
    tracker: DevaSemiOnlineTracker | None = None,
    extractor: BEN2RouteAVideoExtractor | None = None,
) -> Pipeline:
    """DEVA 方式ルートA動画αマットの end-to-end Pipeline を構築する。

    配線は ``build_sam2_ben2_route_a_pipeline`` と同型で、追跡段のみ
    ``DevaSemiOnlineTracker`` に置換する。tracker は ``masks``（BEN2 union 契約 +
    ``per_object_logits``、いずれも source_index でキーイング）を出力するため、
    OwnershipResolver / BEN2 / TrackingOverlay はベースと同一の契約で接続できる。

    Args:
        tracker: 事前構築済みの ``DevaSemiOnlineTracker``。検出島 / propagator / device を
            反映したインスタンスを注入できる。None の場合は既定構成を構築する。
        extractor: 事前構築済みの ``BEN2RouteAVideoExtractor``。BEN2 repo/checkpoint や
            output_dir を反映したインスタンスを注入できる。None の場合は既定を構築する。

    Returns:
        DEVA 方式ルートA動画αマット Pipeline。
    """
    from .components.ownership_resolver import OwnershipResolver

    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component(DEVA_TRACKER_COMPONENT_NAME, tracker or DevaSemiOnlineTracker())
    pipeline.add_component("ownership_resolver", OwnershipResolver())
    pipeline.add_component(BEN2_COMPONENT_NAME, extractor or BEN2RouteAVideoExtractor())
    pipeline.add_component("video_writer", VideoWriter())
    pipeline.add_component("frame_sequence_writer", FrameSequenceWriter())
    pipeline.add_component("tracking_overlay", TrackingOverlayWriter())

    pipeline.connect("video_reader.frames", f"{DEVA_TRACKER_COMPONENT_NAME}.frames")
    pipeline.connect("video_reader.metadata", f"{DEVA_TRACKER_COMPONENT_NAME}.metadata")
    pipeline.connect("video_reader.frames", f"{BEN2_COMPONENT_NAME}.frames")
    pipeline.connect("video_reader.metadata", f"{BEN2_COMPONENT_NAME}.metadata")
    pipeline.connect(f"{DEVA_TRACKER_COMPONENT_NAME}.masks", "ownership_resolver.masks")
    pipeline.connect("ownership_resolver.masks", f"{BEN2_COMPONENT_NAME}.masks")
    pipeline.connect(f"{BEN2_COMPONENT_NAME}.matte", "video_writer.matte")
    pipeline.connect(f"{BEN2_COMPONENT_NAME}.matte", "frame_sequence_writer.matte")
    pipeline.connect("video_reader.frames", "tracking_overlay.frames")
    pipeline.connect("video_reader.metadata", "tracking_overlay.metadata")
    pipeline.connect(f"{DEVA_TRACKER_COMPONENT_NAME}.masks", "tracking_overlay.masks")
    return pipeline
