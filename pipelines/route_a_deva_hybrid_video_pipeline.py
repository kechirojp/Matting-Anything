"""DEVA 方式 + BEN2/TB ハイブリッド背景除去動画 Pipeline。

上流は既存 DEVA 方式 tracker（GroundingDINO → SAM2 → consensus）を再利用し、
下流の alpha 生成だけを ``BEN2TransparentHybridVideoExtractor`` へ差し替える。
I/O 契約は ``masks`` と ``matte`` の dict socket に固定し、tracker / alpha 生成 / writer を
疎結合に保つ。
"""

from __future__ import annotations

from haystack import Pipeline

from .components.deva_semi_online_tracker import DevaSemiOnlineTracker
from .components.hybrid_alpha_components import BEN2TransparentHybridVideoExtractor
from .components.video_model_components import (
    FrameSequenceWriter,
    TrackingOverlayWriter,
    VideoReader,
    VideoWriter,
)

DEVA_TRACKER_COMPONENT_NAME = "deva_semi_online_tracker"
HYBRID_COMPONENT_NAME = "ben2_tb_hybrid_video"


def build_sam2_ben2_tb_deva_hybrid_pipeline(
    tracker: DevaSemiOnlineTracker | None = None,
    extractor: BEN2TransparentHybridVideoExtractor | None = None,
) -> Pipeline:
    """DEVA masks → BEN2/TB hybrid alpha → writers の Pipeline を構築する。

    Args:
        tracker: 事前構築済みの ``DevaSemiOnlineTracker``。None なら既定を使う。
        extractor: 事前構築済みの ``BEN2TransparentHybridVideoExtractor``。None なら既定を使う。

    Returns:
        Haystack ``Pipeline``。
    """
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component(DEVA_TRACKER_COMPONENT_NAME, tracker or DevaSemiOnlineTracker())
    pipeline.add_component(HYBRID_COMPONENT_NAME, extractor or BEN2TransparentHybridVideoExtractor())
    pipeline.add_component("video_writer", VideoWriter())
    pipeline.add_component("frame_sequence_writer", FrameSequenceWriter())
    pipeline.add_component("tracking_overlay", TrackingOverlayWriter())

    pipeline.connect("video_reader.frames", f"{DEVA_TRACKER_COMPONENT_NAME}.frames")
    pipeline.connect("video_reader.metadata", f"{DEVA_TRACKER_COMPONENT_NAME}.metadata")
    pipeline.connect("video_reader.frames", f"{HYBRID_COMPONENT_NAME}.frames")
    pipeline.connect("video_reader.metadata", f"{HYBRID_COMPONENT_NAME}.metadata")
    pipeline.connect(f"{DEVA_TRACKER_COMPONENT_NAME}.masks", f"{HYBRID_COMPONENT_NAME}.masks")
    pipeline.connect(f"{HYBRID_COMPONENT_NAME}.matte", "video_writer.matte")
    pipeline.connect(f"{HYBRID_COMPONENT_NAME}.matte", "frame_sequence_writer.matte")
    pipeline.connect("video_reader.frames", "tracking_overlay.frames")
    pipeline.connect("video_reader.metadata", "tracking_overlay.metadata")
    pipeline.connect(f"{DEVA_TRACKER_COMPONENT_NAME}.masks", "tracking_overlay.masks")
    return pipeline
