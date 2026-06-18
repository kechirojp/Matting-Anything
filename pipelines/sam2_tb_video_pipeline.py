"""SAM2 + transparent-background 動画用 Haystack Pipeline。"""

from __future__ import annotations

from haystack import Pipeline

from .components.video_model_components import (
    FrameSequenceWriter,
    SAM2VideoPropagator,
    TrackingOverlayWriter,
    TransparentBGVideoExtractor,
    VideoReader,
    VideoWriter,
)


def build_video_reader_pipeline() -> Pipeline:
    """動画を frame list と metadata に分解する軽量 Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    return pipeline


def build_sam2_video_propagation_pipeline() -> Pipeline:
    """VideoReader + SAM2 video propagation の Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component("sam2_video_propagator", SAM2VideoPropagator())

    pipeline.connect("video_reader.frames", "sam2_video_propagator.frames")
    pipeline.connect("video_reader.metadata", "sam2_video_propagator.metadata")
    return pipeline


def build_tb_only_video_pipeline() -> Pipeline:
    """背景除去モデル(transparent-background)のみで動画を処理する軽量 Pipeline を構築する。

    SAM2 追跡・所有権解決・tracking overlay を含まず、各フレームを mask 無しで全画面 tb に
    渡す。グリーンバックや単一 salient 対象など、追跡なしで背景除去だけで足りる用途向け。
    ``TransparentBGVideoExtractor`` は ``masks`` 未接続（None）のとき各フレームを mask=None で
    処理し全画面を tb に渡すため、crop も guard も行わない。
    """
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component("transparent_bg_video", TransparentBGVideoExtractor())
    pipeline.add_component("video_writer", VideoWriter())
    pipeline.add_component("frame_sequence_writer", FrameSequenceWriter())

    pipeline.connect("video_reader.frames", "transparent_bg_video.frames")
    pipeline.connect("video_reader.metadata", "transparent_bg_video.metadata")
    pipeline.connect("transparent_bg_video.matte", "video_writer.matte")
    pipeline.connect("transparent_bg_video.matte", "frame_sequence_writer.matte")
    return pipeline


def build_sam2_tb_video_pipeline(propagator: SAM2VideoPropagator | None = None) -> Pipeline:
    """動画背景除去の end-to-end Pipeline を構築する。

    Args:
        propagator: 事前構築済みの SAM2VideoPropagator。tracker モデル選択を反映するために
            registry から構築したインスタンスを注入できる。None の場合は既定 SAM2 を構築する。
    """
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component("sam2_video_propagator", propagator or SAM2VideoPropagator())
    # OwnershipResolver inserted to convert per-object logits -> soft ownership maps
    from .components.ownership_resolver import OwnershipResolver
    pipeline.add_component("ownership_resolver", OwnershipResolver())
    pipeline.add_component("transparent_bg_video", TransparentBGVideoExtractor())
    pipeline.add_component("video_writer", VideoWriter())
    pipeline.add_component("frame_sequence_writer", FrameSequenceWriter())
    pipeline.add_component("tracking_overlay", TrackingOverlayWriter())

    pipeline.connect("video_reader.frames", "sam2_video_propagator.frames")
    pipeline.connect("video_reader.metadata", "sam2_video_propagator.metadata")
    pipeline.connect("video_reader.frames", "transparent_bg_video.frames")
    pipeline.connect("video_reader.metadata", "transparent_bg_video.metadata")
    # expect sam2_video_propagator to emit per-object logits under masks['per_object_logits']
    pipeline.connect("sam2_video_propagator.masks", "ownership_resolver.masks")
    pipeline.connect("ownership_resolver.masks", "transparent_bg_video.masks")
    pipeline.connect("transparent_bg_video.matte", "video_writer.matte")
    pipeline.connect("transparent_bg_video.matte", "frame_sequence_writer.matte")
    pipeline.connect("video_reader.frames", "tracking_overlay.frames")
    pipeline.connect("video_reader.metadata", "tracking_overlay.metadata")
    pipeline.connect("sam2_video_propagator.masks", "tracking_overlay.masks")
    return pipeline
