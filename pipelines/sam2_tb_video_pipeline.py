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


def build_sam2_tb_video_pipeline() -> Pipeline:
    """動画背景除去の end-to-end Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component("sam2_video_propagator", SAM2VideoPropagator())
    pipeline.add_component("transparent_bg_video", TransparentBGVideoExtractor())
    pipeline.add_component("video_writer", VideoWriter())
    pipeline.add_component("frame_sequence_writer", FrameSequenceWriter())
    pipeline.add_component("tracking_overlay", TrackingOverlayWriter())

    pipeline.connect("video_reader.frames", "sam2_video_propagator.frames")
    pipeline.connect("video_reader.metadata", "sam2_video_propagator.metadata")
    pipeline.connect("video_reader.frames", "transparent_bg_video.frames")
    pipeline.connect("video_reader.metadata", "transparent_bg_video.metadata")
    pipeline.connect("sam2_video_propagator.masks", "transparent_bg_video.masks")
    pipeline.connect("transparent_bg_video.matte", "video_writer.matte")
    pipeline.connect("transparent_bg_video.matte", "frame_sequence_writer.matte")
    pipeline.connect("video_reader.frames", "tracking_overlay.frames")
    pipeline.connect("video_reader.metadata", "tracking_overlay.metadata")
    pipeline.connect("sam2_video_propagator.masks", "tracking_overlay.masks")
    return pipeline
