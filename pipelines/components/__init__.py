"""Matting-Anything 用 Haystack Component。"""

from .common import (
    AlphaCompositor,
    BBoxFromMask,
    ImageNormalizer,
    MaskCandidateSelector,
    MaskDilator,
    MaskPreviewComposer,
    MaskUnion,
    ScribbleParser,
)
from .model_components import (
    BackgroundGenerator,
    ColorDecontaminator,
    GroundingDINODetector,
    GroundingDINOMultiBoxDetector,
    MAMAlphaPredictor,
    OutputSaver,
    SAM2GuardFilter,
    SAM2Segmenter,
    TransparentBGExtractor,
)
from .ui_helpers import (
    clamp_prompt_point,
    draw_prompt_overlay,
    extend_box_to_edge,
    normalize_box_from_points,
    select_sam2_prompt,
)
from .video_common import FrameSampler
from .video_model_components import (
    FrameSequenceWriter,
    SAM2VideoPropagator,
    TransparentBGVideoExtractor,
    VideoReader,
    VideoWriter,
)

__all__ = [
    "AlphaCompositor",
    "BBoxFromMask",
    "BackgroundGenerator",
    "ColorDecontaminator",
    "FrameSampler",
    "FrameSequenceWriter",
    "GroundingDINODetector",
    "GroundingDINOMultiBoxDetector",
    "ImageNormalizer",
    "MAMAlphaPredictor",
    "MaskCandidateSelector",
    "MaskDilator",
    "MaskPreviewComposer",
    "MaskUnion",
    "OutputSaver",
    "SAM2GuardFilter",
    "SAM2Segmenter",
    "SAM2VideoPropagator",
    "ScribbleParser",
    "TransparentBGExtractor",
    "TransparentBGVideoExtractor",
    "VideoReader",
    "VideoWriter",
    "clamp_prompt_point",
    "draw_prompt_overlay",
    "extend_box_to_edge",
    "normalize_box_from_points",
    "select_sam2_prompt",
]