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
from .model_registry import (
    ModelEntry,
    build_dropdown_choices,
    clear_registry_cache,
    entries_for,
    entry_by_id,
    is_available,
    load_model_registry,
)
from .hybrid_alpha_components import (
    BEN2TransparentHybridVideoExtractor,
    build_person_region,
    compose_hybrid_alpha,
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
    "BEN2TransparentHybridVideoExtractor",
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
    "build_person_region",
    "clamp_prompt_point",
    "draw_prompt_overlay",
    "extend_box_to_edge",
    "normalize_box_from_points",
    "select_sam2_prompt",
    "compose_hybrid_alpha",
    # model_registry
    "ModelEntry",
    "build_dropdown_choices",
    "clear_registry_cache",
    "entries_for",
    "entry_by_id",
    "is_available",
    "load_model_registry",
]