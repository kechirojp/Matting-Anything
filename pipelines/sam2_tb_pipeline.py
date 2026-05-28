"""SAM2 + transparent-background の Haystack Pipeline。"""

from __future__ import annotations

from haystack import Pipeline

from .components import (
    ImageNormalizer,
    MaskCandidateSelector,
    MaskPreviewComposer,
    MaskUnion,
    OutputSaver,
    SAM2GuardFilter,
    SAM2Segmenter,
    TransparentBGExtractor,
)


def build_sam2_prompt_pipeline() -> Pipeline:
    """SAM2 の point/box prompt だけを実行する Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("image_normalizer", ImageNormalizer())
    pipeline.add_component("sam2_segmenter", SAM2Segmenter())

    pipeline.connect("image_normalizer.image", "sam2_segmenter.image")
    return pipeline


def build_sam2_maskset_pipeline() -> Pipeline:
    """SAM2 の出力を MaskSet 契約として扱う Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("image_normalizer", ImageNormalizer())
    pipeline.add_component("sam2_segmenter", SAM2Segmenter())
    pipeline.add_component("mask_preview", MaskPreviewComposer())

    pipeline.connect("image_normalizer.image", "sam2_segmenter.image")
    pipeline.connect("image_normalizer.image", "mask_preview.image")
    pipeline.connect("sam2_segmenter.mask_set", "mask_preview.mask_set")
    return pipeline


def build_mask_union_pipeline() -> Pipeline:
    """MaskSet から候補選択と union mask 生成を行う Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("mask_selector", MaskCandidateSelector())
    pipeline.add_component("mask_union", MaskUnion())

    pipeline.connect("mask_selector.mask_set", "mask_union.mask_set")
    return pipeline


def build_mask_to_matte_pipeline() -> Pipeline:
    """標準 mask 契約から transparent-background matte を生成する Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("image_normalizer", ImageNormalizer())
    pipeline.add_component("transparent_bg", TransparentBGExtractor())
    pipeline.add_component("sam2_guard", SAM2GuardFilter())
    pipeline.add_component("output_saver", OutputSaver())

    pipeline.connect("image_normalizer.image", "transparent_bg.image")
    pipeline.connect("transparent_bg.alpha", "sam2_guard.alpha")
    pipeline.connect("transparent_bg.rgba", "output_saver.rgba")
    pipeline.connect("sam2_guard.alpha", "output_saver.alpha")
    pipeline.connect("transparent_bg.preview", "output_saver.preview")
    return pipeline


def build_sam2_union_tb_pipeline() -> Pipeline:
    """SAM2 MaskSet → candidate union → tb matte 抽出を 1 DAG で扱う Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("image_normalizer", ImageNormalizer())
    pipeline.add_component("sam2_segmenter", SAM2Segmenter())
    pipeline.add_component("mask_selector", MaskCandidateSelector())
    pipeline.add_component("mask_union", MaskUnion())
    pipeline.add_component("transparent_bg", TransparentBGExtractor())
    pipeline.add_component("sam2_guard", SAM2GuardFilter())
    pipeline.add_component("output_saver", OutputSaver())

    pipeline.connect("image_normalizer.image", "sam2_segmenter.image")
    pipeline.connect("sam2_segmenter.mask_set", "mask_selector.mask_set")
    pipeline.connect("mask_selector.mask_set", "mask_union.mask_set")
    pipeline.connect("image_normalizer.image", "transparent_bg.image")
    pipeline.connect("mask_union.mask", "transparent_bg.mask")
    pipeline.connect("transparent_bg.alpha", "sam2_guard.alpha")
    pipeline.connect("mask_union.mask", "sam2_guard.mask")
    pipeline.connect("transparent_bg.rgba", "output_saver.rgba")
    pipeline.connect("sam2_guard.alpha", "output_saver.alpha")
    pipeline.connect("transparent_bg.preview", "output_saver.preview")
    return pipeline


def build_sam2_tb_pipeline(include_sam2: bool = True) -> Pipeline:
    """SAM2 + tb 推論 Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("image_normalizer", ImageNormalizer())
    if include_sam2:
        pipeline.add_component("sam2_segmenter", SAM2Segmenter())
    pipeline.add_component("transparent_bg", TransparentBGExtractor())
    pipeline.add_component("sam2_guard", SAM2GuardFilter())
    pipeline.add_component("output_saver", OutputSaver())

    if include_sam2:
        pipeline.connect("image_normalizer.image", "sam2_segmenter.image")
    pipeline.connect("image_normalizer.image", "transparent_bg.image")
    pipeline.connect("transparent_bg.alpha", "sam2_guard.alpha")
    pipeline.connect("transparent_bg.rgba", "output_saver.rgba")
    pipeline.connect("sam2_guard.alpha", "output_saver.alpha")
    pipeline.connect("transparent_bg.preview", "output_saver.preview")
    return pipeline