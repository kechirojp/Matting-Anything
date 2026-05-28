"""MAM + GroundingDINO + 背景合成の Haystack Pipeline。"""

from __future__ import annotations

from haystack import Pipeline

from .components import BackgroundGenerator, GroundingDINODetector, ImageNormalizer, MAMAlphaPredictor, OutputSaver, ScribbleParser


def build_mam_pipeline() -> Pipeline:
    """テキストガイド MAM 推論 Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("image_normalizer", ImageNormalizer())
    pipeline.add_component("grounding_dino", GroundingDINODetector())
    pipeline.add_component("alpha_predictor", MAMAlphaPredictor())
    pipeline.add_component("background_generator", BackgroundGenerator())
    pipeline.add_component("output_saver", OutputSaver())

    pipeline.connect("image_normalizer.image", "grounding_dino.image")
    pipeline.connect("image_normalizer.image", "alpha_predictor.image")
    pipeline.connect("grounding_dino.box", "alpha_predictor.box")
    pipeline.connect("image_normalizer.image", "background_generator.image")
    pipeline.connect("alpha_predictor.alpha", "background_generator.alpha")
    pipeline.connect("background_generator.composite", "output_saver.preview")
    pipeline.connect("alpha_predictor.alpha_rgb", "output_saver.alpha")
    return pipeline


def build_mam_scribble_pipeline() -> Pipeline:
    """スクリブルガイド MAM 推論 Pipeline を構築する。"""
    pipeline = Pipeline()
    pipeline.add_component("image_normalizer", ImageNormalizer())
    pipeline.add_component("scribble_parser", ScribbleParser())
    pipeline.add_component("alpha_predictor", MAMAlphaPredictor())
    pipeline.add_component("background_generator", BackgroundGenerator())
    pipeline.add_component("output_saver", OutputSaver())

    pipeline.connect("image_normalizer.image", "alpha_predictor.image")
    pipeline.connect("scribble_parser.box", "alpha_predictor.box")
    pipeline.connect("scribble_parser.points", "alpha_predictor.points")
    pipeline.connect("image_normalizer.image", "background_generator.image")
    pipeline.connect("alpha_predictor.alpha", "background_generator.alpha")
    pipeline.connect("background_generator.composite", "output_saver.preview")
    pipeline.connect("alpha_predictor.alpha_rgb", "output_saver.alpha")
    return pipeline