"""DEVA hybrid 動画アプリの同期実行配線テスト。"""

from __future__ import annotations

import importlib

app = importlib.import_module("gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie")


def _find_dep_by_fn(demo, fn):
    fns = getattr(demo, "fns", {})
    iterator = fns.values() if isinstance(fns, dict) else fns
    for dependency in iterator:
        if getattr(dependency, "fn", None) is fn:
            return dependency
    return None


def _output_labels(dependency) -> list[str | None]:
    return [getattr(component, "label", None) for component in dependency.outputs]


def test_run_btn_wires_deva_hybrid_synchronously_with_overlay_output() -> None:
    dependency = _find_dep_by_fn(app.demo, app.run_deva_hybrid_background_removal)
    assert dependency is not None, "run ボタンが run_deva_hybrid_background_removal へ同期直結していない"
    labels = _output_labels(dependency)
    assert "Tracking Overlay (追跡確認用)" in labels
    assert "Alpha Video" in labels
    assert "Preview Video" in labels
    assert "RGBA Video" in labels


def test_deva_hybrid_pipeline_builder_is_imported() -> None:
    from pipelines.route_a_deva_hybrid_video_pipeline import build_sam2_ben2_tb_deva_hybrid_pipeline

    assert app.build_sam2_ben2_tb_deva_hybrid_pipeline is build_sam2_ben2_tb_deva_hybrid_pipeline


def test_deva_hybrid_pipeline_is_cached_singleton() -> None:
    first = app.get_deva_hybrid_pipeline()
    second = app.get_deva_hybrid_pipeline()
    assert first is second


def test_deva_hybrid_app_exposes_composition_controls() -> None:
    dependency = _find_dep_by_fn(app.demo, app.run_deva_hybrid_background_removal)
    input_labels = [getattr(component, "label", None) for component in dependency.inputs]
    assert "Alpha 合成方式" in input_labels
    assert "transparent-background mode" in input_labels
    assert "人物領域 feather(px)" in input_labels
    # 検出起点フレーム（被写体最大フレームで検出）スライダが run 入力に含まれる。
    assert "検出起点フレーム（サンプリング後 index）" in input_labels
