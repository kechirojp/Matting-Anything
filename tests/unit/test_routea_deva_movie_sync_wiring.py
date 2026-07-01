"""DEVA 方式 RouteA 動画アプリの同期実行配線テスト。

DEVA 版アプリは text_prompt 駆動（point/box prompt canvas は不要）で、run ボタンは
``run_route_a_deva_background_removal`` を同期直結で呼び、4 出力（RGBA/Alpha/Preview/
Tracking Overlay）＋連番2＋status を戻り値から直接描画する契約を固定する。

重い推論は走らせず、Gradio Blocks の依存グラフ（``demo.fns``）のみ検証する。
"""
from __future__ import annotations

import importlib

app = importlib.import_module("gradio_app_sam2_ben2_route_a_deva_for_Movie")


def _find_dep_by_fn(demo, fn):
    """``demo.fns`` から ``fn`` を実体に持つイベント依存を1件返す（無ければ None）。"""
    fns = getattr(demo, "fns", {})
    iterator = fns.values() if isinstance(fns, dict) else fns
    for dependency in iterator:
        if getattr(dependency, "fn", None) is fn:
            return dependency
    return None


def _output_labels(dependency) -> list[str | None]:
    return [getattr(component, "label", None) for component in dependency.outputs]


def test_run_btn_wires_deva_synchronously_with_overlay_output() -> None:
    dependency = _find_dep_by_fn(app.demo, app.run_route_a_deva_background_removal)
    assert dependency is not None, "run ボタンが run_route_a_deva_background_removal へ同期直結していない"
    labels = _output_labels(dependency)
    assert "Tracking Overlay (追跡確認用)" in labels
    assert "Alpha Video" in labels
    assert "Preview Video" in labels
    assert "RGBA Video" in labels


def test_deva_pipeline_builder_is_imported() -> None:
    """アプリは DEVA 方式パイプラインビルダーを使用する（単発 SAM2 ではなく tracker 経路）。"""
    from pipelines.route_a_deva_video_pipeline import build_sam2_ben2_route_a_deva_pipeline

    assert app.build_sam2_ben2_route_a_deva_pipeline is build_sam2_ben2_route_a_deva_pipeline


def test_deva_pipeline_is_cached_singleton() -> None:
    """get_route_a_deva_pipeline は同一インスタンスを返す（再初期化を避ける）。"""
    first = app.get_route_a_deva_pipeline()
    second = app.get_route_a_deva_pipeline()
    assert first is second
