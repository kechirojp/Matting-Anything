"""DEVA 方式 + 手動プロンプト RouteA 動画アプリの同期実行配線テスト。

本アプリは DEVA 方式（周期再検出 × SAM2 伝播 × consensus）にベースアプリの手動 box/point
prompt canvas を統合した版。run ボタンは ``run_route_a_deva_manual_background_removal`` を
同期直結で呼び（ERR064）、4 出力（RGBA/Alpha/Preview/Tracking Overlay）＋連番2＋status を
戻り値から直接描画する契約を固定する。あわせて prompt canvas の主要配線（box/point seed・
個別削除）が存在することを検証する。

重い推論は走らせず、Gradio Blocks の依存グラフ（``demo.fns``）のみ検証する。
"""
from __future__ import annotations

import importlib

app = importlib.import_module("gradio_app_sam2_ben2_route_a_deva_manual_for_Movie")


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


def test_run_btn_wires_deva_manual_synchronously_with_overlay_output() -> None:
    dependency = _find_dep_by_fn(app.demo, app.run_route_a_deva_manual_background_removal)
    assert dependency is not None, "run ボタンが run_route_a_deva_manual_background_removal へ同期直結していない"
    labels = _output_labels(dependency)
    assert "Tracking Overlay (追跡確認用)" in labels
    assert "Alpha Video" in labels
    assert "Preview Video" in labels
    assert "RGBA Video" in labels


def test_run_btn_inputs_include_prompt_state() -> None:
    """手動 seed を渡すため run の入力に prompt_state が含まれる。"""
    dependency = _find_dep_by_fn(app.demo, app.run_route_a_deva_manual_background_removal)
    assert dependency is not None
    input_labels = [getattr(component, "label", None) for component in dependency.inputs]
    # prompt_state は gr.State で label を持たないため、入力数が拡張されていることで間接確認する。
    assert len(dependency.inputs) == 24, "run 入力数が手動 prompt_state + 検出起点フレーム追加後の契約と一致しない"
    assert "Input Video" in input_labels
    # 検出起点フレーム（被写体最大フレームで seed）スライダが run 入力に含まれる。
    assert "検出起点フレーム（サンプリング後 index）" in input_labels


def test_prompt_canvas_select_wires_manual_seed() -> None:
    """prompt canvas クリックが SAM2 prompt 選択ハンドラへ配線されている。"""
    dependency = _find_dep_by_fn(app.demo, app.select_sam2_prompt)
    assert dependency is not None, "prompt_canvas.select が select_sam2_prompt へ配線されていない"


def test_point_removal_handler_is_wired() -> None:
    """point の個別削除ハンドラが配線されている（negative point の取り消し用）。"""
    dependency = _find_dep_by_fn(app.demo, app.remove_selected_prompt_points)
    assert dependency is not None


def test_deva_pipeline_builder_is_imported() -> None:
    """アプリは DEVA 方式パイプラインビルダーを使用する（単発 SAM2 ではなく tracker 経路）。"""
    from pipelines.route_a_deva_video_pipeline import build_sam2_ben2_route_a_deva_pipeline

    assert app.build_sam2_ben2_route_a_deva_pipeline is build_sam2_ben2_route_a_deva_pipeline


def test_deva_pipeline_is_cached_singleton() -> None:
    """get_route_a_deva_pipeline は同一インスタンスを返す（再初期化を避ける）。"""
    first = app.get_route_a_deva_pipeline()
    second = app.get_route_a_deva_pipeline()
    assert first is second
