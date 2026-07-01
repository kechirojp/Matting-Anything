"""RouteA 動画アプリの同期実行配線テスト（ERR064）。

ローカル実行前提では gradio.live トンネルの SSE 切断（ERR058）が無いため、run ボタンは
``run_route_a_background_removal`` / ``run_route_a_only_background_removal`` を同期直結で呼び、
出力（RGBA/Alpha/Preview/Tracking Overlay）が戻り値から直接描画される必要がある。

非同期 Timer ポーリングのままだと出力・標準プログレスが UI に出ない不具合（ERR064）が起きるため、
本テストは「run ボタンのイベントがコア関数へ直結し、4 出力（含 Tracking Overlay）を描く」契約を固定する。
重い推論は走らせず、Gradio Blocks の依存グラフ（``demo.fns``）のみ検証する。
"""
from __future__ import annotations

import importlib

app = importlib.import_module("gradio_app_sam2_ben2_route_a_for_Movie")


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


def test_run_btn_wires_route_a_synchronously_with_overlay_output() -> None:
    dependency = _find_dep_by_fn(app.demo, app.run_route_a_background_removal)
    assert dependency is not None, "run ボタンが run_route_a_background_removal へ同期直結していない"
    labels = _output_labels(dependency)
    assert "Tracking Overlay (追跡確認用)" in labels
    assert "Alpha Video" in labels
    assert "Preview Video" in labels


def test_route_a_only_run_btn_wires_synchronously() -> None:
    dependency = _find_dep_by_fn(app.demo, app.run_route_a_only_background_removal)
    assert dependency is not None, "BEN2 のみ run ボタンが同期直結していない"
    labels = _output_labels(dependency)
    assert "RGBA Video" in labels
    assert "Alpha Video" in labels
    assert "Preview Video" in labels
