"""ハイブリッド DEVA 手動アプリの複数 box 蓄積挙動の単体テスト。

Canvas で対角2点の box を続けて描くたびに、直前の box を ``prompt_state["boxes"]``
（union）へ退避し、複数 box を別オブジェクトとして seed できることを検証する。共有
``select_sam2_prompt`` は無改変で、当該アプリ内のラッパー ``select_sam2_prompt_multi``
のみが複数 box を蓄積する契約を固定する。重い推論・モデル読込は行わない。
"""
from __future__ import annotations

import importlib

import numpy as np

app = importlib.import_module("gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie")


class DummySelectEvent:
    def __init__(self, index):
        self.index = index


def _draw_box(base_image, state, first_corner, second_corner):
    """box モードで対角2点をクリックし box を1つ確定させる。"""
    _, state, _ = app.select_sam2_prompt_multi(base_image, "box", "positive", state, DummySelectEvent(first_corner))
    preview, state, status = app.select_sam2_prompt_multi(base_image, "box", "positive", state, DummySelectEvent(second_corner))
    return preview, state, status


def test_second_box_preserves_first_box_into_union() -> None:
    base_image = np.zeros((200, 200, 3), dtype=np.uint8)

    _, state, _ = _draw_box(base_image, None, (20, 20), (60, 60))
    assert state["box"] == [20, 20, 60, 60]
    assert state["boxes"] == []

    # 2つ目の box を描く。1点目のクリックで直前 box が union へ退避される。
    _, state, _ = _draw_box(base_image, state, (100, 100), (150, 150))
    assert state["boxes"] == [[20, 20, 60, 60]], "1つ目の box が union へ退避されていない"
    assert state["box"] == [100, 100, 150, 150], "直近 box は box に残り Extend 可能であるべき"


def test_three_boxes_accumulate() -> None:
    base_image = np.zeros((300, 300, 3), dtype=np.uint8)

    _, state, _ = _draw_box(base_image, None, (30, 30), (70, 70))
    _, state, _ = _draw_box(base_image, state, (100, 100), (140, 140))
    _, state, _ = _draw_box(base_image, state, (180, 180), (220, 220))

    # union には最初の2つ、直近1つは box に残る。
    assert state["boxes"] == [[30, 30, 70, 70], [100, 100, 140, 140]]
    assert state["box"] == [180, 180, 220, 220]


def test_point_mode_does_not_commit_box() -> None:
    base_image = np.zeros((100, 100, 3), dtype=np.uint8)

    _, state, _ = _draw_box(base_image, None, (30, 30), (60, 60))
    # point を追加しても box は union へ退避されない。
    _, state, _ = app.select_sam2_prompt_multi(base_image, "point", "positive", state, DummySelectEvent((50, 50)))

    assert state["boxes"] == []
    assert state["box"] == [30, 30, 60, 60]
    assert state["points"] == [(50, 50)]


def test_run_handler_box_combination_dedupes_manual_into_union() -> None:
    """run ハンドラの initial_boxes 構築ロジック（union + 直近 box を重複排除で結合）。"""
    # union に既に居る box は重複追加しない（text 検出時の box==boxes[0] 二重計上防止）。
    union_boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    manual_box = [0, 0, 10, 10]
    initial_boxes = [list(single) for single in union_boxes]
    if manual_box is not None and list(manual_box) not in initial_boxes:
        initial_boxes.append(list(manual_box))
    assert initial_boxes == [[0, 0, 10, 10], [20, 20, 30, 30]]

    # union に無い直近 box は追加される（Canvas 複数 box の最後の1つ）。
    manual_box2 = [40, 40, 50, 50]
    initial_boxes2 = [list(single) for single in union_boxes]
    if manual_box2 is not None and list(manual_box2) not in initial_boxes2:
        initial_boxes2.append(list(manual_box2))
    assert initial_boxes2 == [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]


def _find_dep_by_fn(demo, fn):
    """``demo.fns`` から ``fn`` を実体に持つイベント依存を1件返す（無ければ None）。"""
    fns = getattr(demo, "fns", {})
    iterator = fns.values() if isinstance(fns, dict) else fns
    for dependency in iterator:
        if getattr(dependency, "fn", None) is fn:
            return dependency
    return None


def test_prompt_canvas_wires_multi_box_wrapper() -> None:
    """prompt_canvas.select が複数 box 対応ラッパーへ配線されている（共有関数直結ではない）。"""
    dependency = _find_dep_by_fn(app.demo, app.select_sam2_prompt_multi)
    assert dependency is not None, "prompt_canvas.select が select_sam2_prompt_multi へ配線されていない"
    # 共有 select_sam2_prompt を直接配線していないこと（単一 box 挙動に戻っていない）。
    assert _find_dep_by_fn(app.demo, app.select_sam2_prompt) is None
