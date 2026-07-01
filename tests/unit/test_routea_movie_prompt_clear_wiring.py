"""RouteA 動画アプリの prompt クリア／個別削除が UI に反映される配線を担保する回帰テスト。

実機不具合「prompt クリアが効かない／UI に反映されない」の根本原因は、overlay 描画の基準に
overlay 焼き込み済みの prompt_canvas を渡していたこと。クリーンフレーム prompt_base_image を
基準にする配線へ修正したことをソーステキストで回帰防止する。あわせて Colab/tunnel 向け stopgap
（Layer B = prewarm）撤去も検査する。
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTEA_MOVIE_APP = Path("gradio_app_sam2_ben2_route_a_for_Movie.py")


def _read() -> str:
    return ROUTEA_MOVIE_APP.read_text(encoding="utf-8")


def test_prompt_base_image_state_exists() -> None:
    """overlay 描画基準となるクリーンフレーム用 gr.State が宣言されている。"""
    app_source = _read()
    assert "prompt_base_image = gr.State(" in app_source


def test_clear_prompt_renders_from_clean_base() -> None:
    """clear はクリーンフレーム prompt_base_image を入力にする（dirty な prompt_canvas ではない）。"""
    app_source = _read()
    match = re.search(r"clear_prompt_btn\.click\(\s*clear_prompt,\s*inputs=\[(?P<inputs>[^\]]*)\]", app_source)
    assert match is not None, "clear_prompt_btn.click(...) が見つからない"
    assert "prompt_base_image" in match.group("inputs")
    assert "prompt_canvas" not in match.group("inputs")


def test_select_and_remove_handlers_render_from_clean_base() -> None:
    """select / 個別削除 の overlay 描画基準が prompt_base_image である。"""
    app_source = _read()

    select_match = re.search(
        r"prompt_canvas\.select\(\s*select_sam2_prompt,\s*inputs=\[(?P<inputs>[^\]]*)\]",
        app_source,
    )
    assert select_match is not None, "prompt_canvas.select(...) が見つからない"
    assert select_match.group("inputs").lstrip().startswith("prompt_base_image")

    for handler in ("remove_selected_prompt_points", "remove_selected_prompt_boxes"):
        match = re.search(
            handler + r",\s*inputs=\[(?P<inputs>[^\]]*)\]",
            app_source,
        )
        assert match is not None, f"{handler} の配線が見つからない"
        assert match.group("inputs").lstrip().startswith("prompt_base_image")


def test_frame_extraction_outputs_update_base_image() -> None:
    """フレーム取得系は canvas と base の両方へクリーンフレームを出力する。"""
    app_source = _read()
    for trigger in (
        r"input_video\.change\(extract_first_frame_outputs",
        r"load_first_frame_btn\.click\(extract_first_frame_with_base",
        r"show_frame_btn\.click\(\s*extract_prompt_frame_with_base",
        r"prompt_frame_idx\.change\(\s*extract_prompt_frame_with_base",
    ):
        match = re.search(trigger + r".*?outputs=\[(?P<outputs>[^\]]*)\]", app_source, flags=re.DOTALL)
        assert match is not None, f"{trigger} の配線が見つからない"
        assert "prompt_base_image" in match.group("outputs"), f"{trigger} が base を更新しない"


def test_prewarm_stopgap_removed() -> None:
    """Layer B（Colab/tunnel 向け prewarm）は撤去されている。"""
    app_source = _read()
    assert "def prewarm_ben2_models" not in app_source
    assert "prewarm_ben2_models()" not in app_source
    assert "warm_up_ben2_in_pipelines" not in app_source
