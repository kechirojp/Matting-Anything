"""動画版 Gradio アプリの UI 実装ギャップ（Gap A〜D）を担保するソーステキストテスト。

3 度の修正でも壊れていた 4 点（実機不具合）を回帰防止するため、
`gradio_app_sam2_transparent_BG_haystack_for_Movie.py` のソースを検査する。
"""

from __future__ import annotations

import re
from pathlib import Path

MOVIE_APP = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py")


def _read() -> str:
    return MOVIE_APP.read_text(encoding="utf-8")


def test_movie_prompt_canvas_disables_upload_sources() -> None:
    """Gap D: SAM2 Prompt Canvas はアップロード先ではなくクリック編集面（ERR026）。"""
    app_source = _read()

    match = re.search(r"prompt_canvas = gr\.Image\((.*?)\n\s*\)", app_source, flags=re.DOTALL)
    assert match is not None, "prompt_canvas = gr.Image(...) が見つからない"
    prompt_canvas_source = match.group(1)

    assert "value=create_prompt_canvas_placeholder()" in prompt_canvas_source
    assert 'label="SAM2 Prompt Canvas"' in prompt_canvas_source
    assert "sources=[]" in prompt_canvas_source
    assert "interactive=True" in prompt_canvas_source
    assert "interactive=False" not in prompt_canvas_source


def test_movie_model_dropdowns_are_visible_not_buried_in_accordion() -> None:
    """Gap B: tracker/background Dropdown は閉じた Advanced アコーディオンより前で定義される。"""
    app_source = _read()

    tracker_pos = app_source.index("tracker_model = gr.Dropdown(")
    background_pos = app_source.index("background_model = gr.Dropdown(")
    advanced_pos = app_source.index('gr.Accordion("Advanced: 動画処理設定"')

    assert tracker_pos < advanced_pos, "tracker_model dropdown が Advanced アコーディオン内に埋もれている"
    assert background_pos < advanced_pos, "background_model dropdown が Advanced アコーディオン内に埋もれている"


def test_movie_prompt_frame_slider_updates_canvas_and_seek_sync_removed() -> None:
    """スライダー1本集約: prompt_frame_idx.change が extract_prompt_frame を使い、
    不安定な video シーク連動 JS と冗長ボタンは削除されている。"""
    app_source = _read()

    match = re.search(
        r"prompt_frame_idx\.change\(\s*(?P<fn>extract_\w+)[^\)]*?\)",
        app_source,
        flags=re.DOTALL,
    )
    assert match is not None, "prompt_frame_idx.change(...) が見つからない"
    handler_block = match.group(0)

    assert "extract_prompt_frame" in handler_block, (
        "スライダーがシーク連動の extract_prompt_frame に紐づいていない"
    )
    assert "prompt_frame_idx" in handler_block, "スライダーが自分の値 prompt_frame_idx を入力に取らない"

    # 壊れたシーク連動 JS ・冗長ボタン・未使用 fps は削除済み。
    assert "build_video_seek_sync_js" not in app_source
    assert "VIDEO_SEEK_SYNC_JS" not in app_source
    assert 'gr.Button("表示中フレームを再取得")' not in app_source
    assert 'gr.Button("シーク位置を SAM2 に反映")' not in app_source
    assert 'elem_id="movie-video-fps"' not in app_source


def test_movie_get_video_pipeline_builds_propagator_from_registry() -> None:
    """Gap A: get_video_pipeline が registry の tracker config から propagator を構築する。"""
    app_source = _read()

    pipeline_fn = app_source[app_source.index("def get_video_pipeline(") :]
    pipeline_fn = pipeline_fn[: pipeline_fn.index("\ndef ")]

    assert 'entry_by_id("tracker", tracker_model)' in pipeline_fn
    assert "SAM2VideoPropagator(" in pipeline_fn
    assert "build_sam2_tb_video_pipeline(propagator=" in pipeline_fn
