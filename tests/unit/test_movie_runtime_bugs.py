"""動画版 Colab 実機ランタイムエラー 3 件（ERR036-038）の回帰防止テスト。

- Bug A: gr.Dataframe 値（pandas DataFrame）の真偽値曖昧で候補選択肢生成が失敗。
- Bug B: prompt_frame_idx がサンプリング処理 frame 数を超えると 18s 後に範囲外エラー。
- Bug C: SAMURAI config が installed sam2 の検索パスになく MissingConfigException。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

movie_app = importlib.import_module("gradio_app_sam2_transparent_BG_haystack_for_Movie")
video_model_components = importlib.import_module(
    "pipelines.components.video_model_components"
)


# ─────────────────────────────────────────
# Bug A: populate_candidate_choices が pandas DataFrame を安全に扱う
# ─────────────────────────────────────────
def test_populate_candidate_choices_accepts_pandas_dataframe() -> None:
    """gr.Dataframe が渡す pandas DataFrame で真偽値曖昧エラーを出さず選択肢を作る。"""
    frame = pd.DataFrame(
        [
            [1, "person", "0.900", "10, 20, 110, 220"],
            [2, "drums", "0.800", "30, 40, 130, 240"],
        ],
        columns=["rank", "phrase", "confidence", "bbox[x1,y1,x2,y2]"],
    )

    update = movie_app.populate_candidate_choices(frame)

    choices = update["choices"]
    assert len(choices) == 2
    assert choices[0].startswith("#1 person")
    assert choices[1].startswith("#2 drums")
    # top1 が既定 ON。
    assert update["value"] == [choices[0]]


def test_populate_candidate_choices_handles_empty_dataframe() -> None:
    """空 DataFrame でも例外を出さず空選択肢を返す。"""
    empty = pd.DataFrame(columns=["rank", "phrase", "confidence", "bbox[x1,y1,x2,y2]"])

    update = movie_app.populate_candidate_choices(empty)

    assert update["choices"] == []
    assert update["value"] == []


def test_populate_candidate_choices_still_accepts_list_rows() -> None:
    """後方互換: list of list 入力も従来通り扱える。"""
    rows = [[1, "cat", "0.700", "5, 6, 7, 8"]]

    update = movie_app.populate_candidate_choices(rows)

    assert len(update["choices"]) == 1
    assert update["choices"][0].startswith("#1 cat")


# ─────────────────────────────────────────
# Bug B: prompt_frame_idx 範囲を pipeline.run 前に fail-fast 検証
# ─────────────────────────────────────────
def test_run_video_validates_prompt_frame_idx_before_pipeline() -> None:
    """prompt_frame_idx が処理 frame 数以上なら 18s 待たずに gr.Error を出す。"""
    state = movie_app.empty_prompt_state()
    state["box"] = [10, 20, 110, 220]

    import gradio as gr

    with pytest.raises(gr.Error) as excinfo:
        movie_app.run_video_background_removal(
            video_path="dummy.mp4",
            prompt_state=state,
            prompt_frame_idx=75,
            bidirectional=False,
            max_frames=30,
            frame_step=1,
            output_mode_label="動画（mp4/webm）",
            rgba_codec="webm_vp9",
            tracker_model="sam2_hiera_l",
            background_model="tb_base",
            tb_jit=False,
            tb_threshold=0.0,
            tb_output_type="rgba",
            crop_padding=5,
            overlay_enabled=False,
        )

    message = str(excinfo.value)
    assert "75" in message
    assert "30" in message


# ─────────────────────────────────────────
# Bug C: SAMURAI config のローカル検索パス解決
# ─────────────────────────────────────────
def test_samurai_config_root_returns_local_sam2_package_for_samurai_config() -> None:
    """samurai config 名なら configs/ を含むローカル sam2 package root を返す。"""
    root = video_model_components._samurai_config_root(
        "configs/samurai/sam2.1_hiera_l.yaml"
    )

    assert root is not None
    assert root.name == "sam2"
    assert (root / "configs" / "samurai").is_dir()


def test_samurai_config_root_returns_none_for_standard_sam2_config() -> None:
    """非 samurai config では検索パス登録を行わない（None）。"""
    assert (
        video_model_components._samurai_config_root(
            "configs/sam2.1/sam2.1_hiera_l.yaml"
        )
        is None
    )


def test_warm_up_registers_samurai_searchpath_helper_called() -> None:
    """warm_up が samurai 検索パス登録ヘルパを呼ぶ（ソース契約）。"""
    source = Path(video_model_components.__file__).read_text(encoding="utf-8")
    assert "_ensure_samurai_config_searchpath" in source
