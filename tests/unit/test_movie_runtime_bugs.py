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
# Bug B: prompt_frame_idx は任意フレームを許可（読み込み窓を自動拡張）
# ─────────────────────────────────────────
def test_effective_read_frames_expands_window_to_include_prompt_frame() -> None:
    """prompt_frame_idx が max_frames を超える時、読み込み窓を prompt フレームまで拡張する。"""
    # prompt が窓内: 既存の max_frames を維持。
    assert movie_app._effective_read_frames(30, 5) == 30
    assert movie_app._effective_read_frames(30, 29) == 30
    # prompt が窓外: prompt_frame_idx + 1 まで拡張。
    assert movie_app._effective_read_frames(30, 30) == 31
    assert movie_app._effective_read_frames(30, 75) == 76
    # 下限は 1。
    assert movie_app._effective_read_frames(1, 0) == 1


def test_run_video_no_longer_rejects_prompt_frame_idx_beyond_max_frames() -> None:
    """prompt_frame_idx >= max_frames でも『処理フレーム数以上』の範囲外エラーを出さない。

    任意フレーム実行を許可したため、旧 fail-fast バリデーションは撤廃済み。窓拡張後に
    pipeline へ進み、ローカル環境では GPU 不在等で別エラー（動画処理に失敗）となる。
    """
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
            mask_guard_enabled=False,
            mask_guard_feather_ui=0,
            mask_guard_dilate_ui=21,
            overlay_enabled=False,
        )

    message = str(excinfo.value)
    # 旧バリデーションの文言（処理フレーム数 N 以上）は出ないこと。
    assert "処理フレーム数" not in message
    assert "範囲で指定してください" not in message


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


def test_warm_up_missing_checkpoint_raises_actionable_error(tmp_path) -> None:
    """存在しない tracker checkpoint を選ぶと、深い import/torch.load エラーではなく
    ファイル名と代替を明示する fail-fast エラーにする（ERR066）。

    registry は SAM2.1/SAMURAI の Large/B+ 計4 tracker を提供するが、B+ 系 checkpoint
    （sam2.1_hiera_base_plus.pt）が未配置だと選択時に sam2 内部の FileNotFoundError で
    分かりにくく落ちる。build 前に存在検証し、原因と代替を案内する。
    """
    missing = tmp_path / "sam2.1_hiera_base_plus.pt"
    propagator = video_model_components.SAM2VideoPropagator(
        checkpoint_path=str(missing),
        config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    with pytest.raises(RuntimeError) as excinfo:
        propagator.warm_up()
    message = str(excinfo.value)
    assert str(missing) in message
    assert "checkpoint" in message.lower()


# ─────────────────────────────────────────
# Bug D: RGBA(透過)動画は imageio+ffmpeg で alpha 付き書き出す（cv2 は 4ch 不可）（ERR047）
# ─────────────────────────────────────────
def test_select_rgba_codec_returns_alpha_capable_imageio_spec(monkeypatch, tmp_path) -> None:
    """RGBA codec 選択は cv2 VP90 fourcc ではなく alpha 対応 imageio spec を返す。

    cv2.VideoWriter は 4ch(RGBA) frame を書けず全 frame skip するため、透過動画は
    imageio+ffmpeg（webm_vp9=libvpx-vp9/yuva420p, mov_png=png/rgba）で書き出す。
    """
    monkeypatch.setattr(video_model_components, "_require_imageio", lambda: object())
    writer = video_model_components.VideoWriter(output_dir=str(tmp_path))

    spec, fallback = writer._select_rgba_codec((4, 6), preferred_rgba_codec="webm_vp9")
    assert spec.suffix == ".webm"
    assert spec.codec == "libvpx-vp9"
    assert spec.pixelformat == "yuva420p"  # alpha 付き 4:2:0
    assert any(label == "webm_vp9" for label, _ in fallback)

    spec_mov, _ = writer._select_rgba_codec((4, 6), preferred_rgba_codec="mov_png")
    assert spec_mov.suffix == ".mov"
    assert spec_mov.codec == "png"
    assert spec_mov.pixelformat == "rgba"  # PNG-in-MOV で alpha 保持


def test_select_rgba_codec_raises_clear_error_when_imageio_missing(monkeypatch, tmp_path) -> None:
    """imageio[ffmpeg] が無いとき、握り潰さず連番出力を促す明確なエラーにする。"""

    def _no_imageio() -> object:
        raise RuntimeError(
            "RGBA(透過)動画の書き出しには imageio[ffmpeg] が必要です。連番(PNG)出力を選択してください。"
        )

    monkeypatch.setattr(video_model_components, "_require_imageio", _no_imageio)
    writer = video_model_components.VideoWriter(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        writer._select_rgba_codec((4, 6), preferred_rgba_codec="webm_vp9")
    assert "imageio" in str(excinfo.value)


def test_transparent_bg_video_rgba_streams_4channel_to_imageio(monkeypatch, tmp_path) -> None:
    """動画モードの RGBA stream が 4ch RGBA を imageio へ append する（cv2 へは渡さない）。

    alpha/preview は VP9(`_ImageioWebmVideoWriter`) でブラウザ再生互換に書き出す（ERR065）。
    """
    import numpy as np

    appended: list[np.ndarray] = []
    writer_kwargs: dict[str, object] = {}

    class _FakeImageioWriter:
        def append_data(self, frame: np.ndarray) -> None:
            appended.append(np.asarray(frame))

        def close(self) -> None:
            return None

    class _FakeImageio:
        def get_writer(self, *args: object, **kwargs: object) -> "_FakeImageioWriter":
            writer_kwargs.update(kwargs)
            return _FakeImageioWriter()

    h264_frame_channels: list[int] = []

    class _FakeWebmWriter:
        def __init__(self, path, first_frame, fps) -> None:
            return None

        def write(self, frame: np.ndarray) -> None:
            frame_array = np.asarray(frame)
            h264_frame_channels.append(1 if frame_array.ndim == 2 else int(frame_array.shape[2]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(video_model_components, "_require_imageio", lambda: _FakeImageio())
    monkeypatch.setattr(video_model_components, "_ImageioWebmVideoWriter", _FakeWebmWriter)

    extractor = video_model_components.TransparentBGVideoExtractor(output_dir=str(tmp_path))

    def fake_extract(**kwargs):
        image = np.asarray(kwargs["image"])
        height, width = image.shape[:2]
        rgba = np.dstack([image, np.full((height, width), 255, dtype=np.uint8)])
        alpha = np.full((height, width), 255, dtype=np.uint8)
        return {"rgba": rgba, "alpha": alpha, "preview": image}

    extractor.extractor.run = fake_extract
    frames = [
        np.full((4, 6, 3), 10, dtype=np.uint8),
        np.full((4, 6, 3), 20, dtype=np.uint8),
    ]

    matte = extractor.run(frames=frames, output_mode="video", rgba_codec="webm_vp9")["matte"]

    assert matte["rgba_video_path"].endswith(".webm")
    # RGBA は imageio へ 4ch のまま 2 frame 渡る。
    assert len(appended) == 2
    assert appended[0].ndim == 3 and appended[0].shape[2] == 4
    # alpha/preview のみ VP9 writer（RGBA(4ch) は webm writer へ渡さない）。
    assert h264_frame_channels and 4 not in h264_frame_channels
    assert writer_kwargs.get("pixelformat") == "yuva420p"
