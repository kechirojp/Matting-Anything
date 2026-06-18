"""ヘッドレス CLI（run_video_matting_headless）の引数解析と config 駆動 τ の検証。"""

from pathlib import Path

import pytest

from run_video_matting_headless import build_arg_parser, _parse_box, _parse_point


def test_parse_box_valid():
    assert _parse_box("0,1,2,3") == [0.0, 1.0, 2.0, 3.0]


def test_parse_box_invalid_count():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_box("0,1,2")


def test_parse_point_default_label():
    point, label = _parse_point("5,6")
    assert point == [5.0, 6.0]
    assert label == 1


def test_parse_point_explicit_negative_label():
    point, label = _parse_point("5,6,0")
    assert point == [5.0, 6.0]
    assert label == 0


def test_arg_parser_collects_multiple_boxes_and_points():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--video",
            "in.mp4",
            "--box",
            "0,0,2,3",
            "--box",
            "3,0,5,3",
            "--point",
            "1,1",
            "--point",
            "4,2,0",
            "--temperature",
            "0.5",
        ]
    )
    assert args.video == "in.mp4"
    assert args.box == [[0.0, 0.0, 2.0, 3.0], [3.0, 0.0, 5.0, 3.0]]
    assert args.point == [([1.0, 1.0], 1), ([4.0, 2.0], 0)]
    assert args.temperature == 0.5


def test_movie_app_wires_ownership_temperature_from_config():
    """movie app が config の ownership_temperature を ownership_resolver へ配線していること。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")
    assert "ownership_resolver" in source
    assert "ownership_temperature" in source


def test_background_entries_define_ownership_temperature():
    """全 background entry が τ（ownership_temperature）を持つこと（ハードコード禁止）。"""
    from pipelines.components.model_registry import load_model_registry

    registry = load_model_registry()
    backgrounds = registry.get("background", [])
    assert backgrounds, "background entry が存在しません"
    for entry in backgrounds:
        assert "ownership_temperature" in entry, f"{entry.get('id')} に ownership_temperature がありません"
        assert float(entry["ownership_temperature"]) > 0


def test_arg_parser_accepts_matte_mode():
    """--matte-mode が union/per_object を受け付け、未指定で None（config 既定にフォールバック）。"""
    parser = build_arg_parser()
    args = parser.parse_args(["--video", "in.mp4", "--box", "0,0,2,3", "--matte-mode", "per_object"])
    assert args.matte_mode == "per_object"
    default_args = parser.parse_args(["--video", "in.mp4", "--box", "0,0,2,3"])
    assert default_args.matte_mode is None


def test_background_entries_define_video_matte_mode():
    """全 background entry が video_matte_mode を持ち、union/per_object のいずれかであること。"""
    from pipelines.components.model_registry import load_model_registry

    registry = load_model_registry()
    backgrounds = registry.get("background", [])
    assert backgrounds, "background entry が存在しません"
    for entry in backgrounds:
        assert "video_matte_mode" in entry, f"{entry.get('id')} に video_matte_mode がありません"
        assert entry["video_matte_mode"] in {"union", "per_object"}


def test_movie_app_wires_video_matte_mode_from_config():
    """movie app が config の video_matte_mode を transparent_bg_video へ配線していること。"""
    source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")
    assert "video_matte_mode" in source


def test_arg_parser_accepts_diagnose_flag():
    """--diagnose が真偽フラグとして解析されること（既定 False）。"""
    parser = build_arg_parser()
    on = parser.parse_args(["--video", "in.mp4", "--box", "0,0,2,3", "--diagnose"])
    assert on.diagnose is True
    off = parser.parse_args(["--video", "in.mp4", "--box", "0,0,2,3"])
    assert off.diagnose is False


def test_mask_box_stats_detects_mask_inside_box():
    """mask が box 内に収まる対象は in_box≈1・c_in_box=1 で「正常追跡」を数値化する。"""
    import numpy as np

    from run_video_matting_headless import _mask_box_stats

    h, w = 20, 20
    logit = np.full((h, w), -10.0, dtype=np.float32)
    logit[5:10, 5:10] = 10.0  # box 内に前景
    box = [4.0, 4.0, 11.0, 11.0]
    s = _mask_box_stats(logit, box, w, h)
    assert s["area_frac"] == pytest.approx(25 / 400, abs=1e-6)
    assert s["inside_box_frac"] == pytest.approx(1.0, abs=1e-6)
    assert s["centroid_in_box"] == 1.0


def test_mask_box_stats_detects_mask_outside_box_background():
    """mask が box 外（背景側）に乗る対象は in_box≈0・c_in_box=0 で「反転/背景追跡」を数値化する。"""
    import numpy as np

    from run_video_matting_headless import _mask_box_stats

    h, w = 20, 20
    logit = np.full((h, w), -10.0, dtype=np.float32)
    logit[14:19, 14:19] = 10.0  # box の外（右下背景）に前景
    box = [2.0, 2.0, 8.0, 8.0]  # 左上を指定したのに mask は右下
    s = _mask_box_stats(logit, box, w, h)
    assert s["inside_box_frac"] == pytest.approx(0.0, abs=1e-6)
    assert s["centroid_in_box"] == 0.0


def test_mask_box_stats_handles_empty_mask():
    """全画素が背景（logit<0）でも例外を出さず area=0・in_box=NaN を返す。"""
    import math

    import numpy as np

    from run_video_matting_headless import _mask_box_stats

    logit = np.full((10, 10), -5.0, dtype=np.float32)
    s = _mask_box_stats(logit, [0.0, 0.0, 5.0, 5.0], 10, 10)
    assert s["area_frac"] == 0.0
    assert math.isnan(s["inside_box_frac"])
    assert math.isnan(s["centroid_in_box"])


def test_mask_box_stats_scales_box_to_logit_resolution():
    """logit が frame より低解像度でも、box を logit 空間へスケールして box 内判定する。"""
    import numpy as np

    from run_video_matting_headless import _mask_box_stats

    # frame は 200x200、logit は 20x20（1/10 解像度）。前景は logit の左上 5x5。
    logit = np.full((20, 20), -10.0, dtype=np.float32)
    logit[2:7, 2:7] = 10.0
    # frame 座標で対応する box（10倍）を指定 → スケール後 logit 内に収まる。
    box = [10.0, 10.0, 80.0, 80.0]
    s = _mask_box_stats(logit, box, width=200, height=200)
    assert s["logit_w"] == 20.0 and s["logit_h"] == 20.0
    assert s["inside_box_frac"] == pytest.approx(1.0, abs=1e-6)
    assert s["centroid_in_box"] == 1.0


def test_mask_box_stats_point_only_box_none_is_nan():
    """point のみ（box=None）のとき box 系指標は NaN、area/重心は算出される。"""
    import math

    import numpy as np

    from run_video_matting_headless import _mask_box_stats

    logit = np.full((10, 10), -10.0, dtype=np.float32)
    logit[3:6, 3:6] = 10.0
    s = _mask_box_stats(logit, None, 10, 10)
    assert s["area_frac"] == pytest.approx(9 / 100, abs=1e-6)
    assert not math.isnan(s["centroid_x"])
    assert math.isnan(s["inside_box_frac"])
    assert math.isnan(s["centroid_in_box"])
