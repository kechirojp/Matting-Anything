"""ルートA案 純関数ユーティリティ（route_a_common）の単体テスト（GPU 非依存）。"""

from pathlib import Path

import numpy as np
import pytest

from pipelines.components.route_a_common import (
    alpha_to_rgba,
    apply_gate_to_alpha,
    ben2_rgba_to_alpha,
    blur_background_outside_gate,
    combine_alpha_with_mask,
    dilate_mask_to_gate,
    load_route_a_config,
)


def test_load_route_a_config_returns_defaults_for_missing_file(tmp_path):
    config = load_route_a_config(tmp_path / "does_not_exist.toml")
    assert config["alpha"]["ben2_repo_id"] == "PramaLLC/BEN2"
    assert config["blur_guide"]["dilation_px"] == 24
    assert config["composite"]["matte_mode"] == "union"


def test_load_route_a_config_reads_repo_file():
    repo_toml = Path(__file__).resolve().parents[2] / "config" / "route_a.toml"
    config = load_route_a_config(repo_toml)
    assert set(config.keys()) == {"alpha", "blur_guide", "composite", "deva"}
    assert config["alpha"]["ben2_repo_id"]
    assert isinstance(config["blur_guide"]["blur_kernel"], int)
    assert isinstance(config["deva"]["per_object_logits_max_side"], int)


def test_load_route_a_config_merges_partial_overrides(tmp_path):
    toml_path = tmp_path / "route_a.toml"
    toml_path.write_text("[blur_guide]\ndilation_px = 99\n", encoding="utf-8")
    config = load_route_a_config(toml_path)
    # 上書きしたキーは反映され、欠落キーは既定値で補完される。
    assert config["blur_guide"]["dilation_px"] == 99
    assert config["blur_guide"]["blur_kernel"] == 41
    assert config["alpha"]["ben2_repo_id"] == "PramaLLC/BEN2"


def test_dilate_mask_to_gate_expands_foreground():
    mask = np.zeros((21, 21), dtype=np.float32)
    mask[10, 10] = 1.0
    gate = dilate_mask_to_gate(mask, dilation_px=3)
    assert gate.shape == (21, 21)
    assert gate.dtype == np.uint8
    # 膨張により前景画素数が増える。
    assert gate.sum() > 1
    # 中心は前景。
    assert gate[10, 10] == 1


def test_dilate_mask_to_gate_zero_dilation_is_binarize_only():
    mask = np.array([[0.2, 0.8], [0.6, 0.1]], dtype=np.float32)
    gate = dilate_mask_to_gate(mask, dilation_px=0)
    np.testing.assert_array_equal(gate, np.array([[0, 1], [1, 0]], dtype=np.uint8))


def test_blur_background_outside_gate_keeps_inside_sharp():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    gate = np.zeros((40, 40), dtype=np.uint8)
    gate[15:25, 15:25] = 1
    guided = blur_background_outside_gate(image, gate, blur_kernel=11, blur_sigma=0.0, feather_px=0)
    assert guided.shape == image.shape
    assert guided.dtype == np.uint8
    # ゲート内部（羽根なし）は元画像を保持する。
    np.testing.assert_array_equal(guided[18:22, 18:22], image[18:22, 18:22])
    # ゲート外部はブラーで変化する（少なくとも一部の画素が変わる）。
    assert not np.array_equal(guided[0:5, 0:5], image[0:5, 0:5])


def test_blur_background_outside_gate_coerces_even_kernel():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    gate = np.ones((10, 10), dtype=np.uint8)
    # 偶数 kernel を渡しても例外にならず処理できる（内部で奇数へ補正）。
    guided = blur_background_outside_gate(image, gate, blur_kernel=10, feather_px=0)
    assert guided.shape == image.shape


def test_ben2_rgba_to_alpha_extracts_alpha_channel():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    alpha = ben2_rgba_to_alpha(rgba)
    assert alpha.shape == (4, 4)
    assert alpha.dtype == np.uint8
    assert np.all(alpha == 128)


def test_ben2_rgba_to_alpha_rejects_non_rgba():
    with pytest.raises(ValueError):
        ben2_rgba_to_alpha(np.zeros((4, 4, 3), dtype=np.uint8))


def test_alpha_to_rgba_combines_rgb_and_alpha():
    image = np.full((5, 5, 3), 100, dtype=np.uint8)
    alpha = np.full((5, 5), 0.5, dtype=np.float32)
    rgba = alpha_to_rgba(image, alpha)
    assert rgba.shape == (5, 5, 4)
    np.testing.assert_array_equal(rgba[..., :3], image)
    # float [0,1] は 0-255 へスケールされる。
    assert np.all(rgba[..., 3] == 127) or np.all(rgba[..., 3] == 128)


def test_alpha_to_rgba_resizes_mismatched_alpha():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    alpha = np.full((4, 4), 255, dtype=np.uint8)
    rgba = alpha_to_rgba(image, alpha)
    assert rgba.shape == (8, 8, 4)


def test_apply_gate_to_alpha_zeros_outside_gate():
    alpha = np.full((6, 6), 255, dtype=np.uint8)
    gate = np.zeros((6, 6), dtype=np.uint8)
    gate[2:4, 2:4] = 1
    gated = apply_gate_to_alpha(alpha, gate)
    assert gated[0, 0] == 0
    assert gated[2, 2] == 255


def test_load_route_a_config_exposes_mask_floor_mode_default():
    config = load_route_a_config(Path("does_not_exist_xyz.toml"))
    assert config["composite"]["mask_floor_mode"] == "none"


def test_combine_alpha_with_mask_none_is_passthrough():
    alpha = np.array([[10, 200], [0, 255]], dtype=np.uint8)
    out = combine_alpha_with_mask(alpha, np.ones((2, 2), dtype=np.float32), mode="none")
    np.testing.assert_array_equal(out, alpha)
    assert out.dtype == np.uint8


def test_combine_alpha_with_mask_none_mask_is_passthrough():
    alpha = np.array([[10, 200]], dtype=np.uint8)
    out = combine_alpha_with_mask(alpha, None, mode="lighten")
    np.testing.assert_array_equal(out, alpha)


def test_combine_alpha_with_mask_lighten_takes_pixelwise_max():
    alpha = np.array([[0, 255], [128, 0]], dtype=np.uint8)
    mask = np.array([[1.0, 0.0], [0.25, 0.5]], dtype=np.float32)
    out = combine_alpha_with_mask(alpha, mask, mode="lighten")
    # max(alpha_norm, mask_norm) * 255
    np.testing.assert_array_equal(out, np.array([[255, 255], [128, 127]], dtype=np.uint8))


def test_combine_alpha_with_mask_max_alias_matches_lighten():
    alpha = np.array([[0, 200]], dtype=np.uint8)
    mask = np.array([[0.5, 0.1]], dtype=np.float32)
    out_max = combine_alpha_with_mask(alpha, mask, mode="max")
    out_lighten = combine_alpha_with_mask(alpha, mask, mode="lighten")
    np.testing.assert_array_equal(out_max, out_lighten)


def test_combine_alpha_with_mask_screen_floor_raises_alpha():
    alpha = np.zeros((2, 2), dtype=np.uint8)
    mask = np.full((2, 2), 1.0, dtype=np.float32)
    out = combine_alpha_with_mask(alpha, mask, mode="screen")
    # screen: 1-(1-0)(1-1)=1 → 255 全面。SAM2 マスクが床を張る。
    assert np.all(out == 255)


def test_combine_alpha_with_mask_screen_is_at_least_alpha():
    alpha = np.array([[60, 200], [10, 250]], dtype=np.uint8)
    mask = np.array([[0.2, 0.0], [0.5, 0.1]], dtype=np.float32)
    out = combine_alpha_with_mask(alpha, mask, mode="screen")
    # 底上げなので必ず元 α 以上。
    assert np.all(out >= alpha)


def test_combine_alpha_with_mask_accepts_uint8_mask():
    alpha = np.zeros((2, 2), dtype=np.uint8)
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    out = combine_alpha_with_mask(alpha, mask, mode="lighten")
    np.testing.assert_array_equal(out, np.array([[255, 0], [0, 255]], dtype=np.uint8))


def test_combine_alpha_with_mask_accepts_bool_mask():
    alpha = np.zeros((2, 2), dtype=np.uint8)
    mask = np.array([[True, False], [False, True]])
    out = combine_alpha_with_mask(alpha, mask, mode="lighten")
    np.testing.assert_array_equal(out, np.array([[255, 0], [0, 255]], dtype=np.uint8))


def test_combine_alpha_with_mask_resizes_mismatched_mask():
    alpha = np.zeros((8, 8), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.float32)
    out = combine_alpha_with_mask(alpha, mask, mode="lighten")
    assert out.shape == (8, 8)
    assert np.all(out == 255)


def test_combine_alpha_with_mask_rejects_unknown_mode():
    alpha = np.zeros((2, 2), dtype=np.uint8)
    mask = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        combine_alpha_with_mask(alpha, mask, mode="multiply")
