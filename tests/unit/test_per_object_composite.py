"""Phase2: 対象ごと crop tb の最終アルファ合成（比較明 max）の検証。"""

import numpy as np
import pytest

from pipelines.components.video_common import composite_alpha_by_ownership


def test_composite_max_when_alpha_full():
    # 比較明: alpha が全画素 1 のとき、最終アルファ = 画素ごと max = 1.0。
    ownership = np.array(
        [
            [[0.6, 0.2], [0.0, 0.5]],  # obj0
            [[0.3, 0.7], [0.4, 0.1]],  # obj1
            [[0.1, 0.1], [0.6, 0.4]],  # background (last)
        ],
        dtype=np.float32,
    )
    alphas = [np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
    out = composite_alpha_by_ownership(alphas, ownership)
    assert out.shape == (2, 2)
    assert np.allclose(out, np.ones((2, 2), dtype=np.float32), atol=1e-6)


def test_composite_single_object_passthrough():
    # 単独対象で alpha=0.4 → 最終 0.4（tb の連続アルファそのまま）。
    ownership = np.array([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=np.float32)  # (2,1,2): obj0 + bg
    alphas = [np.array([[0.4, 0.4]], dtype=np.float32)]
    out = composite_alpha_by_ownership(alphas, ownership)
    assert np.allclose(out, np.array([[0.4, 0.4]], dtype=np.float32), atol=1e-6)


def test_composite_lighten_takes_max_over_objects():
    # 比較明: 重なり画素は所有権で減衰させず、対象ごと alpha の max を採る。
    # obj0 alpha=0.5, obj1 alpha=1.0 → max = 1.0（背後の残したい対象が黒で潰れない）。
    ownership = np.array([[[0.7]], [[0.3]], [[0.0]]], dtype=np.float32)  # (3,1,1)
    alphas = [np.array([[0.5]], dtype=np.float32), np.array([[1.0]], dtype=np.float32)]
    out = composite_alpha_by_ownership(alphas, ownership)
    assert np.allclose(out, np.array([[1.0]], dtype=np.float32), atol=1e-6)


def test_composite_lighten_keeps_object_even_when_other_is_zero():
    # 手前 obj0 が alpha=0（黒）でも、背後 obj1 の alpha=0.8 が max で生き残る。
    ownership = np.array([[[0.9]], [[0.1]], [[0.0]]], dtype=np.float32)
    alphas = [np.array([[0.0]], dtype=np.float32), np.array([[0.8]], dtype=np.float32)]
    out = composite_alpha_by_ownership(alphas, ownership)
    assert np.allclose(out, np.array([[0.8]], dtype=np.float32), atol=1e-6)


def test_composite_clips_to_unit_range():
    ownership = np.array([[[1.0]], [[0.0]]], dtype=np.float32)
    alphas = [np.array([[2.0]], dtype=np.float32)]  # 異常値でも [0,1] に clip。
    out = composite_alpha_by_ownership(alphas, ownership)
    assert np.allclose(out, np.array([[1.0]], dtype=np.float32), atol=1e-6)


def test_composite_length_mismatch_raises():
    ownership = np.array([[[1.0]], [[0.0]], [[0.0]]], dtype=np.float32)  # N=2 (+bg)
    alphas = [np.array([[1.0]], dtype=np.float32)]  # only 1 alpha
    with pytest.raises(ValueError):
        composite_alpha_by_ownership(alphas, ownership)
