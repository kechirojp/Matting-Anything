"""TransparentBGExtractor が SAM2 mask 形状を最終 alpha に反映することを検証する。

横一直線切れ不具合（調査/2026-06-04_222747_現行動画パイプライン_フロー調査_MAM比較.md）の
回帰テスト。remover が外接矩形クロップ全体に alpha=255 を返しても、最終 alpha は
SAM2 mask の形状に従い、矩形内でも mask 外（直線切れ領域）は 0 になることを確認する。
"""

from __future__ import annotations

import numpy as np

from pipelines.components.model_components import TransparentBGExtractor


class _FullAlphaRemover:
    """クロップ矩形全体に不透明 alpha を返すダミー remover（バグ再現用）。"""

    def process(self, image, type="rgba", threshold=None):  # noqa: A002 - API 互換
        rgb = np.asarray(image)
        height, width = rgb.shape[:2]
        alpha = np.full((height, width), 255, dtype=np.uint8)
        return np.dstack([rgb, alpha])


def _make_extractor() -> TransparentBGExtractor:
    extractor = TransparentBGExtractor()
    extractor._get_remover = lambda mode, jit: _FullAlphaRemover()  # type: ignore[method-assign]
    return extractor


def test_alpha_follows_mask_shape_not_bounding_rectangle() -> None:
    """矩形内でも mask 形状外の領域は alpha=0 になり、横一直線切れを防ぐ。"""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True  # 左上ブロック
    mask[70:90, 70:90] = True  # 右下ブロック（外接矩形は (10,10)-(90,90)）

    result = _make_extractor().run(image=image, mask=mask, crop_padding=0)
    alpha = result["alpha"]

    # mask 内は前景として残る。
    assert alpha[20, 20] > 0
    assert alpha[80, 80] > 0
    # 外接矩形の内側だが mask 形状外（中央）は削られる＝横一直線切れの解消。
    assert alpha[50, 50] == 0


def test_no_mask_keeps_full_alpha() -> None:
    """mask 未指定時は従来通り全面 alpha を維持する（後方互換）。"""
    image = np.full((40, 40, 3), 128, dtype=np.uint8)

    result = _make_extractor().run(image=image, mask=None)
    alpha = result["alpha"]

    assert alpha[20, 20] == 255


def test_guard_is_idempotent_with_sam2_guard_filter() -> None:
    """extractor 内 guard と後段 SAM2GuardFilter の二重適用が冪等であることを確認する。

    build_sam2_union_tb_pipeline は transparent_bg の後段に SAM2GuardFilter を同一 mask で
    接続するため二重適用になるが、二値 guard の乗算は冪等であるべき。
    """
    from pipelines.components.model_components import SAM2GuardFilter

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[70:90, 70:90] = True

    extractor_alpha = _make_extractor().run(image=image, mask=mask, crop_padding=0)["alpha"]
    double_applied = SAM2GuardFilter().run(alpha=extractor_alpha, mask=mask)["alpha"]

    assert np.array_equal(extractor_alpha, double_applied)


def test_apply_mask_guard_can_be_disabled() -> None:
    """apply_mask_guard=False のとき従来の矩形貼り戻し挙動に戻せる。"""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[70:90, 70:90] = True

    result = _make_extractor().run(image=image, mask=mask, crop_padding=0, apply_mask_guard=False)
    alpha = result["alpha"]

    # guard を無効化すると外接矩形全体に alpha が残る（従来挙動）。
    assert alpha[50, 50] == 255


def test_feather_guard_softens_mask_edge() -> None:
    """mask_guard_feather>0 のとき mask 境界が連続値になり 2 値エッジを解消する。

    transparent-background の gradient alpha と SAM2/SAMURAI の二値 union mask を
    合成する際、二値 guard の乗算が黒/白の 2 値エッジを生む。union mask を feather
    した soft guard を乗算することで、境界に中間 alpha が現れることを確認する。
    """
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    result = _make_extractor().run(image=image, mask=mask, crop_padding=0, mask_guard_feather=8)
    alpha = result["alpha"]

    # mask 中心は不透明、遠方は透明。
    assert alpha[50, 50] == 255
    assert alpha[5, 5] == 0
    # 境界に中間 alpha（0 でも 255 でもない値）が存在する＝feather による soft edge。
    assert np.any((alpha > 0) & (alpha < 255))
    # メタデータに feather 値が記録される。
    assert result["matte_result"]["metadata"]["mask_guard_feather"] == 8


def test_feather_zero_keeps_binary_edge() -> None:
    """mask_guard_feather=0（既定）では従来の二値 guard 挙動を維持する（後方互換）。"""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    alpha = _make_extractor().run(image=image, mask=mask, crop_padding=0, mask_guard_feather=0)["alpha"]

    # _FullAlphaRemover は 255 のみ返すため、二値 guard では中間値が出ない。
    assert not np.any((alpha > 0) & (alpha < 255))


def test_sam2_guard_filter_feather_softens_edge() -> None:
    """SAM2GuardFilter も feather>0 で連続値の soft edge を生成する。"""
    from pipelines.components.model_components import SAM2GuardFilter

    alpha_in = np.full((100, 100), 255, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    out = SAM2GuardFilter().run(alpha=alpha_in, mask=mask, feather=8)["alpha"]

    assert out[50, 50] == 255
    assert out[5, 5] == 0
    assert np.any((out > 0) & (out < 255))


def test_float_soft_mask_guard_keeps_interior_alpha_unscaled() -> None:
    """Phase1: float soft mask + feather=0 のとき、guard は形状外ゲートに徹し内部を削らない。

    OwnershipResolver が渡す前景 soft（領域全体の連続確率）を guard としてそのまま乗算
    すると、tb の人物アルファ内部が連続確率で減衰し半透明化する不具合があった。修正後は
    float mask でも feather=0 なら 0.5 閾値で二値化した dilate ゲート（内部 1.0・外部 0）を
    使い、tb の連続アルファ内部を一切減衰させないことを確認する。
    """
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    # 内部が 0.7 など 1.0 未満の連続確率を持つ soft mask。形状は中央ブロック。
    soft = np.zeros((100, 100), dtype=np.float32)
    soft[30:70, 30:70] = 0.7

    alpha = _make_extractor().run(image=image, mask=soft, crop_padding=0, mask_guard_feather=0)["alpha"]

    # 形状内部（soft>=0.5）は tb の不透明アルファ 255 を保ち、0.7 で減衰しない。
    assert alpha[50, 50] == 255
    # 形状外（dilate マージン外）は 0。
    assert alpha[5, 5] == 0
    # 内部に連続確率由来の中間値（178 付近）が現れない＝guard が内部を削っていない。
    interior = alpha[35:65, 35:65]
    assert np.all(interior == 255)


def test_float_soft_mask_guard_feather_opt_in_softens_edge() -> None:
    """Phase1: float soft mask でも feather>0 のときだけ soft guard（境界連続値）になる。"""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    soft = np.zeros((100, 100), dtype=np.float32)
    soft[30:70, 30:70] = 0.9

    alpha = _make_extractor().run(image=image, mask=soft, crop_padding=0, mask_guard_feather=8)["alpha"]

    # feather オプトイン時は境界に中間 alpha が現れる。
    assert np.any((alpha > 0) & (alpha < 255))


def test_feather_binary_mask_helper_produces_continuous_values() -> None:
    """feather_binary_mask は二値 mask を 0..1 連続値の soft guard に変換する。"""
    from pipelines.components.common import feather_binary_mask

    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    soft = feather_binary_mask(mask, dilate_size=1, feather_radius=8)
    assert soft.dtype == np.float32
    assert float(soft.max()) <= 1.0
    assert float(soft.min()) >= 0.0
    assert soft[50, 50] == 1.0
    assert np.any((soft > 0.0) & (soft < 1.0))

    hard = feather_binary_mask(mask, dilate_size=1, feather_radius=0)
    assert not np.any((hard > 0.0) & (hard < 1.0))


def test_feather_binary_mask_handles_extreme_radius_and_small_mask() -> None:
    """極小 mask と過大な feather_radius でも 0..1 範囲を保ち例外を出さない。"""
    from pipelines.components.common import feather_binary_mask

    mask = np.zeros((50, 50), dtype=bool)
    mask[24:27, 24:27] = True  # 3x3 の極小 mask

    soft = feather_binary_mask(mask, dilate_size=21, feather_radius=200)
    assert soft.dtype == np.float32
    assert float(soft.min()) >= 0.0
    assert float(soft.max()) <= 1.0
    # mask 中心は guard で残る（完全 0 にはならない）。
    assert soft[25, 25] > 0.0


def test_feather_binary_mask_empty_mask_is_all_zero() -> None:
    """空 mask では soft guard が全 0 になり、漏れ alpha を完全に削る。"""
    from pipelines.components.common import feather_binary_mask

    mask = np.zeros((40, 40), dtype=bool)
    soft = feather_binary_mask(mask, dilate_size=21, feather_radius=8)

    assert float(soft.max()) == 0.0


def test_soft_probability_mask_uses_soft_guard_no_binary_edge() -> None:
    """修正2: float 確率 mask を渡すと soft guard で gate し、二値エッジを作らない。

    SAM2VideoPropagator の soft union（[0,1] 確率）を最終 alpha の guard に使うと、
    継ぎ目の細い谷が黒線にならず中間 alpha になる（末端 feather）。
    """
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[30:70, 30:70] = 1.0  # 中央のみ確信

    result = _make_extractor().run(image=image, mask=prob, crop_padding=0, mask_guard_feather=8)
    alpha = result["alpha"]

    # mask 中心は不透明、遠方は透明。
    assert alpha[50, 50] == 255
    assert alpha[5, 5] == 0
    # float 確率 mask でも境界に中間 alpha が現れる（二値エッジでない）。
    assert np.any((alpha > 0) & (alpha < 255))
    assert result["matte_result"]["metadata"]["mask_used"] is True


def test_soft_probability_mask_bbox_uses_threshold_not_any_nonzero() -> None:
    """修正2: float 確率 mask の bbox は閾値 0.5 で決め、微小確率で矩形が暴れない。"""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    prob = np.full((100, 100), 0.01, dtype=np.float32)  # 全面に微小確率
    prob[40:60, 40:60] = 1.0  # 中央のみ確信

    result = _make_extractor().run(image=image, mask=prob, crop_padding=0)
    bbox = result["matte_result"]["metadata"]["bbox"]
    x_min, y_min, x_max, y_max = bbox

    # 微小確率(0.01)を前景扱いせず、確信領域(40..60)の外接矩形に収まる。
    assert x_min >= 35 and y_min >= 35
    assert x_max <= 65 and y_max <= 65

