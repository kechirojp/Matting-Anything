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
