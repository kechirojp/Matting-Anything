import numpy as np
from pipelines.components.ownership_resolver import _softmax_across_objects, OwnershipResolver


def test_softmax_stable():
    logits = np.array([[[0.0, 1.0], [2.0, -1.0]], [[0.5, -0.2], [0.0, 0.0]]], dtype=np.float32)
    soft = _softmax_across_objects(logits, temperature=1.0)
    assert soft.shape == logits.shape
    # per-pixel sum to 1
    s = np.sum(soft, axis=0)
    assert np.allclose(s, np.ones_like(s), atol=1e-5)


def test_ownership_component():
    resolver = OwnershipResolver()
    per_object_logits = {0: np.array([[[10.0, -10.0], [0.0, 0.0]], [[-5.0, 5.0], [0.0, 1.0]]], dtype=np.float32)}
    masks = {"per_object_logits": per_object_logits, "frame_masks": {}, "object_ids": [1, 2]}
    out = resolver.run(masks=masks, temperature=1.0)
    result = out["masks"]
    ownership = result["ownership"]
    assert 0 in ownership
    own = ownership[0]
    # objects + background = N+1 channels
    assert own.shape[0] == 3
    # ownership channels sum to 1 per-pixel (softmax across objects + background)
    s = np.sum(own, axis=0)
    assert np.allclose(s, np.ones_like(s), atol=1e-5)
    # frame_masks holds foreground soft = 1 - background ownership
    fg = result["frame_masks"][0]
    assert fg.shape == (2, 2)
    assert np.allclose(fg, np.clip(1.0 - own[-1], 0.0, 1.0), atol=1e-6)
    # carried-over metadata preserved
    assert result["object_ids"] == [1, 2]


def test_foreground_upscaled_to_frame_hw_when_present():
    """frame_hw があれば foreground（frame_masks）を原寸へアップスケールする（ERR068）。

    DEVA tracker は per_object_logits を縮小して host-RAM を抑えるため、OwnershipResolver は
    BEN2 union ゲートが要求する原寸の soft guard を frame_hw から復元する。
    """
    resolver = OwnershipResolver()
    # 2x2 の縮小 logits（2 object）。原寸は 8x6 とする。
    per_object_logits = {0: np.array([[[10.0, -10.0], [0.0, 0.0]], [[-5.0, 5.0], [0.0, 1.0]]], dtype=np.float32)}
    masks = {"per_object_logits": per_object_logits, "frame_masks": {}, "object_ids": [1, 2], "frame_hw": (8, 6)}
    out = resolver.run(masks=masks, temperature=1.0)
    result = out["masks"]
    fg = result["frame_masks"][0]
    # foreground は原寸 (H,W)=(8,6) へ拡大される。
    assert fg.shape == (8, 6)
    assert fg.dtype == np.float32


def test_no_frame_hw_keeps_logits_resolution():
    """frame_hw が無ければ従来通り logits 解像度のまま（後方互換・基底アプリ無影響）。"""
    resolver = OwnershipResolver()
    per_object_logits = {0: np.zeros((2, 4, 5), dtype=np.float32)}
    masks = {"per_object_logits": per_object_logits, "frame_masks": {}, "object_ids": [1, 2]}
    out = resolver.run(masks=masks, temperature=1.0)
    fg = out["masks"]["frame_masks"][0]
    assert fg.shape == (4, 5)

