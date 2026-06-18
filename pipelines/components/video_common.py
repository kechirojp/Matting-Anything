"""動画 Haystack パイプラインで共有する純粋関数 Component。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from haystack import component


VALID_OUTPUT_MODES = {"video", "sequence", "both"}


def normalize_output_mode(output_mode: str) -> str:
    """UI 表示文字列を Component 内部の出力モードへ正規化する。"""
    normalized = str(output_mode or "video").lower()
    if "both" in normalized or "両方" in normalized:
        return "both"
    if "sequence" in normalized or "連番" in normalized:
        return "sequence"
    if "video" in normalized or "動画" in normalized:
        return "video"
    if normalized not in VALID_OUTPUT_MODES:
        raise ValueError(f"未対応の出力形式です: {output_mode}")
    return normalized


def frame_cache_bytes(frame_count: int, height: int, width: int, channels: int = 3) -> int:
    """フレームを uint8 配列として RAM に保持する概算バイト数を返す。"""
    return int(frame_count) * int(height) * int(width) * int(channels)


def sample_frame_indices(frame_count: int, max_frames: int, frame_step: int) -> list[int]:
    """処理対象 frame index を最大枚数と step から決定する。"""
    if frame_count < 0:
        raise ValueError("frame_count は 0 以上である必要があります。")
    if max_frames < 1:
        raise ValueError("max_frames は 1 以上である必要があります。")
    if frame_step < 1:
        raise ValueError("frame_step は 1 以上である必要があります。")
    return list(range(0, int(frame_count), int(frame_step)))[: int(max_frames)]


def build_video_source(
    path: str,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    codec: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """動画メタデータを VideoSource 契約 dict に変換する。"""
    return {
        "path": str(Path(path).resolve()),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "codec": str(codec or ""),
        "metadata": dict(metadata or {}),
    }


def build_frame_mask_sequence(
    frame_masks: dict[int, np.ndarray],
    object_ids: Sequence[int] | None = None,
    source: str = "sam2_video",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """frame index ごとの mask を FrameMaskSequence 契約 dict に変換する。"""
    normalized: dict[int, np.ndarray] = {}
    for frame_index, mask in frame_masks.items():
        mask_array = np.asarray(mask)
        # soft 確率 mask（float）は [0,1] float32 のまま保持し、二値 mask は bool に正規化する。
        if np.issubdtype(mask_array.dtype, np.floating):
            mask_array = np.clip(mask_array, 0.0, 1.0).astype(np.float32)
        else:
            mask_array = mask_array.astype(bool)
        if mask_array.ndim != 2:
            raise ValueError(f"frame mask は (H,W) 形式である必要があります: frame={frame_index}, shape={mask_array.shape}")
        normalized[int(frame_index)] = mask_array
    frame_indices = sorted(normalized.keys())
    return {
        "frame_masks": normalized,
        "object_ids": list(object_ids or [1]),
        "frame_indices": frame_indices,
        "source": source,
        "metadata": dict(metadata or {}),
    }


def composite_alpha_by_ownership(
    per_object_alphas: Sequence[np.ndarray],
    ownership: np.ndarray,
) -> np.ndarray:
    """対象ごと連続アルファを比較明（画素ごと max）で合成し最終アルファを得る（Phase2 ⑤）。

    重ね合成の式::

        alpha_final(p) = max_o alpha_o(p)   (o は前景対象 0..N-1)

    所有権による加重和（Σ ownership_o × alpha_o）は、手前対象のアルファが 0（黒）の画素で
    背後の残したい対象まで減衰させ黒く潰す欠点があった。比較明 max は対象ごとアルファの
    最大値を採るため、どれか 1 対象でも前景なら最終アルファに残り、対象同士の重なりで
    黒抜けが起きない。

    ``ownership`` は形状検証（前景チャネル数 N と per_object_alphas の数の一致）にのみ使う。
    合成自体は max なので所有権の重みは乗じない。

    Args:
        per_object_alphas: 長さ N の list。各要素は (H,W) float [0,1] の対象ごと連続アルファ。
        ownership: (N+1,H,W) float。最終チャネルが背景、先頭 N チャネルが前景対象の所有権。
            合成には使わず、チャネル数で対象数 N を検証するためだけに参照する。

    Returns:
        (H,W) float32 の最終アルファ（[0,1] に clip 済み）。

    Raises:
        ValueError: per_object_alphas の数が ownership の前景チャネル数 (N) と一致しない場合。
    """
    ownership_arr = np.asarray(ownership, dtype=np.float32)
    if ownership_arr.ndim != 3:
        raise ValueError(f"ownership は (N+1,H,W) 形式が必要です: shape={ownership_arr.shape}")
    num_objects = ownership_arr.shape[0] - 1
    if len(per_object_alphas) != num_objects:
        raise ValueError(
            f"per_object_alphas の数 ({len(per_object_alphas)}) が前景対象数 ({num_objects}) と一致しません。"
        )
    height, width = ownership_arr.shape[1:]
    alpha_final = np.zeros((height, width), dtype=np.float32)
    for obj_index in range(num_objects):
        alpha_o = np.clip(np.asarray(per_object_alphas[obj_index], dtype=np.float32), 0.0, 1.0)
        if alpha_o.shape != (height, width):
            raise ValueError(
                f"per_object_alphas[{obj_index}] の形状 {alpha_o.shape} が ownership {(height, width)} と一致しません。"
            )
        # 比較明（lighten）: 画素ごとに対象アルファの max を採り、黒抜けを防ぐ。
        alpha_final = np.maximum(alpha_final, alpha_o)
    return np.clip(alpha_final, 0.0, 1.0).astype(np.float32)


def normalize_rgba_frame(frame: np.ndarray) -> np.ndarray:
    """RGBA/RGB frame を uint8 RGBA に正規化する。"""
    frame_array = np.asarray(frame)
    if frame_array.ndim != 3 or frame_array.shape[2] not in (3, 4):
        raise ValueError(f"frame は HxWx3/4 形式である必要があります: shape={frame_array.shape}")
    if frame_array.shape[2] == 3:
        alpha = np.full(frame_array.shape[:2], 255, dtype=np.uint8)
        frame_array = np.dstack([frame_array, alpha])
    return frame_array.astype(np.uint8, copy=False)


def write_png_frame(path: Path, frame: np.ndarray) -> None:
    """RGB/RGBA/gray frame を PNG として保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_array = np.asarray(frame).astype(np.uint8, copy=False)
    if frame_array.ndim == 2:
        image_to_write = frame_array
    elif frame_array.ndim == 3 and frame_array.shape[2] == 3:
        image_to_write = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    elif frame_array.ndim == 3 and frame_array.shape[2] == 4:
        image_to_write = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"PNG 保存に未対応の frame shape です: {frame_array.shape}")
    if not cv2.imwrite(str(path), image_to_write):
        raise RuntimeError(f"PNG 保存に失敗しました: {path}")


@component
class FrameSampler:
    """動画 frame index を最大枚数と step で間引く Component。"""

    @component.output_types(frame_indices=list)
    def run(self, frame_count: int, max_frames: int = 300, frame_step: int = 1) -> dict[str, list[int]]:
        """処理対象 frame index を返す。"""
        return {"frame_indices": sample_frame_indices(int(frame_count), int(max_frames), int(frame_step))}
