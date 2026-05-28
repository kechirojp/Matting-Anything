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
        mask_array = np.asarray(mask).astype(bool)
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
