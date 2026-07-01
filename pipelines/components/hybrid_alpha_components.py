"""BEN2 + transparent-background の領域分担動画 Component。

DEVA/SAM2 が追跡した人物領域だけを transparent-background に渡し、それ以外は BEN2 の
全画面 alpha を使う。モデル推論、領域分担、alpha 合成、出力保存を 1 つの Component
境界に閉じ、上流の tracker / 下流 writer から独立させる。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from haystack import component


logger = logging.getLogger(__name__)

from .ben2_components import BEN2Extractor
from .common import compose_alpha, ensure_rgb_array, feather_binary_mask
from .model_components import TransparentBGExtractor, default_device
from .route_a_common import alpha_to_rgba, load_route_a_config
from .video_common import normalize_output_mode, normalize_rgba_frame, resolve_run_timestamp, write_png_frame
from .video_model_components import (
    ProgressCallback,
    VideoWriter,
    _ImageioAlphaVideoWriter,
    _ImageioWebmVideoWriter,
    _ProgressKeepAlive,
    _notify_progress,
    _resolve_output_dir,
)


COMPOSITION_MODES = {"lighten", "person_over_ben2", "ben2_over_person", "screen"}


def normalize_alpha_float(alpha: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """alpha を 0..1 float32 に正規化する。"""
    alpha_array = np.asarray(alpha, dtype=np.float32)
    if alpha_array.ndim != 2:
        raise ValueError(f"alpha は (H,W) 形式である必要があります: shape={alpha_array.shape}")
    if alpha_array.max(initial=0.0) > 1.0:
        alpha_array = alpha_array / 255.0
    alpha_array = np.clip(alpha_array, 0.0, 1.0).astype(np.float32)
    if shape is not None and alpha_array.shape != shape:
        alpha_array = cv2.resize(alpha_array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha_array, 0.0, 1.0).astype(np.float32)


def build_person_region(mask: np.ndarray | None, shape: tuple[int, int], dilate_px: int, feather_px: int) -> np.ndarray:
    """SAM2 人物 mask から 0..1 の領域分担 guard を作る。"""
    if mask is None:
        return np.zeros(shape, dtype=np.float32)
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"person mask は (H,W) 形式である必要があります: shape={mask_array.shape}")
    mask_float = normalize_alpha_float(mask_array, shape)
    mask_binary = mask_float >= 0.5
    if not bool(mask_binary.any()):
        return np.zeros(shape, dtype=np.float32)
    kernel = max(1, int(dilate_px) * 2 + 1)
    if int(feather_px) > 0:
        return feather_binary_mask(mask_binary, dilate_size=kernel, feather_radius=int(feather_px))
    return cv2.dilate(mask_binary.astype(np.uint8), np.ones((kernel, kernel), np.uint8), iterations=1).astype(np.float32)


def compose_hybrid_alpha(
    ben2_alpha: np.ndarray,
    tb_alpha: np.ndarray,
    person_region: np.ndarray,
    mode: str = "lighten",
) -> np.ndarray:
    """BEN2 と transparent-background の alpha を領域分担して合成する。

    ``person_region`` 内は transparent-background、外側は BEN2 を基本にする。境界の feather
    領域では ``mode`` に応じて比較明 / over / screen で混ぜる。
    """
    ben2 = normalize_alpha_float(ben2_alpha)
    tb = normalize_alpha_float(tb_alpha, ben2.shape)
    region = normalize_alpha_float(person_region, ben2.shape)
    normalized = str(mode).strip().lower()
    if normalized in {"lighten", "max"}:
        composed = np.maximum(ben2, tb)
    elif normalized == "person_over_ben2":
        composed = tb + ben2 * (1.0 - tb)
    elif normalized == "ben2_over_person":
        composed = ben2 + tb * (1.0 - ben2)
    elif normalized == "screen":
        composed = 1.0 - (1.0 - ben2) * (1.0 - tb)
    else:
        raise ValueError(
            "composition_mode は 'lighten' / 'person_over_ben2' / "
            f"'ben2_over_person' / 'screen' のいずれかです: {mode!r}"
        )
    combined = ben2 * (1.0 - region) + composed * region
    return np.clip(combined * 255.0, 0, 255).astype(np.uint8)


@component
class BEN2TransparentHybridVideoExtractor:
    """人物領域は transparent-background、それ以外は BEN2 で処理して alpha 合成する Component。"""

    def __init__(
        self,
        ben2_extractor: BEN2Extractor | None = None,
        tb_extractor: TransparentBGExtractor | None = None,
        output_dir: str = "outputs",
        config_path: str | None = None,
        device: str | None = None,
    ) -> None:
        config = load_route_a_config(config_path)
        alpha_cfg = config["alpha"]
        resolved_device = device or default_device()
        self.ben2_extractor = ben2_extractor or BEN2Extractor(
            repo_id=str(alpha_cfg.get("ben2_repo_id", "PramaLLC/BEN2")),
            checkpoint_path=str(alpha_cfg.get("ben2_checkpoint_path", "")),
            device=resolved_device,
        )
        self.tb_extractor = tb_extractor or TransparentBGExtractor(device=resolved_device)
        self.output_dir = _resolve_output_dir(output_dir)

    def warm_up(self) -> None:
        """Haystack の no-arg warm_up 契約を守る。モデルは run 内で初回遅延ロードする。"""
        return None

    @staticmethod
    def _build_preview(image_rgb: np.ndarray, alpha_u8: np.ndarray, rgba: np.ndarray, output_type: str) -> np.ndarray:
        alpha_float = alpha_u8.astype(np.float32) / 255.0
        if output_type == "green":
            return compose_alpha(image_rgb, alpha_float, (0, 255, 0))
        if output_type == "white":
            return compose_alpha(image_rgb, alpha_float, (255, 255, 255))
        if output_type == "blur":
            return compose_alpha(image_rgb, alpha_float, cv2.GaussianBlur(image_rgb, (51, 51), 0))
        return rgba

    @component.output_types(matte=dict)
    def run(
        self,
        frames: list,
        masks: dict = None,
        metadata: dict = None,
        output_mode: str = "video",
        tb_mode: str = "base",
        tb_jit: bool = False,
        tb_threshold: float = 0.0,
        tb_crop_padding: int = 40,
        tb_mask_guard_dilate: int = 21,
        tb_mask_guard_feather: int = 0,
        person_region_dilate_px: int = 0,
        person_region_feather_px: int = 8,
        composition_mode: str = "lighten",
        refine_foreground: bool = False,
        output_type: str = "rgba",
        rgba_codec: str = "webm_vp9",
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """frame ごとに BEN2/TB を適用し、合成済み alpha を逐次保存する。"""
        if not frames:
            raise ValueError("frames が空です。")
        normalized_mode = normalize_output_mode(output_mode)
        normalized_composition = str(composition_mode).strip().lower()
        if normalized_composition == "max":
            normalized_composition = "lighten"
        if normalized_composition not in COMPOSITION_MODES:
            raise ValueError(f"未対応の composition_mode です: {composition_mode!r}")

        frame_masks = (masks or {}).get("frame_masks", {})
        source_indices = list((metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(len(frames))))
        timestamp = resolve_run_timestamp(metadata)
        output_root = self.output_dir / timestamp
        video_dir = output_root / "video"
        sequence_root = output_root / "sequence"
        rgba_dir = sequence_root / "rgba"
        alpha_dir = sequence_root / "alpha"
        fps = float((metadata or {}).get("fps", 30.0))
        rgba_video_path: Path | None = None
        alpha_video_path: Path | None = None
        preview_video_path: Path | None = None
        rgba_stream: _ImageioAlphaVideoWriter | None = None
        alpha_stream: _ImageioWebmVideoWriter | None = None
        preview_stream: _ImageioWebmVideoWriter | None = None
        codec_fallback: list[tuple[str, str]] = []
        used_rgba_codec: str | None = None
        total_frames = max(len(frames), 1)
        # 人物mask のヒット状況を集計する。frame_masks があるのに一度も当たらない場合は
        # source_index の写像ズレ（上流 tracker とのキー空間不一致）を疑う手掛かりになる。
        available_mask_frames = sum(
            1 for value in frame_masks.values() if value is not None and bool(np.asarray(value).any())
        )
        person_hit_count = 0

        _notify_progress(progress_callback, "hybrid_alpha", 0.0, "BEN2 + transparent-background を初期化しています")
        _notify_progress(progress_callback, "hybrid_alpha", 0.01, "BEN2 モデルを読み込んでいます")
        self.ben2_extractor.warm_up()
        keepalive = _ProgressKeepAlive(progress_callback, "hybrid_alpha")
        try:
            for local_index, frame in enumerate(frames):
                image_rgb = ensure_rgb_array(frame)
                source_index = int(source_indices[local_index]) if local_index < len(source_indices) else local_index
                person_mask = frame_masks.get(source_index)
                person_region = build_person_region(
                    person_mask,
                    image_rgb.shape[:2],
                    dilate_px=int(person_region_dilate_px),
                    feather_px=int(person_region_feather_px),
                )
                ben2_alpha = self.ben2_extractor.infer_alpha(image_rgb, refine_foreground=bool(refine_foreground))
                if bool(np.any(person_region > 0.0)):
                    person_hit_count += 1
                    tb_result = self.tb_extractor.run(
                        image=image_rgb,
                        mask=person_mask,
                        tb_mode=tb_mode,
                        tb_jit=tb_jit,
                        tb_threshold=float(tb_threshold),
                        tb_output_type="rgba",
                        crop_padding=int(tb_crop_padding),
                        apply_mask_guard=True,
                        mask_guard_dilate=int(tb_mask_guard_dilate),
                        mask_guard_feather=int(tb_mask_guard_feather),
                    )
                    tb_alpha = tb_result["alpha"]
                else:
                    tb_alpha = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                alpha_frame = compose_hybrid_alpha(
                    ben2_alpha=ben2_alpha,
                    tb_alpha=tb_alpha,
                    person_region=person_region,
                    mode=normalized_composition,
                )
                rgba_frame = normalize_rgba_frame(alpha_to_rgba(image_rgb, alpha_frame))
                preview_frame = ensure_rgb_array(self._build_preview(image_rgb, alpha_frame, rgba_frame, output_type))
                if local_index == 0 and normalized_mode in {"video", "both"}:
                    _notify_progress(progress_callback, "hybrid_alpha", 0.03, "動画 codec を確認しています")
                    video_helper = VideoWriter(str(self.output_dir))
                    spec, codec_fallback = video_helper._select_rgba_codec(
                        rgba_frame.shape[:2],
                        preferred_rgba_codec=rgba_codec,
                    )
                    rgba_video_path = video_dir / f"rgba{spec.suffix}"
                    alpha_video_path = video_dir / "alpha.webm"
                    preview_video_path = video_dir / "preview.webm"
                    used_rgba_codec = spec.label
                    rgba_stream = _ImageioAlphaVideoWriter(rgba_video_path, rgba_frame, fps, spec)
                    alpha_stream = _ImageioWebmVideoWriter(alpha_video_path, alpha_frame, fps)
                    preview_stream = _ImageioWebmVideoWriter(preview_video_path, preview_frame, fps)
                if rgba_stream is not None:
                    rgba_stream.write(rgba_frame)
                if alpha_stream is not None:
                    alpha_stream.write(alpha_frame)
                if preview_stream is not None:
                    preview_stream.write(preview_frame)
                if normalized_mode in {"sequence", "both"}:
                    write_png_frame(rgba_dir / f"frame_{local_index:06d}.png", rgba_frame)
                    write_png_frame(alpha_dir / f"frame_{local_index:06d}.png", alpha_frame)
                keepalive.maybe(
                    local_index,
                    total_frames,
                    (local_index + 1) / total_frames,
                    f"BEN2/TB alpha を合成・保存しています ({local_index + 1}/{total_frames})",
                )
        finally:
            for stream in (rgba_stream, alpha_stream, preview_stream):
                if stream is not None:
                    stream.close()

        # ── 人物mask ヒット率チェック（無音フォールバックの検知・日本語注意喚起）──
        # person_hit_count==0 は「transparent-background が一度も適用されず、全 frame が
        # BEN2 単独になった」ことを意味する。原因は (A) 検出/追跡で人物 mask が 0 件、
        # (B) frame_masks はあるが source_index の写像がズレて 1 度も引けなかった、の 2 通り。
        # 特に (B) は静かにデグレードするため、日本語で明確に警告する。
        fallback_warning: str | None = None
        if person_hit_count == 0:
            if available_mask_frames > 0:
                fallback_warning = (
                    "【注意】人物mask は tracker から "
                    f"{available_mask_frames} frame 分渡されましたが、source_index の写像ズレにより "
                    "1 frame も参照できませんでした。transparent-background が全 frame でスキップされ、"
                    "出力は BEN2 単独になっています（人物・髪の切り抜き強化が効いていません）。"
                    "VideoReader の sampled_frame_indices と tracker の frame_masks キー整合を確認してください。"
                )
            else:
                fallback_warning = (
                    "【注意】人物mask が 1 件も得られませんでした（Text Prompt で対象を検出/追跡できていません）。"
                    "transparent-background は全 frame でスキップされ、出力は BEN2 単独です。"
                    "Text Prompt を 'person' など単純な語に変える、Box threshold を下げる、"
                    "再検出周期を短くする、などを試してください。"
                )
            logger.warning(fallback_warning)
            warnings.warn(fallback_warning, RuntimeWarning, stacklevel=2)

        matte = {
            "rgba_video_path": str(rgba_video_path) if rgba_video_path else None,
            "alpha_video_path": str(alpha_video_path) if alpha_video_path else None,
            "preview_video_path": str(preview_video_path) if preview_video_path else None,
            "rgba_sequence_dir": str(rgba_dir) if normalized_mode in {"sequence", "both"} else None,
            "alpha_sequence_dir": str(alpha_dir) if normalized_mode in {"sequence", "both"} else None,
            "preview_sequence_dir": None,
            "sequence_pattern": "frame_{:06d}.png" if normalized_mode in {"sequence", "both"} else None,
            "fps": fps,
            "frame_count": len(frames),
            "output_mode": normalized_mode,
            "rgba_frames": [],
            "alpha_frames": [],
            "preview_frames": [],
            "metadata": {
                "source": "ben2-transparent-hybrid-video",
                "timestamp": timestamp,
                "source_metadata": metadata or {},
                "tb_mode": tb_mode,
                "composition_mode": normalized_composition,
                "person_region_dilate_px": int(person_region_dilate_px),
                "person_region_feather_px": int(person_region_feather_px),
                "streamed_outputs": True,
                "codec_fallback": codec_fallback,
                "used_rgba_codec": used_rgba_codec,
                "person_mask_hit_count": int(person_hit_count),
                "person_mask_available_frames": int(available_mask_frames),
                "person_mask_fallback_warning": fallback_warning,
            },
        }
        return {"matte": matte}
