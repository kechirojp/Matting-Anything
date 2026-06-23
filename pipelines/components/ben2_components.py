"""ルートA案（ブラー誘導 → BEN2 再α化）の BEN2 モデルラッパと動画 Component。

設計方針:
- ``BEN2Extractor``: BEN2 base（PramaLLC/BEN2, MIT）の推論だけを担うプレーンなモデルラッパ。
  重いモデルは import 時ではなく ``warm_up()`` で遅延・冪等に読み込む（Haystack 原則）。
  α 生成モデルとして差し替え可能な単位に閉じる（transparent-background と交換可能な責務境界）。
- ``BEN2RouteAVideoExtractor``: 「合成軸 = ルートA（ブラー誘導）」の責務を持つ Haystack Component。
  per frame で ① マスク膨張 → ② 背景ブラー → ③ BEN2 推論 → ④ RGBA 合成を行い、結果を RAM に
  溜めず逐次書き出す。出力 ``matte`` 契約は ``TransparentBGVideoExtractor`` と同一で、既存の
  ``VideoWriter`` / ``FrameSequenceWriter`` / ``TrackingOverlayWriter`` をそのまま再利用できる。
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from haystack import component

from .common import compose_alpha, ensure_rgb_array, stable_sigmoid
from .model_components import default_device, require_gpu_for_heavy_inference
from .route_a_common import (
    alpha_to_rgba,
    apply_gate_to_alpha,
    ben2_rgba_to_alpha,
    blur_background_outside_gate,
    combine_alpha_with_mask,
    dilate_mask_to_gate,
    load_route_a_config,
    resolve_ben2_device,
)
from .video_common import (
    composite_alpha_by_ownership,
    normalize_output_mode,
    normalize_rgba_frame,
    write_png_frame,
)
from .video_model_components import (
    ProgressCallback,
    VideoWriter,
    _ImageioAlphaVideoWriter,
    _notify_progress,
    _OpenCVFrameVideoWriter,
    _ProgressKeepAlive,
    _resolve_output_dir,
)


class BEN2Extractor:
    """BEN2 base で前景 α を生成するモデルラッパ（推論のみを担う単一責務クラス）。

    α 生成モデルとして差し替え可能な単位に閉じており、ルートAの合成ロジック
    （膨張・ブラー・合成）からは独立している。

    Args:
        repo_id: BEN2 base の Hugging Face repo_id（``from_pretrained`` 用）。
        checkpoint_path: ローカル重みパス。非空なら ``loadcheckpoints`` で読み込む。
        device: 推論デバイス。None なら環境から推定する。
    """

    def __init__(
        self,
        repo_id: str = "PramaLLC/BEN2",
        checkpoint_path: str = "",
        device: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.checkpoint_path = checkpoint_path or ""
        self.device = device or resolve_ben2_device()
        self._model: Any | None = None

    def warm_up(self) -> None:
        """BEN2 base を遅延・冪等に初期化する（import 時には触らない）。"""
        if self._model is not None:
            return
        require_gpu_for_heavy_inference(self.__class__.__name__, self.device)
        from ben2 import BEN_Base

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            model = BEN_Base()
            model.loadcheckpoints(self.checkpoint_path)
        else:
            model = BEN_Base.from_pretrained(self.repo_id)
        model.to(self.device).eval()
        self._model = model

    def infer_alpha(self, image_rgb: np.ndarray, refine_foreground: bool = False) -> np.ndarray:
        """1 フレームの RGB 画像から BEN2 で α（H,W uint8）を生成する。

        Args:
            image_rgb: (H,W,3) の RGB 画像（ブラー誘導済みフレーム I' を渡す）。
            refine_foreground: 境界精緻化の後処理を行うか（精度↑/時間↑）。

        Returns:
            (H,W) uint8 の α。
        """
        self.warm_up()
        from PIL import Image

        assert self._model is not None
        pil_image = Image.fromarray(ensure_rgb_array(image_rgb))
        foreground = self._model.inference(pil_image, refine_foreground=bool(refine_foreground))
        return ben2_rgba_to_alpha(foreground)


@component
class BEN2RouteAVideoExtractor:
    """ルートA（ブラー誘導 → BEN2 再α化）で各 frame を処理し逐次書き出す Component。

    union / per_object の 2 経路を持つ:
    - ``"union"`` (既定): フレームあたり BEN2 1 回。全対象の union マスクから 1 つのゲートを作り、
      G 外をブラーして誘導する軽量経路。
    - ``"per_object"``: フレームあたり BEN2 N 回。対象ごとに誘導・推論し所有権で α 合成する忠実経路。
    """

    def __init__(
        self,
        repo_id: str | None = None,
        ben2_checkpoint_path: str | None = None,
        device: str | None = None,
        output_dir: str = "outputs",
        config_path: str | None = None,
    ) -> None:
        config = load_route_a_config(config_path)
        alpha_cfg = config["alpha"]
        resolved_repo = repo_id or str(alpha_cfg.get("ben2_repo_id", "PramaLLC/BEN2"))
        resolved_ckpt = (
            ben2_checkpoint_path
            if ben2_checkpoint_path is not None
            else str(alpha_cfg.get("ben2_checkpoint_path", ""))
        )
        self.extractor = BEN2Extractor(
            repo_id=resolved_repo,
            checkpoint_path=resolved_ckpt,
            device=device or default_device(),
        )
        self.output_dir = _resolve_output_dir(output_dir)

    def warm_up(self) -> None:
        """Haystack Pipeline の no-arg warm_up 契約に合わせる（BEN2 は推論時に遅延ロード）。"""
        return None

    def _build_preview(self, image_rgb: np.ndarray, alpha_u8: np.ndarray, rgba: np.ndarray, output_type: str) -> np.ndarray:
        """出力種別に応じたプレビュー画像を作る。"""
        alpha_float = alpha_u8.astype(np.float32) / 255.0
        if output_type == "green":
            return compose_alpha(image_rgb, alpha_float, (0, 255, 0))
        if output_type == "white":
            return compose_alpha(image_rgb, alpha_float, (255, 255, 255))
        if output_type == "blur":
            return compose_alpha(image_rgb, alpha_float, cv2.GaussianBlur(image_rgb, (51, 51), 0))
        return rgba

    def _process_union_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None,
        *,
        dilation_px: int,
        blur_kernel: int,
        blur_sigma: float,
        feather_px: int,
        refine_foreground: bool,
        gate_alpha: bool,
        output_type: str,
        mask_floor_mode: str = "none",
    ) -> dict[str, np.ndarray]:
        """union 経路: union マスクから 1 ゲートを作り BEN2 を 1 回適用する（①〜④）。"""
        image_rgb = ensure_rgb_array(frame)
        if mask is None:
            # マスク未供給時は全画面誘導なし（ゲート全面）で BEN2 にそのまま渡す。
            guided = image_rgb
            gate = np.ones(image_rgb.shape[:2], dtype=np.uint8)
        else:
            gate = dilate_mask_to_gate(mask, dilation_px)
            guided = blur_background_outside_gate(image_rgb, gate, blur_kernel, blur_sigma, feather_px)
        alpha_u8 = self.extractor.infer_alpha(guided, refine_foreground=refine_foreground)
        if gate_alpha and mask is not None:
            alpha_u8 = apply_gate_to_alpha(alpha_u8, gate)
        # SAM2 soft マスク M（膨張前）を α の床として加算合成し、BEN2 の抜け落ち（ちらつき）を補う。
        if mask is not None and str(mask_floor_mode).strip().lower() not in {"none", "", "off"}:
            alpha_u8 = combine_alpha_with_mask(alpha_u8, mask, mask_floor_mode)
        rgba = alpha_to_rgba(image_rgb, alpha_u8)
        preview = self._build_preview(image_rgb, alpha_u8, rgba, output_type)
        return {"rgba": rgba, "alpha": alpha_u8, "preview": preview}

    def _process_per_object_frame(
        self,
        frame: np.ndarray,
        logits: np.ndarray,
        ownership: np.ndarray,
        *,
        dilation_px: int,
        blur_kernel: int,
        blur_sigma: float,
        feather_px: int,
        refine_foreground: bool,
        gate_alpha: bool,
        output_type: str,
        mask_floor_mode: str = "none",
    ) -> dict[str, np.ndarray]:
        """per_object 経路: 対象ごとに誘導・BEN2 推論し所有権で α 合成する（①〜⑤）。"""
        image_rgb = ensure_rgb_array(frame)
        logits_array = np.asarray(logits, dtype=np.float32)
        num_objects = logits_array.shape[0]
        per_object_alphas: list[np.ndarray] = []
        union_soft_mask: np.ndarray | None = None
        for obj_index in range(num_objects):
            soft_mask = stable_sigmoid(logits_array[obj_index])
            union_soft_mask = soft_mask if union_soft_mask is None else np.maximum(union_soft_mask, soft_mask)
            gate = dilate_mask_to_gate(soft_mask, dilation_px)
            guided = blur_background_outside_gate(image_rgb, gate, blur_kernel, blur_sigma, feather_px)
            alpha_o = self.extractor.infer_alpha(guided, refine_foreground=refine_foreground).astype(np.float32) / 255.0
            if gate_alpha:
                alpha_o = apply_gate_to_alpha(alpha_o, gate).astype(np.float32) / 255.0
            per_object_alphas.append(alpha_o)
        alpha_final = composite_alpha_by_ownership(per_object_alphas, ownership)
        alpha_u8 = np.clip(alpha_final * 255.0, 0, 255).astype(np.uint8)
        # 全対象の union soft マスクを α の床として加算合成し、BEN2 の抜け落ち（ちらつき）を補う。
        if union_soft_mask is not None and str(mask_floor_mode).strip().lower() not in {"none", "", "off"}:
            alpha_u8 = combine_alpha_with_mask(alpha_u8, union_soft_mask, mask_floor_mode)
        rgba = alpha_to_rgba(image_rgb, alpha_u8)
        preview = self._build_preview(image_rgb, alpha_u8, rgba, output_type)
        return {"rgba": rgba, "alpha": alpha_u8, "preview": preview}

    @component.output_types(matte=dict)
    def run(
        self,
        frames: list,
        masks: dict = None,
        metadata: dict = None,
        output_mode: str = "video",
        dilation_px: int = 24,
        blur_kernel: int = 41,
        blur_sigma: float = 0.0,
        feather_px: int = 12,
        refine_foreground: bool = False,
        matte_mode: str = "union",
        gate_alpha: bool = False,
        mask_floor_mode: str = "none",
        output_type: str = "rgba",
        rgba_codec: str = "webm_vp9",
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """frame ごとにルートA処理を適用し、出力を RAM に溜めず保存する。"""
        if not frames:
            raise ValueError("frames が空です。")
        normalized_mode = normalize_output_mode(output_mode)
        mode = str(matte_mode).strip().lower()
        if mode not in {"union", "per_object"}:
            raise ValueError(f"matte_mode は 'union' か 'per_object' のいずれかです: {matte_mode!r}")
        frame_masks = (masks or {}).get("frame_masks", {})
        per_object_logits = (masks or {}).get("per_object_logits", {})
        ownership_by_frame = (masks or {}).get("ownership", {})
        source_indices = list((metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(len(frames))))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = self.output_dir / timestamp
        video_dir = output_root / "video"
        sequence_root = output_root / "sequence"
        rgba_dir = sequence_root / "rgba"
        alpha_dir = sequence_root / "alpha"
        preview_dir = sequence_root / "preview"
        fps = float((metadata or {}).get("fps", 30.0))
        rgba_video_path: Path | None = None
        alpha_video_path: Path | None = None
        preview_video_path: Path | None = None
        rgba_stream: _ImageioAlphaVideoWriter | None = None
        alpha_stream: _OpenCVFrameVideoWriter | None = None
        preview_stream: _OpenCVFrameVideoWriter | None = None
        codec_fallback: list[tuple[str, str]] = []
        used_rgba_codec: str | None = None
        total_frames = max(len(frames), 1)
        _notify_progress(progress_callback, "ben2_route_a", 0.0, "BEN2（ルートA）を初期化しています")
        ben2_keepalive = _ProgressKeepAlive(progress_callback, "ben2_route_a")
        try:
            for local_index, frame in enumerate(frames):
                source_index = int(source_indices[local_index]) if local_index < len(source_indices) else local_index
                logits = per_object_logits.get(source_index)
                ownership = ownership_by_frame.get(source_index)
                logits_array = np.asarray(logits) if logits is not None else None
                use_per_object = (
                    mode == "per_object"
                    and logits_array is not None
                    and ownership is not None
                    and logits_array.ndim == 3
                    and logits_array.shape[0] >= 1
                )
                if use_per_object:
                    result = self._process_per_object_frame(
                        frame,
                        logits_array,
                        np.asarray(ownership),
                        dilation_px=int(dilation_px),
                        blur_kernel=int(blur_kernel),
                        blur_sigma=float(blur_sigma),
                        feather_px=int(feather_px),
                        refine_foreground=bool(refine_foreground),
                        gate_alpha=bool(gate_alpha),
                        output_type=output_type,
                        mask_floor_mode=str(mask_floor_mode),
                    )
                else:
                    mask = frame_masks.get(source_index)
                    result = self._process_union_frame(
                        frame,
                        mask,
                        dilation_px=int(dilation_px),
                        blur_kernel=int(blur_kernel),
                        blur_sigma=float(blur_sigma),
                        feather_px=int(feather_px),
                        refine_foreground=bool(refine_foreground),
                        gate_alpha=bool(gate_alpha),
                        output_type=output_type,
                        mask_floor_mode=str(mask_floor_mode),
                    )
                rgba_frame = normalize_rgba_frame(result["rgba"])
                alpha_frame = np.asarray(result["alpha"]).astype(np.uint8, copy=False)
                preview_frame = ensure_rgb_array(result["preview"])
                if local_index == 0 and normalized_mode in {"video", "both"}:
                    _notify_progress(progress_callback, "ben2_route_a", 0.03, "動画 codec を確認しています")
                    video_helper = VideoWriter(str(self.output_dir))
                    spec, codec_fallback = video_helper._select_rgba_codec(
                        rgba_frame.shape[:2],
                        preferred_rgba_codec=rgba_codec,
                    )
                    rgba_video_path = video_dir / f"rgba{spec.suffix}"
                    alpha_video_path = video_dir / "alpha.mp4"
                    preview_video_path = video_dir / "preview.mp4"
                    used_rgba_codec = spec.label
                    rgba_stream = _ImageioAlphaVideoWriter(rgba_video_path, rgba_frame, fps, spec)
                    alpha_stream = _OpenCVFrameVideoWriter(alpha_video_path, alpha_frame, fps, "mp4v", channels=1)
                    preview_stream = _OpenCVFrameVideoWriter(preview_video_path, preview_frame, fps, "mp4v", channels=3)
                if rgba_stream is not None:
                    rgba_stream.write(rgba_frame)
                if alpha_stream is not None:
                    alpha_stream.write(alpha_frame)
                if preview_stream is not None:
                    preview_stream.write(preview_frame)
                if normalized_mode in {"sequence", "both"}:
                    write_png_frame(rgba_dir / f"frame_{local_index:06d}.png", rgba_frame)
                    write_png_frame(alpha_dir / f"frame_{local_index:06d}.png", alpha_frame)
                    write_png_frame(preview_dir / f"frame_{local_index:06d}.png", preview_frame)
                ben2_keepalive.maybe(
                    local_index,
                    total_frames,
                    (local_index + 1) / total_frames,
                    f"BEN2（ルートA）を frame ごとに適用・保存しています ({local_index + 1}/{total_frames})",
                )
        finally:
            for stream in (rgba_stream, alpha_stream, preview_stream):
                if stream is not None:
                    stream.close()
        matte = {
            "rgba_video_path": str(rgba_video_path) if rgba_video_path else None,
            "alpha_video_path": str(alpha_video_path) if alpha_video_path else None,
            "preview_video_path": str(preview_video_path) if preview_video_path else None,
            "rgba_sequence_dir": str(rgba_dir) if normalized_mode in {"sequence", "both"} else None,
            "alpha_sequence_dir": str(alpha_dir) if normalized_mode in {"sequence", "both"} else None,
            "preview_sequence_dir": str(preview_dir) if normalized_mode in {"sequence", "both"} else None,
            "sequence_pattern": "frame_{:06d}.png" if normalized_mode in {"sequence", "both"} else None,
            "fps": fps,
            "frame_count": len(frames),
            "output_mode": normalized_mode,
            "rgba_frames": [],
            "alpha_frames": [],
            "preview_frames": [],
            "metadata": {
                "source": "ben2-route-a-video",
                "timestamp": timestamp,
                "source_metadata": metadata or {},
                "route": "A_blur_guidance",
                "matte_mode": mode,
                "dilation_px": int(dilation_px),
                "blur_kernel": int(blur_kernel),
                "feather_px": int(feather_px),
                "refine_foreground": bool(refine_foreground),
                "gate_alpha": bool(gate_alpha),
                "output_type": output_type,
                "streamed_outputs": True,
                "codec_fallback": codec_fallback,
                "used_rgba_codec": used_rgba_codec,
            },
        }
        return {"matte": matte}
