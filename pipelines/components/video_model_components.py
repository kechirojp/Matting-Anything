"""動画 I/O と SAM2 video predictor を扱う Haystack Component。"""

from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from haystack import component

from .common import ensure_rgb_array, render_tracking_overlay_frame
from .model_components import TransparentBGExtractor, default_device, require_gpu_for_heavy_inference
from .video_common import (
    build_frame_mask_sequence,
    build_video_source,
    frame_cache_bytes,
    normalize_output_mode,
    normalize_rgba_frame,
    sample_frame_indices,
    write_png_frame,
)


ProgressCallback = Callable[[str, float, str], None]


def _resolve_output_dir(output_dir: str) -> Path:
    """出力ディレクトリを PROJECT_ROOT 基準の絶対パスへ解決する。"""
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    return project_root / output_path


def _notify_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    fraction: float,
    description: str,
) -> None:
    """Gradio 側へ Component 内部の進捗を通知する。"""
    if progress_callback is None:
        return
    progress_callback(stage, min(max(float(fraction), 0.0), 1.0), description)


class _OpenCVFrameVideoWriter:
    """1 frame ずつ OpenCV 動画へ書き出す軽量 writer。"""

    def __init__(self, path: Path, first_frame: np.ndarray, fps: float, fourcc_name: str, channels: int) -> None:
        first = np.asarray(first_frame)
        height, width = first.shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._channels = channels
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(fps),
            (width, height),
            isColor=channels != 1,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"VideoWriter を開けません: {path}")

    def write(self, frame: np.ndarray) -> None:
        frame_array = np.asarray(frame).astype(np.uint8, copy=False)
        if self._channels == 1:
            if frame_array.ndim == 2:
                self._writer.write(frame_array)
            elif frame_array.shape[2] == 4:
                self._writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGBA2GRAY))
            else:
                self._writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY))
        elif frame_array.ndim == 2:
            self._writer.write(cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR))
        elif frame_array.shape[2] == 4:
            self._writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGRA))
        else:
            self._writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()


@component
class VideoReader:
    """動画ファイルを RGB frame list と metadata に分解する Component。"""

    @component.output_types(frames=list, metadata=dict)
    def run(
        self,
        video_path: str,
        max_frames: int = 300,
        frame_step: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """OpenCV で動画を読み込み、RGB uint8 frame の list を返す。"""
        if not video_path:
            raise ValueError("video_path が空です。")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            codec_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
            codec = "".join(chr((codec_int >> 8 * index) & 0xFF) for index in range(4)).strip()
            indices = set(sample_frame_indices(frame_count, int(max_frames), int(frame_step)))
            target_count = max(len(indices), 1)
            frames: list[np.ndarray] = []
            source_indices: list[int] = []
            _notify_progress(progress_callback, "video_reader", 0.0, "動画を読み込んでいます")
            current_index = 0
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if current_index in indices:
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    source_indices.append(current_index)
                    if len(frames) == 1 or len(frames) % 10 == 0 or len(frames) >= target_count:
                        _notify_progress(
                            progress_callback,
                            "video_reader",
                            len(frames) / target_count,
                            f"動画を読み込んでいます ({len(frames)}/{target_count} frames)",
                        )
                    if len(frames) >= int(max_frames):
                        break
                current_index += 1
        finally:
            capture.release()
        if not frames:
            raise ValueError("動画から frame を読み込めませんでした。")
        metadata = build_video_source(
            str(video_path),
            fps=fps,
            width=width or int(frames[0].shape[1]),
            height=height or int(frames[0].shape[0]),
            frame_count=frame_count or len(frames),
            codec=codec,
            metadata={
                "sampled_frame_indices": source_indices,
                "sampled_count": len(frames),
                "frame_step": int(frame_step),
                "max_frames": int(max_frames),
                "cache_bytes": frame_cache_bytes(len(frames), frames[0].shape[0], frames[0].shape[1], 3),
            },
        )
        return {"frames": frames, "metadata": metadata}


@component
class SAM2VideoPropagator:
    """SAM2 video predictor で first-frame prompt から全 frame の mask を伝搬する Component。"""

    def __init__(self, checkpoint_path: str | None = None, config_name: str | None = None, device: str | None = None) -> None:
        project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "SAM2_CKPT_PATH",
            str(project_root / "checkpoints" / "SAM2" / "sam2.1_hiera_large.pt"),
        )
        self.config_name = config_name or os.environ.get("SAM2_CONFIG_NAME", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.device = device or default_device()
        self._video_predictor: Any | None = None

    def tracker_metadata(self) -> dict[str, Any]:
        """使用中の tracker config / checkpoint と samurai_mode を可視化用に公開する。"""
        config = str(self.config_name)
        return {
            "tracker_config": config,
            "tracker_checkpoint": str(self.checkpoint_path),
            "samurai_mode": "samurai" in config.lower(),
        }

    def warm_up(self) -> None:
        """SAM2 video predictor を遅延・冪等に初期化する。"""
        if self._video_predictor is not None:
            return
        require_gpu_for_heavy_inference(self.__class__.__name__, self.device)
        from sam2.build_sam import build_sam2_video_predictor

        self._video_predictor = build_sam2_video_predictor(self.config_name, self.checkpoint_path, device=str(self.device))

    @component.output_types(masks=dict)
    def run(
        self,
        frames: list,
        metadata: dict = None,
        points: list[tuple[int, int]] | None = None,
        labels: list[int] | None = None,
        box: list[int] | None = None,
        boxes: list[list[int]] | None = None,
        object_id: int = 1,
        prompt_frame_idx: int = 0,
        bidirectional: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """prompt を SAM2 video predictor に登録し、frame mask 列を返す。

        複合対象 union: ``boxes`` を渡すと各 box を obj_id 1..N として登録し、frame ごとに
        全 obj の mask を OR 統合した単一 mask を返す。``bidirectional`` の場合は
        ``prompt_frame_idx`` を起点に forward / reverse の 2 pass を伝搬する。``boxes`` 未指定時は
        従来の単一 box / point・object_id・forward only パスを維持する（後方互換）。
        """
        if not frames:
            raise ValueError("frames が空です。")
        if not points and box is None and not boxes:
            raise ValueError("SAM2 video prompt が空です。points / box / boxes のいずれかを指定してください。")
        prompt_frame_idx = int(prompt_frame_idx)
        if prompt_frame_idx < 0 or prompt_frame_idx >= len(frames):
            raise ValueError(f"prompt_frame_idx が範囲外です: {prompt_frame_idx}（許容 0〜{len(frames) - 1}）")
        _notify_progress(progress_callback, "sam2_video", 0.0, "SAM2 video predictor を初期化しています")
        self.warm_up()
        _notify_progress(progress_callback, "sam2_video", 0.08, "SAM2 用の一時 frame を準備しています")
        assert self._video_predictor is not None
        import torch

        # 複数 box は obj_id 1..N、単一 prompt は object_id を追跡対象とする。
        if boxes:
            target_object_ids = list(range(1, len(boxes) + 1))
        else:
            target_object_ids = [int(object_id)]
        directions = [False, True] if bidirectional else [False]

        frame_masks: dict[int, np.ndarray] = {}
        source_indices = list((metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(len(frames))))
        total_frames = max(len(frames), 1)
        propagation_total = total_frames * len(directions)
        with tempfile.TemporaryDirectory(prefix="sam2_video_frames_") as temp_dir:
            temp_path = Path(temp_dir)
            for frame_index, frame in enumerate(frames):
                frame_rgb = ensure_rgb_array(frame)
                cv2.imwrite(str(temp_path / f"{frame_index:06d}.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                if frame_index == 0 or (frame_index + 1) % 10 == 0 or frame_index + 1 == total_frames:
                    _notify_progress(
                        progress_callback,
                        "sam2_video",
                        0.08 + 0.12 * ((frame_index + 1) / total_frames),
                        f"SAM2 用の一時 frame を準備しています ({frame_index + 1}/{total_frames})",
                    )
            with torch.inference_mode():
                _notify_progress(progress_callback, "sam2_video", 0.22, "SAM2 の video state を初期化しています")
                state = self._video_predictor.init_state(video_path=str(temp_path))
                if boxes:
                    for obj_id, single_box in zip(target_object_ids, boxes):
                        self._video_predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=prompt_frame_idx,
                            obj_id=obj_id,
                            box=np.asarray(single_box, dtype=np.float32),
                        )
                else:
                    add_kwargs: dict[str, Any] = {"inference_state": state, "frame_idx": prompt_frame_idx, "obj_id": int(object_id)}
                    if points:
                        add_kwargs["points"] = np.asarray(points, dtype=np.float32)
                        add_kwargs["labels"] = np.asarray(labels or [1] * len(points), dtype=np.int32)
                    if box is not None:
                        add_kwargs["box"] = np.asarray(box, dtype=np.float32)
                    self._video_predictor.add_new_points_or_box(**add_kwargs)
                _notify_progress(progress_callback, "sam2_video", 0.25, "SAM2 mask を動画全体へ伝搬しています")
                propagated_count = 0
                for reverse in directions:
                    for out_frame_idx, out_obj_ids, out_mask_logits in self._video_predictor.propagate_in_video(state, reverse=reverse):
                        propagated_count += 1
                        object_ids = [int(value) for value in out_obj_ids]
                        union_mask: np.ndarray | None = None
                        for target_obj in target_object_ids:
                            if target_obj not in object_ids:
                                continue
                            mask_logits = out_mask_logits[object_ids.index(target_obj)]
                            if hasattr(mask_logits, "detach"):
                                obj_mask = (mask_logits.detach().cpu().numpy() > 0.0).squeeze()
                            else:
                                obj_mask = (np.asarray(mask_logits) > 0.0).squeeze()
                            obj_mask = obj_mask.astype(bool)
                            union_mask = obj_mask if union_mask is None else (union_mask | obj_mask)
                        if union_mask is None:
                            continue
                        source_index = int(source_indices[int(out_frame_idx)]) if int(out_frame_idx) < len(source_indices) else int(out_frame_idx)
                        # forward / reverse 両 pass で重複する frame は OR 統合する。
                        existing = frame_masks.get(source_index)
                        frame_masks[source_index] = union_mask if existing is None else (existing | union_mask)
                        if propagated_count == 1 or propagated_count % 10 == 0 or propagated_count == propagation_total:
                            _notify_progress(
                                progress_callback,
                                "sam2_video",
                                0.25 + 0.75 * (propagated_count / propagation_total),
                                f"SAM2 mask を動画全体へ伝搬しています ({propagated_count}/{propagation_total})",
                            )
        masks = build_frame_mask_sequence(
            frame_masks,
            object_ids=list(target_object_ids),
            metadata={
                "points": points or [],
                "labels": labels or [],
                "box": box,
                "boxes": [list(single_box) for single_box in (boxes or [])],
                "prompt_frame_idx": prompt_frame_idx,
                "bidirectional": bool(bidirectional),
                "source_metadata": metadata or {},
                **self.tracker_metadata(),
            },
        )
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return {"masks": masks}


@component
class TransparentBGVideoExtractor:
    """各 frame に transparent-background を適用し、結果を逐次書き出す Component。"""

    def __init__(
        self,
        project_root: str | None = None,
        device: str | None = None,
        output_dir: str = "outputs",
    ) -> None:
        self.extractor = TransparentBGExtractor(project_root=project_root, device=device)
        self.output_dir = _resolve_output_dir(output_dir)

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
        tb_output_type: str = "rgba",
        crop_padding: int = 40,
        rgba_codec: str = "webm_vp9",
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """frame ごとに `TransparentBGExtractor` を呼び、出力を RAM に溜めず保存する。"""
        if not frames:
            raise ValueError("frames が空です。")
        normalized_mode = normalize_output_mode(output_mode)
        frame_masks = (masks or {}).get("frame_masks", {})
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
        rgba_stream: _OpenCVFrameVideoWriter | None = None
        alpha_stream: _OpenCVFrameVideoWriter | None = None
        preview_stream: _OpenCVFrameVideoWriter | None = None
        codec_fallback: list[tuple[str, str]] = []
        used_rgba_codec: str | None = None
        total_frames = max(len(frames), 1)
        _notify_progress(progress_callback, "transparent_bg", 0.0, "transparent-background を初期化しています")
        try:
            for local_index, frame in enumerate(frames):
                source_index = int(source_indices[local_index]) if local_index < len(source_indices) else local_index
                mask = frame_masks.get(source_index)
                result = self.extractor.run(
                    image=frame,
                    mask=mask,
                    tb_mode=tb_mode,
                    tb_jit=tb_jit,
                    tb_threshold=tb_threshold,
                    tb_output_type=tb_output_type,
                    crop_padding=int(crop_padding),
                )
                rgba_frame = normalize_rgba_frame(result["rgba"])
                alpha_frame = np.asarray(result["alpha"]).astype(np.uint8, copy=False)
                preview_frame = ensure_rgb_array(result["preview"])
                if local_index == 0 and normalized_mode in {"video", "both"}:
                    _notify_progress(progress_callback, "transparent_bg", 0.03, "動画 codec を確認しています")
                    video_helper = VideoWriter(str(self.output_dir))
                    fourcc, suffix, codec_fallback = video_helper._select_rgba_codec(
                        rgba_frame.shape[:2],
                        preferred_rgba_codec=rgba_codec,
                    )
                    rgba_video_path = video_dir / f"rgba{suffix}"
                    alpha_video_path = video_dir / "alpha.mp4"
                    preview_video_path = video_dir / "preview.mp4"
                    used_rgba_codec = suffix.lstrip(".")
                    rgba_stream = _OpenCVFrameVideoWriter(rgba_video_path, rgba_frame, fps, fourcc, channels=4)
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
                if local_index == 0 or (local_index + 1) % 5 == 0 or local_index + 1 == total_frames:
                    _notify_progress(
                        progress_callback,
                        "transparent_bg",
                        (local_index + 1) / total_frames,
                        f"transparent-background を frame ごとに適用・保存しています ({local_index + 1}/{total_frames})",
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
                "source": "transparent-background-video",
                "timestamp": timestamp,
                "source_metadata": metadata or {},
                "tb_mode": tb_mode,
                "tb_output_type": tb_output_type,
                "crop_padding": int(crop_padding),
                "streamed_outputs": True,
                "codec_fallback": codec_fallback,
                "used_rgba_codec": used_rgba_codec,
            },
        }
        return {"matte": matte}


@component
class TrackingOverlayWriter:
    """各 frame に追跡 mask の輪郭+半透明塗りを重ね、追従確認用 overlay 動画を逐次書き出す Component。"""

    _OBJECT_COLORS = (
        (30, 144, 255),
        (255, 140, 0),
        (46, 204, 113),
        (155, 89, 182),
        (231, 76, 60),
    )

    def __init__(self, output_dir: str = "outputs", fill_alpha: float = 0.45) -> None:
        self.output_dir = _resolve_output_dir(output_dir)
        self.fill_alpha = float(fill_alpha)

    @component.output_types(overlay=dict)
    def run(
        self,
        frames: list,
        masks: dict = None,
        metadata: dict = None,
        enabled: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """frame ごとに mask overlay を描き、追跡確認用の mp4 / PNG 連番を保存する。"""
        if not enabled:
            return {"overlay": {"overlay_video_path": None, "frame_count": 0, "enabled": False}}
        if not frames:
            raise ValueError("frames が空です。")
        frame_masks = (masks or {}).get("frame_masks", {})
        object_ids = list((masks or {}).get("object_ids", [1]))
        source_indices = list((metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(len(frames))))
        color = self._OBJECT_COLORS[(int(object_ids[0]) - 1) % len(self._OBJECT_COLORS)] if object_ids else self._OBJECT_COLORS[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = self.output_dir / timestamp
        video_dir = output_root / "video"
        overlay_sequence_dir = output_root / "sequence" / "overlay"
        fps = float((metadata or {}).get("fps", 30.0))
        overlay_video_path = video_dir / "tracking_overlay.mp4"
        overlay_stream: _OpenCVFrameVideoWriter | None = None
        total_frames = max(len(frames), 1)
        _notify_progress(progress_callback, "tracking_overlay", 0.0, "追跡確認用 overlay を生成しています")
        try:
            for local_index, frame in enumerate(frames):
                source_index = int(source_indices[local_index]) if local_index < len(source_indices) else local_index
                frame_rgb = ensure_rgb_array(frame)
                mask = frame_masks.get(source_index)
                if mask is None:
                    overlay_frame = frame_rgb
                else:
                    overlay_frame = render_tracking_overlay_frame(frame_rgb, mask, color=color, fill_alpha=self.fill_alpha)
                if overlay_stream is None:
                    overlay_stream = _OpenCVFrameVideoWriter(overlay_video_path, overlay_frame, fps, "mp4v", channels=3)
                overlay_stream.write(overlay_frame)
                write_png_frame(overlay_sequence_dir / f"frame_{local_index:06d}.png", overlay_frame)
                if local_index == 0 or (local_index + 1) % 5 == 0 or local_index + 1 == total_frames:
                    _notify_progress(
                        progress_callback,
                        "tracking_overlay",
                        (local_index + 1) / total_frames,
                        f"追跡確認用 overlay を frame ごとに保存しています ({local_index + 1}/{total_frames})",
                    )
        finally:
            if overlay_stream is not None:
                overlay_stream.close()
        tracker_metadata = {key: (masks or {}).get("metadata", {}).get(key) for key in ("tracker_config", "tracker_checkpoint", "samurai_mode")}
        overlay = {
            "overlay_video_path": str(overlay_video_path),
            "overlay_sequence_dir": str(overlay_sequence_dir),
            "sequence_pattern": "frame_{:06d}.png",
            "fps": fps,
            "frame_count": len(frames),
            "enabled": True,
            "metadata": {
                "source": "tracking-overlay",
                "timestamp": timestamp,
                **tracker_metadata,
            },
        }
        return {"overlay": overlay}


@component
class VideoWriter:
    """VideoMatteResult の frame 列を動画ファイルとして保存する Component。"""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = _resolve_output_dir(output_dir)
        self._codec_cache: dict[str, bool] = {}

    def _test_codec(self, codec: str, suffix: str, frame_shape: tuple[int, int], channels: int = 3) -> bool:
        if codec in self._codec_cache:
            return self._codec_cache[codec]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        test_path = self.output_dir / f"_codec_test_{codec}{suffix}"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(test_path), fourcc, 1.0, (frame_shape[1], frame_shape[0]), isColor=channels != 1)
        ok = writer.isOpened()
        writer.release()
        if test_path.exists():
            test_path.unlink()
        self._codec_cache[codec] = ok
        return ok

    def warm_up(self) -> None:
        """Haystack Pipeline の no-arg warm_up 契約に合わせる。"""
        return None

    def _select_rgba_codec(
        self,
        frame_shape: tuple[int, int],
        preferred_rgba_codec: str = "webm_vp9",
    ) -> tuple[str, str, list[tuple[str, str]]]:
        """利用可能な動画 codec を確認し、RGBA 用 codec を選ぶ。"""
        fallback: list[tuple[str, str]] = []
        candidates = []
        if preferred_rgba_codec == "mov_png":
            candidates.append(("png ", ".mov", "mov_png"))
        else:
            candidates.append(("VP90", ".webm", "webm_vp9"))
            candidates.append(("png ", ".mov", "mov_png"))
        for fourcc, suffix, label in candidates:
            if self._test_codec(fourcc, suffix, frame_shape, channels=4):
                fallback.append((label, "ok (used)"))
                return fourcc, suffix, fallback
            fallback.append((label, "failed (skipped)"))
        raise ValueError("RGBA 動画を書き出せる codec が見つかりません。連番出力を選択してください。")

    def _write_video(
        self,
        path: Path,
        frames: list[np.ndarray],
        fps: float,
        fourcc_name: str,
        channels: int = 3,
        progress_callback: ProgressCallback | None = None,
        progress_prefix: str = "動画を書き出しています",
    ) -> None:
        if not frames:
            raise ValueError(f"動画に書き出す frame がありません: {path}")
        first = np.asarray(frames[0])
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), float(fps), (width, height), isColor=channels != 1)
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter を開けません: {path}")
        try:
            total_frames = max(len(frames), 1)
            for frame_index, frame in enumerate(frames):
                frame_array = np.asarray(frame).astype(np.uint8, copy=False)
                if channels == 1:
                    if frame_array.ndim == 2:
                        writer.write(frame_array)
                    elif frame_array.shape[2] == 4:
                        writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGBA2GRAY))
                    else:
                        writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY))
                elif frame_array.ndim == 2:
                    writer.write(cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR))
                elif frame_array.shape[2] == 4:
                    writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGRA))
                else:
                    writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
                written_count = frame_index + 1
                if written_count == 1 or written_count % 20 == 0 or written_count == total_frames:
                    _notify_progress(
                        progress_callback,
                        "video_writer",
                        written_count / total_frames,
                        f"{progress_prefix} ({written_count}/{total_frames})",
                    )
        finally:
            writer.release()

    @component.output_types(matte=dict)
    def run(
        self,
        matte: dict,
        rgba_codec: str = "webm_vp9",
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """出力モードが video/both のとき動画ファイルを書き出す。"""
        output_mode = normalize_output_mode(matte.get("output_mode", "video"))
        if output_mode == "sequence":
            return {"matte": matte}
        timestamp = str(matte.get("metadata", {}).get("timestamp") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        video_dir = self.output_dir / timestamp / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        rgba_frames = [normalize_rgba_frame(frame) for frame in matte.get("rgba_frames", [])]
        alpha_frames = list(matte.get("alpha_frames", []))
        preview_frames = [ensure_rgb_array(frame) for frame in matte.get("preview_frames", [])]
        if not rgba_frames:
            if matte.get("rgba_video_path") and matte.get("alpha_video_path") and matte.get("preview_video_path"):
                return {"matte": matte}
            raise ValueError("RGBA frame が空です。")
        _notify_progress(progress_callback, "video_writer", 0.0, "RGBA 動画 codec を確認しています")
        fourcc, suffix, fallback = self._select_rgba_codec(rgba_frames[0].shape[:2], preferred_rgba_codec=rgba_codec)
        rgba_path = video_dir / f"rgba{suffix}"
        alpha_path = video_dir / "alpha.mp4"
        preview_path = video_dir / "preview.mp4"
        _notify_progress(progress_callback, "video_writer", 0.10, "RGBA 動画を書き出しています")
        self._write_video(
            rgba_path,
            rgba_frames,
            matte.get("fps", 30.0),
            fourcc,
            channels=4,
            progress_callback=progress_callback,
            progress_prefix="RGBA 動画を書き出しています",
        )
        _notify_progress(progress_callback, "video_writer", 0.45, "Alpha 動画を書き出しています")
        self._write_video(
            alpha_path,
            alpha_frames,
            matte.get("fps", 30.0),
            "mp4v",
            channels=1,
            progress_callback=progress_callback,
            progress_prefix="Alpha 動画を書き出しています",
        )
        _notify_progress(progress_callback, "video_writer", 0.75, "Preview 動画を書き出しています")
        self._write_video(
            preview_path,
            preview_frames,
            matte.get("fps", 30.0),
            "mp4v",
            channels=3,
            progress_callback=progress_callback,
            progress_prefix="Preview 動画を書き出しています",
        )
        _notify_progress(progress_callback, "video_writer", 1.0, "動画書き出しが完了しました")
        updated = dict(matte)
        updated["rgba_video_path"] = str(rgba_path)
        updated["alpha_video_path"] = str(alpha_path)
        updated["preview_video_path"] = str(preview_path)
        updated.setdefault("metadata", {})["codec_fallback"] = fallback
        updated.setdefault("metadata", {})["used_rgba_codec"] = suffix.lstrip(".")
        return {"matte": updated}


@component
class FrameSequenceWriter:
    """VideoMatteResult の frame 列を PNG 連番として保存する Component。"""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = _resolve_output_dir(output_dir)

    @component.output_types(matte=dict)
    def run(
        self,
        matte: dict,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """出力モードが sequence/both のとき PNG 連番を書き出す。"""
        output_mode = normalize_output_mode(matte.get("output_mode", "video"))
        if output_mode == "video":
            return {"matte": matte}
        timestamp = str(matte.get("metadata", {}).get("timestamp") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        sequence_root = self.output_dir / timestamp / "sequence"
        rgba_dir = sequence_root / "rgba"
        alpha_dir = sequence_root / "alpha"
        preview_dir = sequence_root / "preview"
        rgba_frames = [normalize_rgba_frame(frame) for frame in matte.get("rgba_frames", [])]
        alpha_frames = list(matte.get("alpha_frames", []))
        preview_frames = [ensure_rgb_array(frame) for frame in matte.get("preview_frames", [])]
        if not rgba_frames and matte.get("rgba_sequence_dir") and matte.get("alpha_sequence_dir") and matte.get("preview_sequence_dir"):
            return {"matte": matte}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        estimated_bytes = sum(frame.nbytes for frame in rgba_frames + alpha_frames + preview_frames)
        free_bytes = shutil.disk_usage(self.output_dir).free
        if free_bytes < estimated_bytes:
            raise RuntimeError("連番出力に必要な空き容量が不足しています。")
        total_writes = max(len(rgba_frames) + len(alpha_frames) + len(preview_frames), 1)
        written_count = 0
        for index, frame in enumerate(rgba_frames):
            write_png_frame(rgba_dir / f"frame_{index:06d}.png", frame)
            written_count += 1
            if written_count == 1 or written_count % 20 == 0 or written_count == total_writes:
                _notify_progress(progress_callback, "frame_sequence_writer", written_count / total_writes, "RGBA PNG 連番を書き出しています")
        for index, frame in enumerate(alpha_frames):
            write_png_frame(alpha_dir / f"frame_{index:06d}.png", frame)
            written_count += 1
            if written_count % 20 == 0 or written_count == total_writes:
                _notify_progress(progress_callback, "frame_sequence_writer", written_count / total_writes, "Alpha PNG 連番を書き出しています")
        for index, frame in enumerate(preview_frames):
            write_png_frame(preview_dir / f"frame_{index:06d}.png", frame)
            written_count += 1
            if written_count % 20 == 0 or written_count == total_writes:
                _notify_progress(progress_callback, "frame_sequence_writer", written_count / total_writes, "Preview PNG 連番を書き出しています")
        updated = dict(matte)
        updated["rgba_sequence_dir"] = str(rgba_dir)
        updated["alpha_sequence_dir"] = str(alpha_dir)
        updated["preview_sequence_dir"] = str(preview_dir)
        updated["sequence_pattern"] = "frame_{:06d}.png"
        return {"matte": updated}
