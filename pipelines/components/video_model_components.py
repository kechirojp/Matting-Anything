"""動画 I/O と SAM2 video predictor を扱う Haystack Component。"""

from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from haystack import component

from .common import ensure_rgb_array
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


def _resolve_output_dir(output_dir: str) -> Path:
    """出力ディレクトリを PROJECT_ROOT 基準の絶対パスへ解決する。"""
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    return project_root / output_path


@component
class VideoReader:
    """動画ファイルを RGB frame list と metadata に分解する Component。"""

    @component.output_types(frames=list, metadata=dict)
    def run(self, video_path: str, max_frames: int = 300, frame_step: int = 1) -> dict[str, Any]:
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
            frames: list[np.ndarray] = []
            source_indices: list[int] = []
            current_index = 0
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if current_index in indices:
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    source_indices.append(current_index)
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
        object_id: int = 1,
    ) -> dict[str, Any]:
        """first-frame prompt を SAM2 video predictor に登録し、frame mask 列を返す。"""
        if not frames:
            raise ValueError("frames が空です。")
        if not points and box is None:
            raise ValueError("SAM2 video prompt が空です。points または box を指定してください。")
        self.warm_up()
        assert self._video_predictor is not None
        import torch

        frame_masks: dict[int, np.ndarray] = {}
        source_indices = list((metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(len(frames))))
        with tempfile.TemporaryDirectory(prefix="sam2_video_frames_") as temp_dir:
            temp_path = Path(temp_dir)
            for frame_index, frame in enumerate(frames):
                frame_rgb = ensure_rgb_array(frame)
                cv2.imwrite(str(temp_path / f"{frame_index:06d}.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            with torch.inference_mode():
                state = self._video_predictor.init_state(video_path=str(temp_path))
                add_kwargs: dict[str, Any] = {"inference_state": state, "frame_idx": 0, "obj_id": int(object_id)}
                if points:
                    add_kwargs["points"] = np.asarray(points, dtype=np.float32)
                    add_kwargs["labels"] = np.asarray(labels or [1] * len(points), dtype=np.int32)
                if box is not None:
                    add_kwargs["box"] = np.asarray(box, dtype=np.float32)
                self._video_predictor.add_new_points_or_box(**add_kwargs)
                for out_frame_idx, out_obj_ids, out_mask_logits in self._video_predictor.propagate_in_video(state):
                    object_ids = [int(value) for value in out_obj_ids]
                    target_index = object_ids.index(int(object_id)) if int(object_id) in object_ids else 0
                    mask_logits = out_mask_logits[target_index]
                    if hasattr(mask_logits, "detach"):
                        mask_array = (mask_logits.detach().cpu().numpy() > 0.0).squeeze()
                    else:
                        mask_array = (np.asarray(mask_logits) > 0.0).squeeze()
                    source_index = int(source_indices[int(out_frame_idx)]) if int(out_frame_idx) < len(source_indices) else int(out_frame_idx)
                    frame_masks[source_index] = mask_array.astype(bool)
        masks = build_frame_mask_sequence(
            frame_masks,
            object_ids=[int(object_id)],
            metadata={"points": points or [], "labels": labels or [], "box": box, "source_metadata": metadata or {}},
        )
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return {"masks": masks}


@component
class TransparentBGVideoExtractor:
    """各 frame に transparent-background を適用し、メモリ上の matte frame 列を作る Component。"""

    def __init__(self, project_root: str | None = None, device: str | None = None) -> None:
        self.extractor = TransparentBGExtractor(project_root=project_root, device=device)

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
    ) -> dict[str, dict[str, Any]]:
        """frame ごとに `TransparentBGExtractor` を呼び、書き出し前の結果をまとめる。"""
        if not frames:
            raise ValueError("frames が空です。")
        normalized_mode = normalize_output_mode(output_mode)
        frame_masks = (masks or {}).get("frame_masks", {})
        source_indices = list((metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(len(frames))))
        rgba_frames: list[np.ndarray] = []
        alpha_frames: list[np.ndarray] = []
        preview_frames: list[np.ndarray] = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            rgba_frames.append(result["rgba"])
            alpha_frames.append(result["alpha"])
            preview_frames.append(result["preview"])
        matte = {
            "rgba_video_path": None,
            "alpha_video_path": None,
            "preview_video_path": None,
            "rgba_sequence_dir": None,
            "alpha_sequence_dir": None,
            "preview_sequence_dir": None,
            "sequence_pattern": None,
            "fps": float((metadata or {}).get("fps", 30.0)),
            "frame_count": len(frames),
            "output_mode": normalized_mode,
            "rgba_frames": rgba_frames,
            "alpha_frames": alpha_frames,
            "preview_frames": preview_frames,
            "metadata": {
                "source": "transparent-background-video",
                "timestamp": timestamp,
                "source_metadata": metadata or {},
                "tb_mode": tb_mode,
                "tb_output_type": tb_output_type,
                "crop_padding": int(crop_padding),
            },
        }
        return {"matte": matte}


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

    def warm_up(self, frame_shape: tuple[int, int], preferred_rgba_codec: str = "webm_vp9") -> tuple[str, str, list[tuple[str, str]]]:
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

    def _write_video(self, path: Path, frames: list[np.ndarray], fps: float, fourcc_name: str, channels: int = 3) -> None:
        if not frames:
            raise ValueError(f"動画に書き出す frame がありません: {path}")
        first = np.asarray(frames[0])
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), float(fps), (width, height), isColor=channels != 1)
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter を開けません: {path}")
        try:
            for frame in frames:
                frame_array = np.asarray(frame).astype(np.uint8, copy=False)
                if frame_array.ndim == 2:
                    writer.write(cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR))
                elif frame_array.shape[2] == 4:
                    writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGRA))
                else:
                    writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    @component.output_types(matte=dict)
    def run(self, matte: dict, rgba_codec: str = "webm_vp9") -> dict[str, dict[str, Any]]:
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
            raise ValueError("RGBA frame が空です。")
        fourcc, suffix, fallback = self.warm_up(rgba_frames[0].shape[:2], preferred_rgba_codec=rgba_codec)
        rgba_path = video_dir / f"rgba{suffix}"
        alpha_path = video_dir / "alpha.mp4"
        preview_path = video_dir / "preview.mp4"
        self._write_video(rgba_path, rgba_frames, matte.get("fps", 30.0), fourcc, channels=4)
        self._write_video(alpha_path, alpha_frames, matte.get("fps", 30.0), "mp4v", channels=1)
        self._write_video(preview_path, preview_frames, matte.get("fps", 30.0), "mp4v", channels=3)
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
    def run(self, matte: dict) -> dict[str, dict[str, Any]]:
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        estimated_bytes = sum(frame.nbytes for frame in rgba_frames + alpha_frames + preview_frames)
        free_bytes = shutil.disk_usage(self.output_dir).free
        if free_bytes < estimated_bytes:
            raise RuntimeError("連番出力に必要な空き容量が不足しています。")
        for index, frame in enumerate(rgba_frames):
            write_png_frame(rgba_dir / f"frame_{index:06d}.png", frame)
        for index, frame in enumerate(alpha_frames):
            write_png_frame(alpha_dir / f"frame_{index:06d}.png", frame)
        for index, frame in enumerate(preview_frames):
            write_png_frame(preview_dir / f"frame_{index:06d}.png", frame)
        updated = dict(matte)
        updated["rgba_sequence_dir"] = str(rgba_dir)
        updated["alpha_sequence_dir"] = str(alpha_dir)
        updated["preview_sequence_dir"] = str(preview_dir)
        updated["sequence_pattern"] = "frame_{:06d}.png"
        return {"matte": updated}
