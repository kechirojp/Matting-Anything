"""動画 I/O と SAM2 video predictor を扱う Haystack Component。"""

from __future__ import annotations

import contextlib
import datetime
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from haystack import component

from .common import (
    assign_points_to_boxes,
    compose_alpha,
    ensure_rgb_array,
    render_tracking_overlay_frame,
    stable_sigmoid,
)
from .model_components import TransparentBGExtractor, default_device, require_gpu_for_heavy_inference
from .video_common import (
    build_frame_mask_sequence,
    build_video_source,
    composite_alpha_by_ownership,
    frame_cache_bytes,
    normalize_output_mode,
    normalize_rgba_frame,
    sample_frame_indices,
    write_png_frame,
)


ProgressCallback = Callable[[str, float, str], None]

# SSE keep-alive 間隔（秒）。Colab / gradio.live の共有トンネルは無通信が続くと
# event SSE 接続を idle 切断し、ブラウザに "Connection errored out" を表示する一方で
# サーバ側 Python は処理を継続する（ERR048）。frame 数ではなく実時間で進捗を流して接続を保つ。
_PROGRESS_KEEPALIVE_SEC = 2.0


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


class _ProgressKeepAlive:
    """長時間ループ中に SSE 接続が idle 切断されないよう、一定間隔で進捗を流す throttle。

    frame 数ベースの間引き（例: 10 frame ごと）だと、低速 GPU で 1 frame に数秒かかる場合に
    通知間隔が数十秒に広がり、Colab / gradio.live の共有トンネルが event SSE を idle で閉じる
    （ブラウザに "Connection errored out" を表示する一方でサーバ処理は継続。ERR048）。
    本 throttle は frame 速度によらず、最初/最後の frame に加えて最低 ``min_interval_sec``
    間隔で進捗を送り、無通信ギャップを上限内に抑えて接続を維持する。

    Args:
        progress_callback: Gradio へ進捗を渡すコールバック（None なら no-op）。
        stage: 進捗 stage 名。
        min_interval_sec: 進捗を流す最小間隔（秒）。
        clock: 単調増加時刻を返す関数。テスト用に注入可能。
    """

    def __init__(
        self,
        progress_callback: ProgressCallback | None,
        stage: str,
        *,
        min_interval_sec: float = _PROGRESS_KEEPALIVE_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._progress_callback = progress_callback
        self._stage = stage
        self._min_interval = float(min_interval_sec)
        self._clock = clock
        self._last_emit = clock()

    def maybe(
        self,
        index: int,
        total: int,
        fraction: float,
        description: str,
        *,
        force: bool = False,
    ) -> None:
        """境界 frame・経過時間・force のいずれかを満たす時のみ進捗を流す。

        Args:
            index: 0 始まりの現在 frame index。
            total: 総 frame 数。
            fraction: 0.0〜1.0 の進捗割合。
            description: 進捗説明文。
            force: True なら間隔に関わらず必ず流す。
        """
        now = self._clock()
        is_boundary = index <= 0 or index + 1 >= total
        if force or is_boundary or (now - self._last_emit) >= self._min_interval:
            _notify_progress(self._progress_callback, self._stage, fraction, description)
            self._last_emit = now


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


def _require_imageio() -> Any:
    """RGBA(透過)動画書き出し用に imageio(v2)+ffmpeg を取得する。

    cv2.VideoWriter は 4ch(RGBA) frame を書けず全 frame を skip する（FFmpeg
    "expected 3 channels but got 4"）。透過動画は imageio+ffmpeg で書き出す必要がある。
    依存が無ければ握り潰さず、連番(PNG)出力を促す明確なエラーにする（ERR047）。

    Returns:
        imageio.v2 モジュール（``get_writer`` を持つ）。

    Raises:
        RuntimeError: imageio または imageio-ffmpeg(ffmpeg 実体) が利用できない場合。
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "RGBA(透過)動画の書き出しには imageio[ffmpeg] が必要です。"
            "`pip install imageio[ffmpeg]` を実行するか、出力モードを連番(PNG)にしてください。"
        ) from exc
    try:
        import imageio_ffmpeg  # noqa: F401  # ffmpeg 実体の同梱を保証する
    except ImportError as exc:
        raise RuntimeError(
            "RGBA(透過)動画の書き出しに必要な ffmpeg(imageio-ffmpeg) が見つかりません。"
            "`pip install imageio[ffmpeg]` を実行するか、出力モードを連番(PNG)にしてください。"
        ) from exc
    return imageio


class _RgbaCodecSpec(NamedTuple):
    """RGBA(透過)動画を imageio+ffmpeg で書き出すための codec パラメータ。"""

    label: str
    suffix: str
    codec: str
    pixelformat: str
    output_params: tuple[str, ...]
    macro_block_size: int


class _ImageioAlphaVideoWriter:
    """imageio+ffmpeg で 1 frame ずつ alpha 付き動画を書き出す軽量 writer。

    cv2.VideoWriter と異なり 4ch(RGBA, RGB order) frame をそのまま受け取り、
    webm(VP9/yuva420p) や mov(PNG/rgba) として透過を保持して書き出す。
    """

    def __init__(self, path: Path, first_frame: np.ndarray, fps: float, spec: _RgbaCodecSpec) -> None:
        imageio = _require_imageio()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        # first_frame は形状参照用（cv2 版と同じ契約）。実書き込みは write() で行う。
        np.asarray(first_frame)
        self._writer = imageio.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=float(fps),
            codec=spec.codec,
            pixelformat=spec.pixelformat,
            output_params=list(spec.output_params),
            macro_block_size=spec.macro_block_size,
        )

    def write(self, frame: np.ndarray) -> None:
        # imageio は RGB order を期待する。normalize_rgba_frame 済みの RGBA をそのまま渡す。
        frame_array = normalize_rgba_frame(frame)
        self._writer.append_data(frame_array)

    def close(self) -> None:
        self._writer.close()


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


def _samurai_config_root(config_name: str | None) -> Path | None:
    """SAMURAI config 利用時、configs/ を含むローカル sam2 package root を返す。

    Colab 等で facebook 版 sam2 が入っていると ``configs/samurai/...`` が
    sam2 の Hydra 検索パスに存在せず ``MissingConfigException`` になる。
    workspace 同梱の samurai fork（``samurai/sam2/sam2``）には configs/samurai が
    あるため、その package root を返して検索パスへ追加できるようにする。
    非 samurai config では None を返し検索パスを変更しない（samurai/ は変更しない）。
    """
    if not config_name or "samurai" not in str(config_name).lower():
        return None
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    sam2_root = project_root / "samurai" / "sam2" / "sam2"
    if (sam2_root / "configs" / "samurai").is_dir():
        return sam2_root
    return None


def _require_samurai_capable_sam2(config_name: str | None) -> None:
    """SAMURAI config 利用時、import される sam2 が SAMURAI 対応 fork かを事前検証する。

    Colab 等で facebook 版 sam2 が入っていると ``configs/samurai/`` も ``samurai_mode``
    対応モデルコードも無く、Hydra の ``MissingConfigException`` や ``SAM2Base`` の
    ``TypeError`` という分かりにくい例外になる（ERR038 の検索パス追加だけでは解決不能）。
    事前に installed sam2 を検査し、SAMURAI fork（``pip install -e samurai/sam2``）導入を
    促す明確なエラーにする。非 samurai config では no-op。

    Args:
        config_name: build_sam2_video_predictor に渡す Hydra config 名。

    Raises:
        RuntimeError: samurai config なのに installed sam2 が SAMURAI fork でない場合。
    """
    if not config_name or "samurai" not in str(config_name).lower():
        return
    import sam2

    installed_root = Path(sam2.__file__).resolve().parent
    if (installed_root / "configs" / "samurai").is_dir():
        return
    raise RuntimeError(
        "SAMURAI tracker を選択しましたが、現在 import される sam2 パッケージは "
        f"SAMURAI 対応 fork ではありません（installed: {installed_root}）。 "
        "SAMURAI は config だけでなく samurai_mode 対応のモデルコードを必要とします。 "
        "同梱の SAMURAI fork を導入してください: `pip install -e samurai/sam2` "
        "（Colab はランタイム再起動後に install セルから再実行）。 "
        "標準 SAM2 で良ければ tracker を SAM2.1（standard）に切り替えてください。"
    )


def _ensure_samurai_config_searchpath(config_name: str | None) -> None:
    """SAMURAI config をローカル sam2 package root から解決できるよう Hydra 検索パスに追加する。

    既に登録済み、または samurai config でない場合は no-op。Hydra 未初期化時は sam2 を
    import して初期化させる。解決できないときはエラーを握り潰さず、後続の build が出す
    例外（MissingConfigException など）をそのまま伝搬させる。
    """
    sam2_root = _samurai_config_root(config_name)
    if sam2_root is None:
        return
    from hydra.core.global_hydra import GlobalHydra

    global_hydra = GlobalHydra.instance()
    if not global_hydra.is_initialized():
        import sam2  # noqa: F401  # import 時に initialize_config_module("sam2") が走る

        global_hydra = GlobalHydra.instance()
    if not global_hydra.is_initialized():
        return
    search_path = global_hydra.config_loader().get_search_path()
    provider_uri = sam2_root.as_uri()
    already_registered = any(
        getattr(entry, "path", None) == provider_uri for entry in search_path.get_path()
    )
    if not already_registered:
        search_path.append(provider="samurai-local", path=provider_uri)


@component
class SAM2VideoPropagator:
    """SAM2 video predictor で first-frame prompt から全 frame の mask を伝搬する Component。"""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_name: str | None = None,
        device: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        autocast_dtype: str | None = "float16",
        single_object_only: bool = False,
    ) -> None:
        project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "SAM2_CKPT_PATH",
            str(project_root / "checkpoints" / "SAM2" / "sam2.1_hiera_large.pt"),
        )
        self.config_name = config_name or os.environ.get("SAM2_CONFIG_NAME", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.device = device or default_device()
        # SAM2 video state の CPU offload。SAMURAI 等で GPU 常駐メモリが大きい場合に
        # 伝搬の最初の重い frame で VRAM 枯渇 stall になるのを防ぐ（ERR049）。registry
        # (config/inference_models.toml) の tracker entry から渡し、既定は現状維持の False。
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)
        # 伝搬を mixed precision (autocast) で回し VRAM を抑え高速化する（ERR050）。SAMURAI 本家
        # (scripts/main_inference.py) も torch.autocast("cuda", float16) を使う。device==cuda のときのみ
        # 適用し、"none"/None/"" で無効化できる。registry の tracker entry から上書き可。
        self.autocast_dtype = autocast_dtype
        # SAMURAI は Kalman filter による単一オブジェクト追跡専用で、KF 状態を予測器インスタンスで
        # 共有するため複数 obj を同時伝搬できない。fork の `_forward_sam_heads` も B=1 前提
        # (`ious[0][best_iou_inds]`) で、複数 obj 時に 'Boolean value of Tensor with more than one
        # value is ambiguous' で落ちる（ERR051）。samurai/ は変更しないため、registry の
        # tracker entry から渡し、複数 obj を伝搬前に actionable に弾く。既定は後方互換で False。
        self.single_object_only = bool(single_object_only)
        self._video_predictor: Any | None = None

    def _autocast_context(self, torch_module: Any):
        """device==cuda かつ autocast_dtype が有効なとき torch.autocast を返す（ERR050）。

        CPU や無効指定時は nullcontext で既存挙動を維持する。
        """
        dtype_name = self.autocast_dtype
        if self.device != "cuda" or dtype_name in (None, "", "none"):
            return contextlib.nullcontext()
        dtype = {
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
        }.get(str(dtype_name), torch_module.float16)
        return torch_module.autocast("cuda", dtype=dtype)

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
        _require_samurai_capable_sam2(self.config_name)
        _ensure_samurai_config_searchpath(self.config_name)
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
        # ERR051: single_object_only な tracker(SAMURAI) は複数 obj を同時追跡できない。warm_up
        # （モデル build）前に fail-fast し、原因と回避策を明示する。
        requested_object_count = len(boxes) if boxes else 1
        if self.single_object_only and requested_object_count > 1:
            raise ValueError(
                f"選択中の tracker は単一オブジェクト専用です（SAMURAI は Kalman filter による単一対象"
                f"追跡のみ対応）。指定オブジェクト数={requested_object_count}。box / オブジェクトを 1 つに"
                f"減らすか、複数対象を扱う場合は標準 SAM2 tracker に切り替えてください。"
            )
        _notify_progress(progress_callback, "sam2_video", 0.0, "SAM2 video predictor を初期化しています")
        self.warm_up()
        _notify_progress(progress_callback, "sam2_video", 0.08, "SAM2 用の一時 frame を準備しています")
        assert self._video_predictor is not None
        import torch

        # 複数 box は obj_id 1..N、単一 prompt は object_id を追跡対象とする。
        # 修正1(方針1): box 群と point 群を併用する場合、各 point を最近傍 box の
        # object prompt に同梱する（point 群を別 obj にまとめると SAM2 が複数インスタンスを
        # 1 mask で表現できず point が落ちるため）。
        points_by_obj: dict[int, list[int]] = {}
        if boxes:
            target_object_ids = list(range(1, len(boxes) + 1))
            points_by_obj = assign_points_to_boxes(points, boxes)
        else:
            target_object_ids = [int(object_id)]
        directions = [False, True] if bidirectional else [False]

        # Collect per-object logits per-frame, keyed by object_id for pass-merge safety:
        # {frame_index: {obj_id: np.ndarray (H,W)}}
        per_object_logits_by_id: dict[int, dict[int, np.ndarray]] = {}
        source_indices = list((metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(len(frames))))
        total_frames = max(len(frames), 1)
        propagation_total = total_frames * len(directions)
        # frame 数ベースの間引きだと低速 GPU で通知間隔が数十秒に広がり SSE が idle 切断されるため（ERR048）、
        # 時間ベース keep-alive throttle で進捗を流し接続を保つ。
        prep_keepalive = _ProgressKeepAlive(progress_callback, "sam2_video")
        propagate_keepalive = _ProgressKeepAlive(progress_callback, "sam2_video")
        with tempfile.TemporaryDirectory(prefix="sam2_video_frames_") as temp_dir:
            temp_path = Path(temp_dir)
            for frame_index, frame in enumerate(frames):
                frame_rgb = ensure_rgb_array(frame)
                cv2.imwrite(str(temp_path / f"{frame_index:06d}.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                prep_keepalive.maybe(
                    frame_index,
                    total_frames,
                    0.08 + 0.12 * ((frame_index + 1) / total_frames),
                    f"SAM2 用の一時 frame を準備しています ({frame_index + 1}/{total_frames})",
                )
            with torch.inference_mode(), self._autocast_context(torch):
                _notify_progress(progress_callback, "sam2_video", 0.22, "SAM2 の video state を初期化しています")
                state = self._video_predictor.init_state(
                    video_path=str(temp_path),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                    offload_state_to_cpu=self.offload_state_to_cpu,
                )
                if boxes:
                    for obj_id, single_box in zip(range(1, len(boxes) + 1), boxes):
                        add_kwargs: dict[str, Any] = {
                            "inference_state": state,
                            "frame_idx": prompt_frame_idx,
                            "obj_id": obj_id,
                            "box": np.asarray(single_box, dtype=np.float32),
                        }
                        # 修正1: この box に割り当てられた補正 point（positive/negative）を同梱する。
                        assigned_indices = points_by_obj.get(obj_id, [])
                        if assigned_indices and points:
                            assigned_points = [points[i] for i in assigned_indices]
                            assigned_labels = [(labels or [1] * len(points))[i] for i in assigned_indices]
                            add_kwargs["points"] = np.asarray(assigned_points, dtype=np.float32)
                            add_kwargs["labels"] = np.asarray(assigned_labels, dtype=np.int32)
                        self._video_predictor.add_new_points_or_box(**add_kwargs)
                else:
                    add_kwargs = {"inference_state": state, "frame_idx": prompt_frame_idx, "obj_id": int(object_id)}
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
                        # 修正2(根治): 各 obj を二値化せず sigmoid 確率で union（max）し、
                        # 継ぎ目を黒線にせず soft 確率 mask として保持する。
                        # 修正3: forward/reverse の 2 pass を object_id をキーに整列してマージする。
                        # 位置ベースで stack すると追跡途切れ/再出現で obj 数が pass 間で変わった際に
                        # 別 obj の logit が混入するため、必ず id で対応付ける。
                        source_index = int(source_indices[int(out_frame_idx)]) if int(out_frame_idx) < len(source_indices) else int(out_frame_idx)
                        obj_logits_by_id = per_object_logits_by_id.setdefault(source_index, {})
                        gathered_any = False
                        for target_obj in target_object_ids:
                            if target_obj not in object_ids:
                                continue
                            mask_logits = out_mask_logits[object_ids.index(target_obj)]
                            if hasattr(mask_logits, "detach"):
                                logits_array = mask_logits.detach().cpu().numpy()
                            else:
                                logits_array = np.asarray(mask_logits)
                            logits_2d = np.asarray(logits_array).squeeze()
                            existing = obj_logits_by_id.get(target_obj)
                            if existing is None:
                                obj_logits_by_id[target_obj] = logits_2d
                            else:
                                obj_logits_by_id[target_obj] = np.maximum(existing, logits_2d)
                            gathered_any = True
                        if not gathered_any:
                            continue
                        propagate_keepalive.maybe(
                            propagated_count - 1,
                            propagation_total,
                            0.25 + 0.75 * (propagated_count / propagation_total),
                            f"SAM2 mask を動画全体へ伝搬しています ({propagated_count}/{propagation_total})",
                        )
        # id ベースで集めた logit を target_object_ids 順に (N,H,W) へ整列する。
        # ある pass で欠損した obj は -1e6（≒確率0）で埋め、チャネル位置と obj_id を固定対応させる。
        per_object_logits: dict[int, np.ndarray] = {}
        for frame_index, obj_logits_by_id in per_object_logits_by_id.items():
            reference = next(iter(obj_logits_by_id.values()))
            ordered: list[np.ndarray] = []
            for target_obj in target_object_ids:
                logits_2d = obj_logits_by_id.get(target_obj)
                if logits_2d is None:
                    logits_2d = np.full(reference.shape, -1e6, dtype=np.float32)
                ordered.append(np.asarray(logits_2d, dtype=np.float32))
            per_object_logits[int(frame_index)] = np.stack(ordered, axis=0)
        # overlay / 後方互換用に per-object logit から union soft mask を派生する。
        # 各 object を sigmoid 確率化し画素ごと max を取って (H,W) soft mask とする。
        union_frame_masks: dict[int, np.ndarray] = {}
        for frame_index, stacked in per_object_logits.items():
            probs = stable_sigmoid(np.asarray(stacked, dtype=np.float32))
            union_frame_masks[int(frame_index)] = np.max(probs, axis=0).astype(np.float32)
        masks = build_frame_mask_sequence(
            union_frame_masks,
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
        # Phase1 契約: OwnershipResolver 用に per-object logit を埋め込んで下流へ渡す。
        masks["per_object_logits"] = per_object_logits
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

    def _run_per_object_frame(
        self,
        frame: np.ndarray,
        logits: np.ndarray,
        ownership: np.ndarray,
        *,
        tb_mode: str,
        tb_jit: bool,
        tb_threshold: float,
        tb_output_type: str,
        crop_padding: int,
        mask_guard_feather: int,
        mask_guard_dilate: int = 21,
    ) -> dict[str, np.ndarray]:
        """per_object モード: 対象ごとに crop tb を実行し、所有権でアルファ合成する（Phase2 ④⑤）。

        各対象の logit を sigmoid した soft mask で `TransparentBGExtractor` を呼び、
        bbox 導出・crop・tb・full frame 配置・soft guard を再利用して対象ごとの連続アルファを得る。
        得た N 枚のアルファを所有権 (先頭 N チャネル) で重み付け合成し、最終アルファを得る。
        RGB は元フレームのまま（アルファのみ合成する）。

        Args:
            frame: 元フレーム (H,W,3)。
            logits: 対象ごとの SAM2 logit (N,H,W)。
            ownership: 所有権 (N+1,H,W)。先頭 N が前景、最終が背景。
            tb_mode/tb_jit/tb_threshold/crop_padding/mask_guard_feather/mask_guard_dilate: tb 推論パラメータ。
            tb_output_type: preview の背景合成種別（green/white/blur/それ以外は rgba）。

        Returns:
            ``{"rgba": (H,W,4) uint8, "alpha": (H,W) uint8, "preview": (H,W,*) uint8}``。
        """
        image_rgb = ensure_rgb_array(frame)
        logits_array = np.asarray(logits, dtype=np.float32)
        num_objects = logits_array.shape[0]
        per_object_alphas: list[np.ndarray] = []
        for obj_index in range(num_objects):
            soft_mask = stable_sigmoid(logits_array[obj_index])
            result = self.extractor.run(
                image=image_rgb,
                mask=soft_mask,
                tb_mode=tb_mode,
                tb_jit=tb_jit,
                tb_threshold=tb_threshold,
                tb_output_type="rgba",
                crop_padding=int(crop_padding),
                mask_guard_feather=int(mask_guard_feather),
                mask_guard_dilate=int(mask_guard_dilate),
            )
            alpha_o = np.asarray(result["alpha"], dtype=np.float32) / 255.0
            per_object_alphas.append(alpha_o)
        alpha_final = composite_alpha_by_ownership(per_object_alphas, ownership)
        alpha_u8 = np.clip(alpha_final * 255.0, 0, 255).astype(np.uint8)
        rgba = np.dstack([image_rgb, alpha_u8])
        if tb_output_type == "green":
            preview = compose_alpha(image_rgb, alpha_final, (0, 255, 0))
        elif tb_output_type == "white":
            preview = compose_alpha(image_rgb, alpha_final, (255, 255, 255))
        elif tb_output_type == "blur":
            preview = compose_alpha(image_rgb, alpha_final, cv2.GaussianBlur(image_rgb, (51, 51), 0))
        else:
            preview = rgba
        return {"rgba": rgba, "alpha": alpha_u8, "preview": preview}

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
        mask_guard_feather: int = 0,
        mask_guard_dilate: int = 21,
        rgba_codec: str = "webm_vp9",
        video_matte_mode: str = "union",
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """frame ごとに `TransparentBGExtractor` を呼び、出力を RAM に溜めず保存する。

        ``video_matte_mode`` で背景透過の経路を切り替える:
        - ``"union"`` (既定): フレームあたり tb 1 回。union mask の外接矩形で 1 度だけ切り抜く軽量経路。
        - ``"per_object"``: フレームあたり tb N 回。対象ごとに crop tb して所有権でアルファ合成する忠実経路。
        """
        if not frames:
            raise ValueError("frames が空です。")
        normalized_mode = normalize_output_mode(output_mode)
        matte_mode = str(video_matte_mode).strip().lower()
        if matte_mode not in {"union", "per_object"}:
            raise ValueError(f"video_matte_mode は 'union' か 'per_object' のいずれかです: {video_matte_mode!r}")
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
        _notify_progress(progress_callback, "transparent_bg", 0.0, "transparent-background を初期化しています")
        # 低速 frame でも SSE 接続を保つため時間ベース keep-alive throttle を使う（ERR048）。
        tb_keepalive = _ProgressKeepAlive(progress_callback, "transparent_bg")
        try:
            for local_index, frame in enumerate(frames):

                source_index = int(source_indices[local_index]) if local_index < len(source_indices) else local_index
                logits = per_object_logits.get(source_index)
                ownership = ownership_by_frame.get(source_index)
                logits_array = np.asarray(logits) if logits is not None else None
                use_per_object = (
                    matte_mode == "per_object"
                    and logits_array is not None
                    and ownership is not None
                    and logits_array.ndim == 3
                    and logits_array.shape[0] >= 1
                )
                if use_per_object:
                    result = self._run_per_object_frame(
                        frame,
                        logits_array,
                        np.asarray(ownership),
                        tb_mode=tb_mode,
                        tb_jit=tb_jit,
                        tb_threshold=tb_threshold,
                        tb_output_type=tb_output_type,
                        crop_padding=int(crop_padding),
                        mask_guard_feather=int(mask_guard_feather),
                        mask_guard_dilate=int(mask_guard_dilate),
                    )
                else:
                    mask = frame_masks.get(source_index)
                    result = self.extractor.run(
                        image=frame,
                        mask=mask,
                        tb_mode=tb_mode,
                        tb_jit=tb_jit,
                        tb_threshold=tb_threshold,
                        tb_output_type=tb_output_type,
                        crop_padding=int(crop_padding),
                        mask_guard_feather=int(mask_guard_feather),
                        mask_guard_dilate=int(mask_guard_dilate),
                    )
                rgba_frame = normalize_rgba_frame(result["rgba"])
                alpha_frame = np.asarray(result["alpha"]).astype(np.uint8, copy=False)
                preview_frame = ensure_rgb_array(result["preview"])
                if local_index == 0 and normalized_mode in {"video", "both"}:
                    _notify_progress(progress_callback, "transparent_bg", 0.03, "動画 codec を確認しています")
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
                tb_keepalive.maybe(
                    local_index,
                    total_frames,
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
                "video_matte_mode": matte_mode,
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
        overlay_keepalive = _ProgressKeepAlive(progress_callback, "tracking_overlay")
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
                overlay_keepalive.maybe(
                    local_index,
                    total_frames,
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

    def warm_up(self) -> None:
        """Haystack Pipeline の no-arg warm_up 契約に合わせる。"""
        return None

    def _select_rgba_codec(
        self,
        frame_shape: tuple[int, int],
        preferred_rgba_codec: str = "webm_vp9",
    ) -> tuple[_RgbaCodecSpec, list[tuple[str, str]]]:
        """RGBA(透過)動画用の imageio+ffmpeg codec spec を選ぶ。

        cv2.VideoWriter は 4ch(RGBA) frame を書けず全 frame skip するため使わない。
        webm_vp9(libvpx-vp9/yuva420p) を既定、mov_png(png/rgba) を代替とする。
        imageio[ffmpeg] が無い場合は握り潰さず連番出力を促すエラーにする（ERR047）。

        Args:
            frame_shape: (height, width)。spec 選択自体には未使用だが契約を維持する。
            preferred_rgba_codec: "webm_vp9" もしくは "mov_png"。

        Returns:
            (選択した _RgbaCodecSpec, 候補ごとの採否ログ)。

        Raises:
            RuntimeError: imageio[ffmpeg] が利用できない場合。
        """
        # ffmpeg(imageio) 可用性を先に検証（cv2 isOpened のような偽陽性を避ける）。
        _require_imageio()
        webm_spec = _RgbaCodecSpec(
            label="webm_vp9",
            suffix=".webm",
            codec="libvpx-vp9",
            pixelformat="yuva420p",
            # VP9 alpha は auto-alt-ref を無効化する必要がある。yuv420p のため偶数に padding。
            output_params=("-auto-alt-ref", "0"),
            macro_block_size=2,
        )
        mov_spec = _RgbaCodecSpec(
            label="mov_png",
            suffix=".mov",
            codec="png",
            pixelformat="rgba",
            output_params=(),
            macro_block_size=1,
        )
        if preferred_rgba_codec == "mov_png":
            chosen = mov_spec
        else:
            chosen = webm_spec
        fallback = [(chosen.label, "ok (used)")]
        return chosen, fallback

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

    def _write_rgba_video(
        self,
        path: Path,
        frames: list[np.ndarray],
        fps: float,
        spec: _RgbaCodecSpec,
        progress_callback: ProgressCallback | None = None,
        progress_prefix: str = "RGBA 動画を書き出しています",
    ) -> None:
        """RGBA(透過)動画を imageio+ffmpeg で書き出す（cv2 は 4ch 不可）。"""
        if not frames:
            raise ValueError(f"動画に書き出す frame がありません: {path}")
        stream = _ImageioAlphaVideoWriter(path, frames[0], fps, spec)
        try:
            total_frames = max(len(frames), 1)
            for frame_index, frame in enumerate(frames):
                stream.write(frame)
                written_count = frame_index + 1
                if written_count == 1 or written_count % 20 == 0 or written_count == total_frames:
                    _notify_progress(
                        progress_callback,
                        "video_writer",
                        written_count / total_frames,
                        f"{progress_prefix} ({written_count}/{total_frames})",
                    )
        finally:
            stream.close()

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
        spec, fallback = self._select_rgba_codec(rgba_frames[0].shape[:2], preferred_rgba_codec=rgba_codec)
        rgba_path = video_dir / f"rgba{spec.suffix}"
        alpha_path = video_dir / "alpha.mp4"
        preview_path = video_dir / "preview.mp4"
        _notify_progress(progress_callback, "video_writer", 0.10, "RGBA 動画を書き出しています")
        self._write_rgba_video(
            rgba_path,
            rgba_frames,
            matte.get("fps", 30.0),
            spec,
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
        updated.setdefault("metadata", {})["used_rgba_codec"] = spec.label
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
