"""外部モデルや副作用を扱う Haystack Component。"""

from __future__ import annotations

import datetime
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from haystack import component
from PIL import Image

from .common import build_mask_set, compose_alpha, dilate_binary_mask, ensure_rgb_array, mask_to_bbox


CPU_FALLBACK_ENV_VAR = "MATTING_ANYTHING_ALLOW_CPU"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def default_device() -> str:
    """CUDA が使える場合は CUDA、なければ CPU を返す。"""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def is_cuda_device(device: str | None) -> bool:
    """device string が CUDA を指すか判定する。"""
    if not str(device or "").startswith("cuda"):
        return False
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def cpu_fallback_allowed() -> bool:
    """CPU 緊急回避モードが明示的に許可されているかを返す。"""
    return os.environ.get(CPU_FALLBACK_ENV_VAR, "").strip().lower() in _TRUTHY_ENV_VALUES


def require_gpu_for_heavy_inference(component_name: str, device: str | None) -> None:
    """SAM2 / GroundingDINO の重い推論で、暗黙の CPU 実行を止める。"""
    if is_cuda_device(device):
        return
    if cpu_fallback_allowed():
        return
    runtime = build_runtime_diagnostics(str(device or ""))
    raise RuntimeError(
        f"{component_name} requires a CUDA GPU for production inference. "
        "CPU execution is reserved for emergency fallback only and is disabled by default. "
        f"Enable a GPU runtime or set {CPU_FALLBACK_ENV_VAR}=1 only when intentionally accepting very slow CPU inference. "
        f"selected_device={runtime.get('selected_device')} cuda_available={runtime.get('cuda_available')} "
        f"torch_available={runtime.get('torch_available')} torch_cuda_version={runtime.get('torch_cuda_version')}"
    )


def build_runtime_diagnostics(selected_device: str) -> dict[str, Any]:
    """実行中プロセスの torch / CUDA 状態を軽量に収集する。"""
    diagnostics: dict[str, Any] = {
        "process_id": os.getpid(),
        "selected_device": selected_device,
        "gpu_required": True,
        "cpu_fallback_allowed": cpu_fallback_allowed(),
        "torch_available": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_device_name": None,
    }
    try:
        import torch
    except ModuleNotFoundError:
        return diagnostics

    diagnostics["torch_available"] = True
    diagnostics["torch_version"] = getattr(torch, "__version__", None)
    diagnostics["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    diagnostics["cuda_available"] = bool(torch.cuda.is_available())
    if diagnostics["cuda_available"]:
        diagnostics["cuda_device_name"] = torch.cuda.get_device_name(0)
    return diagnostics


def build_checkpoint_diagnostics(path: str) -> dict[str, Any]:
    """checkpoint / config path の存在とサイズを診断する。"""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
        checkpoint_path = project_root / checkpoint_path
    exists = checkpoint_path.exists()
    return {
        "path": str(checkpoint_path),
        "exists": exists,
        "size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2) if exists and checkpoint_path.is_file() else None,
        "is_drive_path": "/drive/" in str(checkpoint_path).replace("\\", "/").lower(),
    }


def get_groundingdino_cuda_ops_status() -> dict[str, Any]:
    """GroundingDINO CUDA custom ops の状態を、可能な範囲で副作用少なく確認する。"""
    try:
        sys.path.insert(0, "./GroundingDINO")
        from groundingdino.models.GroundingDINO import ms_deform_attn

        return {
            "available": bool(getattr(ms_deform_attn, "CUDA_OPS_AVAILABLE", False)),
            "source": "groundingdino.models.GroundingDINO.ms_deform_attn",
        }
    except Exception as exc:
        return {"available": None, "source": "import_failed", "error": f"{type(exc).__name__}: {exc}"}


def format_stage_timings(timings: dict[str, float]) -> str:
    """診断表示用に stage timings を短い文字列へ整形する。"""
    return ", ".join(f"{name}={elapsed:.3f}s" for name, elapsed in timings.items())


def patch_transformers_bert_for_groundingdino() -> None:
    """新しい transformers で削除された BERT helper を GroundingDINO 用に補う。"""
    try:
        from transformers.models.bert.modeling_bert import BertModel
    except ModuleNotFoundError:
        return

    if hasattr(BertModel, "get_head_mask"):
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        # GroundingDINO の BertModelWarper は旧 transformers の get_head_mask を前提にする。
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask.to(dtype=self.dtype)

    BertModel.get_head_mask = get_head_mask


@component
class GroundingDINODetector:
    """GroundingDINO でテキストプロンプトから bbox を検出する Component。"""

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.config_path = config_path or os.environ.get(
            "GROUNDING_DINO_CONFIG_PATH",
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        )
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "GROUNDING_DINO_CKPT_PATH",
            "checkpoints/groundingdino_swint_ogc.pth",
        )
        self.device = device or default_device()
        self._model: Any | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "device": self.device,
            "model_cached": self._model is not None,
            "model_id": id(self._model) if self._model is not None else None,
            "runtime": build_runtime_diagnostics(self.device),
            "checkpoint": build_checkpoint_diagnostics(self.checkpoint_path),
            "config": build_checkpoint_diagnostics(self.config_path),
            "cuda_ops": get_groundingdino_cuda_ops_status(),
        }

    def warm_up(self) -> None:
        if self._model is not None:
            return
        require_gpu_for_heavy_inference(self.__class__.__name__, self.device)
        patch_transformers_bert_for_groundingdino()
        sys.path.insert(0, "./GroundingDINO")
        from groundingdino.util.inference import Model

        self._model = Model(
            model_config_path=self.config_path,
            model_checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    @component.output_types(box=np.ndarray, confidence=float, diagnostics=dict)
    def run(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> dict[str, Any]:
        if not text_prompt:
            raise ValueError("テキストプロンプトを入力してください。")
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        cached_before = self._model is not None
        start = time.perf_counter()
        self.warm_up()
        timings["warm_up"] = time.perf_counter() - start
        assert self._model is not None
        import torch
        import torchvision

        start = time.perf_counter()
        image_rgb = ensure_rgb_array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        timings["prepare_image"] = time.perf_counter() - start

        start = time.perf_counter()
        detections, _phrases = self._model.predict_with_caption(
            image=image_bgr,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        timings["predict_with_caption"] = time.perf_counter() - start
        start = time.perf_counter()
        if len(detections.xyxy) > 1:
            keep_indices = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                iou_threshold,
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[keep_indices]
            detections.confidence = detections.confidence[keep_indices]
        timings["nms"] = time.perf_counter() - start
        if len(detections.xyxy) == 0:
            raise ValueError("テキストプロンプトに一致する領域が検出されませんでした。")
        best_index = int(np.argmax(detections.confidence))
        timings["total"] = time.perf_counter() - total_start
        diagnostics = self.diagnostics()
        diagnostics["cached_before"] = cached_before
        diagnostics["timings"] = timings
        return {"box": detections.xyxy[best_index], "confidence": float(detections.confidence[best_index]), "diagnostics": diagnostics}


@component
class GroundingDINOMultiBoxDetector:
    """GroundingDINO で複合対象候補 bbox を複数返す TextToRegion Component。"""

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.config_path = config_path or os.environ.get(
            "GROUNDING_DINO_CONFIG_PATH",
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        )
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "GROUNDING_DINO_CKPT_PATH",
            "checkpoints/groundingdino_swint_ogc.pth",
        )
        self.device = device or default_device()
        self._model: Any | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "device": self.device,
            "model_cached": self._model is not None,
            "model_id": id(self._model) if self._model is not None else None,
            "runtime": build_runtime_diagnostics(self.device),
            "checkpoint": build_checkpoint_diagnostics(self.checkpoint_path),
            "config": build_checkpoint_diagnostics(self.config_path),
            "cuda_ops": get_groundingdino_cuda_ops_status(),
        }

    def warm_up(self) -> None:
        if self._model is not None:
            return
        require_gpu_for_heavy_inference(self.__class__.__name__, self.device)
        patch_transformers_bert_for_groundingdino()
        sys.path.insert(0, "./GroundingDINO")
        from groundingdino.util.inference import Model

        self._model = Model(
            model_config_path=self.config_path,
            model_checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    @component.output_types(boxes=np.ndarray, phrases=list, confidences=np.ndarray, proposals=dict, diagnostics=dict)
    def run(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        top_k: int = 5,
    ) -> dict[str, Any]:
        if not text_prompt:
            raise ValueError("テキストプロンプトを入力してください。")
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        cached_before = self._model is not None
        start = time.perf_counter()
        self.warm_up()
        timings["warm_up"] = time.perf_counter() - start
        assert self._model is not None
        import torch
        import torchvision

        start = time.perf_counter()
        image_rgb = ensure_rgb_array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        timings["prepare_image"] = time.perf_counter() - start

        start = time.perf_counter()
        detections, phrases = self._model.predict_with_caption(
            image=image_bgr,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        timings["predict_with_caption"] = time.perf_counter() - start
        start = time.perf_counter()
        boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        confidences = np.asarray(detections.confidence, dtype=np.float32)
        phrase_list = list(phrases) if phrases is not None else [text_prompt] * len(boxes)
        if len(boxes) == 0:
            raise ValueError("テキストプロンプトに一致する領域が検出されませんでした。")

        # NMS と score 順 top-k で複合対象候補の過剰検出を抑える。
        keep_indices = torchvision.ops.nms(
            torch.from_numpy(boxes),
            torch.from_numpy(confidences),
            iou_threshold,
        ).numpy().tolist()
        keep_indices = sorted(keep_indices, key=lambda index: float(confidences[index]), reverse=True)[: int(top_k)]
        boxes = boxes[keep_indices]
        confidences = confidences[keep_indices]
        phrase_list = [phrase_list[index] if index < len(phrase_list) else text_prompt for index in keep_indices]
        timings["nms_topk"] = time.perf_counter() - start
        proposals = {
            "boxes": boxes,
            "phrases": phrase_list,
            "confidences": confidences,
            "source": "groundingdino",
            "metadata": {"text_prompt": text_prompt, "top_k": int(top_k)},
        }
        timings["total"] = time.perf_counter() - total_start
        diagnostics = self.diagnostics()
        diagnostics["cached_before"] = cached_before
        diagnostics["timings"] = timings
        proposals["metadata"]["diagnostics"] = diagnostics
        return {"boxes": boxes, "phrases": phrase_list, "confidences": confidences, "proposals": proposals, "diagnostics": diagnostics}


@component
class MAMAlphaPredictor:
    """MAM で alpha matte を推論する Component。"""

    def __init__(self, checkpoint_path: str = "checkpoints/mam_vitb.pth", device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or default_device()
        self._model: Any | None = None
        self._transform: Any | None = None

    def warm_up(self) -> None:
        if self._model is not None:
            return
        import torch
        import networks
        import utils

        sys.path.insert(0, "./segment-anything")
        from segment_anything.utils.transforms import ResizeLongestSide

        self._transform = ResizeLongestSide(1024)
        model = networks.get_generator_m2m(seg="sam_vit_b", m2m="sam_decoder_deep")
        model.to(str(self.device))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.m2m.load_state_dict(utils.remove_prefix_state_dict(checkpoint["state_dict"]), strict=True)
        self._model = model.eval()

    @component.output_types(alpha=np.ndarray, alpha_rgb=np.ndarray)
    def run(
        self,
        image: np.ndarray,
        task_type: str,
        box: np.ndarray | None = None,
        points: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        scribble_mode: str = "split",
        guidance_mode: str = "alpha",
    ) -> dict[str, Any]:
        import torch
        import utils
        from torch.nn import functional as F

        self.warm_up()
        assert self._model is not None
        assert self._transform is not None
        image_rgb = ensure_rgb_array(image)
        original_size = image_rgb.shape[:2]
        transformed_image = self._transform.apply_image(image_rgb)
        image_tensor = torch.as_tensor(transformed_image).to(str(self.device))
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()

        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(str(self.device))
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(str(self.device))
        image_tensor = (image_tensor - pixel_mean) / pixel_std
        pad_shape = image_tensor.shape[-2:]
        image_tensor = F.pad(image_tensor, (0, 1024 - pad_shape[1], 0, 1024 - pad_shape[0]))

        if task_type in ("text", "scribble_box"):
            if box is None:
                raise ValueError("box プロンプトが必要です。")
            box_array = np.asarray(box, dtype=np.float32).reshape(1, 4)
            transformed_box = self._transform.apply_boxes(box_array, original_size)
            box_tensor = torch.as_tensor(transformed_box, dtype=torch.float).to(str(self.device)).unsqueeze(0)
            sample = {"image": image_tensor.unsqueeze(0), "bbox": box_tensor, "ori_shape": original_size, "pad_shape": pad_shape}
        elif task_type == "scribble_point":
            if points is None or len(points) == 0:
                raise ValueError("point プロンプトが必要です。")
            transformed_points = self._transform.apply_coords(np.asarray(points), original_size)
            point_tensor = torch.from_numpy(transformed_points).to(str(self.device)).unsqueeze(0)
            label_tensor = torch.from_numpy(np.asarray(point_labels if point_labels is not None else [1] * len(points))).unsqueeze(0).to(str(self.device))
            if scribble_mode == "split":
                point_tensor = point_tensor.permute(1, 0, 2)
                label_tensor = label_tensor.permute(1, 0)
            sample = {"image": image_tensor.unsqueeze(0), "point": point_tensor, "label": label_tensor, "ori_shape": original_size, "pad_shape": pad_shape}
        else:
            raise ValueError(f"無効な task_type: {task_type}")

        with torch.no_grad():
            _features, prediction, post_mask = self._model.forward_inference(sample)
            alpha_os1 = prediction["alpha_os1"][..., : sample["pad_shape"][0], : sample["pad_shape"][1]]
            alpha_os4 = prediction["alpha_os4"][..., : sample["pad_shape"][0], : sample["pad_shape"][1]]
            alpha_os8 = prediction["alpha_os8"][..., : sample["pad_shape"][0], : sample["pad_shape"][1]]
            alpha_os1 = F.interpolate(alpha_os1, sample["ori_shape"], mode="bilinear", align_corners=False)
            alpha_os4 = F.interpolate(alpha_os4, sample["ori_shape"], mode="bilinear", align_corners=False)
            alpha_os8 = F.interpolate(alpha_os8, sample["ori_shape"], mode="bilinear", align_corners=False)
            if guidance_mode == "mask":
                weight_os8 = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False)
                post_mask[weight_os8 > 0] = alpha_os8[weight_os8 > 0]
                alpha = post_mask.clone().detach()
            else:
                weight_os8 = utils.get_unknown_box_from_mask(post_mask)
                alpha_os8[weight_os8 > 0] = post_mask[weight_os8 > 0]
                alpha = alpha_os8.clone().detach()
            weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha, rand_width=20, train_mode=False)
            alpha[weight_os4 > 0] = alpha_os4[weight_os4 > 0]
            weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha, rand_width=10, train_mode=False)
            alpha[weight_os1 > 0] = alpha_os1[weight_os1 > 0]
            alpha_array = alpha[0][0].cpu().numpy()

        alpha_rgb = cv2.cvtColor(np.uint8(alpha_array * 255), cv2.COLOR_GRAY2RGB)
        return {"alpha": alpha_array, "alpha_rgb": alpha_rgb}


@component
class SAM2Segmenter:
    """SAM2 で point/box から候補マスクを生成する Component。"""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_name: str | None = None,
        device: str | None = None,
    ) -> None:
        project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "SAM2_CKPT_PATH",
            str(project_root / "checkpoints" / "SAM2" / "sam2.1_hiera_large.pt"),
        )
        self.config_name = config_name or os.environ.get("SAM2_CONFIG_NAME", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.device = device or default_device()
        self._predictor: Any | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "device": self.device,
            "predictor_cached": self._predictor is not None,
            "predictor_id": id(self._predictor) if self._predictor is not None else None,
            "runtime": build_runtime_diagnostics(self.device),
            "checkpoint": build_checkpoint_diagnostics(self.checkpoint_path),
            "config_name": self.config_name,
        }

    def warm_up(self) -> None:
        if self._predictor is not None:
            return
        require_gpu_for_heavy_inference(self.__class__.__name__, self.device)
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(self.config_name, self.checkpoint_path, device=str(self.device))
        sam2_model.eval()
        self._predictor = SAM2ImagePredictor(sam2_model)

    @component.output_types(masks=np.ndarray, scores=np.ndarray, mask_set=dict, diagnostics=dict)
    def run(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]] | None = None,
        labels: list[int] | None = None,
        box: list[int] | None = None,
        boxes: np.ndarray | None = None,
        multimask: bool = True,
    ) -> dict[str, Any]:
        import torch

        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        cached_before = self._predictor is not None
        start = time.perf_counter()
        self.warm_up()
        timings["warm_up"] = time.perf_counter() - start
        assert self._predictor is not None

        start = time.perf_counter()
        image_rgb = ensure_rgb_array(image)
        timings["prepare_image"] = time.perf_counter() - start
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if is_cuda_device(self.device) else nullcontext()
        with torch.inference_mode(), autocast_context:
            start = time.perf_counter()
            self._predictor.set_image(image_rgb)
            timings["set_image"] = time.perf_counter() - start
            predict_kwargs: dict[str, Any] = {"multimask_output": multimask}
            if points:
                predict_kwargs["point_coords"] = np.array(points, dtype=np.float32)
                predict_kwargs["point_labels"] = np.array(labels, dtype=np.int32)
            # UI の単一 bbox と TextToRegion の複数 bbox のどちらも SAM2 prompt として扱う。
            if box is not None:
                predict_kwargs["box"] = np.array(box, dtype=np.float32)
            elif boxes is not None and len(boxes) > 0:
                predict_kwargs["box"] = np.asarray(boxes, dtype=np.float32)
            start = time.perf_counter()
            masks, scores, _logits = self._predictor.predict(**predict_kwargs)
            timings["predict"] = time.perf_counter() - start
        start = time.perf_counter()
        labels_out = [f"sam2_{index}" for index in range(len(masks))]
        mask_boxes = None
        if box is not None:
            mask_boxes = np.repeat(np.asarray(box, dtype=np.float32).reshape(1, 4), len(masks), axis=0)
        mask_set = build_mask_set(
            masks,
            scores=scores,
            boxes=mask_boxes,
            labels=labels_out,
            source="sam2",
            metadata={"points": points or [], "box": box, "multimask": bool(multimask)},
        )
        timings["build_mask_set"] = time.perf_counter() - start
        timings["total"] = time.perf_counter() - total_start
        diagnostics = self.diagnostics()
        diagnostics["cached_before"] = cached_before
        diagnostics["autocast"] = "cuda_bfloat16" if is_cuda_device(self.device) else "disabled"
        diagnostics["image_shape"] = tuple(int(value) for value in image_rgb.shape)
        diagnostics["timings"] = timings
        mask_set["metadata"]["diagnostics"] = diagnostics
        return {"masks": masks, "scores": scores, "mask_set": mask_set, "diagnostics": diagnostics}


@component
class TransparentBGExtractor:
    """transparent-background で RGBA / alpha / preview を生成する Component。"""

    def __init__(self, project_root: str | None = None, device: str | None = None) -> None:
        self.project_root = Path(project_root or os.environ.get("PROJECT_ROOT", Path.cwd()))
        self.device = device or default_device()
        self._cache: dict[tuple[str, bool, str], Any] = {}

    def _get_remover(self, mode: str, jit: bool) -> Any:
        from transparent_background import Remover

        checkpoint_dir = self.project_root / "checkpoints" / "transparent_BG"
        checkpoint_by_mode = {
            "base": checkpoint_dir / "ckpt_base.pth",
            "fast": checkpoint_dir / "ckpt_fast.pth",
            "base-nightly": checkpoint_dir / "ckpt_base_nightly.pth",
        }
        checkpoint_path = checkpoint_by_mode.get(mode)
        checkpoint_key = str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else ""
        cache_key = (mode, jit, checkpoint_key)
        if cache_key not in self._cache:
            remover_kwargs: dict[str, Any] = {"mode": mode, "jit": jit, "device": self.device}
            if checkpoint_key:
                remover_kwargs["ckpt"] = checkpoint_key
            self._cache[cache_key] = Remover(**remover_kwargs)
        return self._cache[cache_key]

    @component.output_types(rgba=np.ndarray, alpha=np.ndarray, preview=np.ndarray, matte_result=dict)
    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        selected_mask: dict[str, Any] | None = None,
        tb_mode: str = "base",
        tb_jit: bool = False,
        tb_threshold: float = 0.0,
        tb_output_type: str = "rgba",
        crop_padding: int = 40,
        apply_mask_guard: bool = True,
        mask_guard_dilate: int = 21,
    ) -> dict[str, np.ndarray]:
        image_rgb = ensure_rgb_array(image)
        image_height, image_width = image_rgb.shape[:2]
        if mask is None and selected_mask is not None:
            mask = selected_mask.get("mask")
        if mask is not None and mask.any():
            bbox = mask_to_bbox(mask, padding=crop_padding, image_shape=image_rgb.shape)
            if bbox is None:
                bbox = (0, 0, image_width, image_height)
            x_min, y_min, x_max, y_max = bbox
            crop_rgb = image_rgb[y_min:y_max, x_min:x_max]
        else:
            x_min, y_min, x_max, y_max = 0, 0, image_width, image_height
            crop_rgb = image_rgb

        remover = self._get_remover(tb_mode, tb_jit)
        rgba_crop = remover.process(
            Image.fromarray(crop_rgb),
            type="rgba",
            threshold=tb_threshold if tb_threshold > 0 else None,
        )
        rgba_crop_array = np.array(rgba_crop)
        alpha_crop = rgba_crop_array[..., 3].astype(np.float32) / 255.0
        rgb_crop = rgba_crop_array[..., :3]

        full_alpha = np.zeros((image_height, image_width), dtype=np.float32)
        full_alpha[y_min:y_max, x_min:x_max] = alpha_crop
        mask_guard_applied = False
        if apply_mask_guard and mask is not None and mask.any():
            # SAM2 mask の外接矩形でクロップした結果、矩形内・mask 形状外の領域に
            # alpha が残ると「横一直線切れ」として現れる。mask を dilate した guard で
            # mask 形状外の alpha を 0 にし、transparent-background のソフト境界は保つ。
            guard = dilate_binary_mask(mask, kernel_size=mask_guard_dilate).astype(np.float32)
            full_alpha = full_alpha * guard
            mask_guard_applied = True
        full_rgb = image_rgb.copy()
        full_rgb[y_min:y_max, x_min:x_max] = rgb_crop
        alpha_u8 = np.clip(full_alpha * 255, 0, 255).astype(np.uint8)
        rgba = np.dstack([full_rgb, alpha_u8])

        if tb_output_type == "green":
            preview = compose_alpha(full_rgb, full_alpha, (0, 255, 0))
        elif tb_output_type == "white":
            preview = compose_alpha(full_rgb, full_alpha, (255, 255, 255))
        elif tb_output_type == "blur":
            preview = compose_alpha(full_rgb, full_alpha, cv2.GaussianBlur(image_rgb, (51, 51), 0))
        else:
            preview = rgba
        matte_result = {
            "rgba": rgba,
            "alpha": alpha_u8,
            "preview": preview,
            "metadata": {
                "source": "transparent-background",
                "tb_mode": tb_mode,
                "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
                "mask_used": mask is not None,
                "mask_guard_applied": mask_guard_applied,
            },
        }
        return {"rgba": rgba, "alpha": alpha_u8, "preview": preview, "matte_result": matte_result}


@component
class SAM2GuardFilter:
    """SAM2 マスク外の alpha を削る Component。"""

    @component.output_types(alpha=np.ndarray)
    def run(self, alpha: np.ndarray, mask: np.ndarray | None = None, enabled: bool = True, dilate_kernel: int = 21) -> dict[str, np.ndarray]:
        if not enabled or mask is None:
            return {"alpha": alpha}
        alpha_float = alpha.astype(np.float32)
        if alpha_float.max() > 1.0:
            alpha_float = alpha_float / 255.0
        guard = dilate_binary_mask(mask, kernel_size=dilate_kernel).astype(np.float32)
        return {"alpha": np.clip(alpha_float * guard * 255, 0, 255).astype(np.uint8)}


@component
class ColorDecontaminator:
    """pymatting で前景色の色汚染を軽減する Component。"""

    @component.output_types(rgb=np.ndarray)
    def run(self, image: np.ndarray, alpha: np.ndarray, enabled: bool = True) -> dict[str, np.ndarray]:
        image_rgb = ensure_rgb_array(image)
        if not enabled:
            return {"rgb": image_rgb}
        from pymatting import estimate_foreground_ml

        alpha_float = alpha.astype(np.float64)
        if alpha_float.max() > 1.0:
            alpha_float = alpha_float / 255.0
        foreground = estimate_foreground_ml(image_rgb.astype(np.float64) / 255.0, alpha_float)
        return {"rgb": np.clip(foreground * 255.0, 0, 255).astype(np.uint8)}


@component
class BackgroundGenerator:
    """alpha matte から背景合成画像を生成する Component。"""

    def __init__(self, backgrounds_dir: str = "assets/backgrounds", device: str | None = None) -> None:
        self.backgrounds_dir = Path(backgrounds_dir)
        self.device = device or default_device()
        self._generator: Any | None = None

    def _load_generator(self) -> Any:
        import torch
        from diffusers import StableDiffusionPipeline

        if self._generator is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)
            self._generator.to(self.device)
        return self._generator

    @component.output_types(composite=np.ndarray, green_screen=np.ndarray)
    def run(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        background_type: str = "real_world_sample",
        background_prompt: str = "",
    ) -> dict[str, np.ndarray]:
        image_rgb = ensure_rgb_array(image)
        if background_type == "real_world_sample":
            background_files = [path for path in self.backgrounds_dir.iterdir() if path.is_file()]
            if not background_files:
                raise ValueError("assets/backgrounds に背景画像がありません。")
            background = cv2.cvtColor(cv2.imread(str(random.choice(background_files))), cv2.COLOR_BGR2RGB)
        else:
            if not background_prompt:
                raise ValueError("背景プロンプトを入力してください。")
            background = np.array(self._load_generator()(background_prompt).images[0])
        return {
            "composite": compose_alpha(image_rgb, alpha, background),
            "green_screen": compose_alpha(image_rgb, alpha, (51, 255, 146)),
        }


@component
class OutputSaver:
    """推論結果を outputs/<timestamp>/ に保存する Component。"""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = Path(output_dir)

    @component.output_types(paths=dict)
    def run(
        self,
        rgba: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
        preview: np.ndarray | None = None,
        enabled: bool = True,
    ) -> dict[str, dict[str, str]]:
        if not enabled:
            return {"paths": {}}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = self.output_dir / timestamp
        save_root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        if rgba is not None:
            Image.fromarray(rgba).save(save_root / "rgba.png")
            paths["rgba"] = str(save_root / "rgba.png")
        if alpha is not None:
            Image.fromarray(alpha).save(save_root / "alpha.png")
            paths["alpha"] = str(save_root / "alpha.png")
        if preview is not None:
            Image.fromarray(preview).save(save_root / "preview.png")
            paths["preview"] = str(save_root / "preview.png")
        return {"paths": paths}