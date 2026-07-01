"""DEVA方式 DetectionIsland（image-level 検出島）の @component 実装。

DEVA の (a) image-level 検出を本スタックで再構成する:
    GroundingDINO（text→box） → SAM2 画像モード（box→mask）

特徴:
- **検出フレームのみ**実行する（毎フレームではない＝周期再検出 detection_every）。
- 出力は consensus（`pipelines.components.consensus.merge_consensus`）が
  消費する検出仮説 dict。
- 依存（detector / segmenter）は注入可能（テスト容易・差し替え容易）。

I/O 契約（計画書 §3 DetectionIsland）:
    in:
        frames: list[(H, W, 3) uint8]
        detection_frame_indices: list[int]
        text_prompt: str
        閾値類（box/text/iou/top_k）
    out:
        detections: {
            frame_idx: {
                "masks":  (K, H, W) bool,
                "boxes":  (K, 4) float32 (xyxy),
                "scores": (K,) float32,
                "labels": list[str],
            }
        }
"""

from __future__ import annotations

from typing import Any

import numpy as np
from haystack import component

__all__ = ["DetectionIsland", "normalize_segmenter_masks"]

# GroundingDINO が「一致領域なし」で送出する ValueError を識別する語句。
# 周期検出では未検出フレームは正常事象なので空エントリへ変換する（他 ValueError は素通し）。
_NO_DETECTION_MARKER = "検出されませんでした"


def normalize_segmenter_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    num_boxes: int,
) -> np.ndarray:
    """SAM2 画像モードのマスク出力を (K, H, W) bool に正規化する。

    SAM2 の predict は入力 box 数 / multimask の有無で形状が変わる:
        - (K, C, H, W): box ごとに C 候補 → score 最大の候補を選ぶ。
        - (K, H, W):   既に box ごと 1 マスク → そのまま。
        - (H, W):      単一 box・単一マスク → (1, H, W) へ。

    Args:
        masks: SAM2 が返したマスク配列。
        scores: マスクに対応するスコア（候補選択に使う）。
        num_boxes: 入力 box 数（K）。

    Returns:
        (K, H, W) bool マスク。
    """
    masks = np.asarray(masks)
    scores = np.asarray(scores)

    if masks.ndim == 2:
        return masks[None, ...].astype(bool)

    if masks.ndim == 4:
        # (K, C, H, W) → 各 k で score 最大の c を選択。
        k, c = masks.shape[0], masks.shape[1]
        if c == 1:
            return masks[:, 0, :, :].astype(bool)
        score_2d = scores.reshape(k, c) if scores.size == k * c else np.zeros((k, c))
        best = np.argmax(score_2d, axis=1)
        selected = np.stack([masks[i, best[i]] for i in range(k)], axis=0)
        return selected.astype(bool)

    if masks.ndim == 3:
        # 基本は (K, H, W)＝box ごと 1 マスク。
        # 例外的に num_boxes==1 で first 軸が候補数 C のとき（=単一 box の
        # multimask が (C, H, W) で返る実装）だけ score 最大候補を選ぶ。
        if masks.shape[0] != num_boxes and num_boxes == 1 and scores.size == masks.shape[0]:
            best = int(np.argmax(scores))
            return masks[best][None, ...].astype(bool)
        return masks.astype(bool)

    raise ValueError(f"想定外のマスク形状です: {masks.shape}")


@component
class DetectionIsland:
    """検出フレームのみ GroundingDINO→SAM2 画像モードを実行する検出島。

    Args:
        detector: GroundingDINOMultiBoxDetector 互換（``run`` を持つ）。None なら
            warm_up で既定構築。
        segmenter: SAM2Segmenter 互換（``run`` を持つ）。None なら warm_up で既定構築。
        detector_config_path: detector 既定構築時の config パス。
        detector_checkpoint_path: detector 既定構築時の checkpoint パス。
        sam2_config_name: segmenter 既定構築時の SAM2 config 名。
        sam2_checkpoint_path: segmenter 既定構築時の SAM2 checkpoint パス。
        device: 推論デバイス。
    """

    def __init__(
        self,
        detector: Any | None = None,
        segmenter: Any | None = None,
        detector_config_path: str | None = None,
        detector_checkpoint_path: str | None = None,
        sam2_config_name: str | None = None,
        sam2_checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._detector = detector
        self._segmenter = segmenter
        self._detector_config_path = detector_config_path
        self._detector_checkpoint_path = detector_checkpoint_path
        self._sam2_config_name = sam2_config_name
        self._sam2_checkpoint_path = sam2_checkpoint_path
        self._device = device

    def warm_up(self) -> None:
        """detector / segmenter を遅延構築する（import 時に重い初期化をしない）。"""
        if self._detector is None:
            from pipelines.components.model_components import GroundingDINOMultiBoxDetector

            self._detector = GroundingDINOMultiBoxDetector(
                config_path=self._detector_config_path,
                checkpoint_path=self._detector_checkpoint_path,
                device=self._device,
            )
        if self._segmenter is None:
            from pipelines.components.model_components import SAM2Segmenter

            self._segmenter = SAM2Segmenter(
                checkpoint_path=self._sam2_checkpoint_path,
                config_name=self._sam2_config_name,
                device=self._device,
            )
        if hasattr(self._detector, "warm_up"):
            self._detector.warm_up()
        if hasattr(self._segmenter, "warm_up"):
            self._segmenter.warm_up()

    @component.output_types(detections=dict)
    def run(
        self,
        frames: list,
        detection_frame_indices: list,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """検出フレームごとに検出仮説を作る。

        Args:
            frames: RGB フレーム列。
            detection_frame_indices: 検出を走らせるフレーム index のリスト。
            text_prompt: GroundingDINO へのテキストプロンプト。
            box_threshold: GroundingDINO box 閾値。
            text_threshold: GroundingDINO text 閾値。
            iou_threshold: GroundingDINO NMS 閾値。
            top_k: 検出 box の上限。

        Returns:
            ``{"detections": {frame_idx: {masks, boxes, scores, labels}}}``。
        """
        if not text_prompt:
            raise ValueError("text_prompt を入力してください。")
        self.warm_up()
        assert self._detector is not None
        assert self._segmenter is not None

        num_frames = len(frames)
        detections: dict[int, dict[str, Any]] = {}
        for frame_idx in detection_frame_indices:
            if frame_idx < 0 or frame_idx >= num_frames:
                raise ValueError(
                    f"検出フレーム index={frame_idx} が範囲外です（0..{num_frames - 1}）。"
                )
            image = frames[frame_idx]
            height, width = image.shape[:2]
            detections[frame_idx] = self._detect_one(
                image=image,
                height=height,
                width=width,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                iou_threshold=iou_threshold,
                top_k=top_k,
            )
        return {"detections": detections}

    def _detect_one(
        self,
        image: np.ndarray,
        height: int,
        width: int,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        iou_threshold: float,
        top_k: int,
    ) -> dict[str, Any]:
        """1 フレーム分の検出（検出ゼロは空エントリ）。"""
        try:
            det = self._detector.run(
                image=image,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                iou_threshold=iou_threshold,
                top_k=top_k,
            )
        except ValueError as exc:
            # 「一致領域なし」だけ空エントリへ。それ以外の ValueError は素通し（握り潰さない）。
            if _NO_DETECTION_MARKER in str(exc):
                return self._empty_entry(height, width)
            raise

        boxes = np.asarray(det.get("boxes"), dtype=np.float32).reshape(-1, 4)
        phrases = list(det.get("phrases") or [])
        confidences = np.asarray(det.get("confidences"), dtype=np.float32).reshape(-1)
        if boxes.shape[0] == 0:
            return self._empty_entry(height, width)

        seg = self._segmenter.run(image=image, boxes=boxes, multimask=True)
        masks = normalize_segmenter_masks(
            seg.get("masks"), seg.get("scores"), num_boxes=boxes.shape[0]
        )
        labels = [
            phrases[i] if i < len(phrases) else text_prompt for i in range(boxes.shape[0])
        ]
        scores = (
            confidences
            if confidences.shape[0] == boxes.shape[0]
            else np.full((boxes.shape[0],), 0.0, dtype=np.float32)
        )
        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    @staticmethod
    def _empty_entry(height: int, width: int) -> dict[str, Any]:
        """検出ゼロのフレーム用の空エントリ。"""
        return {
            "masks": np.zeros((0, height, width), dtype=bool),
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": [],
        }
