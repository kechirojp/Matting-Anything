"""Haystack パイプラインで共有する純粋関数 Component。"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np
from haystack import component
from PIL import Image
from scipy import ndimage


def ensure_rgb_array(image: dict[str, Any] | np.ndarray | Image.Image) -> np.ndarray:
    """Gradio/PIL/ndarray 入力を RGB uint8 ndarray に正規化する。"""
    if image is None:
        raise ValueError("入力画像が None です。")

    if isinstance(image, dict):
        image_value = None
        for key in ("background", "composite", "image"):
            if image.get(key) is not None:
                image_value = image[key]
                break
    else:
        image_value = image

    if image_value is None:
        raise ValueError("ImageEditor の background/composite/image が取得できません。")

    if isinstance(image_value, Image.Image):
        image_array = np.array(image_value.convert("RGB"))
    elif isinstance(image_value, np.ndarray):
        image_array = image_value
    else:
        raise TypeError(f"未対応の画像型です: {type(image_value)!r}")

    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    if image_array.ndim != 3:
        raise ValueError(f"画像は HxWxC 形式である必要があります: shape={image_array.shape}")
    if image_array.shape[2] == 4:
        image_array = image_array[..., :3]
    if image_array.shape[2] != 3:
        raise ValueError(f"RGB/RGBA 画像のみ対応します: shape={image_array.shape}")

    return image_array.astype(np.uint8, copy=False)


def mask_to_bbox(mask: np.ndarray, padding: int = 20, image_shape: tuple[int, ...] | None = None) -> tuple[int, int, int, int] | None:
    """bool/uint8 マスクから padding 込み bbox を返す。"""
    mask_bool = mask.astype(bool)
    y_indices, x_indices = np.where(mask_bool)
    if len(x_indices) == 0:
        return None

    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    if image_shape is not None:
        image_height, image_width = image_shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image_width, x_max + padding)
        y_max = min(image_height, y_max + padding)
    return x_min, y_min, x_max, y_max


def dilate_binary_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """マスクを OpenCV の矩形 kernel で膨張する。"""
    if kernel_size < 1:
        raise ValueError("kernel_size は 1 以上である必要があります。")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def feather_binary_mask(mask: np.ndarray, dilate_size: int = 21, feather_radius: int = 8) -> np.ndarray:
    """二値マスクを膨張し境界を feather した連続値 soft guard を返す。

    SAM2/SAMURAI の二値 union mask をそのまま guard として乗算すると、境界が黒/白の
    2 値エッジになる。mask 境界を中心に ``feather_radius`` ピクセルかけて 0.0↔1.0 を
    滑らかに遷移させることで、transparent-background の gradient alpha を保ったまま
    mask 外の漏れ alpha を soft に削る。遷移帯が mask 内側の前景 alpha にも食い込むよう、
    膨張量は ``feather_radius`` を超えない範囲に制限する。

    Args:
        mask: 二値マスク（bool もしくは 0/1）。
        dilate_size: guard を外側へ広げる膨張 kernel サイズ（漏れ alpha の許容マージン）。
        feather_radius: 境界を feather する半径（0 で従来の二値 guard）。
            推奨値は 4〜16 程度。画像対角線の半分を超える値では距離変換が飽和し
            遷移帯が頭打ちになるため、画像サイズに対し十分小さい値を指定する。

    Returns:
        0.0..1.0 の float32 soft guard 配列（mask と同 shape）。
    """
    if feather_radius < 1:
        return dilate_binary_mask(mask, kernel_size=dilate_size).astype(np.float32)
    effective_dilate = max(1, min(dilate_size, feather_radius))
    base = dilate_binary_mask(mask, kernel_size=effective_dilate)
    inside_distance = cv2.distanceTransform(base.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~base).astype(np.uint8), cv2.DIST_L2, 5)
    signed_distance = inside_distance - outside_distance
    soft = np.clip(signed_distance / float(feather_radius) * 0.5 + 0.5, 0.0, 1.0)
    return soft.astype(np.float32)


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """数値的に安定なシグモイドで logit を [0,1] の float32 確率に変換する。

    SAM2 の生 logit（mask_logits）は ±数十に達するため、``1/(1+exp(-x))`` を素朴に
    計算すると ``exp`` が overflow して warning や inf を生む。正負で分岐して常に
    ``exp`` の引数を負にすることで overflow を避ける。

    Args:
        x: 任意 shape の logit 配列。

    Returns:
        ``x`` と同 shape の float32 確率配列（値域 [0,1]）。
    """
    arr = np.asarray(x, dtype=np.float32)
    positive = arr >= 0
    result = np.empty_like(arr, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def assign_points_to_boxes(
    points: Sequence[Sequence[float]] | None,
    boxes: Sequence[Sequence[float]] | None,
) -> dict[int, list[int]]:
    """各クリック点を最近傍 box の object prompt に割り当てる（修正1: 方針1）。

    boxes と points を併用する際、全 point を 1 つの追加 object にまとめると SAM2 が
    複数インスタンスを 1 mask で表現できず point が落ちる。代わりに各 point を矩形距離
    が最小の box に割り当て、その box の ``add_new_points_or_box`` に同梱することで、
    positive 点は最寄り box を補強し、negative 点は最寄り box 内部をくり抜ける。

    Args:
        points: クリック点 ``[[x, y], ...]``（None/空可）。
        boxes: 検出 box ``[[x1, y1, x2, y2], ...]``（1-based の obj_id にマップ）。

    Returns:
        ``{obj_id(1..N): [point_index, ...]}`` の辞書。box が無ければ空辞書。
        point が無くても全 obj_id を空リストで含む。
    """
    box_list = [list(map(float, box)) for box in (boxes or [])]
    assignment: dict[int, list[int]] = {obj_id: [] for obj_id in range(1, len(box_list) + 1)}
    if not box_list or not points:
        return assignment

    for point_index, point in enumerate(points):
        px, py = float(point[0]), float(point[1])
        best_obj_id = 1
        best_distance = float("inf")
        for offset, box in enumerate(box_list):
            x1, y1, x2, y2 = box
            dx = max(x1 - px, 0.0, px - x2)
            dy = max(y1 - py, 0.0, py - y2)
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_obj_id = offset + 1
        assignment[best_obj_id].append(point_index)
    return assignment


def soft_probability_guard(
    probability: np.ndarray,
    dilate_size: int = 21,
    feather_radius: int = 8,
) -> np.ndarray:
    """soft 確率 mask の継ぎ目谷を closing で埋め、末端 feather した soft guard を返す（修正2: 根治）。

    SAM2 の soft union 確率をそのまま guard に使うと、複数 object の境界（継ぎ目）で
    確率が落ち込み黒い継ぎ目線になる。grayscale の morphological closing で細い谷を
    橋渡しし、Gaussian blur で末端を feather することで、継ぎ目を暗くせず連続な soft
    guard を生成する。二値化は一切行わない。

    Args:
        probability: [0,1] の soft 確率 mask（float）。
        dilate_size: closing の正方 kernel サイズ（継ぎ目谷を橋渡しする幅）。
        feather_radius: Gaussian feather 半径（ksize = ``feather_radius*2+1``）。

    Returns:
        [0,1] の float32 soft guard 配列（``probability`` と同 shape）。
    """
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if dilate_size >= 1:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        prob = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, kernel)
    if feather_radius >= 1:
        ksize = feather_radius * 2 + 1
        prob = cv2.GaussianBlur(prob, (ksize, ksize), 0)
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def compose_alpha(image: np.ndarray, alpha: np.ndarray, background: np.ndarray | tuple[int, int, int]) -> np.ndarray:
    """RGB 画像と alpha を背景に合成する。"""
    image_rgb = ensure_rgb_array(image)
    alpha_float = alpha.astype(np.float32)
    if alpha_float.max() > 1.0:
        alpha_float = alpha_float / 255.0
    alpha_float = np.clip(alpha_float, 0.0, 1.0)

    if isinstance(background, tuple):
        background_rgb = np.full_like(image_rgb, background, dtype=np.uint8)
    else:
        background_rgb = ensure_rgb_array(background)
        if background_rgb.shape[:2] != image_rgb.shape[:2]:
            background_rgb = cv2.resize(background_rgb, (image_rgb.shape[1], image_rgb.shape[0]))

    composite = alpha_float[..., None] * image_rgb + (1.0 - alpha_float[..., None]) * background_rgb
    return np.clip(composite, 0, 255).astype(np.uint8)


def normalize_masks(masks: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """SAM/SAM2/将来 Segmenter の mask 出力を (N,H,W) bool 配列へ正規化する。"""
    mask_array = np.asarray(masks)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if mask_array.ndim != 3:
        raise ValueError(f"masks は (H,W) または (N,H,W) 形式である必要があります: shape={mask_array.shape}")
    return mask_array.astype(bool, copy=False)


def build_mask_set(
    masks: np.ndarray | Sequence[np.ndarray],
    scores: Sequence[float] | np.ndarray | None = None,
    boxes: Sequence[Sequence[float]] | np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    source: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """候補 mask 群を Haystack Component 間で受け渡す MaskSet 契約 dict に変換する。"""
    # allow passing per-object logits mapping from frame_idx to (N,H,W) arrays
    # but this function remains focused on mask set building for candidate masks.
    mask_array = normalize_masks(masks)
    mask_count = int(mask_array.shape[0])
    score_array = np.zeros((mask_count,), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
    if score_array.shape != (mask_count,):
        raise ValueError(f"scores は mask 数と同じ長さである必要があります: masks={mask_count}, scores={score_array.shape}")

    if boxes is None:
        box_array = np.empty((0, 4), dtype=np.float32)
    else:
        box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(box_array) != mask_count:
            raise ValueError(f"boxes は mask 数と同じ長さである必要があります: masks={mask_count}, boxes={len(box_array)}")

    label_list = list(labels) if labels is not None else [f"mask_{index}" for index in range(mask_count)]
    if len(label_list) != mask_count:
        raise ValueError(f"labels は mask 数と同じ長さである必要があります: masks={mask_count}, labels={len(label_list)}")

    # 高さ・幅などの基本情報は metadata に持たせ、将来の動画 frame_id 等も同じ場所へ拡張する。
    mask_metadata = dict(metadata or {})
    mask_metadata.setdefault("height", int(mask_array.shape[1]))
    mask_metadata.setdefault("width", int(mask_array.shape[2]))
    mask_metadata.setdefault("count", mask_count)
    return {
        "masks": mask_array,
        "scores": score_array,
        "boxes": box_array,
        "labels": label_list,
        "source": source,
        "metadata": mask_metadata,
    }


def build_selected_mask(
    mask: np.ndarray,
    source_indices: Sequence[int] | None = None,
    label: str = "selected",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """単一 mask / union mask を SelectedMask 契約 dict に変換する。"""
    mask_array = np.asarray(mask).astype(bool)
    if mask_array.ndim != 2:
        raise ValueError(f"SelectedMask は (H,W) 形式である必要があります: shape={mask_array.shape}")
    return {
        "mask": mask_array,
        "source_indices": list(source_indices or []),
        "label": label,
        "metadata": dict(metadata or {}),
    }


def select_candidate_masks(
    mask_set: dict,
    indices: Sequence[int] | None = None,
    score_threshold: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """MaskSet から index / score / top-k 条件に合う候補だけを抽出する。"""
    masks = normalize_masks(mask_set["masks"])
    scores = np.asarray(mask_set.get("scores", np.zeros((len(masks),), dtype=np.float32)), dtype=np.float32)
    boxes = np.asarray(mask_set.get("boxes", np.empty((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
    labels = list(mask_set.get("labels", [f"mask_{index}" for index in range(len(masks))]))
    source = str(mask_set.get("source", "unknown"))

    # index 指定がない場合は全候補を対象にし、範囲外 index は明示的にエラーにする。
    candidate_indices = list(range(len(masks))) if indices is None else [int(index) for index in indices]
    if any(index < 0 or index >= len(masks) for index in candidate_indices):
        raise ValueError(f"indices に範囲外の値があります: {candidate_indices}")
    if score_threshold is not None:
        candidate_indices = [index for index in candidate_indices if float(scores[index]) >= float(score_threshold)]
    if top_k is not None:
        candidate_indices = sorted(candidate_indices, key=lambda index: float(scores[index]), reverse=True)[: int(top_k)]
    else:
        candidate_indices = sorted(candidate_indices, key=lambda index: float(scores[index]), reverse=True)

    selected_boxes = boxes[candidate_indices] if len(boxes) else None
    metadata = dict(mask_set.get("metadata", {}))
    metadata["source_indices"] = candidate_indices
    return build_mask_set(
        masks[candidate_indices],
        scores=scores[candidate_indices],
        boxes=selected_boxes,
        labels=[labels[index] for index in candidate_indices],
        source=source,
        metadata=metadata,
    )


def union_masks(
    masks: np.ndarray | Sequence[np.ndarray],
    mode: str = "or",
    dilate_kernel: int = 0,
    min_area: int = 0,
) -> np.ndarray:
    """複数 candidate mask を 1 枚の union mask に統合する。"""
    mask_array = normalize_masks(masks)
    if len(mask_array) == 0:
        raise ValueError("union 対象の mask がありません。")
    if mode != "or":
        raise ValueError(f"未対応の union mode です: {mode}")

    union = np.any(mask_array, axis=0)
    if dilate_kernel and dilate_kernel > 1:
        union = dilate_binary_mask(union, kernel_size=int(dilate_kernel))
    if min_area and min_area > 1:
        labeled, num_labels = ndimage.label(union)
        filtered = np.zeros_like(union, dtype=bool)
        # 小さい連結成分を除外し、背景の誤結合を抑える。
        for label_index in range(1, num_labels + 1):
            region = labeled == label_index
            if int(region.sum()) >= int(min_area):
                filtered |= region
        union = filtered
    return union.astype(bool, copy=False)


def compose_mask_preview(
    image: np.ndarray,
    masks: np.ndarray | Sequence[np.ndarray] | None = None,
    labels: Sequence[str] | None = None,
    selected_indices: Sequence[int] | None = None,
    union_mask: np.ndarray | None = None,
) -> np.ndarray:
    """入力画像へ candidate / selected / union mask を重ねた確認用 preview を作る。"""
    image_rgb = ensure_rgb_array(image).copy()
    selected_set = {int(index) for index in (selected_indices or [])}
    palette = np.array(
        [
            [30, 144, 255],
            [255, 140, 0],
            [46, 204, 113],
            [155, 89, 182],
            [231, 76, 60],
        ],
        dtype=np.uint8,
    )

    if masks is not None:
        mask_array = normalize_masks(masks)
        for index, mask in enumerate(mask_array):
            color = palette[index % len(palette)]
            alpha = 0.55 if index in selected_set else 0.32
            image_rgb[mask] = (image_rgb[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)
            if index in selected_set:
                bbox = mask_to_bbox(mask, padding=0, image_shape=image_rgb.shape)
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), color.tolist(), 2)

    if union_mask is not None:
        union_bool = np.asarray(union_mask).astype(bool)
        if union_bool.shape != image_rgb.shape[:2]:
            raise ValueError(f"union_mask の形状が画像と一致しません: mask={union_bool.shape}, image={image_rgb.shape[:2]}")
        union_color = np.array([255, 215, 0], dtype=np.uint8)
        image_rgb[union_bool] = (image_rgb[union_bool] * 0.45 + union_color * 0.55).astype(np.uint8)
        bbox = mask_to_bbox(union_bool, padding=0, image_shape=image_rgb.shape)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
    return image_rgb


def render_tracking_overlay_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (30, 144, 255),
    fill_alpha: float = 0.45,
    contour_thickness: int = 2,
) -> np.ndarray:
    """1 frame に追跡 mask の半透明塗りと輪郭を重ね、追従確認用 overlay を作る。"""
    overlay = ensure_rgb_array(frame).copy()
    mask_array = np.asarray(mask)
    # soft 確率 mask（float）は閾値 0.5 で前景判定する。二値 mask は従来通り。
    if np.issubdtype(mask_array.dtype, np.floating):
        mask_bool = mask_array >= 0.5
    else:
        mask_bool = mask_array.astype(bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"mask は (H,W) 形式である必要があります: shape={mask_bool.shape}")
    if mask_bool.shape != overlay.shape[:2]:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (overlay.shape[1], overlay.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if not mask_bool.any():
        return overlay
    color_array = np.array(color, dtype=np.float32)
    alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    overlay[mask_bool] = (overlay[mask_bool] * (1.0 - alpha) + color_array * alpha).astype(np.uint8)
    if contour_thickness > 0:
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(value) for value in color), int(contour_thickness))
    return overlay


def mask_set_to_status(mask_set: dict[str, Any]) -> str:
    """MaskSet の件数・最高 score・source を UI 表示用文字列にまとめる。"""
    masks = normalize_masks(mask_set["masks"])
    scores = np.asarray(mask_set.get("scores", np.zeros((len(masks),), dtype=np.float32)), dtype=np.float32)
    source = str(mask_set.get("source", "unknown"))
    if len(masks) == 0:
        return f"0 masks from {source}"
    best_index = int(np.argmax(scores))
    return f"{len(masks)} masks from {source}; best={best_index}; score={float(scores[best_index]):.4f}"


@component
class ImageNormalizer:
    """Gradio/PIL/ndarray 入力を RGB ndarray に正規化する Component。"""

    @component.output_types(image=np.ndarray)
    def run(self, image: dict[str, Any] | np.ndarray | Image.Image) -> dict[str, np.ndarray]:
        return {"image": ensure_rgb_array(image)}


@component
class ScribbleParser:
    """Gradio ImageEditor の描画レイヤーから point/box/mask を抽出する Component。"""

    @component.output_types(points=np.ndarray, box=np.ndarray, mask=np.ndarray)
    def run(self, editor_value: dict[str, Any], mode: str = "point") -> dict[str, np.ndarray]:
        image = ensure_rgb_array(editor_value)
        layers = editor_value.get("layers") if isinstance(editor_value, dict) else None
        if not layers:
            raise ValueError("スクリブルレイヤーがありません。")
        scribble = layers[0]
        if isinstance(scribble, Image.Image):
            scribble = np.array(scribble)
        if scribble.ndim != 3 or scribble.shape[2] < 1:
            raise ValueError("スクリブルは HxWxC 形式である必要があります。")

        mask = scribble[..., 0] >= 255
        labeled_array, num_features = ndimage.label(mask)
        centers = ndimage.center_of_mass(mask, labeled_array, range(1, num_features + 1))
        if len(centers) == 0:
            raise ValueError("スクリブルが検出されませんでした。")

        points_yx = np.array(centers, dtype=np.float32)
        points_xy = points_yx[:, ::-1]
        bbox_value = mask_to_bbox(mask, padding=0, image_shape=image.shape)
        if bbox_value is None:
            raise ValueError("スクリブルの bbox が検出されませんでした。")
        bbox = np.array(bbox_value, dtype=np.float32)
        if mode == "box":
            return {"points": np.empty((0, 2), dtype=np.float32), "box": bbox, "mask": mask}
        return {"points": points_xy, "box": bbox, "mask": mask}


@component
class BBoxFromMask:
    """マスクから bounding box を作る Component。"""

    @component.output_types(box=object)
    def run(self, mask: np.ndarray, padding: int = 20, image_shape: tuple[int, ...] | None = None) -> dict[str, object]:
        return {"box": mask_to_bbox(mask, padding=padding, image_shape=image_shape)}


@component
class MaskDilator:
    """マスクを膨張する Component。"""

    @component.output_types(mask=np.ndarray)
    def run(self, mask: np.ndarray, kernel_size: int = 15) -> dict[str, np.ndarray]:
        return {"mask": dilate_binary_mask(mask, kernel_size=kernel_size)}


@component
class MaskCandidateSelector:
    """MaskSet から UI や score 条件に合う候補を抽出する Component。"""

    @component.output_types(mask_set=dict, selected_indices=list, status=str)
    def run(
        self,
        mask_set: dict,
        indices: list[int] | None = None,
        score_threshold: float | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        selected_mask_set = select_candidate_masks(
            mask_set,
            indices=indices,
            score_threshold=score_threshold,
            top_k=top_k,
        )
        return {
            "mask_set": selected_mask_set,
            "selected_indices": list(selected_mask_set["metadata"].get("source_indices", [])),
            "status": mask_set_to_status(selected_mask_set),
        }


@component
class MaskUnion:
    """MaskSet 内の複数 mask を OR union して SelectedMask 契約で返す Component。"""

    @component.output_types(mask=np.ndarray, selected_mask=dict, status=str)
    def run(
        self,
        mask_set: dict,
        mode: str = "or",
        dilate_kernel: int = 0,
        min_area: int = 0,
    ) -> dict[str, Any]:
        union = union_masks(mask_set["masks"], mode=mode, dilate_kernel=dilate_kernel, min_area=min_area)
        source_indices = list(mask_set.get("metadata", {}).get("source_indices", range(len(mask_set["masks"]))))
        selected_mask = build_selected_mask(
            union,
            source_indices=source_indices,
            label="+".join(mask_set.get("labels", [])) or "union",
            metadata={"source": mask_set.get("source", "unknown")},
        )
        return {"mask": union, "selected_mask": selected_mask, "status": f"Union mask: {int(union.sum())} foreground pixels"}


@component
class MaskPreviewComposer:
    """MaskSet / union mask を入力画像に重ねた preview を生成する Component。"""

    @component.output_types(preview=np.ndarray)
    def run(
        self,
        image: np.ndarray,
        mask_set: dict = None,
        union_mask: np.ndarray | None = None,
        selected_indices: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        masks = mask_set["masks"] if mask_set is not None else None
        labels = mask_set.get("labels") if mask_set is not None else None
        return {
            "preview": compose_mask_preview(
                image,
                masks=masks,
                labels=labels,
                selected_indices=selected_indices,
                union_mask=union_mask,
            )
        }


@component
class AlphaCompositor:
    """alpha matte を背景と合成する Component。"""

    @component.output_types(composite=np.ndarray, green_screen=np.ndarray)
    def run(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        background: np.ndarray | tuple[int, int, int] = (51, 255, 146),
    ) -> dict[str, np.ndarray]:
        return {
            "composite": compose_alpha(image, alpha, background),
            "green_screen": compose_alpha(image, alpha, (51, 255, 146)),
        }