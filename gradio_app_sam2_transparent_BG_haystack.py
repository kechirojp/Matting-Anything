"""Haystack Pipeline 版 SAM2 + transparent-background 複合対象 mask union デモ。"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

try:
    import gradio_client.utils as _gradio_client_utils

    _original_json_schema_to_python_type = getattr(
        _gradio_client_utils,
        "_matting_anything_original_json_schema_to_python_type",
        _gradio_client_utils._json_schema_to_python_type,
    )
    _gradio_client_utils._matting_anything_original_json_schema_to_python_type = _original_json_schema_to_python_type

    def _patched_json_schema_to_python_type(schema, defs=None):
        # gradio_client が JSON Schema の boolean schema を dict として扱う場合の /info 例外を避ける。
        if isinstance(schema, bool):
            return "Any"
        return _original_json_schema_to_python_type(schema, defs)

    _gradio_client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except (ImportError, AttributeError) as exc:
    warnings.warn(f"Gradio bool schema patch was skipped: {exc}", RuntimeWarning)

import cv2
import gradio as gr
import numpy as np

from pipelines.components.common import (
    build_mask_set,
    compose_mask_preview,
    ensure_rgb_array,
    mask_set_to_status,
    mask_to_bbox,
    select_candidate_masks,
    union_masks,
)
from pipelines.components.ui_helpers import (
    clamp_prompt_point,
    copy_prompt_state as _copy_prompt_state,
    draw_prompt_overlay,
    empty_prompt_state,
    extend_box_to_edge,
    normalize_box_from_points,
    select_sam2_prompt,
    set_prompt_overlay_mask_provider,
)
from pipelines.components.model_components import GroundingDINOMultiBoxDetector
from pipelines.components.model_components import format_stage_timings
from pipelines.sam2_tb_pipeline import build_mask_to_matte_pipeline, build_sam2_maskset_pipeline
from pipelines.components.model_registry import build_dropdown_choices, entry_by_id


SAM2_PIPELINE = None
_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}
TEXT_DETECTOR = None
SAM2_STATE: dict[str, Any] = {}
PROMPT_CANVAS_HEIGHT = 420
OUTPUT_WINDOW_HEIGHT = 360
PROMPT_CANVAS_PLACEHOLDER_SIZE = (420, 640)


def create_prompt_canvas_placeholder() -> np.ndarray:
    """SAM2 prompt canvas の初期表示用プレースホルダーを作る。"""
    height, width = PROMPT_CANVAS_PLACEHOLDER_SIZE
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (1, 1), (width - 2, height - 2), (220, 225, 232), 2)
    cv2.putText(canvas, "SAM2 Prompt Canvas", (36, height // 2 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 78, 92), 2)
    cv2.putText(
        canvas,
        "Upload an image in Input Image above.",
        (36, height // 2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (105, 113, 128),
        2,
    )
    cv2.putText(
        canvas,
        "Then click here to place points or box corners.",
        (36, height // 2 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (105, 113, 128),
        2,
    )
    return canvas


def get_sam2_pipeline():
    """SAM2 MaskSet Pipeline を遅延構築する。"""
    global SAM2_PIPELINE
    if SAM2_PIPELINE is None:
        SAM2_PIPELINE = build_sam2_maskset_pipeline()
    return SAM2_PIPELINE


def get_tb_pipeline(background_model_id: str = "tb_base"):
    """transparent-background MatteExtractor Pipeline を遅延構築する。

    同じ model_id の場合はキャッシュを返す（warm_up 再実行なし）。
    """
    key = ("background", background_model_id)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = build_mask_to_matte_pipeline()
    return _PIPELINE_CACHE[key]


def get_text_detector() -> GroundingDINOMultiBoxDetector:
    """GroundingDINO TextToRegion Component を遅延構築する。"""
    global TEXT_DETECTOR
    if TEXT_DETECTOR is None:
        TEXT_DETECTOR = GroundingDINOMultiBoxDetector()
    return TEXT_DETECTOR


def parse_candidate_indices(indices_text: str | None, mask_count: int) -> list[int]:
    """UI の index 文字列を mask index 配列に変換する。"""
    if not indices_text:
        return []
    parsed: list[int] = []
    for chunk in str(indices_text).replace(" ", "").split(","):
        if not chunk:
            continue
        if not chunk.isdigit():
            raise gr.Error(f"Candidate index はカンマ区切りの整数で入力してください: {indices_text}")
        index = int(chunk)
        if index < 0 or index >= mask_count:
            raise gr.Error(f"Candidate index が範囲外です: {index} / count={mask_count}")
        parsed.append(index)
    return sorted(set(parsed))


def format_candidate_table(mask_set: dict[str, Any] | None) -> list[list[Any]]:
    """MaskSet を Gradio Dataframe 表示用の行へ変換する。"""
    if not mask_set:
        return []
    masks = mask_set["masks"]
    scores = mask_set["scores"]
    boxes = mask_set.get("boxes")
    labels = mask_set.get("labels", [])
    rows: list[list[Any]] = []
    for index in range(len(masks)):
        bbox = mask_to_bbox(masks[index], padding=0, image_shape=masks[index].shape)
        if boxes is not None and len(boxes) == len(masks):
            bbox = tuple(int(value) for value in boxes[index])
        rows.append([index, float(scores[index]), labels[index] if index < len(labels) else f"mask_{index}", str(bbox)])
    return rows


def current_overlay_mask() -> np.ndarray | None:
    """表示 overlay では union mask を優先し、なければ best candidate を使う。"""
    if SAM2_STATE.get("union_mask") is not None:
        return SAM2_STATE["union_mask"]
    return SAM2_STATE.get("mask")


set_prompt_overlay_mask_provider(current_overlay_mask)


def clear_sam2_prompt(input_image, prompt_state: dict | None):
    """SAM2 prompt と候補 mask / union mask をクリアする。"""
    state = empty_prompt_state()
    SAM2_STATE.clear()
    preview = ensure_rgb_array(input_image) if input_image is not None else create_prompt_canvas_placeholder()
    return preview, None, None, [], state, "SAM2 prompt and masks cleared"


def sync_prompt_canvas(input_image):
    """アップロード画像を SAM2 prompt 編集用キャンバスへ反映する。"""
    state = empty_prompt_state()
    SAM2_STATE.clear()
    if input_image is None:
        return create_prompt_canvas_placeholder(), state, "Input image is empty."
    prompt_canvas = ensure_rgb_array(input_image).copy()
    return prompt_canvas, state, "SAM2 Prompt Canvas ready. Click this canvas to place points or box corners."


def sync_prompt_canvas_outputs(input_image):
    """UI 更新用に prompt canvas 同期結果へ candidate/union 出力リセットを加える。"""
    prompt_canvas, state, status = sync_prompt_canvas(input_image)
    return prompt_canvas, None, None, [], state, status


def update_image_display_size(display_size: str):
    """prompt / candidate / union / 出力画像の表示高さを切り替える。"""
    if display_size == "original":
        return tuple(gr.update(height=None) for _ in range(6))
    return (
        gr.update(height=PROMPT_CANVAS_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
    )


def update_point_label_visibility(prompt_mode: str):
    """point prompt の時だけ positive / negative 選択を表示する。"""
    return gr.update(visible=prompt_mode == "point")


def format_model_diagnostics(diagnostics: dict[str, Any] | None, callback_elapsed: float | None = None) -> str:
    """モデル診断情報を Gradio status 向けに短く整形する。"""
    if not diagnostics:
        return "diagnostics unavailable"
    runtime = diagnostics.get("runtime", {})
    checkpoint = diagnostics.get("checkpoint", {})
    cuda_ops = diagnostics.get("cuda_ops")
    lines = [
        f"component={diagnostics.get('component')}",
        f"pid={diagnostics.get('process_id', runtime.get('process_id'))}",
        f"device={diagnostics.get('device')} cuda_available={runtime.get('cuda_available')}",
        (
            "gpu_policy=required "
            f"cpu_fallback_allowed={runtime.get('cpu_fallback_allowed')}"
        ),
    ]
    if runtime.get("gpu_required") and not runtime.get("cuda_available"):
        lines.append("WARNING: CPU inference is emergency fallback only. Use a GPU runtime for production/video work.")
    if runtime.get("cuda_device_name"):
        lines.append(f"cuda_device={runtime.get('cuda_device_name')}")
    if "cached_before" in diagnostics:
        lines.append(f"cached_before={diagnostics.get('cached_before')}")
    cache_id = diagnostics.get("predictor_id", diagnostics.get("model_id"))
    if cache_id:
        lines.append(f"cache_id={cache_id}")
    if checkpoint:
        lines.append(
            f"checkpoint_exists={checkpoint.get('exists')} size_mb={checkpoint.get('size_mb')} "
            f"drive_path={checkpoint.get('is_drive_path')}"
        )
    if cuda_ops:
        lines.append(f"groundingdino_cuda_ops={cuda_ops.get('available')} source={cuda_ops.get('source')}")
    if diagnostics.get("image_shape"):
        lines.append(f"image_shape={diagnostics.get('image_shape')}")
    if diagnostics.get("autocast"):
        lines.append(f"autocast={diagnostics.get('autocast')}")
    if diagnostics.get("timings"):
        lines.append(f"timings: {format_stage_timings(diagnostics['timings'])}")
    if callback_elapsed is not None:
        lines.append(f"gradio_callback_total={callback_elapsed:.3f}s")
    return "\n".join(lines)


def detect_text_boxes(
    input_image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    iou_threshold: float,
    top_k: int,
    prompt_state: dict | None,
):
    """Text Prompt を GroundingDINO bbox 候補へ変換し、最上位 bbox を SAM2 prompt に反映する。"""
    try:
        callback_start = time.perf_counter()
        if input_image is None:
            raise gr.Error("先に画像をアップロードしてください。")
        if not text_prompt:
            raise gr.Error("Text Prompt を入力してください。")
        result = get_text_detector().run(
            ensure_rgb_array(input_image),
            text_prompt=text_prompt,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            iou_threshold=float(iou_threshold),
            top_k=int(top_k),
        )
        SAM2_STATE["detected_boxes"] = result["proposals"]
        state = _copy_prompt_state(prompt_state)
        state["box"] = [int(value) for value in result["boxes"][0].tolist()]
        state["box_buffer"] = []
        preview = draw_prompt_overlay(input_image, state)
        rows = [
            [index, float(confidence), phrase, str(tuple(int(value) for value in box))]
            for index, (box, phrase, confidence) in enumerate(
                zip(result["boxes"], result["phrases"], result["confidences"])
            )
        ]
        status = (
            f"Detected {len(rows)} text boxes. Top box was copied to SAM2 prompt.\n"
            f"{format_model_diagnostics(result.get('diagnostics'), time.perf_counter() - callback_start)}"
        )
        return preview, rows, state, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Text Prompt 検出に失敗しました: {exc}") from exc


def predict_masks(input_image, prompt_state: dict | None, multimask: bool):
    """SAM2 Component で MaskSet 候補を生成する。"""
    try:
        callback_start = time.perf_counter()
        state = _copy_prompt_state(prompt_state)
        points = state.get("points") or []
        labels = state.get("labels") or []
        box = state.get("box")
        if not points and box is None:
            raise gr.Error("画像をクリックするか Text Prompt で bbox を検出してください。")
        segmenter_input: dict[str, Any] = {"multimask": multimask}
        if points:
            segmenter_input["points"] = points
            segmenter_input["labels"] = labels
        if box is not None:
            segmenter_input["box"] = box
        result = get_sam2_pipeline().run(
            {
                "image_normalizer": {"image": input_image},
                "sam2_segmenter": segmenter_input,
            },
            include_outputs_from={"sam2_segmenter", "mask_preview"},
        )
        mask_set = result["sam2_segmenter"]["mask_set"]
        scores = mask_set["scores"]
        best_index = int(np.argmax(scores))
        SAM2_STATE["mask_set"] = mask_set
        SAM2_STATE["mask"] = mask_set["masks"][best_index].astype(bool)
        SAM2_STATE["union_mask"] = SAM2_STATE["mask"]
        SAM2_STATE["union_indices"] = [best_index]
        prompt_preview = draw_prompt_overlay(input_image, state, SAM2_STATE["union_mask"])
        candidate_preview = compose_mask_preview(input_image, mask_set["masks"], selected_indices=[best_index], union_mask=SAM2_STATE["union_mask"])
        diagnostics = result["sam2_segmenter"].get("diagnostics") or mask_set.get("metadata", {}).get("diagnostics")
        status = (
            f"{mask_set_to_status(mask_set)}; best candidate {best_index} is ready for transparent-background\n"
            f"{format_model_diagnostics(diagnostics, time.perf_counter() - callback_start)}"
        )
        return (
            prompt_preview,
            candidate_preview,
            candidate_preview,
            format_candidate_table(mask_set),
            status,
            state,
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"SAM2 推論に失敗しました: {exc}") from exc


def _require_mask_set() -> dict[str, Any]:
    mask_set = SAM2_STATE.get("mask_set")
    if not mask_set:
        raise gr.Error("先に Predict SAM2 Candidate Masks を実行してください。")
    return mask_set


def add_candidates_to_union(input_image, candidate_indices: str, dilate_kernel: int, min_area: int):
    """指定 candidate index を現在の union mask に追加する。"""
    mask_set = _require_mask_set()
    indices = parse_candidate_indices(candidate_indices, len(mask_set["masks"]))
    if not indices:
        raise gr.Error("追加する Candidate Mask Indices を入力してください。")
    union_indices = sorted(set(SAM2_STATE.get("union_indices", [])) | set(indices))
    selected = select_candidate_masks(mask_set, indices=union_indices)
    union = union_masks(selected["masks"], dilate_kernel=int(dilate_kernel), min_area=int(min_area))
    SAM2_STATE["union_indices"] = union_indices
    SAM2_STATE["union_mask"] = union
    preview = compose_mask_preview(input_image, mask_set["masks"], selected_indices=union_indices, union_mask=union)
    return preview, f"Union indices: {union_indices}; foreground pixels={int(union.sum())}"


def remove_candidates_from_union(input_image, candidate_indices: str, dilate_kernel: int, min_area: int):
    """指定 candidate index を現在の union mask から除外する。"""
    mask_set = _require_mask_set()
    remove_indices = set(parse_candidate_indices(candidate_indices, len(mask_set["masks"])))
    union_indices = [index for index in SAM2_STATE.get("union_indices", []) if index not in remove_indices]
    if not union_indices:
        SAM2_STATE.pop("union_mask", None)
        SAM2_STATE["union_indices"] = []
        return compose_mask_preview(input_image, mask_set["masks"]), "Union is empty."
    selected = select_candidate_masks(mask_set, indices=union_indices)
    union = union_masks(selected["masks"], dilate_kernel=int(dilate_kernel), min_area=int(min_area))
    SAM2_STATE["union_indices"] = union_indices
    SAM2_STATE["union_mask"] = union
    preview = compose_mask_preview(input_image, mask_set["masks"], selected_indices=union_indices, union_mask=union)
    return preview, f"Union indices: {union_indices}; foreground pixels={int(union.sum())}"


def clear_union_mask(input_image):
    """Union mask だけをクリアし、candidate mask は保持する。"""
    mask_set = _require_mask_set()
    SAM2_STATE.pop("union_mask", None)
    SAM2_STATE["union_indices"] = []
    return compose_mask_preview(input_image, mask_set["masks"]), "Union cleared."


def run_transparent_bg(
    input_image,
    mask_source: str,
    background_model_id: str,
    tb_jit: bool,
    tb_threshold: float,
    tb_output_type: str,
    crop_padding: int,
):
    """標準 mask 契約から transparent-background MatteExtractor を実行する。"""
    try:
        entry = entry_by_id("background", background_model_id)
        tb_mode = entry.get("tb_mode", "base")
        mask_feather = int(entry.get("mask_feather", 0))
        mask = None
        if mask_source == "Union Mask":
            mask = SAM2_STATE.get("union_mask")
            if mask is None:
                raise gr.Error("Union Mask がありません。複合対象だけ Add to Union で mask を追加してください。")
        elif mask_source == "Best Candidate Mask":
            mask = SAM2_STATE.get("mask")
            if mask is None:
                raise gr.Error("Best Candidate Mask がありません。先に SAM2 候補を生成してください。")
        # feather>0 のときは extractor 段で soft guard を一元適用し、後段 sam2_guard の
        # 二値 guard との二重適用（soft × 二値 = 2 値エッジ再発）を避けるため無効化する。
        sam2_guard_enabled = mask is not None and mask_feather <= 0
        result = get_tb_pipeline(background_model_id).run(
            {
                "image_normalizer": {"image": input_image},
                "transparent_bg": {
                    "mask": mask,
                    "tb_mode": tb_mode,
                    "tb_jit": tb_jit,
                    "tb_threshold": tb_threshold,
                    "tb_output_type": tb_output_type,
                    "crop_padding": int(crop_padding),
                    "mask_guard_feather": mask_feather,
                },
                "sam2_guard": {"mask": mask, "enabled": sam2_guard_enabled},
                "output_saver": {"enabled": True},
            },
            include_outputs_from={"transparent_bg", "sam2_guard", "output_saver"},
        )
        rgba = result["transparent_bg"]["rgba"]
        preview = result["transparent_bg"]["preview"]
        alpha = result["sam2_guard"]["alpha"]
        paths = result.get("output_saver", {}).get("paths", {})
        bbox = mask_to_bbox(mask, padding=int(crop_padding), image_shape=rgba.shape) if mask is not None else None
        return rgba, alpha, preview, str(paths), str(bbox)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"transparent-background 推論に失敗しました: {exc}") from exc


with gr.Blocks(title="SAM2 + Transparent Background Haystack") as demo:
    gr.Markdown("# SAM2 + Transparent Background Haystack")
    gr.Markdown(
        "Text Prompt / SAM2 multimask / mask union を標準 I/O 契約で接続し、"
        "ドラムをたたく人・自転車に乗る人のような複合対象 mask を作る実験 UI です。"
    )
    gr.Markdown(
        """
### 使い方（日本語ガイド）

**すべてを上から順に入力する必要はありません。** 必須なのは **Input Image** と、
SAM2 に渡す対象範囲の指定だけです。対象範囲は **Text Prompt で自動検出**するか、
**SAM2 Prompt Canvas をクリックして手動指定**するか、どちらか一方で進められます。

**最短フロー**

1. **必須**: `Input Image` に画像を入れる。
2. **必須**: `SAM2 Prompt Canvas` を2クリックして bbox を指定する。
3. **必須**: `Predict SAM2 Candidate Masks` を押して候補 mask を作る。
4. **必須**: `Run transparent-background` を押す。既定では最高スコア候補 mask を使います。

**任意・複合対象（人 + ドラム / 人 + 自転車）**

- Text Prompt で自動 bbox を作りたい場合だけ `Optional: Text Prompt to Box` を開きます。
- 複数候補を統合したい場合だけ `Optional: Mask Union` を開き、
   `SAM2 Candidate Mask Table` の index を見て
   `Candidate Mask Indices` に `0,1` のように入力し、`Add to Union` で統合する。

**迷ったとき**: まず `Input Image` → bbox を2クリック → `Predict SAM2 Candidate Masks` →
`Run transparent-background` の最短ルートで試してください。
"""
    )
    prompt_state = gr.State(value=empty_prompt_state())
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Input Image", height=300, interactive=True)
            prompt_canvas = gr.Image(
                value=create_prompt_canvas_placeholder(),
                type="numpy",
                label="SAM2 Prompt Canvas",
                height=PROMPT_CANVAS_HEIGHT,
                sources=[],
                interactive=True,
                show_download_button=False,
                show_fullscreen_button=False,
                placeholder="Input Image に画像をアップロードすると、ここに編集用コピーが表示されます。",
            )
            gr.Markdown(
                "1. Upload image. 2. Use Text Prompt or click Prompt Canvas. "
                "3. Predict Candidate Masks. 4. Run transparent-background. Union is optional."
            )
            with gr.Accordion("Optional: Text Prompt to Box", open=False):
                text_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="person playing drums / person riding bicycle",
                    info="例: person playing drums / person riding bicycle。GroundingDINO で bbox 候補を作ります。",
                )
                box_threshold = gr.Slider(
                    0.0, 1.0, value=0.25, step=0.01, label="Box Threshold",
                    info="GroundingDINO の物体信頼度しきい値（0〜1 のスコア、単位なし）。高いほど検出が厳しく誤検出が減るが取りこぼしが増える。低いほど候補が増える。目安 0.25。",
                )
                text_threshold = gr.Slider(
                    0.0, 1.0, value=0.25, step=0.01, label="Text Threshold",
                    info="検出 box と text 語句の一致しきい値（0〜1、単位なし）。高いほど語句との合致を厳しく要求する。目安 0.25。",
                )
                iou_threshold = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="NMS IoU Threshold",
                    info="重複 box を間引く NMS の IoU しきい値（0〜1、単位なし）。低いほど重なりを強く除去し重複検出が減る。高いほど近接 box を残す。目安 0.5。",
                )
                top_k = gr.Slider(
                    1, 10, value=5, step=1, label="Top-K Text Boxes",
                    info="保持する検出 box の最大個数（個数、整数）。スコア上位から K 個まで採用する。複数対象を union する時は大きめにする。目安 5。",
                )
                detect_button = gr.Button("Detect Text Boxes", variant="secondary")
                detected_boxes = gr.Dataframe(headers=["index", "confidence", "phrase", "box"], label="Detected Text Boxes")
            with gr.Accordion("SAM2 Candidate Masks", open=True):
                prompt_mode = gr.Radio(
                    choices=["point", "box"],
                    value="box",
                    label="Prompt Mode",
                    info="point はクリック位置を前景/除外として指定し、box は2クリックで範囲を指定します。",
                )
                point_label = gr.Radio(
                    choices=["positive", "negative"],
                    value="positive",
                    label="Point Label (positive=対象 / negative=除外)",
                    info="positive は残したい領域、negative は除外したい領域のヒントです。",
                    visible=False,
                )
                multimask = gr.Checkbox(
                    value=True, label="Multimask",
                    info="ON: SAM2 が候補 mask を最大3枚返し、Candidate から選べる。OFF: スコア最上位1枚のみ。真偽値。",
                )
                gr.Markdown("Box は2クリックで確定。被写体が画面端に接している時は Extend Box Edge を使います。")
                with gr.Row():
                    extend_left_button = gr.Button("Extend Left")
                    extend_right_button = gr.Button("Extend Right")
                with gr.Row():
                    extend_top_button = gr.Button("Extend Top")
                    extend_bottom_button = gr.Button("Extend Bottom")
                clear_button = gr.Button("Clear SAM2 Prompt")
                sam2_button = gr.Button("Predict SAM2 Candidate Masks", variant="primary")
            with gr.Accordion("Optional: Mask Union for composite subjects", open=False):
                candidate_indices = gr.Textbox(
                    label="Candidate Mask Indices",
                    value="0",
                    info="union したい候補 mask の index を 0,1,2 のようにカンマ区切りで入力。index は Candidate Masks の番号（0 起点の整数）。例: person と drums を結合するなら 0,1。",
                )
                union_dilate_kernel = gr.Slider(
                    0, 51, value=0, step=1, label="Union Dilate Kernel",
                    info="union mask を太らせる膨張カーネルサイズ（px、奇数推奨）。0 で膨張なし。値が大きいほどマスク間の隔たりを埋めてつなげるが輪郭が太る。目安 0〜15。",
                )
                union_min_area = gr.Slider(
                    0, 5000, value=0, step=1, label="Union Min Area",
                    info="union 結果から除去する微小領域の面積しきい値（px²、ピクセル数）。0 で除去なし。値以下の島ノイズを消す。目安 0〜500。",
                )
                with gr.Row():
                    add_union_button = gr.Button("Add to Union", variant="primary")
                    remove_union_button = gr.Button("Remove from Union")
                    clear_union_button = gr.Button("Clear Union")
            run_button = gr.Button("Run transparent-background", variant="primary")
            with gr.Accordion("Advanced: transparent-background settings", open=False):
                mask_source = gr.Radio(
                    choices=["Union Mask", "Best Candidate Mask", "No SAM2 Mask"],
                    value="Best Candidate Mask",
                    label="Mask Source for MatteExtractor",
                    info="通常は Best Candidate Mask のままで進めます。複合対象を統合した時だけ Union Mask を選びます。",
                )
                background_model = gr.Dropdown(
                    choices=build_dropdown_choices("background"),
                    value="tb_base",
                    label="背景除去モデル（Background Model）",
                    info="transparent-background モデルを選択します。切替時は再初期化が走ります。tb_base: 標準品質（推奨）。tb_fast: 軽量・高速だが精度低。tb_base_nightly: 最新実験版 base。単位なしの選択値。",
                )
                tb_jit = gr.Checkbox(
                    value=False, label="JIT",
                    info="ON: TorchScript JIT で推論を高速化（初回コンパイルに時間）。OFF: 通常実行。真偽値。",
                )
                tb_threshold = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01, label="Threshold",
                    info="アルファを二値化するしきい値（0.0〜1.0 の正規化アルファ、単位なし）。0.0: 二値化せず髪などのソフトな半透明を残す。値を上げるほどその値未満を透明にし輪郭が硬くなる。目安 0.0。",
                )
                tb_output_type = gr.Radio(
                    choices=["rgba", "green", "white", "blur"], value="rgba", label="Preview",
                    info="プレビュー背景の表示方法。rgba: 透明（市松模様）。green: 緑背景。white: 白背景。blur: 元背景をぼかす。出力 RGBA 本体には影響しない。単位なしの選択値。",
                )
                crop_padding = gr.Slider(
                    0, 64, value=5, step=1, label="Crop Padding",
                    info="SAM2 mask の外接矩形の外側に加える余白（px、整数）。主目的は髪・手足先など細部の検出漏れ防止。2K 解像度基準で目安 5px。大きすぎると mask が背景を巻き込み壊れるため、細部が切れる時のみ 10〜30 に増やす。",
                )
        with gr.Column(scale=1):
            display_size = gr.Radio(
                choices=["window", "original"], value="window", label="Image Display Size",
                info="プレビュー画像の表示サイズ。window: ウィンドウに収まるよう縮小表示。original: 等倉大で表示。出力画像の解像度には影響しない。単位なしの選択値。",
            )
            sam2_status = gr.Textbox(label="SAM2 Status")
            union_status = gr.Textbox(label="Union Status")
            candidate_table = gr.Dataframe(headers=["index", "score", "label", "box"], label="SAM2 Candidate Mask Table")
            candidate_preview = gr.Image(label="SAM2 Candidate Masks", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            union_preview = gr.Image(label="Union Mask Preview", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            rgba_output = gr.Image(label="RGBA", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            alpha_output = gr.Image(label="Alpha", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            preview_output = gr.Image(label="Preview", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            paths_output = gr.Textbox(label="Saved Paths")
            bbox_output = gr.Textbox(label="Mask BBox")

    input_image.change(
        sync_prompt_canvas_outputs,
        inputs=[input_image],
        outputs=[prompt_canvas, candidate_preview, union_preview, candidate_table, prompt_state, sam2_status],
    )
    prompt_canvas.select(
        select_sam2_prompt,
        inputs=[input_image, prompt_mode, point_label, prompt_state],
        outputs=[prompt_canvas, prompt_state, sam2_status],
    )
    prompt_mode.change(update_point_label_visibility, inputs=[prompt_mode], outputs=[point_label])
    for button, side in (
        (extend_left_button, "left"),
        (extend_right_button, "right"),
        (extend_top_button, "top"),
        (extend_bottom_button, "bottom"),
    ):
        button.click(
            extend_box_to_edge,
            inputs=[input_image, prompt_state, gr.State(side)],
            outputs=[prompt_canvas, prompt_state, sam2_status],
        )
    clear_button.click(
        clear_sam2_prompt,
        inputs=[input_image, prompt_state],
        outputs=[prompt_canvas, candidate_preview, union_preview, candidate_table, prompt_state, sam2_status],
    )
    detect_button.click(
        detect_text_boxes,
        inputs=[input_image, text_prompt, box_threshold, text_threshold, iou_threshold, top_k, prompt_state],
        outputs=[prompt_canvas, detected_boxes, prompt_state, sam2_status],
    )
    sam2_button.click(
        predict_masks,
        inputs=[input_image, prompt_state, multimask],
        outputs=[prompt_canvas, candidate_preview, union_preview, candidate_table, sam2_status, prompt_state],
    )
    add_union_button.click(
        add_candidates_to_union,
        inputs=[input_image, candidate_indices, union_dilate_kernel, union_min_area],
        outputs=[union_preview, union_status],
    )
    remove_union_button.click(
        remove_candidates_from_union,
        inputs=[input_image, candidate_indices, union_dilate_kernel, union_min_area],
        outputs=[union_preview, union_status],
    )
    clear_union_button.click(clear_union_mask, inputs=[input_image], outputs=[union_preview, union_status])
    run_button.click(
        run_transparent_bg,
        inputs=[input_image, mask_source, background_model, tb_jit, tb_threshold, tb_output_type, crop_padding],
        outputs=[rgba_output, alpha_output, preview_output, paths_output, bbox_output],
    )
    display_size.change(
        update_image_display_size,
        inputs=[display_size],
        outputs=[prompt_canvas, candidate_preview, union_preview, rgba_output, alpha_output, preview_output],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 + transparent-background Haystack Gradio demo")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server host")
    parser.add_argument("--server-port", type=int, default=7862, help="Server port")
    args = parser.parse_args()

    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        show_api=False,
    )
