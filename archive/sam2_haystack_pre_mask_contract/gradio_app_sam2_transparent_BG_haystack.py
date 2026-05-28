"""Haystack Pipeline 版 SAM2 + transparent-background Gradio デモ。"""

from __future__ import annotations

import argparse
import warnings

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

from pipelines.components.common import ensure_rgb_array, mask_to_bbox
from pipelines.sam2_tb_pipeline import build_sam2_prompt_pipeline, build_sam2_tb_pipeline


SAM2_PIPELINE = None
TB_PIPELINE = None
SAM2_STATE: dict[str, np.ndarray] = {}
# 端付近の click を画像端に吸着させる範囲（px）。bbox を画面端まで届かせるための余裕。
EDGE_SNAP_PIXELS = 16
# bbox のどの辺を画像端へ延長するか指定するラベル。
EDGE_SIDES = ("left", "right", "top", "bottom")
PROMPT_CANVAS_HEIGHT = 420
OUTPUT_WINDOW_HEIGHT = 360
PROMPT_CANVAS_PLACEHOLDER_SIZE = (420, 640)


def empty_prompt_state() -> dict[str, object]:
    """SAM2 prompt の UI state を初期化する。"""
    return {
        "points": [],
        "labels": [],
        "box": None,
        "box_buffer": [],
        "input_mode": "point",
    }


def _copy_prompt_state(prompt_state: dict | None) -> dict[str, object]:
    state = empty_prompt_state()
    if prompt_state:
        state.update(prompt_state)
        state["points"] = list(state.get("points") or [])
        state["labels"] = list(state.get("labels") or [])
        state["box_buffer"] = list(state.get("box_buffer") or [])
        state["box"] = list(state["box"]) if state.get("box") is not None else None
    return state


def clamp_prompt_point(
    point: tuple[int, int] | list[int],
    image_shape: tuple[int, ...],
    edge_snap: int = EDGE_SNAP_PIXELS,
) -> tuple[int, int]:
    """クリック座標を画像内に収め、端付近は画像端へ吸着する。"""
    image_height, image_width = image_shape[:2]
    x_value = int(round(float(point[0])))
    y_value = int(round(float(point[1])))
    x_value = min(max(x_value, 0), image_width - 1)
    y_value = min(max(y_value, 0), image_height - 1)
    if x_value <= edge_snap:
        x_value = 0
    elif x_value >= image_width - 1 - edge_snap:
        x_value = image_width - 1
    if y_value <= edge_snap:
        y_value = 0
    elif y_value >= image_height - 1 - edge_snap:
        y_value = image_height - 1
    return x_value, y_value


def normalize_box_from_points(
    first_point: tuple[int, int] | list[int],
    second_point: tuple[int, int] | list[int],
    image_shape: tuple[int, ...],
) -> list[int]:
    """2クリックから順序に依存しない SAM2 bbox を作る。"""
    x_first, y_first = clamp_prompt_point(first_point, image_shape)
    x_second, y_second = clamp_prompt_point(second_point, image_shape)
    return [min(x_first, x_second), min(y_first, y_second), max(x_first, x_second), max(y_first, y_second)]


def draw_prompt_overlay(
    input_image,
    prompt_state: dict | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """SAM2 prompt の点・bbox・マスクを画像に重ねる。"""
    image_rgb = ensure_rgb_array(input_image).copy()
    state = _copy_prompt_state(prompt_state)
    if mask is not None:
        mask_bool = mask.astype(bool)
        overlay_color = np.array([30, 144, 255], dtype=np.uint8)
        image_rgb[mask_bool] = (image_rgb[mask_bool] * 0.5 + overlay_color * 0.5).astype(np.uint8)
    if state.get("box") is not None:
        x_min, y_min, x_max, y_max = [int(value) for value in state["box"]]
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 215, 0), 3)
    for point, label in zip(state.get("points") or [], state.get("labels") or []):
        x_value, y_value = [int(value) for value in point]
        color = (0, 255, 0) if int(label) == 1 else (255, 0, 0)
        cv2.circle(image_rgb, (x_value, y_value), 7, color, -1)
        cv2.circle(image_rgb, (x_value, y_value), 9, (255, 255, 255), 2)
    for point in state.get("box_buffer") or []:
        x_value, y_value = [int(value) for value in point]
        cv2.circle(image_rgb, (x_value, y_value), 7, (255, 215, 0), -1)
        cv2.circle(image_rgb, (x_value, y_value), 9, (255, 255, 255), 2)
    return image_rgb


def create_prompt_canvas_placeholder() -> np.ndarray:
    """SAM2 prompt canvas の初期表示用プレースホルダーを作る。"""
    height, width = PROMPT_CANVAS_PLACEHOLDER_SIZE
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (1, 1), (width - 2, height - 2), (220, 225, 232), 2)
    cv2.putText(
        canvas,
        "SAM2 Prompt Canvas",
        (36, height // 2 - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (70, 78, 92),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Upload an image in Input Image above.",
        (36, height // 2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (105, 113, 128),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Then click here to place points or box corners.",
        (36, height // 2 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (105, 113, 128),
        2,
        cv2.LINE_AA,
    )
    return canvas


def get_sam2_pipeline():
    """SAM2 prompt Pipeline を遅延構築する。"""
    global SAM2_PIPELINE
    if SAM2_PIPELINE is None:
        SAM2_PIPELINE = build_sam2_prompt_pipeline()
    return SAM2_PIPELINE


def get_tb_pipeline():
    """transparent-background Pipeline を遅延構築する。"""
    global TB_PIPELINE
    if TB_PIPELINE is None:
        TB_PIPELINE = build_sam2_tb_pipeline(include_sam2=False)
    return TB_PIPELINE


def select_sam2_prompt(input_image, prompt_mode: str, point_label: str | bool, prompt_state: dict | None, evt: gr.SelectData):
    """画像クリックで SAM2 point / bbox prompt を蓄積する。

    `point_label` は Radio 値 "positive" / "negative" を想定するが、
    後方互換のため bool も受け付ける（True=positive）。
    """
    if input_image is None:
        raise gr.Error("先に画像をアップロードしてください。")
    image_rgb = ensure_rgb_array(input_image)
    state = _copy_prompt_state(prompt_state)
    if state.get("input_mode") != prompt_mode:
        state["box_buffer"] = []
    state["input_mode"] = prompt_mode
    clicked_point = clamp_prompt_point(evt.index, image_rgb.shape)

    # Radio ("positive" / "negative") と bool の両方を受理する
    if isinstance(point_label, str):
        is_positive = point_label.lower() == "positive"
    else:
        is_positive = bool(point_label)

    if prompt_mode == "point":
        state["points"].append(clicked_point)
        state["labels"].append(1 if is_positive else 0)
        status = f"Point selected: {clicked_point}, label={'positive' if is_positive else 'negative'}"
    else:
        state["box_buffer"].append(clicked_point)
        if len(state["box_buffer"]) >= 2:
            state["box"] = normalize_box_from_points(state["box_buffer"][0], state["box_buffer"][1], image_rgb.shape)
            state["box_buffer"] = []
            status = f"Box selected: {state['box']}"
        else:
            status = f"Box first corner selected: {clicked_point}"
    return draw_prompt_overlay(image_rgb, state), state, status


def extend_box_to_edge(input_image, prompt_state: dict | None, side: str):
    """確定済み bbox の指定辺を画像端へ延長する。

    Gradio の click イベントは画像の外側を拾えないため、被写体が画面端に
    接している場合に bbox を端まで届かせる手段としてエッジボタンを提供する。
    """
    if input_image is None:
        raise gr.Error("先に画像をアップロードしてください。")
    if side not in EDGE_SIDES:
        raise gr.Error(f"未知のエッジ指定です: {side}")
    image_rgb = ensure_rgb_array(input_image)
    image_height, image_width = image_rgb.shape[:2]
    state = _copy_prompt_state(prompt_state)
    if state.get("box") is None:
        raise gr.Error("先に画像を2回クリックして bbox を作成してから端を延長してください。")
    x_min, y_min, x_max, y_max = [int(value) for value in state["box"]]
    if side == "left":
        x_min = 0
    elif side == "right":
        x_max = image_width - 1
    elif side == "top":
        y_min = 0
    elif side == "bottom":
        y_max = image_height - 1
    state["box"] = [x_min, y_min, x_max, y_max]
    preview = draw_prompt_overlay(image_rgb, state, SAM2_STATE.get("mask"))
    status = f"Box {side} edge extended → {state['box']}"
    return preview, state, status


def clear_sam2_prompt(input_image, prompt_state: dict | None):
    """SAM2 prompt と候補マスクをクリアする。"""
    state = empty_prompt_state()
    SAM2_STATE.clear()
    preview = ensure_rgb_array(input_image) if input_image is not None else create_prompt_canvas_placeholder()
    return preview, state, "SAM2 prompt cleared"


def sync_prompt_canvas(input_image):
    """アップロード画像を SAM2 prompt 編集用キャンバスへ反映する。"""
    state = empty_prompt_state()
    SAM2_STATE.clear()
    if input_image is None:
        return create_prompt_canvas_placeholder(), state, "Input image is empty."
    # 入力画像は推論用の原本として保持し、prompt 操作用には別キャンバスへコピーする。
    prompt_canvas = ensure_rgb_array(input_image).copy()
    return prompt_canvas, state, "SAM2 Prompt Canvas ready. Click this canvas to place points or box corners."


def update_image_display_size(display_size: str):
    """prompt / 予測画像の表示高さをウィンドウサイズと原寸で切り替える。"""
    if display_size == "original":
        return (gr.update(height=None), gr.update(height=None), gr.update(height=None), gr.update(height=None))
    return (
        gr.update(height=PROMPT_CANVAS_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
        gr.update(height=OUTPUT_WINDOW_HEIGHT),
    )


def predict_masks(input_image, prompt_state: dict | None, multimask: bool):
    """SAM2 Component で候補マスクを生成する。"""
    try:
        state = _copy_prompt_state(prompt_state)
        points = state.get("points") or []
        labels = state.get("labels") or []
        box = state.get("box")
        if not points and box is None:
            raise gr.Error("画像をクリックして Point または Box を指定してください。")
        segmenter_input = {"multimask": multimask}
        if points:
            segmenter_input["points"] = points
            segmenter_input["labels"] = labels
        if box is not None:
            segmenter_input["box"] = box
        result = get_sam2_pipeline().run(
            {
                "image_normalizer": {"image": input_image},
                "sam2_segmenter": segmenter_input,
            }
        )
        masks = result["sam2_segmenter"]["masks"]
        scores = result["sam2_segmenter"]["scores"]
        best_index = int(np.argmax(scores))
        SAM2_STATE["mask"] = masks[best_index].astype(bool)
        preview = draw_prompt_overlay(input_image, state, SAM2_STATE["mask"])
        return preview, f"selected={best_index}, score={float(scores[best_index]):.4f}", state
    except Exception as exc:
        raise gr.Error(f"SAM2 推論に失敗しました: {exc}") from exc


def run_transparent_bg(
    input_image,
    use_sam2_mask: bool,
    tb_mode: str,
    tb_jit: bool,
    tb_threshold: float,
    tb_output_type: str,
    crop_padding: int,
):
    """tb Component を実行し、必要に応じて SAM2 mask を guard に使う。"""
    try:
        mask = SAM2_STATE.get("mask") if use_sam2_mask else None
        result = get_tb_pipeline().run(
            {
                "image_normalizer": {"image": input_image},
                "transparent_bg": {
                    "mask": mask,
                    "tb_mode": tb_mode,
                    "tb_jit": tb_jit,
                    "tb_threshold": tb_threshold,
                    "tb_output_type": tb_output_type,
                    "crop_padding": int(crop_padding),
                },
                "sam2_guard": {"mask": mask, "enabled": use_sam2_mask},
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
    except Exception as exc:
        raise gr.Error(f"transparent-background 推論に失敗しました: {exc}") from exc


with gr.Blocks(title="SAM2 + Transparent Background Haystack") as demo:
    gr.Markdown("# SAM2 + Transparent Background Haystack")
    prompt_state = gr.State(value=empty_prompt_state())
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="numpy",
                label="Input Image",
                height=300,
                interactive=True,
            )
            # SAM2 prompt は入力画像とは別の編集キャンバスで扱い、アップロード欄の役割を混ぜない。
            prompt_canvas = gr.Image(
                value=create_prompt_canvas_placeholder(),
                type="numpy",
                label="SAM2 Prompt Canvas",
                height=PROMPT_CANVAS_HEIGHT,
                sources=[],
                interactive=False,
                show_download_button=False,
                show_fullscreen_button=False,
                placeholder="Input Image に画像をアップロードすると、ここに編集用コピーが表示されます。",
            )
            gr.Markdown(
                "1. Upload an image above. 2. Click the SAM2 Prompt Canvas for points or box corners. "
                "3. Predict the SAM2 mask. 4. Run transparent-background."
            )
            with gr.Accordion("SAM2", open=True):
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
                )
                multimask = gr.Checkbox(
                    value=True,
                    label="Multimask",
                    info="ON では SAM2 が複数候補を出し、スコアが最も高い mask を採用します。",
                )
                gr.Markdown(
                    "Box は2クリックで確定。被写体が画面端に接している時は"
                    "下の **Extend Box Edge** で対応する辺を画像端まで延長してください。"
                )
                with gr.Row():
                    extend_left_button = gr.Button("⬅ Extend Left")
                    extend_right_button = gr.Button("Extend Right ➡")
                with gr.Row():
                    extend_top_button = gr.Button("⬆ Extend Top")
                    extend_bottom_button = gr.Button("Extend Bottom ⬇")
                clear_button = gr.Button("Clear SAM2 Prompt")
                sam2_button = gr.Button("Predict SAM2 Mask", variant="primary")
            with gr.Accordion("transparent-background", open=True):
                use_sam2_mask = gr.Checkbox(
                    value=True,
                    label="Use SAM2 Mask",
                    info="ON では SAM2 mask の範囲を優先して背景除去します。OFF では画像全体を処理します。",
                )
                tb_mode = gr.Radio(
                    choices=["base", "fast", "base-nightly"],
                    value="base",
                    label="Mode",
                    info="base は品質優先、fast は速度優先、base-nightly は新しい重みを使うモードです。",
                )
                tb_jit = gr.Checkbox(value=False, label="JIT", info="TorchScript による高速化を試します。初回は遅くなる場合があります。")
                tb_threshold = gr.Slider(
                    0.0,
                    1.0,
                    value=0.0,
                    step=0.01,
                    label="Threshold",
                    info="0.0 は柔らかい alpha を保持し、値を上げるほど透明/不透明を強く分けます。",
                )
                tb_output_type = gr.Radio(
                    choices=["rgba", "green", "white", "blur"],
                    value="rgba",
                    label="Preview",
                    info="確認用プレビューの背景表現を切り替えます。保存される RGBA の alpha とは別です。",
                )
                crop_padding = gr.Slider(
                    0,
                    160,
                    value=40,
                    step=1,
                    label="Crop Padding",
                    info="SAM2 mask の周囲を何 px 広げて tb に渡すか。髪や輪郭が切れる場合は増やします。",
                )
                run_button = gr.Button("Run Haystack Pipeline", variant="primary")
        with gr.Column(scale=1):
            display_size = gr.Radio(
                choices=["window", "original"],
                value="window",
                label="Image Display Size",
                info="window は画面内に収まる高さで表示し、original は画像の原寸表示に切り替えます。",
            )
            sam2_status = gr.Textbox(label="SAM2 Status")
            rgba_output = gr.Image(label="RGBA", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            alpha_output = gr.Image(label="Alpha", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            preview_output = gr.Image(label="Preview", type="numpy", height=OUTPUT_WINDOW_HEIGHT)
            paths_output = gr.Textbox(label="Saved Paths")
            bbox_output = gr.Textbox(label="Mask BBox")

    input_image.change(
        sync_prompt_canvas,
        inputs=[input_image],
        outputs=[prompt_canvas, prompt_state, sam2_status],
    )
    prompt_canvas.select(
        select_sam2_prompt,
        inputs=[input_image, prompt_mode, point_label, prompt_state],
        outputs=[prompt_canvas, prompt_state, sam2_status],
    )
    # 4 つのエッジ延長ボタン: 確定済み bbox の辺を画像端へ届かせる
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
        outputs=[prompt_canvas, prompt_state, sam2_status],
    )
    sam2_button.click(
        predict_masks,
        inputs=[input_image, prompt_state, multimask],
        outputs=[prompt_canvas, sam2_status, prompt_state],
    )
    run_button.click(
        run_transparent_bg,
        inputs=[input_image, use_sam2_mask, tb_mode, tb_jit, tb_threshold, tb_output_type, crop_padding],
        outputs=[rgba_output, alpha_output, preview_output, paths_output, bbox_output],
    )
    display_size.change(
        update_image_display_size,
        inputs=[display_size],
        outputs=[prompt_canvas, rgba_output, alpha_output, preview_output],
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
        show_api=False,  # API 表示は隠し、/info 自体の crash は上の schema patch で防ぐ。
    )