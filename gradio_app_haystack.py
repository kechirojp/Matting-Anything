"""Haystack Pipeline 版 Matting-Anything Gradio デモ。"""

from __future__ import annotations

import argparse

import gradio as gr

from pipelines.mam_pipeline import build_mam_pipeline, build_mam_scribble_pipeline


TEXT_PIPELINE = None
SCRIBBLE_PIPELINE = None


def get_text_pipeline():
    """テキストガイド Pipeline を遅延構築する。"""
    global TEXT_PIPELINE
    if TEXT_PIPELINE is None:
        TEXT_PIPELINE = build_mam_pipeline()
    return TEXT_PIPELINE


def get_scribble_pipeline():
    """スクリブルガイド Pipeline を遅延構築する。"""
    global SCRIBBLE_PIPELINE
    if SCRIBBLE_PIPELINE is None:
        SCRIBBLE_PIPELINE = build_mam_scribble_pipeline()
    return SCRIBBLE_PIPELINE


def run_haystack_mam(
    input_image,
    text_prompt: str,
    task_type: str,
    background_prompt: str,
    background_type: str,
    box_threshold: float,
    text_threshold: float,
    iou_threshold: float,
    scribble_mode: str,
    guidance_mode: str,
):
    """Gradio 入力を Haystack Pipeline に渡し、表示用結果を返す。"""
    try:
        if task_type == "text":
            result = get_text_pipeline().run(
                {
                    "image_normalizer": {"image": input_image},
                    "grounding_dino": {
                        "text_prompt": text_prompt,
                        "box_threshold": box_threshold,
                        "text_threshold": text_threshold,
                        "iou_threshold": iou_threshold,
                    },
                    "alpha_predictor": {"task_type": "text", "guidance_mode": guidance_mode},
                    "background_generator": {
                        "background_type": background_type,
                        "background_prompt": background_prompt,
                    },
                    "output_saver": {"enabled": True},
                }
            )
        else:
            parser_mode = "box" if task_type == "scribble_box" else "point"
            result = get_scribble_pipeline().run(
                {
                    "image_normalizer": {"image": input_image},
                    "scribble_parser": {"editor_value": input_image, "mode": parser_mode},
                    "alpha_predictor": {
                        "task_type": task_type,
                        "scribble_mode": scribble_mode,
                        "guidance_mode": guidance_mode,
                    },
                    "background_generator": {
                        "background_type": background_type,
                        "background_prompt": background_prompt,
                    },
                    "output_saver": {"enabled": True},
                }
            )
        background_result = result["background_generator"]
        alpha_result = result["alpha_predictor"]
        saver_result = result.get("output_saver", {})
        return (
            alpha_result["alpha_rgb"],
            background_result["green_screen"],
            background_result["composite"],
            str(saver_result.get("paths", {})),
        )
    except Exception as exc:
        raise gr.Error(f"Haystack 推論に失敗しました: {exc}") from exc


with gr.Blocks(title="Matting-Anything Haystack") as demo:
    gr.Markdown("# Matting-Anything Haystack")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.ImageEditor(type="numpy", label="Input / Scribble", height=420)
            task_type = gr.Radio(choices=["text", "scribble_box", "scribble_point"], value="text", label="Task")
            text_prompt = gr.Textbox(value="person", label="Text Prompt")
            with gr.Accordion("Advanced", open=False):
                background_type = gr.Radio(
                    choices=["real_world_sample", "stable_diffusion"],
                    value="real_world_sample",
                    label="Background Type",
                )
                background_prompt = gr.Textbox(value="a clean studio background", label="Background Prompt")
                box_threshold = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Box Threshold")
                text_threshold = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Text Threshold")
                iou_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="NMS IoU Threshold")
                scribble_mode = gr.Radio(choices=["split", "merge"], value="split", label="Scribble Mode")
                guidance_mode = gr.Radio(choices=["alpha", "mask"], value="alpha", label="Guidance Mode")
            run_button = gr.Button("Run Haystack Pipeline", variant="primary")
        with gr.Column(scale=1):
            alpha_output = gr.Image(label="Alpha", type="numpy")
            green_output = gr.Image(label="Green Screen", type="numpy")
            composite_output = gr.Image(label="Composite", type="numpy")
            paths_output = gr.Textbox(label="Saved Paths")

    run_button.click(
        run_haystack_mam,
        inputs=[
            input_image,
            text_prompt,
            task_type,
            background_prompt,
            background_type,
            box_threshold,
            text_threshold,
            iou_threshold,
            scribble_mode,
            guidance_mode,
        ],
        outputs=[alpha_output, green_output, composite_output, paths_output],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matting-Anything Haystack Gradio demo")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server host")
    parser.add_argument("--server-port", type=int, default=7861, help="Server port")
    args = parser.parse_args()

    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share, debug=args.debug)