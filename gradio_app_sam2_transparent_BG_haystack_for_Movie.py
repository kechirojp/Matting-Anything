"""Haystack Pipeline 版 SAM2 + transparent-background 動画背景除去デモ。"""

from __future__ import annotations

import argparse
import gc
import time
import warnings
from pathlib import Path
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

import gradio as gr

from pipelines.components.model_components import GroundingDINOMultiBoxDetector
from pipelines.components.ui_helpers import (
    copy_prompt_state,
    draw_prompt_overlay,
    empty_prompt_state,
    extend_box_to_edge,
    select_sam2_prompt,
)
from pipelines.components.video_common import normalize_output_mode
from pipelines.components.model_registry import build_dropdown_choices, entry_by_id
from pipelines.sam2_tb_video_pipeline import build_sam2_tb_video_pipeline, build_video_reader_pipeline


READER_PIPELINE = None
_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}
TEXT_DETECTOR = None
PROMPT_CANVAS_PLACEHOLDER_SIZE = (420, 640)
OUTPUT_MODE_CHOICES = ["動画 (video)", "連番静止画 (sequence)", "両方 (both)"]
STAGE_PROGRESS_RANGES = {
    "video_reader": (0.05, 0.15),
    "sam2_video": (0.15, 0.55),
    "transparent_bg": (0.55, 0.88),
    "video_writer": (0.88, 0.96),
    "frame_sequence_writer": (0.88, 0.96),
    "tracking_overlay": (0.88, 0.96),
}


def get_reader_pipeline():
    """第 1 フレーム抽出用 Pipeline を遅延構築する。"""
    global READER_PIPELINE
    if READER_PIPELINE is None:
        READER_PIPELINE = build_video_reader_pipeline()
    return READER_PIPELINE


def get_video_pipeline(tracker_model: str = "sam2_hiera_l", background_model: str = "tb_base"):
    """動画背景除去 Pipeline を遅延構築する。"""
    key = (tracker_model, background_model)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = build_sam2_tb_video_pipeline()
    return _PIPELINE_CACHE[key]


def get_text_detector():
    """GroundingDINO Text Prompt 検出器をボタン押下時まで遅延構築する。"""
    global TEXT_DETECTOR
    if TEXT_DETECTOR is None:
        TEXT_DETECTOR = GroundingDINOMultiBoxDetector()
    return TEXT_DETECTOR


def release_text_detector() -> None:
    """動画処理前に GroundingDINO / BERT cache を解放し、Colab RAM ピークを下げる。"""
    global TEXT_DETECTOR
    TEXT_DETECTOR = None
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_prompt_canvas_placeholder():
    """動画 prompt canvas の初期表示用プレースホルダーを作る。"""
    import cv2
    import numpy as np

    height, width = PROMPT_CANVAS_PLACEHOLDER_SIZE
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (1, 1), (width - 2, height - 2), (220, 225, 232), 2)
    cv2.putText(canvas, "SAM2 Video Prompt", (36, height // 2 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 78, 92), 2)
    cv2.putText(canvas, "Load a frame, then click here.", (36, height // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (105, 113, 128), 2)
    cv2.putText(canvas, "Pick the prompt frame below; multiple boxes are unioned.", (36, height // 2 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (105, 113, 128), 2)
    return canvas


def extract_first_frame(video_path: str | None):
    """入力動画から第 1 フレームを取り出し、prompt canvas に表示する。"""
    try:
        if not video_path:
            raise gr.Error("先に動画をアップロードしてください。")
        result = get_reader_pipeline().run(
            {"video_reader": {"video_path": video_path, "max_frames": 1, "frame_step": 1}},
            include_outputs_from={"video_reader"},
        )
        frame = result["video_reader"]["frames"][0]
        metadata = result["video_reader"]["metadata"]
        state = empty_prompt_state()
        status = f"第 1 フレームを取得しました: {metadata['width']}x{metadata['height']}, fps={metadata['fps']:.2f}"
        return frame, state, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"第 1 フレームの取得に失敗しました: {exc}") from exc


def extract_first_frame_outputs(video_path: str | None):
    """動画アップロード時に第 1 フレームを自動で prompt canvas へ反映する。"""
    if not video_path:
        return create_prompt_canvas_placeholder(), empty_prompt_state(), "動画をアップロードすると第 1 フレームを自動取得します。"
    return extract_first_frame(video_path)


def clear_prompt(first_frame):
    """SAM2 video prompt を初期化する。"""
    state = empty_prompt_state()
    preview = first_frame if first_frame is not None else create_prompt_canvas_placeholder()
    return preview, state, "SAM2 video prompt cleared"


def detect_text_boxes_for_video(
    first_frame,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    top_k: int,
    prompt_state: dict | None,
):
    """第 1 フレームで Text Prompt 物体検出を行い、top bbox を SAM2 prompt に反映する。"""
    try:
        if first_frame is None:
            raise gr.Error("先に動画をアップロードして第 1 フレームを取得してください。")
        if not text_prompt or not text_prompt.strip():
            raise gr.Error("Text Prompt を入力してください。例: person playing drums / person riding bicycle")
        result = get_text_detector().run(
            image=first_frame,
            text_prompt=text_prompt.strip(),
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            top_k=int(top_k),
        )
        boxes = result["boxes"]
        phrases = result["phrases"]
        confidences = result["confidences"]
        if len(boxes) == 0:
            raise gr.Error("Text Prompt に一致する候補が見つかりませんでした。")
        state = copy_prompt_state(prompt_state)
        state["box"] = [int(round(float(value))) for value in boxes[0].tolist()]
        state["box_buffer"] = []
        rows = []
        candidate_boxes = []
        for index, box in enumerate(boxes):
            phrase = phrases[index] if index < len(phrases) else text_prompt.strip()
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            box_values = [int(round(float(value))) for value in box.tolist()]
            candidate_boxes.append(box_values)
            rows.append([index + 1, phrase, f"{confidence:.3f}", ", ".join(str(value) for value in box_values)])
        # 複合対象 union 用に全候補 bbox を保持する（apply_selected_boxes で部分採用される）。
        state["boxes"] = candidate_boxes
        preview = draw_prompt_overlay(first_frame, state)
        status = f"GroundingDINO detected {len(rows)} candidate(s). Top bbox copied to SAM2 prompt: {state['box']}"
        return preview, state, rows, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Text Prompt 検出に失敗しました: {exc}") from exc


def _candidate_choice_label(rank: int, phrase: str, box_values: list[int]) -> str:
    """CheckboxGroup 用に候補 bbox を一意な文字列ラベルへ変換する。"""
    bbox_text = ",".join(str(int(value)) for value in box_values)
    return f"#{rank} {phrase} [{bbox_text}]"


def populate_candidate_choices(detected_rows):
    """検出結果 Dataframe から候補選択 CheckboxGroup の選択肢を作る（top1 を既定 ON）。"""
    rows = list(detected_rows or [])
    labels: list[str] = []
    for row in rows:
        try:
            rank = int(row[0])
            phrase = str(row[1])
            box_values = [int(round(float(value))) for value in str(row[3]).split(",")]
        except (ValueError, IndexError):
            continue
        labels.append(_candidate_choice_label(rank, phrase, box_values))
    default = [labels[0]] if labels else []
    return gr.update(choices=labels, value=default)


def apply_selected_boxes(first_frame, prompt_state: dict | None, selected_labels):
    """選択された候補 bbox を複合対象 union 用に prompt_state["boxes"] へ反映する。"""
    try:
        if first_frame is None:
            raise gr.Error("先に第 1 フレームを取得してください。")
        labels = list(selected_labels or [])
        if not labels:
            raise gr.Error("少なくとも 1 つの候補 bbox を選択してください。")
        selected_boxes: list[list[int]] = []
        for label in labels:
            # ラベル末尾の "[x1,y1,x2,y2]" を抽出する。phrase 内に "[]" を含んでも
            # 最後の括弧群が bbox なので rsplit で確実に取り出す。
            tail = str(label).rsplit("[", 1)
            if len(tail) != 2:
                continue
            bbox_text = tail[1].rstrip("] ").strip()
            try:
                box_values = [int(round(float(value))) for value in bbox_text.split(",")]
            except ValueError:
                continue
            if len(box_values) == 4:
                selected_boxes.append(box_values)
        if not selected_boxes:
            raise gr.Error("選択した候補から bbox を解釈できませんでした。")
        state = copy_prompt_state(prompt_state)
        state["boxes"] = selected_boxes
        # 複合対象 union を使う場合は単一 box を解除して描画と伝搬の二重指定を避ける。
        state["box"] = None
        state["box_buffer"] = []
        preview = draw_prompt_overlay(first_frame, state)
        status = f"複合対象 union を {len(selected_boxes)} 個の bbox で構成しました（各 box を別 obj として追跡し OR 統合）。"
        return preview, state, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"候補 bbox の適用に失敗しました: {exc}") from exc


def extract_prompt_frame(video_path: str | None, prompt_frame_idx: int, frame_step: int):
    """指定したサンプリング後フレーム位置を取り出し、SAM2 prompt の起点フレームにする。"""
    try:
        if not video_path:
            raise gr.Error("先に動画をアップロードしてください。")
        # sampled_index: VideoReader が frame_step でサンプリングした後の frames list 上の位置。
        # 元動画の frame 番号 ≈ sampled_index * frame_step。propagator の prompt_frame_idx も
        # 同じ frames list index として渡すため両者が一致する。
        sampled_index = max(int(prompt_frame_idx), 0)
        result = get_reader_pipeline().run(
            {"video_reader": {"video_path": video_path, "max_frames": sampled_index + 1, "frame_step": int(frame_step)}},
            include_outputs_from={"video_reader"},
        )
        frames = result["video_reader"]["frames"]
        metadata = result["video_reader"]["metadata"]
        frame = frames[min(sampled_index, len(frames) - 1)]
        state = empty_prompt_state()
        raw_index = sampled_index * int(frame_step)
        status = (
            f"プロンプト起点フレームをシーケンス位置 {sampled_index}（元動画フレーム≈{raw_index}）に設定しました。"
            f"このフレームで bbox / point / Text Prompt を作り直してください。"
            f" 画像サイズ {metadata['width']}x{metadata['height']}。"
        )
        return frame, state, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"プロンプトフレームの取得に失敗しました: {exc}") from exc


def _sequence_preview_files(result: dict[str, Any]) -> list[str]:
    """連番出力から Gradio Files 表示用に先頭数枚を返す。"""
    files: list[str] = []
    for key in ("rgba_sequence_dir", "alpha_sequence_dir", "preview_sequence_dir"):
        directory = result.get(key)
        if directory:
            files.extend(str(path) for path in sorted(Path(directory).glob("*.png"))[:3])
    return files


def _merge_matte_results(*results: dict[str, Any] | None) -> dict[str, Any]:
    """writer ごとの VideoMatteResult を path / dir 優先で統合する。"""
    merged: dict[str, Any] = {}
    for result in results:
        if not result:
            continue
        for key, value in result.items():
            if value is not None or key not in merged:
                merged[key] = value
    return merged


def _estimate_processed_frames(max_frames: int, frame_step: int) -> int:
    """UI 表示用に今回処理する最大 frame 数を見積もる。"""
    return max(1, int(max_frames))


def build_video_progress_callback(progress, stage_state: dict[str, Any]):
    """Component 内部進捗を Gradio 全体進捗に変換する callback を作る。"""

    def _callback(stage: str, fraction: float, description: str) -> None:
        stage_state["name"] = stage
        stage_state["description"] = description
        start, end = STAGE_PROGRESS_RANGES.get(stage, (0.05, 0.95))
        value = start + (end - start) * min(max(float(fraction), 0.0), 1.0)
        progress(value, desc=description)

    return _callback


def run_video_background_removal(
    video_path: str | None,
    prompt_state: dict | None,
    prompt_frame_idx: int,
    bidirectional: bool,
    max_frames: int,
    frame_step: int,
    output_mode_label: str,
    rgba_codec: str,
    tracker_model: str,
    background_model: str,
    tb_jit: bool,
    tb_threshold: float,
    tb_output_type: str,
    crop_padding: int,
    overlay_enabled: bool,
    progress=gr.Progress(),
):
    """動画背景除去 Pipeline を実行し、動画または PNG 連番を出力する。"""
    stage_state: dict[str, Any] = {"name": "startup", "description": "Pipeline を起動しています"}
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    try:
        if not video_path:
            raise gr.Error("先に動画をアップロードしてください。")
        state = copy_prompt_state(prompt_state)
        points = state.get("points") or []
        labels = state.get("labels") or []
        box = state.get("box")
        boxes = state.get("boxes") or []
        if not points and box is None and not boxes:
            raise gr.Error("SAM2 Prompt Canvas 上で point / bbox を指定するか、候補 bbox を選択してください。")
        output_mode = normalize_output_mode(output_mode_label)
        processed_frames = _estimate_processed_frames(int(max_frames), int(frame_step))
        progress_callback = build_video_progress_callback(progress, stage_state)
        release_text_detector()
        progress(
            0.03,
            desc=f"Pipeline を起動しています。初回はモデル読込を含みます（最大 {processed_frames} frames）",
        )
        bg_entry = entry_by_id("background", background_model)
        tb_mode = bg_entry.get("tb_mode", "base")
        result = get_video_pipeline(tracker_model, background_model).run(
            {
                "video_reader": {
                    "video_path": video_path,
                    "max_frames": int(max_frames),
                    "frame_step": int(frame_step),
                    "progress_callback": progress_callback,
                },
                "sam2_video_propagator": {
                    "points": points,
                    "labels": labels,
                    "box": box,
                    "boxes": boxes,
                    "prompt_frame_idx": int(prompt_frame_idx),
                    "bidirectional": bool(bidirectional),
                    "progress_callback": progress_callback,
                },
                "transparent_bg_video": {
                    "output_mode": output_mode,
                    "tb_mode": tb_mode,
                    "tb_jit": bool(tb_jit),
                    "tb_threshold": float(tb_threshold),
                    "tb_output_type": tb_output_type,
                    "crop_padding": int(crop_padding),
                    "rgba_codec": rgba_codec,
                    "progress_callback": progress_callback,
                },
                "video_writer": {"rgba_codec": rgba_codec, "progress_callback": progress_callback},
                "frame_sequence_writer": {"progress_callback": progress_callback},
                "tracking_overlay": {"enabled": bool(overlay_enabled), "progress_callback": progress_callback},
            },
            include_outputs_from={"video_writer", "frame_sequence_writer", "tracking_overlay"},
        )
        progress(0.95, desc="出力を整理しています")
        base = {}
        video_result = result.get("video_writer", {}).get("matte")
        sequence_result = result.get("frame_sequence_writer", {}).get("matte")
        merged = _merge_matte_results(base, video_result, sequence_result)
        overlay_result = result.get("tracking_overlay", {}).get("overlay", {})
        overlay_video_path = overlay_result.get("overlay_video_path")
        sequence_files = _sequence_preview_files(merged)
        sequence_dirs = [merged.get(key) for key in ("rgba_sequence_dir", "alpha_sequence_dir", "preview_sequence_dir") if merged.get(key)]
        fallback = merged.get("metadata", {}).get("codec_fallback", [])
        status = f"完了: output_mode={output_mode}, frames={merged.get('frame_count')}"
        if fallback:
            status += f"\ncodec fallback: {fallback}"
        timings["total"] = time.perf_counter() - total_start
        status += f"\n処理時間: total={timings['total']:.1f}s"
        return (
            merged.get("rgba_video_path"),
            merged.get("alpha_video_path"),
            merged.get("preview_video_path"),
            overlay_video_path,
            sequence_files,
            "\n".join(sequence_dirs),
            status,
        )
    except gr.Error:
        raise
    except Exception as exc:
        stage = stage_state.get("description") or stage_state.get("name") or "unknown"
        elapsed = time.perf_counter() - total_start
        raise gr.Error(f"動画処理に失敗しました（stage={stage}, elapsed={elapsed:.1f}s）: {exc}") from exc


def update_codec_visibility(output_mode_label: str):
    """連番のみ選択時は動画 codec UI を無効化する。"""
    sequence_only = normalize_output_mode(output_mode_label) == "sequence"
    return gr.update(interactive=not sequence_only, visible=not sequence_only)


def update_point_label_visibility(prompt_mode: str):
    """point prompt の時だけ positive / negative 選択を表示する。"""
    return gr.update(visible=prompt_mode == "point")


with gr.Blocks(title="SAM2 + Transparent Background Haystack for Movie") as demo:
    gr.Markdown("# SAM2 + Transparent Background Haystack for Movie")
    gr.Markdown(
        """
### 使い方（クイックプレビュー推奨）
1. **Input Video** に動画をアップロードします。アップロード後、右側の **SAM2 Prompt Canvas** に動画の第 1 フレームが自動表示されます。
2. 複合対象を意味で選びたい場合は **Optional: Text Prompt to Box** を開き、`person playing drums` や `person riding bicycle` のように入力して bbox 候補を作ります。
3. **SAM2 Prompt Canvas** 上で対象を確認します。手動 bbox の場合は、対象を囲む四角形の **対角 2 点**（例: 左上→右下、または右下→左上）をクリックします。
4. 必要なら **Extend Left/Right/Top/Bottom** で bbox の辺を画像端まで伸ばします。Point mode では positive/negative 点で補正します。
5. まずは既定のクイックプレビュー（最大 30 frames）で **動画背景除去を実行** し、結果を確認してから Advanced で処理 frame 数を増やします。

**処理順の考え方**: 静止画版・動画版とも本質的には `Text Prompt（意味解釈）→ SAM2（マスク/トラッキング）→ transparent-background（背景除去）` です。動画版の **SAM2 Prompt Canvas** は SAM2 への入力先で、Text Prompt はその Canvas に bbox を自動で書き込む補助機能です。
"""
    )

    with gr.Row():
        input_video = gr.Video(label="Input Video", sources=["upload"])
        prompt_canvas = gr.Image(value=create_prompt_canvas_placeholder(), label="SAM2 Prompt Canvas", type="numpy", interactive=True)

    with gr.Row():
        load_first_frame_btn = gr.Button("第1フレームを再取得")
        clear_prompt_btn = gr.Button("Prompt をクリア")

    with gr.Row():
        prompt_mode = gr.Radio(["point", "box"], value="box", label="Input Mode",
            info="box: Prompt Canvas 上で対角 2 点をクリックし対象を囲む矩形を1個指定。point: クリック1点ごとに前景/背景ヒントを与える（複数点可）。単位なしの選択値。",
        )
        point_label = gr.Radio(
            ["positive", "negative"], value="positive", label="Point Label", visible=False,
            info="point モード時のみ表示。positive: クリック位置を前景（追跡対象）とする。negative: クリック位置を背景（除外）とする。",
        )

    with gr.Row():
        extend_left_btn = gr.Button("Extend Left")
        extend_right_btn = gr.Button("Extend Right")
        extend_top_btn = gr.Button("Extend Top")
        extend_bottom_btn = gr.Button("Extend Bottom")

    with gr.Row():
        prompt_frame_idx = gr.Slider(
            0, 1999, value=0, step=1, label="プロンプト起点フレーム位置",
            info="SAM2 プロンプトを与えるフレームの位置（サンプリング後シーケンスの 0 始まりの番号、整数）。0 で第1フレーム。最大処理フレーム数より小さい値にする。途中フレームから前後双方向に追跡したい時に上げる。目安 0。",
        )
        show_frame_btn = gr.Button("このフレームを表示")
    bidirectional = gr.Checkbox(
        value=False, label="双方向伝播（前後フレームへ追跡）",
        info="ON: 起点フレームから過去方向と未来方向の両方へ追跡を伝搬し、各 frame のマスクを OR 統合する（途中フレームを起点にした時に有効。処理時間は約2倍）。OFF: 起点フレームから未来方向のみ。真偽値。",
    )

    prompt_state = gr.State(value=empty_prompt_state())
    prompt_status = gr.Textbox(label="Prompt Status", interactive=False)

    with gr.Accordion("Optional: Text Prompt to Box (GroundingDINO)", open=False):
        gr.Markdown(
            "第 1 フレームに対して意味プロンプトで候補 bbox を作成し、最上位候補を SAM2 Prompt Canvas にコピーします。"
        )
        text_prompt = gr.Textbox(
            label="Text Prompt",
            placeholder="person playing drums / person riding bicycle / dog jumping through hoop",
        )
        with gr.Row():
            text_box_threshold = gr.Slider(
                0.05, 0.95, value=0.25, step=0.01, label="Box threshold",
                info="GroundingDINO の物体信頼度しきい値（0.05〜0.95 のスコア、単位なし）。高いほど検出が厳しく誤検出が減る。低いほど候補が増える。目安 0.25。",
            )
            text_text_threshold = gr.Slider(
                0.05, 0.95, value=0.25, step=0.01, label="Text threshold",
                info="検出 box と text 語句の一致しきい値（0.05〜0.95、単位なし）。高いほど語句との合致を厳しく要求する。目安 0.25。",
            )
            text_top_k = gr.Slider(
                1, 10, value=5, step=1, label="候補数 top-k",
                info="保持する検出 box の最大個数（個数、整数）。スコア上位から K 個まで表示し、最上位を SAM2 prompt にコピーする。目安 5。",
            )
        detect_text_btn = gr.Button("Text Prompt から bbox を検出")
        detected_boxes = gr.Dataframe(
            headers=["rank", "phrase", "confidence", "bbox[x1,y1,x2,y2]"],
            datatype=["number", "str", "str", "str"],
            label="Detected boxes",
            interactive=False,
        )
        box_candidate_group = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="複合対象に使う候補 bbox を選択（union 用）",
            info="複数選択すると各 bbox を別オブジェクトとして追跡し、frame ごとに OR 統合する（例: person playing drums で人とドラムを両方選ぶ）。検出後に top1 が自動 ON。選択後に下のボタンで反映。単位なしの選択値。",
        )
        apply_boxes_btn = gr.Button("選択した候補 bbox を複合対象として反映")

    run_btn = gr.Button("動画背景除去を実行", variant="primary")

    with gr.Accordion("Advanced: 動画処理設定", open=False):
        gr.Markdown(
            """
| パラメーター | 何を変えるか | 目安 |
|---|---|---|
| 最大処理フレーム数 | 先頭から処理する frame 数。多いほど長く、出力も重くなります。 | 初回 30、最終確認 300+ |
| フレーム間引きステップ | 何 frame ごとに処理するか。大きいほど速いが追跡が粗くなります。 | 通常 1、確認用のみ 2 以上 |
| 出力形式 | 動画、PNG 連番、両方。codec エラー時は連番が安全です。 | 初回は動画、失敗時は連番 |
| RGBA 動画コーデック | 透明付き動画の保存方式。環境により使えない場合があります。 | webm_vp9 優先、だめなら mov_png |
| transparent-background mode | 背景除去モデルの速度/品質。 | base が標準、fast は軽量 |
| Alpha threshold | 透明度を二値化する強さ。0 はソフトな髪/境界を残します。 | 通常 0 |
| Crop padding | SAM2 mask bbox の外側余白（髪など細部の検出漏れ防止）。大きすぎると mask が壊れます。 | 2K 基準 5、細部は 10〜30 |
"""
        )
        with gr.Row():
            max_frames = gr.Slider(1, 2000, value=30, step=1, label="最大処理フレーム数",
                info="先頭から処理する frame 数（frame 数、整数）。多いほど処理が長く出力も重くなる。初回クイックプレビュー 30、最終確認 300 以上が目安。",
            )
            frame_step = gr.Slider(1, 10, value=1, step=1, label="フレーム間引きステップ",
                info="何 frame ごとに1枚処理するか（frame 間隔、整数）。1 で全 frame。2 以上は速いが追跡が粗くなる。通常 1、確認用のみ 2 以上。",
            )
        with gr.Row():
            output_mode = gr.Radio(
                OUTPUT_MODE_CHOICES, value="動画 (video)", label="出力形式",
                info="動画: 1つの動画ファイル。連番静止画: frame ごとの PNG。両方: 動画と PNG の両方。codec エラー時は連番が安全。単位なしの選択値。",
            )
            rgba_codec = gr.Radio(
                ["webm_vp9", "mov_png"], value="webm_vp9", label="RGBA 動画コーデック",
                info="透明付き動画の保存方式。webm_vp9: VP9/WebM（軽量・推奨）。mov_png: PNG コーデックの MOV（高互換・大容量）。環境により使えない場合は自動で他方式に fallback。単位なしの選択値。",
            )
        with gr.Row():
            tracker_model = gr.Dropdown(
                choices=build_dropdown_choices("tracker"),
                value="sam2_hiera_l",
                label="トラッカーモデル（Tracker Model）",
                info="SAM2/SAMURAI のモデル。変更時はパイプラインが再初期化されます（初回のみ時間がかかります）。",
            )
            background_model = gr.Dropdown(
                choices=build_dropdown_choices("background"),
                value="tb_base",
                label="背景除去モデル（Background Model）",
                info="transparent-background のモード。base: 標準品質。fast: 軽量・高速だが精度低。base-nightly: 最新版。",
            )
            tb_jit = gr.Checkbox(
                value=False, label="JIT",
                info="ON: TorchScript JIT で推論を高速化（初回コンパイルに時間）。OFF: 通常実行。真偽値。",
            )
            tb_threshold = gr.Slider(
                0.0, 1.0, value=0.0, step=0.01, label="Alpha threshold",
                info="アルファを二値化するしきい値（0.0〜1.0 の正規化アルファ、単位なし）。0.0: 二値化せず髪などのソフトな半透明を残す。上げるほどその値未満を透明にし輪郭が硬くなる。目安 0.0。",
            )
            crop_padding = gr.Slider(
                0, 64, value=5, step=1, label="Crop padding",
                info="SAM2 mask の外接矩形の外側に加える余白（px、整数）。主目的は髪・手足先など細部の検出漏れ防止。2K 解像度基準で目安 5px。大きすぎると mask が背景を巻き込み壊れるため、細部が切れる時のみ 10〜30 に増やす。",
            )
        tb_output_type = gr.Radio(
            ["rgba", "green", "white", "blur"], value="rgba", label="Preview type",
            info="プレビュー動画の背景表示方法。rgba: 透明。green: 緑背景。white: 白背景。blur: 元背景をぼかす。RGBA 出力本体には影響しない。単位なしの選択値。",
        )
        overlay_enabled = gr.Checkbox(
            value=True, label="Tracking Overlay を生成",
            info="ON: SAM2/SAMURAI の追跡 mask を元動画に半透明で重ねた確認用動画を生成し、追従が正しいか目視できる（処理がわずかに増えます）。OFF: 生成しない。真偽値。",
        )

    with gr.Row():
        rgba_video = gr.Video(label="RGBA Video")
        alpha_video = gr.Video(label="Alpha Video")
        preview_video = gr.Video(label="Preview Video")
    tracking_overlay_video = gr.Video(
        label="Tracking Overlay (追跡確認用)",
    )
    sequence_files = gr.Files(label="連番 PNG サンプル")
    sequence_dirs = gr.Textbox(label="連番出力フォルダ", interactive=False)
    run_status = gr.Markdown()

    input_video.change(extract_first_frame_outputs, inputs=[input_video], outputs=[prompt_canvas, prompt_state, prompt_status])
    load_first_frame_btn.click(extract_first_frame, inputs=[input_video], outputs=[prompt_canvas, prompt_state, prompt_status])
    prompt_canvas.select(select_sam2_prompt, inputs=[prompt_canvas, prompt_mode, point_label, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status])
    clear_prompt_btn.click(clear_prompt, inputs=[prompt_canvas], outputs=[prompt_canvas, prompt_state, prompt_status])
    extend_left_btn.click(lambda image, state: extend_box_to_edge(image, state, "left"), inputs=[prompt_canvas, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status])
    extend_right_btn.click(lambda image, state: extend_box_to_edge(image, state, "right"), inputs=[prompt_canvas, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status])
    extend_top_btn.click(lambda image, state: extend_box_to_edge(image, state, "top"), inputs=[prompt_canvas, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status])
    extend_bottom_btn.click(lambda image, state: extend_box_to_edge(image, state, "bottom"), inputs=[prompt_canvas, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status])
    detect_text_btn.click(
        detect_text_boxes_for_video,
        inputs=[prompt_canvas, text_prompt, text_box_threshold, text_text_threshold, text_top_k, prompt_state],
        outputs=[prompt_canvas, prompt_state, detected_boxes, prompt_status],
    ).then(
        populate_candidate_choices,
        inputs=[detected_boxes],
        outputs=[box_candidate_group],
    )
    apply_boxes_btn.click(
        apply_selected_boxes,
        inputs=[prompt_canvas, prompt_state, box_candidate_group],
        outputs=[prompt_canvas, prompt_state, prompt_status],
    )
    show_frame_btn.click(
        extract_prompt_frame,
        inputs=[input_video, prompt_frame_idx, frame_step],
        outputs=[prompt_canvas, prompt_state, prompt_status],
    )
    output_mode.change(update_codec_visibility, inputs=[output_mode], outputs=[rgba_codec])
    prompt_mode.change(update_point_label_visibility, inputs=[prompt_mode], outputs=[point_label])
    run_btn.click(
        run_video_background_removal,
        inputs=[input_video, prompt_state, prompt_frame_idx, bidirectional, max_frames, frame_step, output_mode, rgba_codec, tracker_model, background_model, tb_jit, tb_threshold, tb_output_type, crop_padding, overlay_enabled],
        outputs=[rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status],
    )


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="SAM2 + transparent-background Haystack movie demo")
    parser.add_argument("--share", action="store_true", help="Gradio public link を有効化")
    parser.add_argument("--debug", action="store_true", help="Gradio debug mode")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server name")
    parser.add_argument("--server-port", type=int, default=7861, help="Gradio server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.queue()
    demo.launch(share=args.share, debug=args.debug, server_name=args.server_name, server_port=args.server_port, show_api=False)
