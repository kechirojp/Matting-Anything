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
from pipelines.components.video_model_components import SAM2VideoPropagator
from pipelines.sam2_tb_video_pipeline import (
    build_sam2_tb_video_pipeline,
    build_tb_only_video_pipeline,
    build_video_reader_pipeline,
)


READER_PIPELINE = None
_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}
_TB_ONLY_PIPELINE: Any = None
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
    """動画背景除去 Pipeline を遅延構築する（tracker 選択を registry 経由で propagator へ反映）。"""
    key = (tracker_model, background_model)
    if key not in _PIPELINE_CACHE:
        tracker_entry = entry_by_id("tracker", tracker_model)
        propagator = SAM2VideoPropagator(
            checkpoint_path=tracker_entry["checkpoint_path"],
            config_name=tracker_entry["config_name"],
            offload_video_to_cpu=bool(tracker_entry.get("offload_video_to_cpu", False)),
            offload_state_to_cpu=bool(tracker_entry.get("offload_state_to_cpu", False)),
            autocast_dtype=tracker_entry.get("autocast_dtype", "none"),
            single_object_only=bool(tracker_entry.get("single_object_only", False)),
        )
        _PIPELINE_CACHE[key] = build_sam2_tb_video_pipeline(propagator=propagator)
    return _PIPELINE_CACHE[key]


def get_tb_only_pipeline():
    """背景除去モデルのみの動画 Pipeline を遅延構築する（SAM2 追跡なし）。

    tb のモード（base/fast 等）は run 時の ``tb_mode`` で切り替わるため、Pipeline は背景モデルに
    依存せず単一インスタンスを再利用する。
    """
    global _TB_ONLY_PIPELINE
    if _TB_ONLY_PIPELINE is None:
        _TB_ONLY_PIPELINE = build_tb_only_video_pipeline()
    return _TB_ONLY_PIPELINE


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


# ─────────────────────────────────────────
# ## 1. フレーム取得系
# ─────────────────────────────────────────
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
    """動画アップロード時に第 1 フレームを自動反映し、プロンプト起点スライダーを 0 に戻す。"""
    prompt_frame_reset = gr.update(value=0)
    if not video_path:
        return (
            create_prompt_canvas_placeholder(),
            empty_prompt_state(),
            "動画をアップロードすると第 1 フレームを自動取得します。",
            prompt_frame_reset,
        )
    frame, state, status = extract_first_frame(video_path)
    return frame, state, status, prompt_frame_reset


def clear_prompt(first_frame):
    """SAM2 video prompt を初期化する。"""
    state = empty_prompt_state()
    preview = first_frame if first_frame is not None else create_prompt_canvas_placeholder()
    return preview, state, "SAM2 video prompt cleared"


# ─────────────────────────────────────────
# ## 2. DINO系
# ─────────────────────────────────────────
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


def _normalize_detected_rows(detected_rows) -> list[list[Any]]:
    """gr.Dataframe 値（pandas DataFrame / list / None）を行リストへ正規化する（ERR036）。"""
    if detected_rows is None:
        return []
    # pandas DataFrame は真偽評価が曖昧なので type で分岐し .values.tolist() を使う。
    if hasattr(detected_rows, "values") and hasattr(detected_rows, "columns"):
        return detected_rows.values.tolist()
    return list(detected_rows)


def populate_candidate_choices(detected_rows):
    """検出結果 Dataframe から候補選択 CheckboxGroup の選択肢を作る（top1 を既定 ON）。"""
    rows = _normalize_detected_rows(detected_rows)
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


def _effective_read_frames(max_frames: int, prompt_frame_idx: int) -> int:
    """prompt フレームを必ず読み込み窓に含むよう、実効的な読み込み frame 数を返す。

    任意フレームを起点に実行できるよう、`max_frames` を「最低限処理する frame 数」と扱い、
    `prompt_frame_idx` がそれを超える場合は prompt フレームを含むよう窓を引き上げる。
    これにより SAM2VideoPropagator の範囲外エラーを起こさず任意フレームで実行できる。

    Args:
        max_frames: UI 指定の最大処理 frame 数（先頭から処理する目安）。
        prompt_frame_idx: サンプリング後シーケンスにおける prompt フレームの 0 始まり番号。

    Returns:
        prompt フレームを必ず含む実効的な読み込み frame 数。
    """
    return max(1, int(max_frames), int(prompt_frame_idx) + 1)


def build_video_progress_callback(progress, stage_state: dict[str, Any]):
    """Component 内部進捗を Gradio 全体進捗に変換する callback を作る。"""

    def _callback(stage: str, fraction: float, description: str) -> None:
        stage_state["name"] = stage
        stage_state["description"] = description
        start, end = STAGE_PROGRESS_RANGES.get(stage, (0.05, 0.95))
        value = start + (end - start) * min(max(float(fraction), 0.0), 1.0)
        progress(value, desc=description)

    return _callback


# ─────────────────────────────────────────
# ## 3. SAM系
# ─────────────────────────────────────────
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
    mask_guard_enabled: bool,
    mask_guard_feather_ui: int,
    mask_guard_dilate_ui: int,
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
        # 任意フレームを起点に実行できるよう、読み込み窓を prompt フレームまで自動拡張する。
        # （範囲外バリデーションは廃止。max_frames より後ろの prompt でもその frame まで読み込む。）
        effective_max_frames = _effective_read_frames(int(max_frames), int(prompt_frame_idx))
        processed_frames = _estimate_processed_frames(effective_max_frames, int(frame_step))
        output_mode = normalize_output_mode(output_mode_label)
        progress_callback = build_video_progress_callback(progress, stage_state)
        release_text_detector()
        progress(
            0.03,
            desc=f"Pipeline を起動しています。初回はモデル読込を含みます（最大 {processed_frames} frames）",
        )
        bg_entry = entry_by_id("background", background_model)
        tb_mode = bg_entry.get("tb_mode", "base")
        mask_feather = int(bg_entry.get("mask_feather", 0))
        # mask guard 手動調整（既定 OFF）。OFF のときは config 既定 feather と extractor 既定 dilate(21) を使い既存挙動を維持する。
        if bool(mask_guard_enabled):
            mask_guard_feather = int(mask_guard_feather_ui)
            mask_guard_dilate = int(mask_guard_dilate_ui)
        else:
            mask_guard_feather = mask_feather
            mask_guard_dilate = 21
        ownership_temperature = float(bg_entry.get("ownership_temperature", 1.0))
        video_matte_mode = str(bg_entry.get("video_matte_mode", "union"))
        result = get_video_pipeline(tracker_model, background_model).run(
            {
                "video_reader": {
                    "video_path": video_path,
                    "max_frames": effective_max_frames,
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
                # ## 3.5 所有権解決: per-object logit から画素 softmax で前景 soft ゲートを算出する。
                "ownership_resolver": {
                    "temperature": ownership_temperature,
                },
                # ## 4. 背景透過系: transparent-background で frame ごとに matte を生成する。
                "transparent_bg_video": {
                    "output_mode": output_mode,
                    "tb_mode": tb_mode,
                    "tb_jit": bool(tb_jit),
                    "tb_threshold": float(tb_threshold),
                    "tb_output_type": tb_output_type,
                    "crop_padding": int(crop_padding),
                    "mask_guard_feather": mask_guard_feather,
                    "mask_guard_dilate": mask_guard_dilate,
                    "video_matte_mode": video_matte_mode,
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


def run_tb_only_background_removal(
    video_path: str | None,
    max_frames: int,
    frame_step: int,
    output_mode_label: str,
    rgba_codec: str,
    background_model: str,
    tb_jit: bool,
    tb_threshold: float,
    tb_output_type: str,
    progress=gr.Progress(),
):
    """背景除去モデルのみで動画を処理する（SAM2 追跡なし・全画面 tb・prompt 不要）。

    グリーンバックや単一 salient 対象など、追跡なしで背景除去だけで足りる用途向けの軽量経路。
    各フレームを mask 無しで全画面 transparent-background に渡すため prompt 入力は不要。
    """
    stage_state: dict[str, Any] = {"name": "startup", "description": "Pipeline を起動しています"}
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    try:
        if not video_path:
            raise gr.Error("先に動画をアップロードしてください。")
        processed_frames = _estimate_processed_frames(int(max_frames), int(frame_step))
        output_mode = normalize_output_mode(output_mode_label)
        progress_callback = build_video_progress_callback(progress, stage_state)
        release_text_detector()
        progress(
            0.03,
            desc=f"Pipeline を起動しています。初回はモデル読込を含みます（最大 {processed_frames} frames）",
        )
        bg_entry = entry_by_id("background", background_model)
        tb_mode = bg_entry.get("tb_mode", "base")
        result = get_tb_only_pipeline().run(
            {
                "video_reader": {
                    "video_path": video_path,
                    "max_frames": int(max_frames),
                    "frame_step": int(frame_step),
                    "progress_callback": progress_callback,
                },
                # 背景透過系: mask 未接続なので各フレームを全画面 tb に渡す（crop/guard なし）。
                "transparent_bg_video": {
                    "output_mode": output_mode,
                    "tb_mode": tb_mode,
                    "tb_jit": bool(tb_jit),
                    "tb_threshold": float(tb_threshold),
                    "tb_output_type": tb_output_type,
                    "rgba_codec": rgba_codec,
                    "progress_callback": progress_callback,
                },
                "video_writer": {"rgba_codec": rgba_codec, "progress_callback": progress_callback},
                "frame_sequence_writer": {"progress_callback": progress_callback},
            },
            include_outputs_from={"video_writer", "frame_sequence_writer"},
        )
        progress(0.95, desc="出力を整理しています")
        video_result = result.get("video_writer", {}).get("matte")
        sequence_result = result.get("frame_sequence_writer", {}).get("matte")
        merged = _merge_matte_results({}, video_result, sequence_result)
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
            sequence_files,
            "\n".join(sequence_dirs),
            status,
        )
    except gr.Error:
        raise
    except Exception as exc:
        stage = stage_state.get("description") or stage_state.get("name") or "unknown"
        elapsed = time.perf_counter() - total_start
        raise gr.Error(f"背景除去のみ処理に失敗しました（stage={stage}, elapsed={elapsed:.1f}s）: {exc}") from exc


def update_codec_visibility(output_mode_label: str):
    """連番のみ選択時は動画 codec UI を無効化する。"""
    sequence_only = normalize_output_mode(output_mode_label) == "sequence"
    return gr.update(interactive=not sequence_only, visible=not sequence_only)


def update_point_label_visibility(prompt_mode: str):
    """point prompt の時だけ positive / negative 選択を表示する。"""
    return gr.update(visible=prompt_mode == "point")


def update_bidirectional_for_tracker(tracker_id: str):
    """tracker registry の supports_bidirectional に従い双方向伝播 UI を切り替える。

    SAMURAI など forward-only（Kalman filter）の tracker を選んだ場合は、双方向伝播を
    自動で OFF にし、操作できないように無効化する（ERR050）。standard SAM2 では有効化する。
    config 駆動なので tracker を追加してもこの関数は変更不要。
    """
    try:
        entry = entry_by_id("tracker", tracker_id) if tracker_id else None
    except KeyError:
        # 未知 id は安全側（操作可能）に倒す。エラーは握り潰さず interactive=True を明示。
        return gr.update(interactive=True)
    supports = bool(entry.get("supports_bidirectional", True)) if entry else True
    if supports:
        return gr.update(interactive=True)
    return gr.update(value=False, interactive=False)


with gr.Blocks(title="SAM2 + Transparent Background Haystack for Movie") as demo:
    gr.Markdown("# SAM2 + Transparent Background Haystack for Movie")
    gr.Markdown(
        """
> ## ⚠️ SAMURAI トラッカー推奨設定（必読）
> SAMURAI（motion-aware）は Kalman filter による **forward-only（前方向のみ）** 設計です。Colab T4 等の
> 小 VRAM 環境ではメモリ枯渇で伝搬が `propagate 1/N` のまま凍結する stall が起きやすいため、以下を守ってください（ERR049 / ERR050）。
>
> | 項目 | 推奨 | 理由 |
> |---|---|---|
> | **対象オブジェクト数** | **1 個のみ**（複数は標準 SAM2 へ） | SAMURAI は Kalman filter による単一対象追跡専用。複数 box を渡すと fork が `Boolean value of Tensor ... ambiguous` で伝搬失敗（ERR051） |
> | **双方向伝播** | **OFF**（SAMURAI 選択時は自動で OFF・無効化） | 逆方向は Kalman の速度ベクトルが反転し追跡が崩れ、per-frame memory も 2 倍 |
> | **プロンプト起点フレーム** | **0（先頭）** | forward-only のため先頭起点が安定。末尾起点は逆走 stall を誘発 |
> | **CPU offload** | 有効（config で自動 ON） | 常駐 VRAM を抑え stall を回避（`offload_video_to_cpu` / `offload_state_to_cpu`） |
> | **autocast** | fp16（config で自動 ON） | 伝搬を mixed precision で回し VRAM を削減・高速化（SAMURAI 本家と同じ） |
> | **最大処理フレーム数** | まず 30 でプレビュー | 初回から大量フレームにすると stall リスクが上がる |
>
> これらの値はすべて `config/inference_models.toml` の tracker entry 駆動です。UI 側は registry を見て双方向 UI を自動制御します。
"""
    )
    gr.Markdown(
        """
### 使い方（クイックプレビュー推奨）
1. **Input Video** に動画をアップロードします。アップロード後、右側の **SAM2 Prompt Canvas** に動画の第 1 フレームが自動表示されます。
2. 複合対象を意味で選びたい場合は **Optional: Text Prompt to Box** を開き、`person playing drums` や `person riding bicycle` のように入力して bbox 候補を作ります。
3. **SAM2 Prompt Canvas** 上で対象を確認します。手動 bbox の場合は、対象を囲む四角形の **対角 2 点**（例: 左上→右下、または右下→左上）をクリックします。
4. 必要なら **Extend Left/Right/Top/Bottom** で bbox の辺を画像端まで伸ばします。Point mode では positive/negative 点で補正します。
5. まずは既定のクイックプレビュー（最大 30 frames）で **動画背景除去を実行** し、結果を確認してから Advanced で処理 frame 数を増やします。

**処理順の考え方**: 動画版の本質は `フレーム取得 → DINO で候補生成 → SAM2 で対象ごとに prompt / 追跡（logit 保持・2値化しない）→ 所有権解決（ピクセル softmax で重なりを各対象へ排他割当）→ 背景透過（連続アルファ）→ 所有権でアルファ合成` です。背景透過は既定で union モード（union mask の外接矩形で 1 回だけ切り抜く軽量モード）で動きます。config の `video_matte_mode = "per_object"` に変えると対象ごとに切り抜いて合成しますが、多 box 選択時は tb 呼び出しが対象数×フレームとなり重くなります。エッジの半透明を硬くしたい場合は **Alpha threshold** スライダを上げて tb の連続アルファを二値化します。動画版の **SAM2 Prompt Canvas** は SAM2 への入力先で、Text Prompt はその Canvas に bbox を自動で書き込む補助機能です。
"""
    )

    with gr.Tabs():
        with gr.Tab("SAM2 追跡 + 背景除去"):
            with gr.Row():
                input_video = gr.Video(label="Input Video", sources=["upload"], elem_id="movie-input-video")
                prompt_canvas = gr.Image(
                    value=create_prompt_canvas_placeholder(),
                    label="SAM2 Prompt Canvas",
                    type="numpy",
                    sources=[],
                    interactive=True,
                )

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
                    0, 1999, value=0, step=1, label="プロンプト起点フレーム位置（ドラッグで Canvas 更新）",
                    elem_id="prompt-frame-idx",
                    info="SAM2 プロンプトを与えるフレームの位置（サンプリング後シーケンスの 0 始まりの番号、整数）。0 で第1フレーム。最大処理フレーム数を超える値でも、その frame まで自動で読み込んで実行する（範囲外エラーなし。その分だけ読み込み時間が増える）。途中フレームから前後双方向に追跡したい時に上げる。SAMURAI は forward-only なので 0（先頭）推奨。目安 0。",
                )
                show_frame_btn = gr.Button("このフレームを表示")
            bidirectional = gr.Checkbox(
                value=False, label="双方向伝播（前後フレームへ追跡）",
                info="ON: 起点フレームから過去方向と未来方向の両方へ追跡を伝搬し、各 frame のマスクを OR 統合する（途中フレームを起点にした時に有効。処理時間は約2倍）。OFF: 起点フレームから未来方向のみ。真偽値。SAMURAI 選択時は forward-only のため自動で OFF ・無効化される。",
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
                        1, 20, value=20, step=1, label="候補数 top-k",
                        info="保持する検出 box の最大個数（個数、整数）。スコア上位から K 個まで表示し、最上位を SAM2 prompt にコピーする。目安 20。",
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
| Crop padding | SAM2 mask bbox の外側余白（髪など細部の検出漏れ防止）。大きすぎると mask が壊れます。 | 既定 5、細部が切れる時のみ 5〜30 |
"""
                )
                with gr.Row():
                    max_frames = gr.Slider(1, 2000, value=30, step=1, label="最大処理フレーム数",
                        info="先頭から処理する frame 数（frame 数、整数）。多いほど処理が長く出力も重くなる。初回クイックプレビュー 30、最終確認 300 以上が目安。",
                    )
                    frame_step = gr.Slider(1, 10, value=1, step=1, label="フレーム間引きステップ",
                        elem_id="movie-frame-step",
                        info="何 frame ごとに1枚処理するか（frame 間隔、整数）。1 で全 frame。2 以上は速いが追跡が粗くなる。通常 1、確認用のみ 2 以上。",
                    )
                with gr.Row():
                    output_mode = gr.Radio(
                        OUTPUT_MODE_CHOICES, value="動画 (video)", label="出力形式",
                        info="動画: 1つの動画ファイル。連番静止画: frame ごとの PNG。両方: 動画と PNG の両方。codec エラー時は連番が安全。単位なしの選択値。",
                    )
                    rgba_codec = gr.Radio(
                        ["webm_vp9", "mov_png"], value="webm_vp9", label="RGBA 動画コーデック",
                        info="透明付き動画の保存方式（imageio+ffmpeg で alpha 保持）。webm_vp9: VP9/WebM（軽量・推奨）。mov_png: PNG コーデックの MOV（高互換・大容量）。書き出せない環境では連番(PNG)出力が確実。単位なしの選択値。",
                    )
                with gr.Row():
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
                        info="SAM2 mask の外接矩形の外側に加える余白（px、整数）。既定 5。主目的は髪・手足先など細部の検出漏れ防止。大きすぎると mask が背景を巻き込み壊れるため、細部が切れる時のみ 5〜30 に増やす。",
                    )
                tb_output_type = gr.Radio(
                    ["rgba", "green", "white", "blur"], value="rgba", label="Preview type",
                    info="プレビュー動画の背景表示方法。rgba: 透明。green: 緑背景。white: 白背景。blur: 元背景をぼかす。RGBA 出力本体には影響しない。単位なしの選択値。",
                )
                mask_guard_enabled = gr.Checkbox(
                    value=False, label="Mask guard を手動調整",
                    info="OFF（既定）: 従来挙動（config の feather と dilate=21）を維持し出力は変わりません。ON: 下の feather/dilate スライダで SAM2 mask の外側ゲートを手動調整します。真偽値。",
                )
                with gr.Row():
                    mask_guard_feather = gr.Slider(
                        0, 64, value=0, step=1, label="Mask guard feather",
                        info="SAM2 mask 境界をぼかす幅（px、整数）。0: 二値ゲート（内部 1.0・外部 0）で tb の内部アルファを削らない。上げるほど境界が連続化し馴染むが、追跡ずれ時は背景を巻き込みやすい。手動調整 ON のときのみ有効。",
                    )
                    mask_guard_dilate = gr.Slider(
                        1, 81, value=21, step=2, label="Mask guard dilate",
                        info="二値ゲート（feather=0 時）の膨張カーネルサイズ（px、奇数）。既定 21。大きいほど SAM2 mask の外側に余白を足し細部の取りこぼしを減らすが、背景の巻き込みも増える。手動調整 ON のときのみ有効。",
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

        with gr.Tab("背景除去のみ (tb only)"):
            gr.Markdown(
                """
### 背景除去のみ（SAM2 追跡なし）
グリーンバックや単一 salient 対象など、追跡が不要で背景除去モデルだけで足りる用途向けの軽量経路です。
**prompt は不要**で、各フレームを全画面のまま transparent-background に渡します（crop / guard / 所有権合成・tracking overlay は行いません）。
複数人物・複合対象を意味で選び分けたい場合や、対象を限定して追跡したい場合は「SAM2 追跡 + 背景除去」タブを使ってください。
"""
            )
            tb_only_input_video = gr.Video(label="Input Video", sources=["upload"], elem_id="tb-only-input-video")
            tb_only_background_model = gr.Dropdown(
                choices=build_dropdown_choices("background"),
                value="tb_base",
                label="背景除去モデル（Background Model）",
                info="transparent-background のモード。base: 標準品質。fast: 軽量・高速だが精度低。base-nightly: 最新版。",
            )
            tb_only_run_btn = gr.Button("背景除去のみを実行", variant="primary")

            with gr.Accordion("Advanced: 動画処理設定", open=False):
                with gr.Row():
                    tb_only_max_frames = gr.Slider(1, 2000, value=30, step=1, label="最大処理フレーム数",
                        info="先頭から処理する frame 数（frame 数、整数）。多いほど処理が長く出力も重くなる。初回クイックプレビュー 30、最終確認 300 以上が目安。",
                    )
                    tb_only_frame_step = gr.Slider(1, 10, value=1, step=1, label="フレーム間引きステップ",
                        info="何 frame ごとに1枚処理するか（frame 間隔、整数）。1 で全 frame。2 以上は速いが粗くなる。通常 1、確認用のみ 2 以上。",
                    )
                with gr.Row():
                    tb_only_output_mode = gr.Radio(
                        OUTPUT_MODE_CHOICES, value="動画 (video)", label="出力形式",
                        info="動画: 1つの動画ファイル。連番静止画: frame ごとの PNG。両方: 動画と PNG の両方。codec エラー時は連番が安全。単位なしの選択値。",
                    )
                    tb_only_rgba_codec = gr.Radio(
                        ["webm_vp9", "mov_png"], value="webm_vp9", label="RGBA 動画コーデック",
                        info="透明付き動画の保存方式（imageio+ffmpeg で alpha 保持）。webm_vp9: VP9/WebM（軽量・推奨）。mov_png: PNG コーデックの MOV（高互換・大容量）。書き出せない環境では連番(PNG)出力が確実。単位なしの選択値。",
                    )
                with gr.Row():
                    tb_only_jit = gr.Checkbox(
                        value=False, label="JIT",
                        info="ON: TorchScript JIT で推論を高速化（初回コンパイルに時間）。OFF: 通常実行。真偽値。",
                    )
                    tb_only_threshold = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.01, label="Alpha threshold",
                        info="アルファを二値化するしきい値（0.0〜1.0 の正規化アルファ、単位なし）。0.0: 二値化せず髪などのソフトな半透明を残す。上げるほどその値未満を透明にし輪郭が硬くなる。目安 0.0。",
                    )
                tb_only_output_type = gr.Radio(
                    ["rgba", "green", "white", "blur"], value="rgba", label="Preview type",
                    info="プレビュー動画の背景表示方法。rgba: 透明。green: 緑背景。white: 白背景。blur: 元背景をぼかす。RGBA 出力本体には影響しない。単位なしの選択値。",
                )

            with gr.Row():
                tb_only_rgba_video = gr.Video(label="RGBA Video")
                tb_only_alpha_video = gr.Video(label="Alpha Video")
                tb_only_preview_video = gr.Video(label="Preview Video")
            tb_only_sequence_files = gr.Files(label="連番 PNG サンプル")
            tb_only_sequence_dirs = gr.Textbox(label="連番出力フォルダ", interactive=False)
            tb_only_run_status = gr.Markdown()

    input_video.change(extract_first_frame_outputs, inputs=[input_video], outputs=[prompt_canvas, prompt_state, prompt_status, prompt_frame_idx])
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
    # スライダー1本に集約: 起点フレーム位置を変えると即座に SAM2 prompt canvas を更新する。
    prompt_frame_idx.change(
        extract_prompt_frame,
        inputs=[input_video, prompt_frame_idx, frame_step],
        outputs=[prompt_canvas, prompt_state, prompt_status],
    )
    output_mode.change(update_codec_visibility, inputs=[output_mode], outputs=[rgba_codec])
    prompt_mode.change(update_point_label_visibility, inputs=[prompt_mode], outputs=[point_label])
    tracker_model.change(update_bidirectional_for_tracker, inputs=[tracker_model], outputs=[bidirectional])
    run_btn.click(
        run_video_background_removal,
        inputs=[input_video, prompt_state, prompt_frame_idx, bidirectional, max_frames, frame_step, output_mode, rgba_codec, tracker_model, background_model, tb_jit, tb_threshold, tb_output_type, crop_padding, mask_guard_enabled, mask_guard_feather, mask_guard_dilate, overlay_enabled],
        outputs=[rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status],
    )

    tb_only_output_mode.change(update_codec_visibility, inputs=[tb_only_output_mode], outputs=[tb_only_rgba_codec])
    tb_only_run_btn.click(
        run_tb_only_background_removal,
        inputs=[tb_only_input_video, tb_only_max_frames, tb_only_frame_step, tb_only_output_mode, tb_only_rgba_codec, tb_only_background_model, tb_only_jit, tb_only_threshold, tb_only_output_type],
        outputs=[tb_only_rgba_video, tb_only_alpha_video, tb_only_preview_video, tb_only_sequence_files, tb_only_sequence_dirs, tb_only_run_status],
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
