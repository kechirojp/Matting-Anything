"""Haystack Pipeline 版 SAM2 + BEN2（ルートA: ブラー誘導 → 再α化）動画αマットデモ。

既存の ``gradio_app_sam2_transparent_BG_haystack_for_Movie`` と同型の UI / イベント配線を保ちつつ、
背景除去段を BEN2（ルートA）に差し替えた版。追跡（SAM2/SAMURAI）・所有権解決・tracking overlay・
GroundingDINO テキスト→bbox・複合対象 union はそのまま再利用する。

ルートA案: SAM2 下地マスク M を膨張してゲート G を作り、G 外を強くブラーした誘導フレーム I' を
BEN2 に渡して α を再生成する（仕様書 2026-06-22 ルートA案）。膨張量・ブラー強度・羽根幅などの
チューニング値は ``config/route_a.toml`` 既定値を UI 初期値として読み込み、画面から微調整できる。
"""

from __future__ import annotations

import argparse
import gc
import sys
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

if sys.platform == "win32":
    # Gradio が内部で使う uvicorn は Windows でも Proactor ループを使い続けるため、
    # WindowsSelectorEventLoopPolicy への切替だけでは connection_lost の WinError 10054
    # ノイズを抑えられない。クライアント切断時の無害な ConnectionResetError だけを
    # Proactor トランスポートの _call_connection_lost で握って黙らせる（他の例外は素通し）。
    from asyncio.proactor_events import _ProactorBasePipeTransport
    from functools import wraps as _wraps

    def _silence_proactor_connection_reset(func):
        @_wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except ConnectionResetError:
                # WinError 10054: クライアント切断時の競合で発生する無害なシャットダウン例外のみ無視。
                return None

        return wrapper

    _ProactorBasePipeTransport._call_connection_lost = _silence_proactor_connection_reset(
        _ProactorBasePipeTransport._call_connection_lost
    )

import gradio as gr

from pipelines.components.ben2_components import BEN2RouteAVideoExtractor
from pipelines.components.model_components import GroundingDINOMultiBoxDetector
from pipelines.components.model_registry import build_dropdown_choices, entry_by_id
from pipelines.components.route_a_common import load_route_a_config
from pipelines.components.ui_helpers import (
    build_prompt_selection_choices,
    copy_prompt_state,
    draw_prompt_overlay,
    empty_prompt_state,
    extend_box_to_edge,
    remove_selected_boxes,
    remove_selected_points,
    select_sam2_prompt,
)
from pipelines.components.common import assign_points_to_boxes
from pipelines.components.video_common import normalize_output_mode
from pipelines.components.video_model_components import SAM2VideoPropagator
from pipelines.job_manager import JobManager
from pipelines.route_a_video_pipeline import (
    build_ben2_route_a_only_video_pipeline,
    build_sam2_ben2_route_a_pipeline,
)
from pipelines.sam2_tb_video_pipeline import build_video_reader_pipeline


READER_PIPELINE = None
_PIPELINE_CACHE: dict[str, Any] = {}
_ROUTE_A_ONLY_PIPELINE: Any = None
TEXT_DETECTOR = None
PROMPT_CANVAS_PLACEHOLDER_SIZE = (420, 640)
OUTPUT_MODE_CHOICES = ["動画 (video)", "連番静止画 (sequence)", "両方 (both)"]
MATTE_MODE_CHOICES = ["union（高速・1回/frame）", "per_object（忠実・対象数×/frame）"]
MASK_FLOOR_MODE_CHOICES = [
    "none（無効）",
    "screen（底上げ・自然）",
    "lighten / 比較明（確実に塗る）",
]
STAGE_PROGRESS_RANGES = {
    "video_reader": (0.05, 0.15),
    "sam2_video": (0.15, 0.55),
    "ben2_route_a": (0.55, 0.88),
    "video_writer": (0.88, 0.96),
    "frame_sequence_writer": (0.88, 0.96),
    "tracking_overlay": (0.88, 0.96),
}
_ROUTE_A_DEFAULTS = load_route_a_config()


def _blur_defaults() -> dict[str, Any]:
    """config/route_a.toml の blur_guide 既定値を UI 初期値として返す。"""
    return dict(_ROUTE_A_DEFAULTS.get("blur_guide", {}))


def _composite_defaults() -> dict[str, Any]:
    """config/route_a.toml の composite 既定値を UI 初期値として返す。"""
    return dict(_ROUTE_A_DEFAULTS.get("composite", {}))


def _alpha_defaults() -> dict[str, Any]:
    """config/route_a.toml の alpha 既定値を UI 初期値として返す。"""
    return dict(_ROUTE_A_DEFAULTS.get("alpha", {}))


def _normalize_matte_mode(matte_mode_label: str) -> str:
    """UI ラベルを matte_mode（'union' / 'per_object'）へ正規化する。"""
    return "per_object" if str(matte_mode_label).startswith("per_object") else "union"


def _normalize_mask_floor_mode(mask_floor_label: str) -> str:
    """UI ラベルを mask_floor_mode（'none' / 'screen' / 'lighten'）へ正規化する。"""
    text = str(mask_floor_label).strip().lower()
    if text.startswith("screen"):
        return "screen"
    if text.startswith("lighten") or "比較明" in str(mask_floor_label):
        return "lighten"
    return "none"


def _mask_floor_label_from_value(value: str) -> str:
    """config 値（none/screen/lighten）を UI ラベルへ対応付ける。"""
    text = str(value).strip().lower()
    if text == "screen":
        return MASK_FLOOR_MODE_CHOICES[1]
    if text in {"lighten", "max"}:
        return MASK_FLOOR_MODE_CHOICES[2]
    return MASK_FLOOR_MODE_CHOICES[0]


def get_reader_pipeline():
    """第 1 フレーム抽出用 Pipeline を遅延構築する。"""
    global READER_PIPELINE
    if READER_PIPELINE is None:
        READER_PIPELINE = build_video_reader_pipeline()
    return READER_PIPELINE


def get_route_a_pipeline(tracker_model: str = "sam2_hiera_l"):
    """ルートA動画αマット Pipeline を遅延構築する（tracker 選択を registry 経由で反映）。"""
    if tracker_model not in _PIPELINE_CACHE:
        tracker_entry = entry_by_id("tracker", tracker_model)
        propagator = SAM2VideoPropagator(
            checkpoint_path=tracker_entry["checkpoint_path"],
            config_name=tracker_entry["config_name"],
            offload_video_to_cpu=bool(tracker_entry.get("offload_video_to_cpu", False)),
            offload_state_to_cpu=bool(tracker_entry.get("offload_state_to_cpu", False)),
            autocast_dtype=tracker_entry.get("autocast_dtype", "none"),
            single_object_only=bool(tracker_entry.get("single_object_only", False)),
        )
        _PIPELINE_CACHE[tracker_model] = build_sam2_ben2_route_a_pipeline(propagator=propagator)
    return _PIPELINE_CACHE[tracker_model]


def get_route_a_only_pipeline():
    """SAM2 追跡なしで BEN2（ルートA）のみ動画処理する Pipeline を遅延構築する。"""
    global _ROUTE_A_ONLY_PIPELINE
    if _ROUTE_A_ONLY_PIPELINE is None:
        _ROUTE_A_ONLY_PIPELINE = build_ben2_route_a_only_video_pipeline()
    return _ROUTE_A_ONLY_PIPELINE


def get_text_detector():
    """GroundingDINO Text Prompt 検出器をボタン押下時まで遅延構築する。"""
    global TEXT_DETECTOR
    if TEXT_DETECTOR is None:
        TEXT_DETECTOR = GroundingDINOMultiBoxDetector()
    return TEXT_DETECTOR


def release_text_detector() -> None:
    """動画処理前に GroundingDINO / BERT を解放し、GPU VRAM のピークを下げる。

    SAM2 + BEN2 の重い動画パイプラインをロードする前に検出器を解放して
    ``torch.cuda.empty_cache()`` を呼び、VRAM 余裕を確保する（小容量 GPU・長尺で有効）。
    """
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
    """動画アップロード時に第 1 フレームを自動反映し、プロンプト起点スライダーを 0 に戻す。

    戦略: prompt canvas 表示用画像（overlay 焦き込み済み）とは別に、描画基準となる
    クリーンフレームを prompt_base_image へ返す。これにより clear / 個別削除が
    クリーン基準から再描画され、UI に冪等に反映される（overlay 累積バグ防止）。
    出力順: (canvas, base, state, status, prompt_frame_reset)。
    """
    prompt_frame_reset = gr.update(value=0)
    if not video_path:
        placeholder = create_prompt_canvas_placeholder()
        return (
            placeholder,
            placeholder,
            empty_prompt_state(),
            "動画をアップロードすると第 1 フレームを自動取得します。",
            prompt_frame_reset,
        )
    frame, state, status = extract_first_frame(video_path)
    return frame, frame, state, status, prompt_frame_reset


def extract_first_frame_with_base(video_path: str | None):
    """第 1 フレームを取得し、canvas と描画基準 base の両方にクリーンフレームを返す。

    出力順: (canvas, base, state, status)。
    """
    frame, state, status = extract_first_frame(video_path)
    return frame, frame, state, status


def clear_prompt(base_image):
    """SAM2 video prompt を初期化し、canvas をクリーン基準フレームへ戻す。

    base_image は overlay を含まないクリーンフレーム（prompt_base_image State）。
    これを返すことで初期化が canvas 表示に確実に反映される。
    """
    state = empty_prompt_state()
    preview = base_image if base_image is not None else create_prompt_canvas_placeholder()
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


def extract_prompt_frame_with_base(video_path: str | None, prompt_frame_idx: int, frame_step: int):
    """指定フレームを取得し、canvas と描画基準 base の両方にクリーンフレームを返す。

    出力順: (canvas, base, state, status)。
    """
    frame, state, status = extract_prompt_frame(video_path, prompt_frame_idx, frame_step)
    return frame, frame, state, status


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
    """prompt フレームを必ず読み込み窓に含むよう、実効的な読み込み frame 数を返す。"""
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
# ## 3. ルートA実行系
# ─────────────────────────────────────────
def run_route_a_background_removal(
    video_path: str | None,
    prompt_state: dict | None,
    prompt_frame_idx: int,
    bidirectional: bool,
    max_frames: int,
    frame_step: int,
    output_mode_label: str,
    rgba_codec: str,
    tracker_model: str,
    matte_mode_label: str,
    dilation_px: int,
    blur_kernel: int,
    blur_sigma: float,
    feather_px: int,
    refine_foreground: bool,
    gate_alpha: bool,
    mask_floor_mode_label: str,
    output_type: str,
    overlay_enabled: bool,
    progress=gr.Progress(),
):
    """ルートA動画αマット Pipeline を実行し、動画または PNG 連番を出力する。"""
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
        effective_max_frames = _effective_read_frames(int(max_frames), int(prompt_frame_idx))
        processed_frames = _estimate_processed_frames(effective_max_frames, int(frame_step))
        output_mode = normalize_output_mode(output_mode_label)
        matte_mode = _normalize_matte_mode(matte_mode_label)
        mask_floor_mode = _normalize_mask_floor_mode(mask_floor_mode_label)
        positive_count = sum(1 for value in labels if int(value) == 1)
        negative_count = sum(1 for value in labels if int(value) == 0)
        prompt_debug_lines = [
            f"prompt: points={len(points)} (pos={positive_count}, neg={negative_count}), manual_box={'yes' if box is not None else 'no'}, union_boxes={len(boxes)}"
        ]
        if boxes and points:
            assignment = assign_points_to_boxes(points, boxes)
            mapping_text = ", ".join(f"obj{obj_id}:{len(indexes)}" for obj_id, indexes in sorted(assignment.items()))
            prompt_debug_lines.append(f"point assignment: {mapping_text}")
        if points and not bool(bidirectional):
            try:
                tracker_entry = entry_by_id("tracker", tracker_model)
                if bool(tracker_entry.get("supports_bidirectional", True)):
                    prompt_debug_lines.append("flicker hint: 標準 SAM2 では『双方向伝播=ON』がちらつき抑制に有効です。")
            except KeyError:
                pass
        if points and matte_mode == "union":
            prompt_debug_lines.append("flicker hint: 複雑シーンでは『per_object』の方が mask ちらつきを抑えやすいです。")
        # ルートAでは SAM2 マスク（=point の反映先）は『背景ブラーのゲート G』を作るためだけに使われ、
        # gate_alpha=OFF の場合は最終 α を BEN2 が単独で決める（BEN2 はマスク入力ポートを持たない＝仕様 A-2）。
        # そのため point を打っても最終 α / ちらつきは直接変わらず『伝わっていない』ように見える。
        # point を最終 α に効かせるには gate_alpha=ON（α をゲート G 内に限定）を案内する。
        if points and not bool(gate_alpha):
            prompt_debug_lines.append(
                "flicker hint: ルートAでは point/SAM2 マスクは現在『背景ブラーの範囲』にのみ使われ、"
                "最終 α は BEN2 が単独生成します（BEN2 はマスク入力なし）。"
                "ちらつき箇所を point で直接抑えるには『ゲートでαを制限（gate_alpha）』を ON にしてください。"
            )
            prompt_debug_lines.append(
                "確認方法: Tracking Overlay は point を反映した SAM2 マスクを描きます。"
                "point 追加で overlay マスクが変われば SAM2 へは伝わっており、ちらつきは BEN2 側要因です。"
            )
        if mask_floor_mode == "none":
            prompt_debug_lines.append(
                "flicker hint: SAM2 が最後まで追跡できているなら『SAM2マスクでα底上げ（合成）』を "
                "screen か lighten/比較明 にすると、安定マスクを α の床にして BEN2 のちらつきを直接補えます。"
            )
        progress_callback = build_video_progress_callback(progress, stage_state)
        release_text_detector()
        progress(
            0.03,
            desc=f"Pipeline を起動しています。初回はモデル読込（BEN2 含む）を伴います（最大 {processed_frames} frames）",
        )
        result = get_route_a_pipeline(tracker_model).run(
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
                "ownership_resolver": {
                    "temperature": 1.0,
                },
                # ## 4. ルートA背景透過: 下地マスク膨張 → 背景ブラー → BEN2 再α化。
                "ben2_route_a_video": {
                    "output_mode": output_mode,
                    "matte_mode": matte_mode,
                    "dilation_px": int(dilation_px),
                    "blur_kernel": int(blur_kernel),
                    "blur_sigma": float(blur_sigma),
                    "feather_px": int(feather_px),
                    "refine_foreground": bool(refine_foreground),
                    "gate_alpha": bool(gate_alpha),
                    "mask_floor_mode": mask_floor_mode,
                    "output_type": output_type,
                    "rgba_codec": rgba_codec,
                    "progress_callback": progress_callback,
                },
                "video_writer": {"rgba_codec": rgba_codec, "progress_callback": progress_callback},
                "frame_sequence_writer": {"progress_callback": progress_callback},
                "tracking_overlay": {"enabled": bool(overlay_enabled), "output_mode": output_mode, "progress_callback": progress_callback},
            },
            include_outputs_from={"video_writer", "frame_sequence_writer", "tracking_overlay"},
        )
        progress(0.95, desc="出力を整理しています")
        video_result = result.get("video_writer", {}).get("matte")
        sequence_result = result.get("frame_sequence_writer", {}).get("matte")
        merged = _merge_matte_results({}, video_result, sequence_result)
        overlay_result = result.get("tracking_overlay", {}).get("overlay", {})
        overlay_video_path = overlay_result.get("overlay_video_path")
        sequence_files = _sequence_preview_files(merged)
        sequence_dirs = [merged.get(key) for key in ("rgba_sequence_dir", "alpha_sequence_dir", "preview_sequence_dir") if merged.get(key)]
        fallback = merged.get("metadata", {}).get("codec_fallback", [])
        status = f"完了: output_mode={output_mode}, matte_mode={matte_mode}, frames={merged.get('frame_count')}"
        status += "\n" + "\n".join(prompt_debug_lines)
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
        raise gr.Error(f"ルートA動画処理に失敗しました（stage={stage}, elapsed={elapsed:.1f}s）: {exc}") from exc


def run_route_a_only_background_removal(
    video_path: str | None,
    max_frames: int,
    frame_step: int,
    output_mode_label: str,
    rgba_codec: str,
    refine_foreground: bool,
    output_type: str,
    progress=gr.Progress(),
):
    """SAM2 追跡なしで BEN2（ルートA）のみ動画を処理する（prompt 不要・全画面 BEN2）。

    単一 salient 対象など追跡が不要な用途向けの軽量経路。マスク未供給のため誘導ブラーは行わず、
    各フレームをそのまま BEN2 に渡して α を生成する。
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
            desc=f"Pipeline を起動しています。初回はモデル読込（BEN2 含む）を伴います（最大 {processed_frames} frames）",
        )
        result = get_route_a_only_pipeline().run(
            {
                "video_reader": {
                    "video_path": video_path,
                    "max_frames": int(max_frames),
                    "frame_step": int(frame_step),
                    "progress_callback": progress_callback,
                },
                "ben2_route_a_video": {
                    "output_mode": output_mode,
                    "matte_mode": "union",
                    "refine_foreground": bool(refine_foreground),
                    "output_type": output_type,
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
        raise gr.Error(f"BEN2 のみ処理に失敗しました（stage={stage}, elapsed={elapsed:.1f}s）: {exc}") from exc


# ─────────────────────────────────────────
# ## 3b. 非同期ジョブ実行（ERR058）
# ─────────────────────────────────────────
# 数分かかるパイプラインを 1 本の同期リクエストとして gradio.live トンネル越しに保持すると、
# idle / 総接続時間の上限で切断され全出力が「Error」になる（ERR048-057 はこの対症療法）。
# submit で即 job_id を返してリクエストを <1s に短縮し、gr.Timer がジョブ状態をポーリングして
# テキスト進捗を描画する。これにより長時間 SSE が存在しなくなり切断クラスのバグを根治する。
_JOB_MANAGER = JobManager()
# 一度 UI へ通知済みのエラー job_id（Timer 再発火による多重トースト抑止）。
_REPORTED_JOB_ERRORS: set[str] = set()


class _ProgressBridge:
    """``gr.Progress`` 互換の ``__call__(value, desc=...)`` を JobState 進捗へ橋渡しする。

    既存の :func:`build_video_progress_callback` は ``progress(value, desc=...)`` のみ使用するため、
    本ブリッジを ``progress`` 引数へ渡すだけで Pipeline 内部進捗が JobState に反映される。
    """

    def __init__(self, report: Any) -> None:
        self._report = report

    def __call__(self, value: float, desc: str = "") -> None:
        self._report(float(value), str(desc))


def _progress_text(fraction: float, description: str) -> str:
    """進捗率と説明をテキスト進捗表示に整形する。"""
    percent = int(round(min(max(float(fraction), 0.0), 1.0) * 100))
    return f"処理中… {percent}%　{description}".rstrip()


def start_route_a_job(
    video_path: str | None,
    prompt_state: dict | None,
    prompt_frame_idx: int,
    bidirectional: bool,
    max_frames: int,
    frame_step: int,
    output_mode_label: str,
    rgba_codec: str,
    tracker_model: str,
    matte_mode_label: str,
    dilation_px: int,
    blur_kernel: int,
    blur_sigma: float,
    feather_px: int,
    refine_foreground: bool,
    gate_alpha: bool,
    mask_floor_mode_label: str,
    output_type: str,
    overlay_enabled: bool,
):
    """ルートA動画ジョブを起動し、即座に job_id と Timer 活性化を返す（非同期/ERR058）。"""
    # fail-fast 検証（瞬時。重い処理前に通知 / ERR037）。
    if not video_path:
        raise gr.Error("先に動画をアップロードしてください。")
    state = copy_prompt_state(prompt_state)
    if not (state.get("points") or state.get("box") is not None or (state.get("boxes") or [])):
        raise gr.Error("SAM2 Prompt Canvas 上で point / bbox を指定するか、候補 bbox を選択してください。")

    args = (
        video_path, prompt_state, prompt_frame_idx, bidirectional, max_frames, frame_step,
        output_mode_label, rgba_codec, tracker_model, matte_mode_label, dilation_px, blur_kernel,
        blur_sigma, feather_px, refine_foreground, gate_alpha, mask_floor_mode_label, output_type,
        overlay_enabled,
    )

    def work(report):
        return run_route_a_background_removal(*args, progress=_ProgressBridge(report))

    job_id = _JOB_MANAGER.submit(work)
    return (
        job_id,
        "処理を開始しました…（進捗はこの欄に表示されます）",
        gr.Timer(active=True),
        gr.update(interactive=False),
    )


def poll_route_a_job(job_id: str):
    """ルートAジョブ状態を 1 件読み、出力 / 進捗テキスト / Timer / ボタンを更新する（ERR058）。"""
    no_change = (gr.update(),) * 6
    if not job_id:
        return (*no_change, gr.update(), gr.Timer(active=False), gr.update(interactive=True))
    snapshot = _JOB_MANAGER.snapshot(job_id)
    if snapshot.status == "running":
        return (*no_change, _progress_text(snapshot.fraction, snapshot.description), gr.update(), gr.update())
    if snapshot.status == "error":
        message = snapshot.error or "ルートA動画処理に失敗しました。"
        if job_id in _REPORTED_JOB_ERRORS:
            # 2 回目以降の tick: Timer を止め UI を復帰させる。
            return (*no_change, f"❌ 失敗: {message}", gr.Timer(active=False), gr.update(interactive=True))
        _REPORTED_JOB_ERRORS.add(job_id)
        # 初回: gr.Error で通知（握り潰さない / Hard Rule）。次 tick で Timer 停止・ボタン復帰。
        raise gr.Error(message)
    rgba, alpha, preview, overlay, seq_files, seq_dirs, status = snapshot.result
    return (rgba, alpha, preview, overlay, seq_files, seq_dirs, status, gr.Timer(active=False), gr.update(interactive=True))


def start_route_a_only_job(
    video_path: str | None,
    max_frames: int,
    frame_step: int,
    output_mode_label: str,
    rgba_codec: str,
    refine_foreground: bool,
    output_type: str,
):
    """BEN2 のみ動画ジョブを起動し、即座に job_id と Timer 活性化を返す（非同期/ERR058）。"""
    if not video_path:
        raise gr.Error("先に動画をアップロードしてください。")
    args = (video_path, max_frames, frame_step, output_mode_label, rgba_codec, refine_foreground, output_type)

    def work(report):
        return run_route_a_only_background_removal(*args, progress=_ProgressBridge(report))

    job_id = _JOB_MANAGER.submit(work)
    return (
        job_id,
        "処理を開始しました…（進捗はこの欄に表示されます）",
        gr.Timer(active=True),
        gr.update(interactive=False),
    )


def poll_route_a_only_job(job_id: str):
    """BEN2 のみジョブ状態を 1 件読み、出力 / 進捗テキスト / Timer / ボタンを更新する（ERR058）。"""
    no_change = (gr.update(),) * 5
    if not job_id:
        return (*no_change, gr.update(), gr.Timer(active=False), gr.update(interactive=True))
    snapshot = _JOB_MANAGER.snapshot(job_id)
    if snapshot.status == "running":
        return (*no_change, _progress_text(snapshot.fraction, snapshot.description), gr.update(), gr.update())
    if snapshot.status == "error":
        message = snapshot.error or "BEN2 のみ処理に失敗しました。"
        if job_id in _REPORTED_JOB_ERRORS:
            return (*no_change, f"❌ 失敗: {message}", gr.Timer(active=False), gr.update(interactive=True))
        _REPORTED_JOB_ERRORS.add(job_id)
        raise gr.Error(message)
    rgba, alpha, preview, seq_files, seq_dirs, status = snapshot.result
    return (rgba, alpha, preview, seq_files, seq_dirs, status, gr.Timer(active=False), gr.update(interactive=True))


def update_codec_visibility(output_mode_label: str):
    """連番のみ選択時は動画 codec UI を無効化する。"""
    sequence_only = normalize_output_mode(output_mode_label) == "sequence"
    return gr.update(interactive=not sequence_only, visible=not sequence_only)


def update_point_label_visibility(prompt_mode: str):
    """point prompt の時だけ positive / negative 選択を表示する。"""
    return gr.update(visible=prompt_mode == "point")


def refresh_prompt_selection_widgets(prompt_state: dict | None):
    """prompt_state から point / bbox 個別削除 UI の選択肢を再生成する。"""
    choices = build_prompt_selection_choices(prompt_state)
    return (
        gr.update(choices=choices["point_choices"], value=[]),
        gr.update(choices=choices["box_choices"], value=[]),
    )


def _render_prompt_preview(first_frame, state: dict | None):
    """現在 state を prompt canvas に描画して返す。"""
    preview_base = first_frame if first_frame is not None else create_prompt_canvas_placeholder()
    return draw_prompt_overlay(preview_base, state)


def remove_selected_prompt_points(first_frame, prompt_state: dict | None, selected_point_labels):
    """選択された point prompt（positive / negative）を削除する。"""
    selected = list(selected_point_labels or [])
    if not selected:
        raise gr.Error("削除する point prompt を選択してください。")
    state = remove_selected_points(prompt_state, selected)
    preview = _render_prompt_preview(first_frame, state)
    point_choices, box_choices = refresh_prompt_selection_widgets(state)
    status = f"選択 point を {len(selected)} 件削除しました。残り {len(state.get('points') or [])} 件。"
    return preview, state, status, point_choices, box_choices


def remove_selected_prompt_boxes(first_frame, prompt_state: dict | None, selected_box_labels):
    """選択された bbox（manual / union）を削除する。"""
    selected = list(selected_box_labels or [])
    if not selected:
        raise gr.Error("削除する bbox を選択してください。")
    state = remove_selected_boxes(prompt_state, selected)
    preview = _render_prompt_preview(first_frame, state)
    point_choices, box_choices = refresh_prompt_selection_widgets(state)
    manual = "あり" if state.get("box") is not None else "なし"
    status = f"選択 bbox を {len(selected)} 件削除しました。manual={manual}, union={len(state.get('boxes') or [])} 件。"
    return preview, state, status, point_choices, box_choices


def update_bidirectional_for_tracker(tracker_id: str):
    """tracker registry の supports_bidirectional に従い双方向伝播 UI を切り替える（ERR050）。"""
    try:
        entry = entry_by_id("tracker", tracker_id) if tracker_id else None
    except KeyError:
        return gr.update(interactive=True)
    supports = bool(entry.get("supports_bidirectional", True)) if entry else True
    if supports:
        return gr.update(interactive=True)
    return gr.update(value=False, interactive=False)


_BLUR_DEFAULTS = _blur_defaults()
_COMPOSITE_DEFAULTS = _composite_defaults()
_ALPHA_DEFAULTS = _alpha_defaults()

with gr.Blocks(title="SAM2 + BEN2 Route A (ブラー誘導) for Movie") as demo:
    gr.Markdown("# SAM2 + BEN2 Route A（ブラー誘導 → 再α化）for Movie")
    gr.Markdown(
        """
> ## ルートA案（ブラー誘導 → BEN2 再α化）とは
> SAM2 が出す下地マスク **M** を少し膨張させてゲート **G** を作り、**G の外側だけを強くブラー**した
> 誘導フレーム **I'** を **BEN2**（前景マッティングモデル）に渡して、α を高品質に作り直す方式です。
> 背景がボケて「ここが前景」と誘導されるため、髪や手足の細部のα品質が上がりやすくなります。
>
> | パラメーター | 役割 | 目安 |
> |---|---|---|
> | **下地マスク膨張(px)** | G を M からどれだけ広げるか。大きいほど前景候補を広く残す。 | 16〜32 |
> | **背景ブラーカーネル(px)** | G 外をどれだけ強くぼかすか（奇数）。大きいほど誘導が強い。 | 31〜61 |
> | **羽根幅(px)** | G 境界の馴染ませ幅。大きいほど境界が滑らか。 | 8〜16 |
> | **境界精緻化** | BEN2 の髪エッジ後処理。ON で精度↑/時間↑。 | 必要時のみ ON |
> | **合成モード** | union: frameあたりBEN2 1回（高速）。per_object: 対象数×回（忠実）。 | まず union |
> | **SAM2マスクでα底上げ（合成）** | SAM2 の安定マスクを α の「床」に合成し、BEN2 の取りこぼし（ちらつき）を補う。難しい動画向け。 | ちらつき時に screen / 比較明 |
>
> 既定値は `config/route_a.toml` から読み込んでいます。微調整は Advanced から行えます。
"""
    )
    gr.Markdown(
        """
### 使い方（クイックプレビュー推奨）
1. **Input Video** に動画をアップロードします。アップロード後、右側の **SAM2 Prompt Canvas** に第 1 フレームが自動表示されます。
2. 複合対象を意味で選びたい場合は **Optional: Text Prompt to Box** を開き、`person playing drums` のように入力して bbox 候補を作ります。
3. **SAM2 Prompt Canvas** 上で対象を確認します。手動 bbox は対象を囲む **対角 2 点** をクリックします。
4. まずは既定値（最大 30 frames・union）で **ルートA実行** し、結果を確認してから Advanced で膨張量・ブラー強度を微調整します。

**処理順**: `フレーム取得 → DINO で候補生成 → SAM2 で追跡（logit 保持）→ 所有権解決 → ルートA（下地マスク膨張→背景ブラー→BEN2 再α化）→ 出力`。
"""
    )

    with gr.Tabs():
        with gr.Tab("SAM2 追跡 + ルートA"):
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
                    info="SAM2 プロンプトを与えるフレームの位置（サンプリング後シーケンスの 0 始まりの番号、整数）。0 で第1フレーム。最大処理フレーム数を超える値でも、その frame まで自動で読み込んで実行する。SAMURAI は forward-only なので 0（先頭）推奨。目安 0。",
                )
                show_frame_btn = gr.Button("このフレームを表示")
            bidirectional = gr.Checkbox(
                value=False, label="双方向伝播（前後フレームへ追跡）",
                info="ON: 起点フレームから過去・未来方向の両方へ追跡を伝搬し各 frame の mask を OR 統合する（処理時間は約2倍）。OFF: 未来方向のみ。真偽値。SAMURAI 選択時は forward-only のため自動で OFF・無効化される。",
            )

            prompt_state = gr.State(value=empty_prompt_state())
            # prompt_base_image: overlay を含まないクリーンフレーム。全 overlay 描画の基準とし、
            # clear / 個別削除 を冪等に UI 反映させる（ERR036 系の overlay 焼き込み防止）。
            prompt_base_image = gr.State(value=create_prompt_canvas_placeholder())
            prompt_status = gr.Textbox(label="Prompt Status", interactive=False)
            with gr.Accordion("Prompt 編集（個別削除）", open=False):
                gr.Markdown("誤って追加した point（positive/negative）や bbox（manual/union）だけを選択して削除できます。")
                with gr.Row():
                    point_selection_group = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="削除する point prompt を選択",
                        info="positive / negative を個別に削除します。",
                    )
                    box_selection_group = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="削除する bbox を選択",
                        info="manual box と union boxes を個別に削除します。",
                    )
                with gr.Row():
                    remove_selected_points_btn = gr.Button("選択した point を削除")
                    remove_selected_boxes_btn = gr.Button("選択した bbox を削除")

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
                        info="GroundingDINO の物体信頼度しきい値（0.05〜0.95 のスコア、単位なし）。高いほど検出が厳しく誤検出が減る。目安 0.25。",
                    )
                    text_text_threshold = gr.Slider(
                        0.05, 0.95, value=0.25, step=0.01, label="Text threshold",
                        info="検出 box と text 語句の一致しきい値（0.05〜0.95、単位なし）。高いほど語句との合致を厳しく要求する。目安 0.25。",
                    )
                    text_top_k = gr.Slider(
                        1, 20, value=20, step=1, label="候補数 top-k",
                        info="保持する検出 box の最大個数（個数、整数）。スコア上位から K 個まで表示し最上位を SAM2 prompt にコピーする。目安 20。",
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
                    info="複数選択すると各 bbox を別オブジェクトとして追跡し frame ごとに OR 統合する。検出後に top1 が自動 ON。選択後に下のボタンで反映。単位なしの選択値。",
                )
                apply_boxes_btn = gr.Button("選択した候補 bbox を複合対象として反映")

            with gr.Row():
                tracker_model = gr.Dropdown(
                    choices=build_dropdown_choices("tracker"),
                    value="sam2_hiera_l",
                    label="トラッカーモデル（Tracker Model）",
                    info="SAM2/SAMURAI のモデル。変更時はパイプラインが再初期化されます（初回のみ時間がかかります）。複数対象は標準 SAM2 を推奨。",
                )
                matte_mode = gr.Radio(
                    MATTE_MODE_CHOICES, value=MATTE_MODE_CHOICES[0], label="合成モード（Matte Mode）",
                    info="union: union マスクから 1 ゲートを作り BEN2 を frame あたり 1 回（高速）。per_object: 対象ごとに誘導・BEN2 推論し所有権で α 合成（忠実だが対象数×回で重い）。単位なしの選択値。",
                )

            run_btn = gr.Button("ルートA実行", variant="primary")

            with gr.Accordion("Advanced: ルートA / 動画処理設定", open=False):
                gr.Markdown(
                    """
| パラメーター | 何を変えるか | 目安 |
|---|---|---|
| 下地マスク膨張(px) | ゲート G を SAM2 マスクからどれだけ広げるか。 | 16〜32 |
| 背景ブラーカーネル(px) | G 外をどれだけ強くぼかすか（奇数）。 | 31〜61 |
| 背景ブラーσ | ガウシアンの標準偏差。0 でカーネルから自動。 | 0 |
| 羽根幅(px) | G 境界の馴染ませ幅。 | 8〜16 |
| 境界精緻化 | BEN2 の髪エッジ後処理（精度↑/時間↑）。 | 必要時のみ ON |
| ゲートでα制限 | BEN2 αをゲート内に限定し背景漏れを抑える（αを絞る）。 | 通常 OFF |
| SAM2マスクでα底上げ | SAM2 の安定マスクを α の床にし BEN2 のちらつきを補う（αを底上げ）。 | none、ちらつき時 screen/比較明 |
| 最大処理フレーム数 | 先頭から処理する frame 数。 | 初回 30、最終 300+ |
"""
                )
                with gr.Row():
                    dilation_px = gr.Slider(
                        0, 128, value=int(_BLUR_DEFAULTS.get("dilation_px", 24)), step=1, label="下地マスク膨張(px)",
                        info="SAM2 下地マスク M を膨張してゲート G を作る量（px、整数）。0 で二値化のみ。大きいほど前景候補を広く残すが背景も入りやすい。目安 16〜32。",
                    )
                    blur_kernel = gr.Slider(
                        1, 151, value=int(_BLUR_DEFAULTS.get("blur_kernel", 41)), step=2, label="背景ブラーカーネル(px)",
                        info="ゲート外をぼかすガウシアンカーネルサイズ（px、奇数）。大きいほど誘導が強い。偶数は自動で +1 補正。目安 31〜61。",
                    )
                with gr.Row():
                    blur_sigma = gr.Slider(
                        0.0, 50.0, value=float(_BLUR_DEFAULTS.get("blur_sigma", 0.0)), step=0.5, label="背景ブラーσ",
                        info="ガウシアンブラーの標準偏差（単位なし）。0.0 でカーネルサイズから自動算出。大きいほど強くぼかす。目安 0.0。",
                    )
                    feather_px = gr.Slider(
                        0, 64, value=int(_BLUR_DEFAULTS.get("feather_px", 12)), step=1, label="羽根幅(px)",
                        info="ゲート G 境界をぼかして鮮明部と背景を馴染ませる幅（px、整数）。大きいほど境界が滑らか。目安 8〜16。",
                    )
                with gr.Row():
                    refine_foreground = gr.Checkbox(
                        value=bool(_ALPHA_DEFAULTS.get("refine_foreground", False)), label="境界精緻化（refine foreground）",
                        info="ON: BEN2 の髪エッジ後処理を行い境界品質を上げる（時間↑）。OFF: 標準。真偽値。",
                    )
                    gate_alpha = gr.Checkbox(
                        value=bool(_COMPOSITE_DEFAULTS.get("gate_alpha", False)), label="ゲートでαを制限",
                        info="ON: BEN2 α をゲート G 内に限定し、ゲート外の背景誤検出を 0 にする。OFF: BEN2 α をそのまま使う。真偽値。",
                    )
                gr.Markdown(
                    """
**SAM2マスクでα底上げ（合成）＝ちらつき対策**
SAM2 が最後まで追跡できているのに BEN2 の α がフレームごとに揺れて被写体が点滅する場合、
SAM2 の安定マスク **M** を α の「床（最低値）」として合成し、抜け落ちを埋めて点滅を抑えます。
`gate_alpha`（α を絞る）とは逆向きの **α 底上げ** で、両者は併用できます。

- **none**: 無効（BEN2 の α をそのまま使う・既定）
- **screen**: 1-(1-α)(1-M) で柔らかく底上げ（境界が自然）
- **lighten / 比較明**: 画素ごと max(α, M) で確実に塗る（前景ブラー・背景同系色・高速な被写体向け）
"""
                )
                with gr.Row():
                    mask_floor_mode = gr.Radio(
                        MASK_FLOOR_MODE_CHOICES,
                        value=_mask_floor_label_from_value(_COMPOSITE_DEFAULTS.get("mask_floor_mode", "none")),
                        label="SAM2マスクでα底上げ（合成）",
                        info=(
                            "SAM2 の安定マスクを α の『床』として加算合成し BEN2 の抜け落ち（ちらつき）を補う。"
                            "screen=1-(1-α)(1-M) で自然に底上げ。lighten=max(α,M) で確実に塗る（前景ブラー/背景同系色の動画向け）。"
                            "gate_alpha が α を絞るのに対し、本機能は α を底上げ（逆向き）。"
                        ),
                    )
                with gr.Row():
                    max_frames = gr.Slider(1, 2000, value=30, step=1, label="最大処理フレーム数",
                        info="先頭から処理する frame 数（frame 数、整数）。多いほど処理が長く出力も重くなる。初回 30、最終確認 300 以上が目安。",
                    )
                    frame_step = gr.Slider(1, 10, value=1, step=1, label="フレーム間引きステップ",
                        elem_id="movie-frame-step",
                        info="何 frame ごとに1枚処理するか（frame 間隔、整数）。1 で全 frame。2 以上は速いが追跡が粗くなる。通常 1。",
                    )
                with gr.Row():
                    output_mode = gr.Radio(
                        OUTPUT_MODE_CHOICES, value="動画 (video)", label="出力形式",
                        info="動画: 1つの動画ファイル。連番静止画: frame ごとの PNG。両方: 動画と PNG の両方。codec エラー時は連番が安全。単位なしの選択値。",
                    )
                    rgba_codec = gr.Radio(
                        ["webm_vp9", "mov_png"], value="webm_vp9", label="RGBA 動画コーデック",
                        info="透明付き動画の保存方式。webm_vp9: VP9/WebM（軽量・推奨）。mov_png: PNG コーデックの MOV（高互換・大容量）。書き出せない環境では連番(PNG)が確実。単位なしの選択値。",
                    )
                output_type = gr.Radio(
                    ["rgba", "green", "white", "blur"], value="rgba", label="Preview type",
                    info="プレビュー動画の背景表示方法。rgba: 透明。green: 緑背景。white: 白背景。blur: 元背景をぼかす。RGBA 出力本体には影響しない。単位なしの選択値。",
                )
                overlay_enabled = gr.Checkbox(
                    value=True, label="Tracking Overlay を生成",
                    info="ON: SAM2/SAMURAI の追跡 mask を元動画に半透明で重ねた確認用動画を生成する（処理がわずかに増えます）。OFF: 生成しない。真偽値。",
                )

            with gr.Row():
                rgba_video = gr.Video(label="RGBA Video")
                alpha_video = gr.Video(label="Alpha Video")
                preview_video = gr.Video(label="Preview Video")
            tracking_overlay_video = gr.Video(label="Tracking Overlay (追跡確認用)")
            sequence_files = gr.Files(label="連番 PNG サンプル")
            sequence_dirs = gr.Textbox(label="連番出力フォルダ", interactive=False)
            run_status = gr.Markdown()
            route_a_job_id = gr.State("")
            route_a_timer = gr.Timer(1.0, active=False)

        with gr.Tab("BEN2 のみ (追跡なし)"):
            gr.Markdown(
                """
### BEN2 のみ（SAM2 追跡なし）
単一 salient 対象など、追跡が不要で BEN2 だけで足りる用途向けの軽量経路です。
**prompt は不要**で、各フレームをそのまま BEN2 に渡して α を生成します（誘導ブラー・所有権合成・tracking overlay は行いません）。
複数人物・複合対象を選び分けたい場合や対象を限定して追跡したい場合は「SAM2 追跡 + ルートA」タブを使ってください。
"""
            )
            route_a_only_input_video = gr.Video(label="Input Video", sources=["upload"], elem_id="ben2-only-input-video")
            route_a_only_run_btn = gr.Button("BEN2 のみを実行", variant="primary")

            with gr.Accordion("Advanced: 動画処理設定", open=False):
                with gr.Row():
                    route_a_only_max_frames = gr.Slider(1, 2000, value=30, step=1, label="最大処理フレーム数",
                        info="先頭から処理する frame 数（frame 数、整数）。多いほど処理が長く出力も重くなる。初回 30、最終確認 300 以上が目安。",
                    )
                    route_a_only_frame_step = gr.Slider(1, 10, value=1, step=1, label="フレーム間引きステップ",
                        info="何 frame ごとに1枚処理するか（frame 間隔、整数）。1 で全 frame。2 以上は速いが粗くなる。通常 1。",
                    )
                with gr.Row():
                    route_a_only_output_mode = gr.Radio(
                        OUTPUT_MODE_CHOICES, value="動画 (video)", label="出力形式",
                        info="動画: 1つの動画ファイル。連番静止画: frame ごとの PNG。両方: 動画と PNG の両方。codec エラー時は連番が安全。単位なしの選択値。",
                    )
                    route_a_only_rgba_codec = gr.Radio(
                        ["webm_vp9", "mov_png"], value="webm_vp9", label="RGBA 動画コーデック",
                        info="透明付き動画の保存方式。webm_vp9: VP9/WebM（軽量・推奨）。mov_png: PNG コーデックの MOV（高互換・大容量）。書き出せない環境では連番(PNG)が確実。単位なしの選択値。",
                    )
                route_a_only_refine = gr.Checkbox(
                    value=bool(_ALPHA_DEFAULTS.get("refine_foreground", False)), label="境界精緻化（refine foreground）",
                    info="ON: BEN2 の髪エッジ後処理を行い境界品質を上げる（時間↑）。OFF: 標準。真偽値。",
                )
                route_a_only_output_type = gr.Radio(
                    ["rgba", "green", "white", "blur"], value="rgba", label="Preview type",
                    info="プレビュー動画の背景表示方法。rgba: 透明。green: 緑背景。white: 白背景。blur: 元背景をぼかす。RGBA 出力本体には影響しない。単位なしの選択値。",
                )

            with gr.Row():
                route_a_only_rgba_video = gr.Video(label="RGBA Video")
                route_a_only_alpha_video = gr.Video(label="Alpha Video")
                route_a_only_preview_video = gr.Video(label="Preview Video")
            route_a_only_sequence_files = gr.Files(label="連番 PNG サンプル")
            route_a_only_sequence_dirs = gr.Textbox(label="連番出力フォルダ", interactive=False)
            route_a_only_run_status = gr.Markdown()
            route_a_only_job_id = gr.State("")
            route_a_only_timer = gr.Timer(1.0, active=False)

    input_video.change(extract_first_frame_outputs, inputs=[input_video], outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status, prompt_frame_idx]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    load_first_frame_btn.click(extract_first_frame_with_base, inputs=[input_video], outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    prompt_canvas.select(select_sam2_prompt, inputs=[prompt_base_image, prompt_mode, point_label, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    clear_prompt_btn.click(clear_prompt, inputs=[prompt_base_image], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    extend_left_btn.click(lambda image, state: extend_box_to_edge(image, state, "left"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    extend_right_btn.click(lambda image, state: extend_box_to_edge(image, state, "right"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    extend_top_btn.click(lambda image, state: extend_box_to_edge(image, state, "top"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    extend_bottom_btn.click(lambda image, state: extend_box_to_edge(image, state, "bottom"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    detect_text_btn.click(
        detect_text_boxes_for_video,
        inputs=[prompt_base_image, text_prompt, text_box_threshold, text_text_threshold, text_top_k, prompt_state],
        outputs=[prompt_canvas, prompt_state, detected_boxes, prompt_status],
    ).then(
        populate_candidate_choices,
        inputs=[detected_boxes],
        outputs=[box_candidate_group],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    apply_boxes_btn.click(
        apply_selected_boxes,
        inputs=[prompt_base_image, prompt_state, box_candidate_group],
        outputs=[prompt_canvas, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    show_frame_btn.click(
        extract_prompt_frame_with_base,
        inputs=[input_video, prompt_frame_idx, frame_step],
        outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    prompt_frame_idx.change(
        extract_prompt_frame_with_base,
        inputs=[input_video, prompt_frame_idx, frame_step],
        outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    remove_selected_points_btn.click(
        remove_selected_prompt_points,
        inputs=[prompt_base_image, prompt_state, point_selection_group],
        outputs=[prompt_canvas, prompt_state, prompt_status, point_selection_group, box_selection_group],
    )
    remove_selected_boxes_btn.click(
        remove_selected_prompt_boxes,
        inputs=[prompt_base_image, prompt_state, box_selection_group],
        outputs=[prompt_canvas, prompt_state, prompt_status, point_selection_group, box_selection_group],
    )
    output_mode.change(update_codec_visibility, inputs=[output_mode], outputs=[rgba_codec])
    prompt_mode.change(update_point_label_visibility, inputs=[prompt_mode], outputs=[point_label])
    tracker_model.change(update_bidirectional_for_tracker, inputs=[tracker_model], outputs=[bidirectional])
    # ローカル実行前提では gradio.live トンネルの SSE 切断（ERR058）が起きないため、非同期ジョブ方式
    # （JobManager + gr.Timer ポーリング）を同期直結に戻す（ERR064）。これにより Gradio 標準の推論
    # プログレス（スピナー）が復活し、描画系 4 出力（RGBA/Alpha/Preview/Tracking Overlay）＋連番2＋status が
    # run_btn の戻り値から直接描画される。非同期関数群（start_*/poll_* と JobManager）は Colab 再利用に備え温存。
    run_btn.click(
        run_route_a_background_removal,
        inputs=[input_video, prompt_state, prompt_frame_idx, bidirectional, max_frames, frame_step, output_mode, rgba_codec, tracker_model, matte_mode, dilation_px, blur_kernel, blur_sigma, feather_px, refine_foreground, gate_alpha, mask_floor_mode, output_type, overlay_enabled],
        outputs=[rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status],
    )

    route_a_only_output_mode.change(update_codec_visibility, inputs=[route_a_only_output_mode], outputs=[route_a_only_rgba_codec])
    route_a_only_run_btn.click(
        run_route_a_only_background_removal,
        inputs=[route_a_only_input_video, route_a_only_max_frames, route_a_only_frame_step, route_a_only_output_mode, route_a_only_rgba_codec, route_a_only_refine, route_a_only_output_type],
        outputs=[route_a_only_rgba_video, route_a_only_alpha_video, route_a_only_preview_video, route_a_only_sequence_files, route_a_only_sequence_dirs, route_a_only_run_status],
    )


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="SAM2 + BEN2 Route A (blur-guidance) Haystack movie demo")
    parser.add_argument("--share", action="store_true", help="Gradio public link を有効化")
    parser.add_argument("--debug", action="store_true", help="Gradio debug mode")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server name")
    parser.add_argument("--server-port", type=int, default=7862, help="Gradio server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.queue()
    demo.launch(share=args.share, debug=args.debug, server_name=args.server_name, server_port=args.server_port, show_api=False)
