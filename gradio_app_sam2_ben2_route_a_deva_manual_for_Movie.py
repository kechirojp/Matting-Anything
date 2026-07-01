"""Haystack Pipeline 版 DEVA 方式 + 手動プロンプト SAM2 + BEN2（ルートA）動画αマットデモ。

``gradio_app_sam2_ben2_route_a_deva_for_Movie`` を土台に、ベースアプリ
（``gradio_app_sam2_ben2_route_a_for_Movie``）の **手動 box / point（positive / negative）
プロンプト canvas** を統合した版。DEVA 方式の「周期再検出 × SAM2 伝播 × consensus 統合」を
維持しつつ、先頭フレームで対象を正確に seed したり、negative point で誤検出領域を除外できる。

2 つの動作モード（``DevaSemiOnlineTracker`` が引数で自動判定）:
    - モードA（手動 seed のみ）: Text Prompt を空にし、Canvas の box/point だけで追跡。
      検出島を走らせず、全フレームを単一クリップで伝播する（ベースアプリ相当）。
    - モードB（ハイブリッド・推奨）: Text Prompt も入力。先頭フレームを手動 box/point で
      正確に seed し、以降は GroundingDINO(text) が ``detection_every`` ごとに再検出して
      consensus で「はがれ→自動復帰」する。手動 point/label は先頭クリップの seed にのみ適用。

NOTE: DEVA「ライブラリ」は採用していない。DEVA の「方式」を SAM2.1 / BEN2 / GroundingDINO 上に
再構成したものである（プロジェクト方針）。手動 seed の SAM2 伝播は既存
``SAM2VideoPropagator`` の points/labels/boxes 経路（negative=label 0 含む）をそのまま利用する。
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
    # クライアント切断時の無害な ConnectionResetError（WinError 10054）だけを黙らせる。
    from asyncio.proactor_events import _ProactorBasePipeTransport
    from functools import wraps as _wraps

    def _silence_proactor_connection_reset(func):
        @_wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except ConnectionResetError:
                return None

        return wrapper

    _ProactorBasePipeTransport._call_connection_lost = _silence_proactor_connection_reset(
        _ProactorBasePipeTransport._call_connection_lost
    )

import gradio as gr

from pipelines.components.model_components import GroundingDINOMultiBoxDetector
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
from pipelines.components.video_common import normalize_output_mode
from pipelines.route_a_deva_video_pipeline import build_sam2_ben2_route_a_deva_pipeline
from pipelines.sam2_tb_video_pipeline import build_video_reader_pipeline


_DEVA_PIPELINE: Any = None
READER_PIPELINE: Any = None
TEXT_DETECTOR: Any = None
PROMPT_CANVAS_PLACEHOLDER_SIZE = (420, 640)
OUTPUT_MODE_CHOICES = ["動画 (video)", "連番静止画 (sequence)", "両方 (both)"]
MATTE_MODE_CHOICES = ["union（高速・1回/frame）", "per_object（忠実・対象数×/frame）"]
MASK_FLOOR_MODE_CHOICES = [
    "none（無効）",
    "screen（底上げ・自然）",
    "lighten / 比較明（確実に塗る）",
]
STAGE_PROGRESS_RANGES = {
    "video_reader": (0.05, 0.12),
    "deva_tracker": (0.12, 0.58),
    "ben2_route_a": (0.58, 0.88),
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


def _deva_per_object_logits_max_side() -> int:
    """config/route_a.toml [deva] の per_object_logits 長辺上限（px）を返す（既定 1024）。"""
    deva_cfg = _ROUTE_A_DEFAULTS.get("deva", {}) or {}
    try:
        return int(deva_cfg.get("per_object_logits_max_side", 1024))
    except (TypeError, ValueError):
        return 1024


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


def get_route_a_deva_pipeline():
    """DEVA 方式ルートA動画αマット Pipeline を遅延構築する（singleton）。"""
    global _DEVA_PIPELINE
    if _DEVA_PIPELINE is None:
        _DEVA_PIPELINE = build_sam2_ben2_route_a_deva_pipeline()
    return _DEVA_PIPELINE


def get_reader_pipeline():
    """第 1 フレーム抽出用 Pipeline を遅延構築する。"""
    global READER_PIPELINE
    if READER_PIPELINE is None:
        READER_PIPELINE = build_video_reader_pipeline()
    return READER_PIPELINE


def get_text_detector():
    """GroundingDINO Text Prompt 検出器（Canvas 補助の Text→Box 用）を遅延構築する。"""
    global TEXT_DETECTOR
    if TEXT_DETECTOR is None:
        TEXT_DETECTOR = GroundingDINOMultiBoxDetector()
    return TEXT_DETECTOR


def release_text_detector() -> None:
    """動画処理前に Canvas 補助の GroundingDINO を解放し、GPU VRAM のピークを下げる。"""
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
    cv2.putText(canvas, "box: drag 2 corners / point: click foreground or background.", (28, height // 2 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (105, 113, 128), 2)
    return canvas


# ─────────────────────────────────────────
# ## フレーム取得系（手動プロンプトの先頭フレーム）
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
    """動画アップロード時に第 1 フレームを自動反映する。

    canvas 表示用画像（overlay 焼き込み済み）とは別に、描画基準のクリーンフレームを
    prompt_base_image へ返す。clear / 個別削除がクリーン基準から再描画され冪等に反映される。
    出力順: (canvas, base, state, status)。
    """
    if not video_path:
        placeholder = create_prompt_canvas_placeholder()
        return (
            placeholder,
            placeholder,
            empty_prompt_state(),
            "動画をアップロードすると第 1 フレームを自動取得します。",
        )
    frame, state, status = extract_first_frame(video_path)
    return frame, frame, state, status


def extract_first_frame_with_base(video_path: str | None):
    """第 1 フレームを取得し、canvas と描画基準 base の両方にクリーンフレームを返す。

    出力順: (canvas, base, state, status)。
    """
    frame, state, status = extract_first_frame(video_path)
    return frame, frame, state, status


def extract_prompt_frame(video_path: str | None, detection_start_frame: int, frame_step: int):
    """検出起点フレーム位置のクリーンフレームを取り出し、prompt canvas を張り替える。

    被写体が最大に映るフレームに box/point を描けるよう、``detection_start_frame``
    （VideoReader が ``frame_step`` でサンプリングした後の index）のフレームを返す。
    フレームが変わると既存の box 座標は無効になるため prompt_state は初期化する。
    出力順: (canvas, base, state, status)。この index は DEVA トラッカーへ渡す
    ``detection_start_frame`` と同一規約で、clip0 の seed 位置と一致する。
    """
    try:
        if not video_path:
            raise gr.Error("先に動画をアップロードしてください。")
        sampled_index = max(int(detection_start_frame), 0)
        result = get_reader_pipeline().run(
            {"video_reader": {"video_path": video_path, "max_frames": sampled_index + 1, "frame_step": int(frame_step)}},
            include_outputs_from={"video_reader"},
        )
        frames = result["video_reader"]["frames"]
        metadata = result["video_reader"]["metadata"]
        effective_index = min(sampled_index, len(frames) - 1)
        frame = frames[effective_index]
        raw_index = effective_index * int(frame_step)
        state = empty_prompt_state()
        status = (
            f"検出起点フレームをシーケンス位置 {effective_index}（元動画フレーム≈{raw_index}）に更新しました。"
            f"被写体が最大に映るこのフレームに box/point を描いてください。"
            f"サイズ {metadata['width']}x{metadata['height']}。"
        )
        return frame, frame, state, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"検出起点フレームの取得に失敗しました: {exc}") from exc


def clear_prompt(base_image):
    """SAM2 video prompt を初期化し、canvas をクリーン基準フレームへ戻す。"""
    state = empty_prompt_state()
    preview = base_image if base_image is not None else create_prompt_canvas_placeholder()
    return preview, state, "SAM2 video prompt cleared"


# ─────────────────────────────────────────
# ## Text Prompt → Box（GroundingDINO 補助）
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
    """gr.Dataframe 値（pandas DataFrame / list / None）を行リストへ正規化する。"""
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


# ─────────────────────────────────────────
# ## Prompt 編集（個別削除）・モード切替
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# ## 出力整形ヘルパー（ベースアプリと同一契約）
# ─────────────────────────────────────────
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


def _effective_read_frames(max_frames: int, detection_start_frame: int) -> int:
    """検出起点フレームを必ず読み込み窓に含むよう、実効的な読み込み frame 数を返す。

    ``max_frames`` を「最低限処理する frame 数」と扱い、``detection_start_frame`` がそれを
    超える場合は起点フレームを含むよう窓を引き上げる。これにより DEVA トラッカーの
    範囲外エラー（detection_start_frame >= num_frames）を避けて任意フレーム起点で seed できる。
    """
    return max(1, int(max_frames), int(detection_start_frame) + 1)


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
# ## DEVA 方式 + 手動プロンプト 実行系
# ─────────────────────────────────────────
def run_route_a_deva_manual_background_removal(
    video_path: str | None,
    prompt_state: dict | None,
    text_prompt: str,
    detection_every: int,
    max_missed_detection_count: int,
    iou_threshold: float,
    box_threshold: float,
    text_threshold: float,
    top_k: int,
    detection_start_frame: int,
    max_frames: int,
    frame_step: int,
    output_mode_label: str,
    rgba_codec: str,
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
    """DEVA 方式 + 手動プロンプト ルートA動画αマット Pipeline を実行する。"""
    stage_state: dict[str, Any] = {"name": "startup", "description": "Pipeline を起動しています"}
    total_start = time.perf_counter()
    try:
        if not video_path:
            raise gr.Error("先に動画をアップロードしてください。")

        # ## 手動 seed（box / point pos・neg）を prompt_state から取り出す。
        state = copy_prompt_state(prompt_state)
        points = state.get("points") or []
        labels = state.get("labels") or []
        manual_box = state.get("box")
        union_boxes = state.get("boxes") or []
        # union（複数 box）を優先し、無ければ手動 single box を採用する。
        initial_boxes = (
            [list(single) for single in union_boxes]
            if union_boxes
            else ([list(manual_box)] if manual_box is not None else [])
        )
        initial_points = [list(point) for point in points] if points else None
        initial_labels = [int(value) for value in labels] if points else None

        text = str(text_prompt or "").strip()
        if not text and not initial_boxes:
            raise gr.Error(
                "Text Prompt を入力するか、SAM2 Prompt Canvas で box / point を指定してください"
                "（手動 seed のみのモードA、または Text 併用のモードB）。"
            )
        if initial_points and not initial_boxes:
            raise gr.Error(
                "point は box とあわせて指定してください。box で対象を囲み、point（positive / negative）で"
                "前景・背景を補正します。DEVA 方式では point 単独追跡はできません。"
            )
        if int(detection_every) <= 0:
            raise gr.Error("再検出周期（detection_every）は 1 以上で指定してください。")

        detection_start_frame = max(int(detection_start_frame), 0)
        # 起点フレームを必ず読み込み窓に含める（範囲外エラー回避、任意フレーム起点で seed 可能に）。
        effective_max_frames = _effective_read_frames(int(max_frames), detection_start_frame)
        processed_frames = _estimate_processed_frames(effective_max_frames, int(frame_step))
        output_mode = normalize_output_mode(output_mode_label)
        matte_mode = _normalize_matte_mode(matte_mode_label)
        mask_floor_mode = _normalize_mask_floor_mode(mask_floor_mode_label)
        positive_count = sum(1 for value in (initial_labels or []) if int(value) == 1)
        negative_count = sum(1 for value in (initial_labels or []) if int(value) == 0)
        mode_label = "B（手動 seed + text 再検出）" if text else "A（手動 seed のみ・再検出なし）"

        # Canvas 補助の GroundingDINO を解放し、重い DEVA Pipeline ロード前に VRAM を空ける。
        release_text_detector()

        progress_callback = build_video_progress_callback(progress, stage_state)
        progress(
            0.03,
            desc=f"Pipeline を起動しています（モード{mode_label}）。初回はモデル読込（SAM2/BEN2/GroundingDINO）を伴います（最大 {processed_frames} frames）",
        )
        result = get_route_a_deva_pipeline().run(
            {
                "video_reader": {
                    "video_path": video_path,
                    "max_frames": effective_max_frames,
                    "frame_step": int(frame_step),
                    "progress_callback": progress_callback,
                },
                # ## DEVA 方式追跡 + 手動 seed: 先頭フレームを box/point で seed し、text があれば周期再検出。
                "deva_semi_online_tracker": {
                    "text_prompt": text,
                    "detection_every": int(detection_every),
                    "max_missed_detection_count": int(max_missed_detection_count),
                    "iou_threshold": float(iou_threshold),
                    "box_threshold": float(box_threshold),
                    "text_threshold": float(text_threshold),
                    "top_k": int(top_k),
                    "initial_boxes": initial_boxes or None,
                    "initial_points": initial_points,
                    "initial_labels": initial_labels,
                    "detection_start_frame": detection_start_frame,
                    # union 経路は per_object_logits を縮小して host-RAM を抑える（ERR068）。
                    # per_object 経路は所有権合成が原寸 logit を要するため 0（縮小なし）にする。
                    "per_object_logits_max_side": (
                        0 if matte_mode == "per_object" else _deva_per_object_logits_max_side()
                    ),
                    "progress_callback": progress_callback,
                },
                "ownership_resolver": {
                    "temperature": 1.0,
                },
                # ## ルートA背景透過: 下地マスク膨張 → 背景ブラー → BEN2 再α化。
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
                "tracking_overlay": {"enabled": bool(overlay_enabled), "progress_callback": progress_callback},
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
        sequence_dirs = [
            merged.get(key)
            for key in ("rgba_sequence_dir", "alpha_sequence_dir", "preview_sequence_dir")
            if merged.get(key)
        ]
        fallback = merged.get("metadata", {}).get("codec_fallback", [])
        status = (
            f"完了: モード{mode_label}, output_mode={output_mode}, matte_mode={matte_mode}, "
            f"frames={merged.get('frame_count')}, 手動box={len(initial_boxes)}, "
            f"point(pos={positive_count}, neg={negative_count}), "
            f"text='{text}', detection_every={int(detection_every)}, start_frame={detection_start_frame}, "
            f"max_missed={int(max_missed_detection_count)}, iou={float(iou_threshold):.2f}"
        )
        if fallback:
            status += f"\ncodec fallback: {fallback}"
        status += f"\n処理時間: total={time.perf_counter() - total_start:.1f}s"
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
        raise gr.Error(f"DEVA + 手動プロンプト 動画処理に失敗しました（stage={stage}, elapsed={elapsed:.1f}s）: {exc}") from exc


def update_codec_visibility(output_mode_label: str):
    """連番のみ選択時は動画 codec UI を無効化する。"""
    sequence_only = normalize_output_mode(output_mode_label) == "sequence"
    return gr.update(interactive=not sequence_only, visible=not sequence_only)


_BLUR_DEFAULTS = _blur_defaults()
_COMPOSITE_DEFAULTS = _composite_defaults()
_ALPHA_DEFAULTS = _alpha_defaults()

with gr.Blocks(title="SAM2 + BEN2 Route A (DEVA 方式 + 手動プロンプト) for Movie") as demo:
    gr.Markdown("# SAM2 + BEN2 Route A（DEVA 方式 + 手動 box / point プロンプト）for Movie")
    gr.Markdown(
        """
> ## このアプリの特徴
> DEVA 方式（**周期再検出 × SAM2 伝播 × consensus 統合**）に、ベースアプリの
> **手動 box / point（positive・negative）プロンプト**を統合した版です。
>
> | モード | Text Prompt | Canvas の box/point | 挙動 |
> |---|---|---|---|
> | **A. 手動 seed のみ** | 空 | 必須 | 先頭フレームの box/point から SAM2 伝播（再検出なし・ベース相当） |
> | **B. ハイブリッド（推奨）** | 入力 | 任意 | 先頭を box/point で精緻 seed → 以降 text で周期再検出し「はがれ復帰」 |
>
> - **point の positive/negative** で前景・背景を補正できます（negative で誤検出領域を除外）。
> - 手動 point は**先頭フレームの seed にのみ**反映されます（対象が動くため後続へ再投影しません）。
> - ルートA（ブラー誘導→BEN2 再α化）と出力系はベースアプリと同一です。既定値は `config/route_a.toml`。
"""
    )

    with gr.Tab("DEVA + 手動プロンプト + ルートA"):
        with gr.Row():
            input_video = gr.Video(label="Input Video", sources=["upload"], elem_id="movie-input-video")
            prompt_canvas = gr.Image(
                value=create_prompt_canvas_placeholder(),
                label="SAM2 Prompt Canvas（先頭フレーム）",
                type="numpy",
                sources=[],
                interactive=True,
            )

        with gr.Row():
            load_first_frame_btn = gr.Button("第1フレームを再取得")
            clear_prompt_btn = gr.Button("Prompt をクリア")

        with gr.Accordion("検出起点フレーム（被写体が最大に映るフレームで seed）", open=False):
            gr.Markdown(
                "被写体が先頭フレームで小さい/映っていない場合、**被写体が最大に映るフレーム**を"
                "検出起点にすると seed（box/point）と初回検出の精度が上がります。スライダを動かすと"
                "**Canvas がそのフレームに張り替わる**ので、その位置で box/point を描いてください。"
                "起点より前のフレームは**双方向逆伝播**でカバーされます（0 で従来通り先頭起点）。"
            )
            with gr.Row():
                detection_start_frame = gr.Slider(
                    0, 2000, value=0, step=1, label="検出起点フレーム（サンプリング後 index）",
                    elem_id="movie-detection-start-frame",
                    info="seed/検出を最初に確立するフレーム位置（0 始まり、frame_step 適用後の index）。動かすと Canvas がそのフレームへ更新され、既存 prompt は初期化される。0 は先頭。",
                )
                show_start_frame_btn = gr.Button("起点フレームを Canvas へ表示")

        with gr.Row():
            prompt_mode = gr.Radio(
                ["point", "box"], value="box", label="Input Mode",
                info="box: Prompt Canvas 上で対角 2 点をクリックし対象を囲む矩形を指定。point: クリック1点ごとに前景/背景ヒントを与える（複数点可）。単位なしの選択値。",
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

        prompt_state = gr.State(value=empty_prompt_state())
        # prompt_base_image: overlay を含まないクリーンフレーム。全 overlay 描画の基準とし、
        # clear / 個別削除 を冪等に UI 反映させる（ERR036 系の overlay 焼き込み防止）。
        prompt_base_image = gr.State(value=create_prompt_canvas_placeholder())
        prompt_status = gr.Textbox(label="Prompt Status", interactive=False)

        with gr.Accordion("Prompt 編集（個別削除）", open=False):
            gr.Markdown("誤って追加した point（positive/negative）や bbox（manual/union）だけを選択して削除できます。")
            with gr.Row():
                point_selection_group = gr.CheckboxGroup(
                    choices=[], value=[], label="削除する point prompt を選択",
                    info="positive / negative を個別に削除します。",
                )
                box_selection_group = gr.CheckboxGroup(
                    choices=[], value=[], label="削除する bbox を選択",
                    info="manual box と union boxes を個別に削除します。",
                )
            with gr.Row():
                remove_selected_points_btn = gr.Button("選択した point を削除")
                remove_selected_boxes_btn = gr.Button("選択した bbox を削除")

        text_prompt = gr.Textbox(
            label="Text Prompt（モードB: 周期再検出に使用。空ならモードA=手動 seed のみ）",
            placeholder="person playing drums / person riding bicycle（空欄可）",
            info="入力すると GroundingDINO が detection_every ごとに再検出し、はがれを自動復帰する（モードB）。空欄なら再検出せず手動 seed のみで伝播する（モードA）。",
        )

        with gr.Accordion("Optional: Text Prompt から box を自動作成（GroundingDINO）", open=False):
            gr.Markdown(
                "上の Text Prompt（または下の欄）で先頭フレームの候補 bbox を作り、最上位候補を SAM2 Prompt Canvas にコピーします。"
                "複数候補を選んで union（複合対象）にもできます。"
            )
            canvas_text_prompt = gr.Textbox(
                label="検出用 Text Prompt（box 自動作成専用）",
                placeholder="person playing drums / dog jumping through hoop",
            )
            detect_text_btn = gr.Button("Text Prompt から bbox を検出")
            detected_boxes = gr.Dataframe(
                headers=["rank", "phrase", "confidence", "bbox[x1,y1,x2,y2]"],
                datatype=["number", "str", "str", "str"],
                interactive=False,
                label="検出候補",
            )
            box_candidate_group = gr.CheckboxGroup(
                choices=[], value=[], label="union にする候補 bbox を選択",
                info="複数選ぶと各 box を別オブジェクトとして追跡し OR 統合します。",
            )
            apply_boxes_btn = gr.Button("選択した候補で union を構成")

        with gr.Row():
            detection_every = gr.Slider(
                1, 60, value=10, step=1, label="再検出周期（frames・モードBのみ有効）",
                info="検出島（GroundingDINO→SAM2）を走らせる間隔（frame 数、整数）。短いほど追従が速いが重い。Text Prompt 空のモードAでは無視される。目安 8〜15。",
            )
            max_missed_detection_count = gr.Slider(
                0, 10, value=3, step=1, label="未検出保持回数",
                info="連続で再検出に現れなかった track を、この回数を超えたら削除する（回数、整数）。目安 2〜4。",
            )
            iou_threshold = gr.Slider(
                0.05, 0.95, value=0.5, step=0.05, label="consensus IoU しきい値",
                info="伝播マスクと再検出マスクを同一対象と見なす IoU しきい値（0.05〜0.95、単位なし）。目安 0.4〜0.6。",
            )

        matte_mode = gr.Radio(
            MATTE_MODE_CHOICES, value=MATTE_MODE_CHOICES[0], label="合成モード（Matte Mode）",
            info="union: union マスクから 1 ゲートを作り BEN2 を frame あたり 1 回（高速）。per_object: 対象ごとに誘導・BEN2 推論し所有権で α 合成（忠実だが重い）。単位なしの選択値。",
        )

        run_btn = gr.Button("DEVA + 手動 実行", variant="primary")

        with gr.Accordion("Advanced: 検出島しきい値（GroundingDINO）", open=False):
            with gr.Row():
                box_threshold = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.01, label="Box threshold",
                    info="GroundingDINO の物体信頼度しきい値（0.05〜0.95、単位なし）。高いほど検出が厳しく誤検出が減る。目安 0.25。",
                )
                text_threshold = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.01, label="Text threshold",
                    info="検出 box と text 語句の一致しきい値（0.05〜0.95、単位なし）。目安 0.25。",
                )
                top_k = gr.Slider(
                    1, 20, value=20, step=1, label="検出 box 上限 top-k",
                    info="1 回の検出で保持する box の最大個数（個数、整数）。目安 20。",
                )

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
| ゲートでα制限 | BEN2 αをゲート内に限定し背景漏れを抑える。 | 通常 OFF |
| SAM2マスクでα底上げ | SAM2 の安定マスクを α の床にし BEN2 のちらつきを補う。 | none、ちらつき時 screen/比較明 |
| 最大処理フレーム数 | 先頭から処理する frame 数。 | 初回 30、最終 300+ |
"""
            )
            with gr.Row():
                dilation_px = gr.Slider(
                    0, 128, value=int(_BLUR_DEFAULTS.get("dilation_px", 24)), step=1, label="下地マスク膨張(px)",
                    info="SAM2 下地マスク M を膨張してゲート G を作る量（px、整数）。目安 16〜32。",
                )
                blur_kernel = gr.Slider(
                    1, 151, value=int(_BLUR_DEFAULTS.get("blur_kernel", 41)), step=2, label="背景ブラーカーネル(px)",
                    info="ゲート外をぼかすガウシアンカーネルサイズ（px、奇数）。偶数は自動で +1 補正。目安 31〜61。",
                )
            with gr.Row():
                blur_sigma = gr.Slider(
                    0.0, 50.0, value=float(_BLUR_DEFAULTS.get("blur_sigma", 0.0)), step=0.5, label="背景ブラーσ",
                    info="ガウシアンブラーの標準偏差（単位なし）。0.0 でカーネルサイズから自動算出。目安 0.0。",
                )
                feather_px = gr.Slider(
                    0, 64, value=int(_BLUR_DEFAULTS.get("feather_px", 12)), step=1, label="羽根幅(px)",
                    info="ゲート G 境界をぼかして鮮明部と背景を馴染ませる幅（px、整数）。目安 8〜16。",
                )
            with gr.Row():
                refine_foreground = gr.Checkbox(
                    value=bool(_ALPHA_DEFAULTS.get("refine_foreground", False)), label="境界精緻化（refine foreground）",
                    info="ON: BEN2 の髪エッジ後処理を行い境界品質を上げる（時間↑）。真偽値。",
                )
                gate_alpha = gr.Checkbox(
                    value=bool(_COMPOSITE_DEFAULTS.get("gate_alpha", False)), label="ゲートでαを制限",
                    info="ON: BEN2 α をゲート G 内に限定し、ゲート外の背景誤検出を 0 にする。真偽値。",
                )
            with gr.Row():
                mask_floor_mode = gr.Radio(
                    MASK_FLOOR_MODE_CHOICES,
                    value=_mask_floor_label_from_value(_COMPOSITE_DEFAULTS.get("mask_floor_mode", "none")),
                    label="SAM2マスクでα底上げ（合成）",
                    info=(
                        "DEVA の追跡マスクを α の『床』として加算合成し BEN2 の抜け落ち（ちらつき）を補う。"
                        "screen=1-(1-α)(1-M) で自然に底上げ。lighten=max(α,M) で確実に塗る。"
                    ),
                )
            with gr.Row():
                max_frames = gr.Slider(
                    1, 2000, value=30, step=1, label="最大処理フレーム数",
                    info="先頭から処理する frame 数（frame 数、整数）。初回 30、最終確認 300 以上が目安。",
                )
                frame_step = gr.Slider(
                    1, 10, value=1, step=1, label="フレーム間引きステップ", elem_id="movie-frame-step",
                    info="何 frame ごとに1枚処理するか（frame 間隔、整数）。1 で全 frame。通常 1。",
                )
            with gr.Row():
                output_mode = gr.Radio(
                    OUTPUT_MODE_CHOICES, value="動画 (video)", label="出力形式",
                    info="動画: 1つの動画ファイル。連番静止画: frame ごとの PNG。両方: 両方。codec エラー時は連番が安全。単位なしの選択値。",
                )
                rgba_codec = gr.Radio(
                    ["webm_vp9", "mov_png"], value="webm_vp9", label="RGBA 動画コーデック",
                    info="webm_vp9: VP9/WebM（軽量・推奨）。mov_png: PNG コーデックの MOV（高互換・大容量）。単位なしの選択値。",
                )
            output_type = gr.Radio(
                ["rgba", "green", "white", "blur"], value="rgba", label="Preview type",
                info="プレビュー動画の背景表示方法。rgba: 透明。green: 緑背景。white: 白背景。blur: 元背景をぼかす。単位なしの選択値。",
            )
            overlay_enabled = gr.Checkbox(
                value=True, label="Tracking Overlay を生成",
                info="ON: DEVA の追跡 mask を元動画に半透明で重ねた確認用動画を生成する。真偽値。",
            )

        with gr.Row():
            rgba_video = gr.Video(label="RGBA Video")
            alpha_video = gr.Video(label="Alpha Video")
            preview_video = gr.Video(label="Preview Video")
        tracking_overlay_video = gr.Video(label="Tracking Overlay (追跡確認用)")
        sequence_files = gr.Files(label="連番 PNG サンプル")
        sequence_dirs = gr.Textbox(label="連番出力フォルダ", interactive=False)
        run_status = gr.Markdown()

    # ── 手動プロンプト canvas 配線 ──
    input_video.change(
        extract_first_frame_outputs,
        inputs=[input_video],
        outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    load_first_frame_btn.click(
        extract_first_frame_with_base,
        inputs=[input_video],
        outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    # 検出起点フレーム: スライダ操作 / ボタンで Canvas を該当サンプリング後フレームへ張り替える。
    # フレームが変わると既存 box 座標は無効になるため prompt_state は初期化され、選択 UI も再生成する。
    detection_start_frame.change(
        extract_prompt_frame,
        inputs=[input_video, detection_start_frame, frame_step],
        outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    show_start_frame_btn.click(
        extract_prompt_frame,
        inputs=[input_video, detection_start_frame, frame_step],
        outputs=[prompt_canvas, prompt_base_image, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    prompt_canvas.select(
        select_sam2_prompt,
        inputs=[prompt_base_image, prompt_mode, point_label, prompt_state],
        outputs=[prompt_canvas, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    clear_prompt_btn.click(
        clear_prompt,
        inputs=[prompt_base_image],
        outputs=[prompt_canvas, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets,
        inputs=[prompt_state],
        outputs=[point_selection_group, box_selection_group],
    )
    prompt_mode.change(update_point_label_visibility, inputs=[prompt_mode], outputs=[point_label])
    extend_left_btn.click(lambda image, state: extend_box_to_edge(image, state, "left"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets, inputs=[prompt_state], outputs=[point_selection_group, box_selection_group])
    extend_right_btn.click(lambda image, state: extend_box_to_edge(image, state, "right"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets, inputs=[prompt_state], outputs=[point_selection_group, box_selection_group])
    extend_top_btn.click(lambda image, state: extend_box_to_edge(image, state, "top"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets, inputs=[prompt_state], outputs=[point_selection_group, box_selection_group])
    extend_bottom_btn.click(lambda image, state: extend_box_to_edge(image, state, "bottom"), inputs=[prompt_base_image, prompt_state], outputs=[prompt_canvas, prompt_state, prompt_status]).then(
        refresh_prompt_selection_widgets, inputs=[prompt_state], outputs=[point_selection_group, box_selection_group])
    detect_text_btn.click(
        detect_text_boxes_for_video,
        inputs=[prompt_base_image, canvas_text_prompt, box_threshold, text_threshold, top_k, prompt_state],
        outputs=[prompt_canvas, prompt_state, detected_boxes, prompt_status],
    ).then(
        populate_candidate_choices, inputs=[detected_boxes], outputs=[box_candidate_group],
    ).then(
        refresh_prompt_selection_widgets, inputs=[prompt_state], outputs=[point_selection_group, box_selection_group],
    )
    apply_boxes_btn.click(
        apply_selected_boxes,
        inputs=[prompt_base_image, prompt_state, box_candidate_group],
        outputs=[prompt_canvas, prompt_state, prompt_status],
    ).then(
        refresh_prompt_selection_widgets, inputs=[prompt_state], outputs=[point_selection_group, box_selection_group],
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
    # ローカル実行前提では gradio.live トンネルの SSE 切断（ERR058）が起きないため、run ボタンは
    # コア関数へ同期直結する（ERR064）。これにより標準プログレスが復活し、4 出力＋連番2＋status が
    # run_btn の戻り値から直接描画される。
    run_btn.click(
        run_route_a_deva_manual_background_removal,
        inputs=[
            input_video, prompt_state, text_prompt, detection_every, max_missed_detection_count, iou_threshold,
            box_threshold, text_threshold, top_k, detection_start_frame, max_frames, frame_step, output_mode, rgba_codec,
            matte_mode, dilation_px, blur_kernel, blur_sigma, feather_px, refine_foreground,
            gate_alpha, mask_floor_mode, output_type, overlay_enabled,
        ],
        outputs=[rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status],
    )


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="SAM2 + BEN2 Route A (DEVA method + manual prompt) Haystack movie demo")
    parser.add_argument("--share", action="store_true", help="Gradio public link を有効化")
    parser.add_argument("--debug", action="store_true", help="Gradio debug mode")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server name")
    parser.add_argument("--server-port", type=int, default=7864, help="Gradio server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.queue()
    demo.launch(share=args.share, debug=args.debug, server_name=args.server_name, server_port=args.server_port, show_api=False)
