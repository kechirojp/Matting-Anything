"""Haystack Pipeline 版 DEVA 方式 SAM2 + BEN2（ルートA: ブラー誘導 → 再α化）動画αマットデモ。

ベースの ``gradio_app_sam2_ben2_route_a_for_Movie`` と同じ BEN2（ルートA）背景透過・所有権解決・
tracking overlay・出力系をそのまま再利用しつつ、追跡段を **DEVA 方式**（周期再検出 ×
SAM2 伝播 × consensus 統合 × メモリ整理を semi-online に回す ``DevaSemiOnlineTracker``）へ
置換した版。

DEVA は **text_prompt 駆動**で、GroundingDINO→SAM2 の検出島を ``detection_every`` フレームごとに
内部で自動実行するため、ベースアプリのような手動 point/box prompt canvas は不要。ユーザーは
意味プロンプト（例: ``person playing drums``）と再検出周期・consensus 閾値・メモリ保持回数だけを
指定する。ルートA のブラー誘導パラメータ（膨張/ブラー/羽根/底上げ）はベースと同じく
``config/route_a.toml`` 既定値を初期値として読み込み、画面から微調整できる。

NOTE: DEVA「ライブラリ」は採用していない。DEVA の「方式」を SAM2.1 / BEN2 / GroundingDINO 上に
再構成したものである（プロジェクト方針）。
"""

from __future__ import annotations

import argparse
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

from pipelines.components.route_a_common import load_route_a_config
from pipelines.components.video_common import normalize_output_mode
from pipelines.route_a_deva_video_pipeline import build_sam2_ben2_route_a_deva_pipeline


_DEVA_PIPELINE: Any = None
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
    """DEVA 方式ルートA動画αマット Pipeline を遅延構築する（singleton）。

    ``DevaSemiOnlineTracker`` が検出島（GroundingDINO→SAM2）と SAM2 伝播器を内部で
    遅延構築するため、ここでは Pipeline の組み立て（軽量なコンストラクタ呼び出し）のみ行う。
    """
    global _DEVA_PIPELINE
    if _DEVA_PIPELINE is None:
        _DEVA_PIPELINE = build_sam2_ben2_route_a_deva_pipeline()
    return _DEVA_PIPELINE


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
# ## DEVA 方式実行系
# ─────────────────────────────────────────
def run_route_a_deva_background_removal(
    video_path: str | None,
    text_prompt: str,
    detection_every: int,
    max_missed_detection_count: int,
    iou_threshold: float,
    box_threshold: float,
    text_threshold: float,
    top_k: int,
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
    """DEVA 方式ルートA動画αマット Pipeline を実行し、動画または PNG 連番を出力する。"""
    stage_state: dict[str, Any] = {"name": "startup", "description": "Pipeline を起動しています"}
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    try:
        if not video_path:
            raise gr.Error("先に動画をアップロードしてください。")
        if not str(text_prompt).strip():
            raise gr.Error("Text Prompt を入力してください（例: person playing drums）。DEVA 方式は検出島で自動追跡します。")
        if int(detection_every) <= 0:
            raise gr.Error("再検出周期（detection_every）は 1 以上で指定してください。")
        processed_frames = _estimate_processed_frames(int(max_frames), int(frame_step))
        output_mode = normalize_output_mode(output_mode_label)
        matte_mode = _normalize_matte_mode(matte_mode_label)
        mask_floor_mode = _normalize_mask_floor_mode(mask_floor_mode_label)
        progress_callback = build_video_progress_callback(progress, stage_state)
        progress(
            0.03,
            desc=f"Pipeline を起動しています。初回はモデル読込（GroundingDINO/SAM2/BEN2）を伴います（最大 {processed_frames} frames）",
        )
        result = get_route_a_deva_pipeline().run(
            {
                "video_reader": {
                    "video_path": video_path,
                    "max_frames": int(max_frames),
                    "frame_step": int(frame_step),
                    "progress_callback": progress_callback,
                },
                # ## DEVA 方式追跡: 周期再検出 × SAM2 伝播 × consensus 統合 × メモリ整理。
                "deva_semi_online_tracker": {
                    "text_prompt": str(text_prompt).strip(),
                    "detection_every": int(detection_every),
                    "max_missed_detection_count": int(max_missed_detection_count),
                    "iou_threshold": float(iou_threshold),
                    "box_threshold": float(box_threshold),
                    "text_threshold": float(text_threshold),
                    "top_k": int(top_k),
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
        sequence_dirs = [
            merged.get(key)
            for key in ("rgba_sequence_dir", "alpha_sequence_dir", "preview_sequence_dir")
            if merged.get(key)
        ]
        fallback = merged.get("metadata", {}).get("codec_fallback", [])
        status = (
            f"完了: output_mode={output_mode}, matte_mode={matte_mode}, frames={merged.get('frame_count')}, "
            f"text='{str(text_prompt).strip()}', detection_every={int(detection_every)}, "
            f"max_missed={int(max_missed_detection_count)}, iou={float(iou_threshold):.2f}"
        )
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
        raise gr.Error(f"DEVA 方式動画処理に失敗しました（stage={stage}, elapsed={elapsed:.1f}s）: {exc}") from exc


def update_codec_visibility(output_mode_label: str):
    """連番のみ選択時は動画 codec UI を無効化する。"""
    sequence_only = normalize_output_mode(output_mode_label) == "sequence"
    return gr.update(interactive=not sequence_only, visible=not sequence_only)


_BLUR_DEFAULTS = _blur_defaults()
_COMPOSITE_DEFAULTS = _composite_defaults()
_ALPHA_DEFAULTS = _alpha_defaults()

with gr.Blocks(title="SAM2 + BEN2 Route A (DEVA 方式) for Movie") as demo:
    gr.Markdown("# SAM2 + BEN2 Route A（DEVA 方式: 周期再検出 × SAM2 伝播 × consensus）for Movie")
    gr.Markdown(
        """
> ## DEVA 方式とは（ライブラリではなく「方式」の再構成）
> 意味プロンプト（例: `person playing drums`）から **GroundingDINO→SAM2 の検出島**を
> **一定フレームごと（再検出周期）** に走らせ、その間は **SAM2 で伝播**して追跡します。
> 周期境界では伝播マスクと再検出マスクを **consensus（IoU 統合）** で突き合わせ、
> 一致を維持・新規を追加・連続未検出（最大保持回数超過）を削除します。これにより、
> 新規対象の出現・既存対象の消失に半オンラインで追従します。
>
> | パラメーター | 役割 | 目安 |
> |---|---|---|
> | **Text Prompt** | 検出島が探す対象（意味プロンプト）。 | person playing drums など |
> | **再検出周期(frames)** | 検出島を走らせる間隔。短いほど追従が速いが重い。 | 8〜15 |
> | **未検出保持回数** | 連続未検出でこの回数を超えた track を削除。 | 2〜4 |
> | **consensus IoU しきい値** | 伝播 vs 再検出の同一対象判定しきい値。 | 0.4〜0.6 |
>
> ルートA（ブラー誘導→BEN2 再α化）と出力系はベースアプリと同一です。
> 既定値は `config/route_a.toml` から読み込んでいます。
"""
    )
    gr.Markdown(
        """
### 使い方
1. **Input Video** に動画をアップロードします。
2. **Text Prompt** に追跡したい対象を意味で入力します（手動の point/box 指定は不要です）。
3. 必要に応じて **再検出周期 / 未検出保持回数 / consensus IoU** を調整します。
4. まずは既定値（最大 30 frames・union）で **DEVA 実行** し、結果を確認してから Advanced で
   膨張量・ブラー強度を微調整します。

**処理順**: `フレーム取得 → DEVA（検出島×SAM2 伝播×consensus 統合）→ 所有権解決 → ルートA（下地マスク膨張→背景ブラー→BEN2 再α化）→ 出力`。
"""
    )

    with gr.Tab("DEVA 方式追跡 + ルートA"):
        input_video = gr.Video(label="Input Video", sources=["upload"], elem_id="movie-input-video")

        text_prompt = gr.Textbox(
            label="Text Prompt（必須）",
            placeholder="person playing drums / person riding bicycle / dog jumping through hoop",
            info="検出島（GroundingDINO）が探す対象の意味プロンプト。DEVA 方式は手動 point/box 指定は不要で、ここで指定した対象を周期再検出＋SAM2 伝播で自動追跡する。",
        )

        with gr.Row():
            detection_every = gr.Slider(
                1, 60, value=10, step=1, label="再検出周期（frames）",
                info="検出島（GroundingDINO→SAM2）を走らせる間隔（frame 数、整数）。短いほど新規/消失への追従が速いが重い。目安 8〜15。",
            )
            max_missed_detection_count = gr.Slider(
                0, 10, value=3, step=1, label="未検出保持回数",
                info="連続で再検出に現れなかった track を、この回数を超えたら削除する（回数、整数）。大きいほど一時的な隠れに強いが残像が残りやすい。目安 2〜4。",
            )
            iou_threshold = gr.Slider(
                0.05, 0.95, value=0.5, step=0.05, label="consensus IoU しきい値",
                info="伝播マスクと再検出マスクを同一対象と見なす IoU しきい値（0.05〜0.95、単位なし）。高いほど厳密。目安 0.4〜0.6。",
            )

        matte_mode = gr.Radio(
            MATTE_MODE_CHOICES, value=MATTE_MODE_CHOICES[0], label="合成モード（Matte Mode）",
            info="union: union マスクから 1 ゲートを作り BEN2 を frame あたり 1 回（高速）。per_object: 対象ごとに誘導・BEN2 推論し所有権で α 合成（忠実だが対象数×回で重い）。単位なしの選択値。",
        )

        run_btn = gr.Button("DEVA 実行", variant="primary")

        with gr.Accordion("Advanced: 検出島しきい値（GroundingDINO）", open=False):
            with gr.Row():
                box_threshold = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.01, label="Box threshold",
                    info="GroundingDINO の物体信頼度しきい値（0.05〜0.95、単位なし）。高いほど検出が厳しく誤検出が減る。目安 0.25。",
                )
                text_threshold = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.01, label="Text threshold",
                    info="検出 box と text 語句の一致しきい値（0.05〜0.95、単位なし）。高いほど語句との合致を厳しく要求する。目安 0.25。",
                )
                top_k = gr.Slider(
                    1, 20, value=20, step=1, label="検出 box 上限 top-k",
                    info="1 回の検出で保持する box の最大個数（個数、整数）。スコア上位から K 個まで。目安 20。",
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
                info="ON: DEVA の追跡 mask を元動画に半透明で重ねた確認用動画を生成する（処理がわずかに増えます）。OFF: 生成しない。真偽値。",
            )

        with gr.Row():
            rgba_video = gr.Video(label="RGBA Video")
            alpha_video = gr.Video(label="Alpha Video")
            preview_video = gr.Video(label="Preview Video")
        tracking_overlay_video = gr.Video(label="Tracking Overlay (追跡確認用)")
        sequence_files = gr.Files(label="連番 PNG サンプル")
        sequence_dirs = gr.Textbox(label="連番出力フォルダ", interactive=False)
        run_status = gr.Markdown()

    output_mode.change(update_codec_visibility, inputs=[output_mode], outputs=[rgba_codec])
    # ローカル実行前提では gradio.live トンネルの SSE 切断（ERR058）が起きないため、run ボタンは
    # コア関数へ同期直結する（ERR064）。これにより標準プログレスが復活し、4 出力＋連番2＋status が
    # run_btn の戻り値から直接描画される。
    run_btn.click(
        run_route_a_deva_background_removal,
        inputs=[
            input_video, text_prompt, detection_every, max_missed_detection_count, iou_threshold,
            box_threshold, text_threshold, top_k, max_frames, frame_step, output_mode, rgba_codec,
            matte_mode, dilation_px, blur_kernel, blur_sigma, feather_px, refine_foreground,
            gate_alpha, mask_floor_mode, output_type, overlay_enabled,
        ],
        outputs=[rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status],
    )


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="SAM2 + BEN2 Route A (DEVA method) Haystack movie demo")
    parser.add_argument("--share", action="store_true", help="Gradio public link を有効化")
    parser.add_argument("--debug", action="store_true", help="Gradio debug mode")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server name")
    parser.add_argument("--server-port", type=int, default=7863, help="Gradio server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.queue()
    demo.launch(share=args.share, debug=args.debug, server_name=args.server_name, server_port=args.server_port, show_api=False)
