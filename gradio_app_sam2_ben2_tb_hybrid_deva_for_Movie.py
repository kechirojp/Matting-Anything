"""DEVA 方式 SAM2 + BEN2 / transparent-background ハイブリッド動画αマットデモ。

``gradio_app_sam2_ben2_route_a_deva_for_Movie`` を土台に、alpha 生成段を
``BEN2TransparentHybridVideoExtractor`` へ差し替えた版。GroundingDINO が text prompt から
解釈した人物 bbox を SAM2.1 で追跡し、その mask 領域だけ transparent-background を適用する。
領域外は BEN2 で処理し、最後に 2 つの alpha を UI 選択の合成方式で統合する。

NOTE: DEVA「ライブラリ」は採用していない。DEVA の「方式」を SAM2.1 / BEN2 /
GroundingDINO 上に再構成したものである（プロジェクト方針）。
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
from pipelines.route_a_deva_hybrid_video_pipeline import build_sam2_ben2_tb_deva_hybrid_pipeline
from pipelines.sam2_tb_video_pipeline import build_video_reader_pipeline


_HYBRID_PIPELINE: Any = None
READER_PIPELINE: Any = None
OUTPUT_MODE_CHOICES = ["動画 (video)", "連番静止画 (sequence)", "両方 (both)"]
COMPOSITION_MODE_CHOICES = [
    "lighten / 比較明（推奨）",
    "person_over_ben2（人物TBを手前）",
    "ben2_over_person（BEN2を手前）",
    "screen（スクリーン）",
]
STAGE_PROGRESS_RANGES = {
    "video_reader": (0.05, 0.12),
    "deva_tracker": (0.12, 0.58),
    "hybrid_alpha": (0.58, 0.88),
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


def _normalize_composition_mode(composition_mode_label: str) -> str:
    """UI ラベルを hybrid alpha composition mode へ正規化する。"""
    text = str(composition_mode_label).strip().lower()
    if text.startswith("person_over_ben2"):
        return "person_over_ben2"
    if text.startswith("ben2_over_person"):
        return "ben2_over_person"
    if text.startswith("screen"):
        return "screen"
    return "lighten"


def get_deva_hybrid_pipeline():
    """DEVA 方式 BEN2/TB ハイブリッド動画αマット Pipeline を遅延構築する（singleton）。"""
    global _HYBRID_PIPELINE
    if _HYBRID_PIPELINE is None:
        _HYBRID_PIPELINE = build_sam2_ben2_tb_deva_hybrid_pipeline()
    return _HYBRID_PIPELINE


def get_reader_pipeline():
    """検出起点フレーム抽出用の VideoReader Pipeline を遅延構築する（singleton）。"""
    global READER_PIPELINE
    if READER_PIPELINE is None:
        READER_PIPELINE = build_video_reader_pipeline()
    return READER_PIPELINE


def extract_detection_start_frame(video_path: str | None, detection_start_frame: int, frame_step: int):
    """指定したサンプリング後フレーム位置を取り出し、検出起点フレームのプレビューにする。

    被写体が最大に映るフレームを目視で選べるよう、スライダ位置のフレーム画像を返す。
    ``detection_start_frame`` は VideoReader が ``frame_step`` でサンプリングした後の
    frames list 上の位置（元動画フレーム番号 ≈ index * frame_step）で、DEVA トラッカーへ
    渡す ``detection_start_frame`` と同じ index 規約である。
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
        status = (
            f"検出起点フレームをシーケンス位置 {effective_index}（元動画フレーム≈{raw_index}）に設定しました。"
            f"被写体が最大に映るフレームを選んでください。画像サイズ {metadata['width']}x{metadata['height']}。"
        )
        return frame, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"検出起点フレームの取得に失敗しました: {exc}") from exc


def extract_start_frame_preview_on_upload(video_path: str | None):
    """動画アップロード時に先頭フレームを起点プレビューへ自動反映する。"""
    if not video_path:
        return None, "動画をアップロードすると検出起点フレームのプレビューを取得できます。"
    return extract_detection_start_frame(video_path, 0, 1)


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
    範囲外エラー（detection_start_frame >= num_frames）を避けて任意フレーム起点で実行できる。
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
# ## DEVA + BEN2/TB hybrid 実行系
# ─────────────────────────────────────────
def run_deva_hybrid_background_removal(
    video_path: str | None,
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
    composition_mode_label: str,
    tb_mode: str,
    tb_threshold: float,
    tb_crop_padding: int,
    tb_mask_guard_feather: int,
    tb_mask_guard_dilate: int,
    person_region_dilate_px: int,
    person_region_feather_px: int,
    refine_foreground: bool,
    output_type: str,
    overlay_enabled: bool,
    progress=gr.Progress(),
):
    """DEVA 追跡 mask を使い、人物TB＋その他BEN2の hybrid alpha を出力する。"""
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
        detection_start_frame = max(int(detection_start_frame), 0)
        # 起点フレームを必ず読み込み窓に含める（範囲外エラー回避、任意フレーム起点で実行可能に）。
        effective_max_frames = _effective_read_frames(int(max_frames), detection_start_frame)
        processed_frames = _estimate_processed_frames(effective_max_frames, int(frame_step))
        output_mode = normalize_output_mode(output_mode_label)
        composition_mode = _normalize_composition_mode(composition_mode_label)
        progress_callback = build_video_progress_callback(progress, stage_state)
        progress(
            0.03,
            desc=(
                "Pipeline を起動しています。初回はモデル読込"
                f"（GroundingDINO/SAM2/BEN2/transparent-background）を伴います（最大 {processed_frames} frames）"
            ),
        )
        result = get_deva_hybrid_pipeline().run(
            {
                "video_reader": {
                    "video_path": video_path,
                    "max_frames": effective_max_frames,
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
                    "detection_start_frame": detection_start_frame,
                    "per_object_logits_max_side": _deva_per_object_logits_max_side(),
                    "progress_callback": progress_callback,
                },
                # ## Hybrid α: 人物 mask 内は transparent-background、外側は BEN2。
                "ben2_tb_hybrid_video": {
                    "output_mode": output_mode,
                    "tb_mode": tb_mode,
                    "tb_threshold": float(tb_threshold),
                    "tb_crop_padding": int(tb_crop_padding),
                    "tb_mask_guard_dilate": int(tb_mask_guard_dilate),
                    "tb_mask_guard_feather": int(tb_mask_guard_feather),
                    "person_region_dilate_px": int(person_region_dilate_px),
                    "person_region_feather_px": int(person_region_feather_px),
                    "composition_mode": composition_mode,
                    "refine_foreground": bool(refine_foreground),
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
            f"完了: output_mode={output_mode}, composition={composition_mode}, frames={merged.get('frame_count')}, "
            f"text='{str(text_prompt).strip()}', detection_every={int(detection_every)}, "
            f"start_frame={detection_start_frame}, "
            f"max_missed={int(max_missed_detection_count)}, iou={float(iou_threshold):.2f}"
        )
        person_warning = merged.get("metadata", {}).get("person_mask_fallback_warning")
        if person_warning:
            status += f"\n\n⚠️ {person_warning}"
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
        raise gr.Error(f"DEVA hybrid 動画処理に失敗しました（stage={stage}, elapsed={elapsed:.1f}s）: {exc}") from exc


def update_codec_visibility(output_mode_label: str):
    """連番のみ選択時は動画 codec UI を無効化する。"""
    sequence_only = normalize_output_mode(output_mode_label) == "sequence"
    return gr.update(interactive=not sequence_only, visible=not sequence_only)


_BLUR_DEFAULTS = _blur_defaults()
_COMPOSITE_DEFAULTS = _composite_defaults()
_ALPHA_DEFAULTS = _alpha_defaults()

with gr.Blocks(title="SAM2 + BEN2/TB Hybrid (DEVA 方式) for Movie") as demo:
    gr.Markdown("# SAM2 + BEN2/TB Hybrid（DEVA 方式）for Movie")
    gr.Markdown(
        """
> ## 目的
> **BEN2 は主被写体をつかむ力**、**transparent-background は人物・髪の毛の繊細な切り抜き**
> を担当します。Text Prompt で指定した人物を GroundingDINO→SAM2.1 で追跡し、その mask 領域
> だけ transparent-background を使います。mask 外側は BEN2 の alpha を使い、最後に合成します。
>
> | パラメーター | 役割 | 目安 |
> |---|---|---|
> | **Text Prompt** | transparent-background を使う人物領域の検出語。 | person / person playing drums |
> | **再検出周期(frames)** | 検出島を走らせる間隔。短いほど追従が速いが重い。 | 8〜15 |
> | **Alpha 合成方式** | BEN2 alpha と人物TB alpha の重ね方。 | 比較明または人物TBを手前 |
"""
    )
    gr.Markdown(
        """
### 使い方
1. **Input Video** に動画をアップロードします。
2. **Text Prompt** に追跡したい対象を意味で入力します（手動の point/box 指定は不要です）。
3. 必要に応じて **再検出周期 / 未検出保持回数 / consensus IoU** を調整します。
4. まずは既定値（最大 30 frames・比較明）で **Hybrid 実行** し、Tracking Overlay で mask を確認します。

**処理順**: `フレーム取得 → DEVA（GroundingDINO×SAM2 伝播×consensus）→ 人物領域TB / 外側BEN2 → alpha合成 → 出力`。
"""
    )

    with gr.Tab("DEVA 方式追跡 + BEN2/TB Hybrid"):
        input_video = gr.Video(label="Input Video", sources=["upload"], elem_id="movie-input-video")

        text_prompt = gr.Textbox(
            label="Text Prompt（必須）",
            placeholder="person playing drums / person riding bicycle / dog jumping through hoop",
            info="検出島（GroundingDINO）が探す対象の意味プロンプト。DEVA 方式は手動 point/box 指定は不要で、ここで指定した対象を周期再検出＋SAM2 伝播で自動追跡する。",
        )

        with gr.Accordion("検出起点フレーム（被写体が最大に映るフレームで検出）", open=False):
            gr.Markdown(
                "被写体が最初のフレームで小さい/映っていない場合、**被写体が最大に映るフレーム**を"
                "検出起点に指定すると GroundingDINO の初回検出精度が上がります。起点より前のフレームは"
                "**双方向逆伝播**でカバーされ取りこぼしません（標準 SAM2 のみ・0 で従来通り先頭起点）。"
            )
            with gr.Row():
                detection_start_frame = gr.Slider(
                    0, 2000, value=0, step=1, label="検出起点フレーム（サンプリング後 index）",
                    elem_id="movie-detection-start-frame",
                    info="検出/seed を最初に確立するフレーム位置（0 始まり、frame_step 適用後の index）。スライダを動かすと右のプレビューが更新される。被写体が最大に映る位置を選ぶ。0 は先頭。",
                )
                show_start_frame_btn = gr.Button("起点フレームを表示")
            start_frame_preview = gr.Image(label="検出起点フレーム プレビュー", type="numpy", interactive=False)
            start_frame_status = gr.Markdown()

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

        composition_mode = gr.Radio(
            COMPOSITION_MODE_CHOICES,
            value=COMPOSITION_MODE_CHOICES[0],
            label="Alpha 合成方式",
            info=(
                "BEN2 alpha（mask外側）と transparent-background alpha（人物mask内）の合成方式。"
                "lighten=比較明で黒抜けを避ける推奨値。person_over_ben2=人物TBを手前、"
                "ben2_over_person=BEN2を手前、screen=スクリーン合成。単位なし。"
            ),
        )

        run_btn = gr.Button("Hybrid 実行", variant="primary")

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

        with gr.Accordion("Advanced: Hybrid α / 動画処理設定", open=False):
            gr.Markdown(
                """
| パラメーター | 何を変えるか | 目安 |
|---|---|---|
| TB mode | transparent-background のモデルモード。 | base |
| 人物領域feather(px) | TB/BEN2 の切替境界をどれだけぼかすか。 | 8〜16 |
| TB mask guard | TB alpha が人物mask外へ漏れる量を制御。 | dilate 21 / feather 0 |
| 境界精緻化 | BEN2 の髪エッジ後処理（精度↑/時間↑）。 | 必要時のみ ON |
| 最大処理フレーム数 | 先頭から処理する frame 数。 | 初回 30、最終 300+ |
"""
            )
            with gr.Row():
                tb_mode = gr.Radio(
                    ["base", "fast", "base-nightly"],
                    value="base",
                    label="transparent-background mode",
                    info="TB の推論モード。base=標準・推奨、fast=軽量、base-nightly=実験的。単位なし。",
                )
                tb_threshold = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01, label="TB threshold",
                    info="transparent-background の閾値（0.0〜1.0）。0.0 はモデル既定。高いほど薄い alpha を切りやすい。目安 0.0。",
                )
            with gr.Row():
                tb_crop_padding = gr.Slider(
                    0, 160, value=40, step=1, label="TB crop padding(px)",
                    info="SAM2 人物mask bbox の外側に足す余白（px、整数）。大きいほど髪や手を拾いやすいが背景も入りやすい。目安 30〜60。",
                )
                person_region_feather_px = gr.Slider(
                    0, 64, value=8, step=1, label="人物領域 feather(px)",
                    info="TB alpha と BEN2 alpha の切替境界をぼかす幅（px、整数）。大きいほど段差が減るが混ざりが増える。目安 8〜16。",
                )
            with gr.Row():
                person_region_dilate_px = gr.Slider(
                    0, 64, value=0, step=1, label="人物領域 dilate(px)",
                    info="TB が担当する SAM2 人物領域を外側へ広げる量（px、整数）。髪が切れる時に少し上げる。目安 0〜12。",
                )
                tb_mask_guard_dilate = gr.Slider(
                    1, 101, value=21, step=2, label="TB mask guard dilate(px)",
                    info="TB alpha が SAM2 mask 外へ漏れるのを許すマージン（px相当の奇数kernel）。大きいほど髪を残すが背景漏れも増える。目安 21。",
                )
            with gr.Row():
                tb_mask_guard_feather = gr.Slider(
                    0, 64, value=0, step=1, label="TB mask guard feather(px)",
                    info="TB guard の境界をぼかす幅（px、整数）。0 は二値guard。髪先の段差が目立つ時だけ 4〜12。",
                )
                refine_foreground = gr.Checkbox(
                    value=bool(_ALPHA_DEFAULTS.get("refine_foreground", False)), label="BEN2 境界精緻化（refine foreground）",
                    info="ON: BEN2 の境界後処理を行う（精度↑/時間↑）。TB 側ではなく BEN2 側に効く。通常 OFF。",
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
    # 動画アップロード時に先頭フレームを起点プレビューへ自動反映（被写体最大フレーム探索の起点）。
    input_video.change(
        extract_start_frame_preview_on_upload,
        inputs=[input_video],
        outputs=[start_frame_preview, start_frame_status],
    )
    # 検出起点フレーム スライダ操作 / ボタン押下で該当フレームをプレビュー更新。
    detection_start_frame.change(
        extract_detection_start_frame,
        inputs=[input_video, detection_start_frame, frame_step],
        outputs=[start_frame_preview, start_frame_status],
    )
    show_start_frame_btn.click(
        extract_detection_start_frame,
        inputs=[input_video, detection_start_frame, frame_step],
        outputs=[start_frame_preview, start_frame_status],
    )
    # ローカル実行前提では gradio.live トンネルの SSE 切断（ERR058）が起きないため、run ボタンは
    # コア関数へ同期直結する（ERR064）。これにより標準プログレスが復活し、4 出力＋連番2＋status が
    # run_btn の戻り値から直接描画される。
    run_btn.click(
        run_deva_hybrid_background_removal,
        inputs=[
            input_video, text_prompt, detection_every, max_missed_detection_count, iou_threshold,
            box_threshold, text_threshold, top_k, detection_start_frame, max_frames, frame_step, output_mode, rgba_codec,
            composition_mode, tb_mode, tb_threshold, tb_crop_padding, tb_mask_guard_feather,
            tb_mask_guard_dilate, person_region_dilate_px, person_region_feather_px,
            refine_foreground, output_type, overlay_enabled,
        ],
        outputs=[rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status],
    )


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="SAM2 + BEN2/TB hybrid (DEVA method) Haystack movie demo")
    parser.add_argument("--share", action="store_true", help="Gradio public link を有効化")
    parser.add_argument("--debug", action="store_true", help="Gradio debug mode")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server name")
    parser.add_argument("--server-port", type=int, default=7865, help="Gradio server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.queue()
    demo.launch(share=args.share, debug=args.debug, server_name=args.server_name, server_port=args.server_port, show_api=False)
