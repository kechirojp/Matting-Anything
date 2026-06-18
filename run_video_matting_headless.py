"""SAM2 + transparent-background 動画背景除去のヘッドレス実行 CLI。

Gradio を起動せずに end-to-end でパイプラインを実行する検証用エントリポイント。
box / point プロンプトを CLI 引数で受け取り、per-object logit 保持 → OwnershipResolver
所有権解決 → transparent-background 合成までを実行して出力パスを表示する。

GroundingDINO によるテキストプロンプト検出は本 CLI では扱わない（box/point 直接指定）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.components.common import stable_sigmoid
from pipelines.components.model_registry import entry_by_id
from pipelines.components.video_common import normalize_output_mode
from pipelines.components.video_model_components import SAM2VideoPropagator, VideoReader
from pipelines.sam2_tb_video_pipeline import build_sam2_tb_video_pipeline


def _parse_box(value: str) -> list[float]:
    """"x1,y1,x2,y2" を float 4 要素のリストへ変換する。"""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--box は 'x1,y1,x2,y2' 形式で指定してください: {value!r}")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--box の数値変換に失敗しました: {value!r}") from exc


def _parse_point(value: str) -> tuple[list[float], int]:
    """"x,y[,label]" を (point, label) へ変換する。label 省略時は 1（positive）。"""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"--point は 'x,y' または 'x,y,label' 形式で指定してください: {value!r}")
    try:
        x, y = float(parts[0]), float(parts[1])
        label = int(parts[2]) if len(parts) == 3 else 1
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--point の数値変換に失敗しました: {value!r}") from exc
    return [x, y], label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="入力動画ファイルのパス")
    parser.add_argument(
        "--box",
        action="append",
        type=_parse_box,
        default=[],
        help="対象の bbox 'x1,y1,x2,y2'。複数指定で複数オブジェクトを追跡する。",
    )
    parser.add_argument(
        "--point",
        action="append",
        type=_parse_point,
        default=[],
        help="補正点 'x,y[,label]'（label: 1=positive, 0=negative）。複数指定可。",
    )
    parser.add_argument("--tracker", default="sam2_hiera_l", help="tracker モデル id（config/inference_models.toml）")
    parser.add_argument("--background", default="tb_base", help="background モデル id（config/inference_models.toml）")
    parser.add_argument("--max-frames", type=int, default=0, help="処理する最大フレーム数（0 で全フレーム）")
    parser.add_argument("--frame-step", type=int, default=1, help="フレーム間引き間隔")
    parser.add_argument("--prompt-frame-idx", type=int, default=0, help="プロンプト起点フレーム位置")
    parser.add_argument("--bidirectional", action="store_true", help="プロンプトフレームから前後双方向に伝播する")
    parser.add_argument(
        "--output-mode",
        default="video",
        choices=["video", "sequence", "both"],
        help="出力形式（動画 / 連番静止画 / 両方）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="OwnershipResolver の softmax 温度 τ（未指定で config の ownership_temperature を使用）",
    )
    parser.add_argument("--crop-padding", type=int, default=40, help="transparent-background の crop padding")
    parser.add_argument(
        "--matte-mode",
        choices=["union", "per_object"],
        default=None,
        help="動画背景透過の経路（未指定時は config の video_matte_mode に従う。config 既定は union）",
    )
    parser.add_argument("--rgba-codec", default="webm_vp9", help="RGBA 動画の codec")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "tb 合成を行わず SAM2 伝搬のみ実行し、prompt→mask 対応を数値化する診断モード。"
            "各 obj の mask 面積率・box 内被覆率・重心・logit 統計を出力する（反転/背景追跡の切り分け用）。"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI 引数からパイプラインを構築・実行し、matte 結果 dict を返す。"""
    boxes = [list(b) for b in (args.box or [])]
    points = [p for p, _ in (args.point or [])]
    labels = [label for _, label in (args.point or [])]
    if not boxes and not points:
        raise ValueError("--box または --point を 1 つ以上指定してください。")
    if not Path(args.video).is_file():
        raise ValueError(f"--video が見つかりません: {args.video}")

    tracker_entry = entry_by_id("tracker", args.tracker)
    bg_entry = entry_by_id("background", args.background)
    tb_mode = bg_entry.get("tb_mode", "base")
    mask_feather = int(bg_entry.get("mask_feather", 0))
    temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(bg_entry.get("ownership_temperature", 1.0))
    )
    video_matte_mode = (
        str(args.matte_mode)
        if args.matte_mode is not None
        else str(bg_entry.get("video_matte_mode", "union"))
    )

    propagator = SAM2VideoPropagator(
        checkpoint_path=tracker_entry["checkpoint_path"],
        config_name=tracker_entry["config_name"],
    )
    pipeline = build_sam2_tb_video_pipeline(propagator=propagator)
    output_mode = normalize_output_mode(args.output_mode)

    result = pipeline.run(
        {
            "video_reader": {
                "video_path": args.video,
                "max_frames": int(args.max_frames),
                "frame_step": int(args.frame_step),
            },
            "sam2_video_propagator": {
                "points": points,
                "labels": labels,
                "boxes": boxes,
                "prompt_frame_idx": int(args.prompt_frame_idx),
                "bidirectional": bool(args.bidirectional),
            },
            "ownership_resolver": {"temperature": temperature},
            "transparent_bg_video": {
                "output_mode": output_mode,
                "tb_mode": tb_mode,
                "crop_padding": int(args.crop_padding),
                "mask_guard_feather": mask_feather,
                "video_matte_mode": video_matte_mode,
                "rgba_codec": args.rgba_codec,
            },
            "video_writer": {"rgba_codec": args.rgba_codec},
        },
        include_outputs_from={"video_writer", "frame_sequence_writer"},
    )
    return result


def _mask_box_stats(
    logit_2d: np.ndarray,
    box: list[float] | None,
    width: int,
    height: int,
) -> dict[str, float]:
    """1 対象の logit から mask 面積率・box 内被覆率・重心・logit 統計を計算する。

    反転（mask が背景側に乗る）や box→mask 不整合を数値で切り分けるための純粋関数。
    SAM2 の logit は frame と解像度が異なる場合があるため、判定は logit 自身の解像度
    (Hl,Wl) を基準とし、frame ピクセル座標の box を logit 空間へスケールしてから比較する。
    """
    logit = np.asarray(logit_2d, dtype=np.float32)
    h_l, w_l = logit.shape[:2]
    prob = stable_sigmoid(logit)
    mask = prob >= 0.5
    area_frac = float(mask.mean())
    stats: dict[str, float] = {
        "area_frac": area_frac,
        "logit_min": float(np.min(logit)),
        "logit_max": float(np.max(logit)),
        "logit_mean": float(np.mean(logit)),
        "logit_h": float(h_l),
        "logit_w": float(w_l),
        "centroid_x": float("nan"),
        "centroid_y": float("nan"),
        "inside_box_frac": float("nan"),
        "box_area_frac": float("nan"),
        "centroid_in_box": float("nan"),
    }
    if mask.any():
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        # 重心は logit 解像度で正規化（0..1）。frame 正規化と一致する。
        stats["centroid_x"] = cx / max(w_l, 1)
        stats["centroid_y"] = cy / max(h_l, 1)
        if box is not None and len(box) == 4:
            # frame ピクセル座標の box を logit 解像度へスケールする。
            sx = w_l / max(int(width), 1)
            sy = h_l / max(int(height), 1)
            x1, y1, x2, y2 = (float(v) for v in box)
            x_lo, x_ho = min(x1, x2) * sx, max(x1, x2) * sx
            y_lo, y_ho = min(y1, y2) * sy, max(y1, y2) * sy
            inside = (xs >= x_lo) & (xs <= x_ho) & (ys >= y_lo) & (ys <= y_ho)
            stats["inside_box_frac"] = float(inside.mean())
            box_area = max(x_ho - x_lo, 0.0) * max(y_ho - y_lo, 0.0)
            stats["box_area_frac"] = box_area / max(w_l * h_l, 1)
            stats["centroid_in_box"] = float(x_lo <= cx <= x_ho and y_lo <= cy <= y_ho)
    return stats


def run_diagnose(args: argparse.Namespace) -> int:
    """tb を回さず SAM2 伝搬のみ実行し、prompt→mask 対応を数値化して出力する。"""
    boxes = [list(b) for b in (args.box or [])]
    points = [p for p, _ in (args.point or [])]
    labels = [label for _, label in (args.point or [])]
    if not boxes and not points:
        raise ValueError("--box または --point を 1 つ以上指定してください。")
    if not Path(args.video).is_file():
        raise ValueError(f"--video が見つかりません: {args.video}")

    tracker_entry = entry_by_id("tracker", args.tracker)
    reader = VideoReader()
    read = reader.run(
        video_path=args.video,
        # --max-frames 0 のときは VideoReader 既定（300）を踏襲する（診断は通常先頭数十枚で十分）。
        max_frames=int(args.max_frames) if int(args.max_frames) > 0 else 300,
        frame_step=int(args.frame_step),
    )
    frames = read["frames"]
    metadata = read["metadata"]
    width = int(metadata.get("width") or frames[0].shape[1])
    height = int(metadata.get("height") or frames[0].shape[0])

    propagator = SAM2VideoPropagator(
        checkpoint_path=tracker_entry["checkpoint_path"],
        config_name=tracker_entry["config_name"],
    )
    out = propagator.run(
        frames=frames,
        metadata=metadata,
        points=points,
        labels=labels,
        boxes=boxes,
        prompt_frame_idx=int(args.prompt_frame_idx),
        bidirectional=bool(args.bidirectional),
    )
    per_object_logits: dict[int, np.ndarray] = out["masks"].get("per_object_logits", {})
    if not per_object_logits:
        print("per_object_logits が空です。SAM2 伝搬が mask を返していません。")
        return 1

    target_box_for = lambda obj_index: (boxes[obj_index] if obj_index < len(boxes) else None)
    print("=== SAM2 伝搬 診断 ===")
    print(f"video={args.video}  frames={len(frames)}  WxH={width}x{height}")
    print(f"boxes={boxes}  points={points}  labels={labels}  prompt_frame_idx={args.prompt_frame_idx}")
    if boxes:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = (float(v) for v in b)
            ba = max(abs(x2 - x1), 0.0) * max(abs(y2 - y1), 0.0) / max(width * height, 1)
            print(f"  box obj{i + 1}: {b}  画面占有率={ba:.3f}")
    print("列: area=mask面積率 / in_box=mask画素のbox内割合 / box_area=box占有率 / c=(重心x,y正規化) / c_in_box / logit(min/mean/max)")

    # 集計用アキュムレータ（obj_index -> list）
    agg_area: dict[int, list[float]] = {}
    agg_inbox: dict[int, list[float]] = {}
    sorted_frames = sorted(per_object_logits.keys())
    show_frames = set(sorted_frames[:3] + sorted_frames[-1:])  # 先頭3 + 末尾1 を逐次表示
    for frame_index in sorted_frames:
        stacked = np.asarray(per_object_logits[frame_index], dtype=np.float32)
        for obj_index in range(stacked.shape[0]):
            s = _mask_box_stats(stacked[obj_index], target_box_for(obj_index), width, height)
            agg_area.setdefault(obj_index, []).append(s["area_frac"])
            if not np.isnan(s["inside_box_frac"]):
                agg_inbox.setdefault(obj_index, []).append(s["inside_box_frac"])
            if frame_index in show_frames:
                print(
                    f"frame {frame_index:>5} obj{obj_index + 1}: "
                    f"area={s['area_frac']:.3f} in_box={s['inside_box_frac']:.3f} "
                    f"box_area={s['box_area_frac']:.3f} "
                    f"c=({s['centroid_x']:.2f},{s['centroid_y']:.2f}) c_in_box={s['centroid_in_box']:.0f} "
                    f"logit({s['logit_min']:.1f}/{s['logit_mean']:.1f}/{s['logit_max']:.1f}) "
                    f"logit_res={int(s['logit_w'])}x{int(s['logit_h'])}"
                )
    print("--- 全フレーム集計 ---")
    for obj_index in sorted(agg_area.keys()):
        areas = agg_area[obj_index]
        inbox = agg_inbox.get(obj_index, [])
        mean_area = float(np.mean(areas))
        mean_inbox = float(np.mean(inbox)) if inbox else float("nan")
        print(
            f"obj{obj_index + 1}: mean_area={mean_area:.3f}  mean_in_box={mean_inbox:.3f}  "
            f"(in_box が低い=mask が box 外/背景側、area が box_area より大きい=対象外へ膨張)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if getattr(args, "diagnose", False):
        return run_diagnose(args)
    result = run(args)
    matte = result.get("video_writer", {}).get("matte") or result.get("frame_sequence_writer", {}).get("matte") or {}
    print("=== 出力 ===")
    for key in (
        "rgba_video_path",
        "alpha_video_path",
        "preview_video_path",
        "rgba_sequence_dir",
        "alpha_sequence_dir",
        "preview_sequence_dir",
    ):
        value = matte.get(key)
        if value:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
