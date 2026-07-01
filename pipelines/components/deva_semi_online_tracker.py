"""DEVA方式 DevaSemiOnlineTracker（@component コーディネータ）の実装。

DEVA の semi-online 方式を本スタックで再構成する中核。検出島・伝播・consensus を
**クリップ周回**で束ね、再シードの状態（track memory）を内部に隔離する。

外部から見た I/O は前向き疎結合（Haystack DAG）を保ち、フィードバック・ループ
（はがれ→再検出→再シード）は本コンポーネント内部にのみ閉じ込める（計画書 §2.3）:

    VideoReader ─►[DevaSemiOnlineTracker]─► OwnershipResolver ─► BEN2 ─► Writers
                        │
                        └─ 内部でのみ ①検出島 ②伝播 ③consensus を周回

クリップ周回の流れ（detection_every ごと）:
    1. 検出フレーム d_i で DetectionIsland（image-level 仮説）。
    2. 直前クリップが d_i に伝播したマスクと検出を ``merge_consensus`` で突き合わせ、
       track memory を更新（維持/新規/missed++/削除）。
    3. consensus 後の track box でクリップ [d_i, d_{i+1}] を SAM2 伝播（再シード）。
    4. クリップ末（次の検出フレーム d_{i+1}）の per-object マスクを次の consensus 用に保持。

出力 ``masks`` は **既存 BEN2 union 契約**
``{frame_masks:{idx:(H,W)float32}, object_ids, frame_indices, source, metadata}`` に一致させ、
下流（OwnershipResolver / BEN2RouteAVideoExtractor / Writers）へ無改変で接続できる。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from haystack import component

from pipelines.components.consensus import merge_consensus
from pipelines.components.video_common import build_frame_mask_sequence

__all__ = ["DevaSemiOnlineTracker"]


def _notify(progress_callback, stage: str, fraction: float, description: str) -> None:
    """進捗コールバックがあれば (stage, fraction[0..1], description) を通知する。"""
    if progress_callback is None:
        return
    progress_callback(stage, min(max(float(fraction), 0.0), 1.0), description)


def _downsample_per_object_logits(stacked: np.ndarray, max_side: int) -> np.ndarray:
    """per-object logits (N,H,W) を long side <= max_side へ縮小する（アスペクト比保持）。

    OwnershipResolver が消費する soft guard 用 logit のメモリを抑える（ERR068: 4K×多対象で
    フル解像度 float32 を全 frame 蓄積すると host-RAM が枯渇するため）。``max_side <= 0`` または
    既に収まる場合は入力をそのまま返す。``N == 0``（対象なし frame）も縮小形状で返す。

    Args:
        stacked: (N,H,W) float32 の per-object logits。
        max_side: 縮小後の長辺上限（px）。0 以下で縮小しない。

    Returns:
        (N, h, w) float32（h,w は max_side 以下、アスペクト比保持）。
    """
    arr = np.asarray(stacked, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"per_object_logits は (N,H,W) 形式が必要です: shape={arr.shape}")
    n, h, w = arr.shape
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return arr
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if n == 0:
        return np.zeros((0, new_h, new_w), dtype=np.float32)
    out = np.empty((n, new_h, new_w), dtype=np.float32)
    for i in range(n):
        out[i] = cv2.resize(arr[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out


@component
class DevaSemiOnlineTracker:
    """検出島＋SAM2 伝播＋consensus をクリップ周回で束ねる semi-online コーディネータ。

    Args:
        detection_island: DetectionIsland 互換（``run`` を持つ）。None なら warm_up で既定構築。
        propagator: SAM2VideoPropagator 互換（``run`` を持つ）。None なら warm_up で既定構築。
        device: 推論デバイス（既定構築時に伝える）。
    """

    def __init__(
        self,
        detection_island: Any | None = None,
        propagator: Any | None = None,
        device: str | None = None,
    ) -> None:
        self._detection_island = detection_island
        self._propagator = propagator
        self._device = device

    def warm_up(self) -> None:
        """依存（検出島・伝播）を遅延構築する（import 時に重い初期化をしない）。"""
        if self._detection_island is None:
            from pipelines.components.detection_island import DetectionIsland

            self._detection_island = DetectionIsland(device=self._device)
        if self._propagator is None:
            from pipelines.components.video_model_components import SAM2VideoPropagator

            self._propagator = SAM2VideoPropagator(device=self._device)
        if hasattr(self._detection_island, "warm_up"):
            self._detection_island.warm_up()
        if hasattr(self._propagator, "warm_up"):
            self._propagator.warm_up()

    @component.output_types(masks=dict)
    def run(
        self,
        frames: list,
        metadata: dict | None = None,
        text_prompt: str = "",
        detection_every: int = 10,
        max_missed_detection_count: int = 3,
        iou_threshold: float = 0.5,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        top_k: int = 20,
        per_object_logits_max_side: int = 0,
        initial_boxes: list | None = None,
        initial_points: list | None = None,
        initial_labels: list | None = None,
        detection_start_frame: int = 0,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """クリップ周回で追跡し、BEN2 union 契約の masks を返す。

        Args:
            frames: RGB フレーム列 [(H, W, 3) uint8]。
            metadata: VideoReader 由来のメタデータ（出力 metadata に引き継ぐ）。
            text_prompt: GroundingDINO へのテキストプロンプト。空文字でも ``initial_boxes`` が
                あれば手動 seed のみで動作する（モードA: 周期再検出なし）。
            detection_every: 検出島を走らせる周期（フレーム数）。``text_prompt`` が空の
                モードA では無視され、全フレームを単一クリップで伝播する（ベースアプリ相当）。
            max_missed_detection_count: この回数を超えて未検出の track を削除する。
            iou_threshold: consensus の IoU マッチ閾値。
            box_threshold: GroundingDINO box 閾値。
            text_threshold: GroundingDINO text 閾値。
            top_k: 検出 box 上限。
            per_object_logits_max_side: 蓄積する per_object_logits の長辺上限（px）。0 で原寸
                （後方互換）。>0 のとき各 frame の per_object_logits を縮小して host-RAM を抑え、
                原寸を ``masks["frame_hw"]`` に残して OwnershipResolver が soft guard を原寸へ復元
                できるようにする（ERR068: 4K×多対象 OOM 対策。soft guard は膨張・ブラーされる
                低周波ガイドのため縮小しても実害が小さい。最終 α は BEN2 が原寸で生成する）。
            initial_boxes: 手動 box（xyxy）のリスト。第1クリップ（``clip_start`` =
                ``detection_start_frame``。既定 0 なら先頭フレーム）の初期 track として
                pre-populate し、その起点フレームの SAM2 seed にする。``text_prompt`` ありの場合は
                consensus が text 検出と IoU 統合する（同一対象なら重複を作らない）。
            initial_points: 手動補正 point 座標 ``[[x, y], ...]``（pos/neg 混在可）。各 point は
                最近傍 box（``initial_boxes``）に割り当てられ、**第1クリップの seed にのみ**渡る
                （対象が動くため後続クリップへ同じ座標を再投影しない）。``initial_boxes`` が無い
                場合は割り当て先が無いため ValueError。
            initial_labels: ``initial_points`` の各 label（1=positive / 0=negative）。省略時は
                全て positive(1)。指定時は ``initial_points`` と同長が必要。
            detection_start_frame: 検出/seed を最初に確立する **起点フレーム**（サンプリング後
                シーケンス上の 0 始まり local index）。既定 0（従来通りフレーム0起点・前向きのみ）。
                >0 のとき、被写体が最大に映るフレーム等で seed してから、``[0, detection_start_frame)``
                を **逆伝播（双方向）** でカバーする。text モードでは周期再検出も起点フレームから開始する
                （``range(detection_start_frame, num_frames, detection_every)``）。逆伝播は標準 SAM2 系
                tracker のみ対応で、forward-only（SAMURAI: ``single_object_only``）では ValueError。
            progress_callback: 進捗通知 (stage, fraction, description)。

        Returns:
            ``{"masks": {frame_masks, object_ids, frame_indices, source, metadata, per_object_logits}}``。

        Note:
            出力 ``masks`` の frame_index キー（``frame_masks`` / ``per_object_logits``）は
            **source_index** で統一される。これは VideoReader が返す
            ``metadata["metadata"]["sampled_frame_indices"]`` と同じ値で、frame_step /
            max_frames でサンプリングされた場合も元動画の frame 番号になる。
            下流（OwnershipResolver / BEN2RouteAVideoExtractor / TrackingOverlayWriter）は
            ``source_index`` でマスクを引くため、本 Component は SAM2VideoPropagator と
            同一のキー規約でドロップイン可能である。``per_object_logits`` は
            全 source frame を網羅し（対象なし frame は (0,H,W)）、OwnershipResolver が
            frame を取りこぼさないよう保証する。
        """
        if not frames:
            raise ValueError("frames が空です。")
        if detection_every <= 0:
            raise ValueError(f"detection_every は 1 以上が必要です: {detection_every}")
        has_manual_seed = bool(initial_boxes)
        if not text_prompt and not has_manual_seed:
            raise ValueError(
                "text_prompt または手動 box（initial_boxes）のいずれかを指定してください。"
            )
        if initial_points and not has_manual_seed:
            raise ValueError(
                "手動 point（initial_points）には初期 box（initial_boxes）が必要です。"
            )
        if initial_points and initial_labels is not None and len(initial_labels) != len(initial_points):
            raise ValueError(
                "initial_labels の長さは initial_points と一致が必要です: "
                f"points={len(initial_points)} labels={len(initial_labels)}"
            )

        self.warm_up()
        assert self._detection_island is not None
        assert self._propagator is not None

        num_frames = len(frames)
        height, width = frames[0].shape[:2]
        detection_start_frame = int(detection_start_frame)
        if detection_start_frame < 0 or detection_start_frame >= num_frames:
            raise ValueError(
                f"detection_start_frame が範囲外です: {detection_start_frame}"
                f"（許容 0〜{num_frames - 1}）"
            )
        # 起点フレーム>0 は [0, start) への逆伝播（双方向）を要する。forward-only tracker
        # （SAMURAI: Kalman filter で reverse 不可・single_object_only）では実行できないため
        # fail-fast する（ERR051 と同系のガード）。
        if detection_start_frame > 0 and getattr(self._propagator, "single_object_only", False):
            raise ValueError(
                "検出起点フレーム>0（起点より前への逆伝播が必要）は forward-only tracker"
                "（SAMURAI）では使用できません。標準 SAM2 系 tracker を使用するか、"
                "起点フレームを 0 にしてください。"
            )
        # モードA（text 空・手動 seed のみ）は周期再検出を行わず、全フレームを単一クリップで
        # 手動 seed から伝播する（ベースアプリ相当）。これにより stale-box / missed 削除を回避する。
        if text_prompt:
            detection_frame_indices = list(range(detection_start_frame, num_frames, detection_every))
        else:
            detection_frame_indices = [detection_start_frame]
        # 出力 frame_masks のキーは下流（OwnershipResolver/BEN2/TrackingOverlay）が引く
        # source_index（VideoReader の sampled_frame_indices）に揃える。サンプリングで
        # 非連続になりうるため、内部の local 位置 0..N-1 を source_index に写像する。
        source_indices = list(
            (metadata or {}).get("metadata", {}).get("sampled_frame_indices", range(num_frames))
        )

        if text_prompt:
            _notify(progress_callback, "deva_tracker", 0.02, "検出島（GroundingDINO→SAM2）を実行しています")
            det_out = self._detection_island.run(
                frames=frames,
                detection_frame_indices=detection_frame_indices,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                iou_threshold=iou_threshold,
                top_k=top_k,
            )
            detections = det_out["detections"]
        else:
            # モードA: 検出島を呼ばず空検出。consensus は手動 track をそのまま維持する。
            _notify(progress_callback, "deva_tracker", 0.02, "手動 seed で SAM2 伝播します（再検出なし）")
            detections = {}

        # 再シード状態（フィードバック）はコーディネータ内部にのみ保持する。
        tracks: list[dict[str, Any]] = []
        next_object_id = 1
        propagated_at_start: dict[int, np.ndarray] = {}
        global_frame_masks: dict[int, np.ndarray] = {}
        global_per_object_logits: dict[int, np.ndarray] = {}
        seen_object_ids: list[int] = []
        all_new_object_ids: list[int] = []
        all_deleted_object_ids: list[int] = []
        # 起点フレーム>0 のとき、第1クリップ（clip_start=detection_start_frame）で確立した seed を
        # 逆伝播（[0, detection_start_frame)）用に退避する。
        backward_seed_boxes: list | None = None
        backward_seed_points: list | None = None
        backward_seed_labels: list | None = None

        # 手動 box を初期 track として pre-populate する。propagated_at_start にも同マスクを
        # 入れることで、モードB の第1クリップで text 検出と IoU 統合され、同一対象の重複 track を
        # 防ぐ（マッチ時は text box へ再アンカー、未マッチ時は手動 track を維持して seed する）。
        if has_manual_seed:
            for obj_id, box in enumerate(initial_boxes, start=1):
                x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                xa, xb = int(round(min(x0, x1))), int(round(max(x0, x1)))
                ya, yb = int(round(min(y0, y1))), int(round(max(y0, y1)))
                xa, xb = max(0, xa), min(width, xb)
                ya, yb = max(0, ya), min(height, yb)
                manual_mask = np.zeros((height, width), dtype=bool)
                manual_mask[ya:yb, xa:xb] = True
                tracks.append(
                    {
                        "object_id": obj_id,
                        "missed": 0,
                        "mask": manual_mask,
                        "box": (x0, y0, x1, y1),
                        "label": "manual",
                        "score": 1.0,
                    }
                )
                propagated_at_start[obj_id] = manual_mask
            next_object_id = len(initial_boxes) + 1

        num_clips = len(detection_frame_indices)
        for clip_i, clip_start in enumerate(detection_frame_indices):
            is_last = clip_i == num_clips - 1
            clip_stop_exclusive = num_frames if is_last else detection_frame_indices[clip_i + 1]
            # クリップ末（次の検出フレーム）を 1 frame だけ重ねて伝播し、その位置の
            # per-object マスクを次クリップの consensus（propagated）に使う。
            clip_frames = frames[clip_start : clip_stop_exclusive + (0 if is_last else 1)]

            detected = detections.get(clip_start, self._empty_detection(height, width))
            consensus = merge_consensus(
                tracks=tracks,
                propagated=propagated_at_start,
                detected=detected,
                iou_threshold=iou_threshold,
                max_missed=max_missed_detection_count,
                next_object_id=next_object_id,
            )
            tracks = consensus["tracks"]
            next_object_id = consensus["next_object_id"]
            all_new_object_ids.extend(consensus["new_object_ids"])
            all_deleted_object_ids.extend(consensus["deleted_object_ids"])

            # box を持つ track だけを再シード対象にする（順序＝クリップ内 obj_id 1..N）。
            seed_tracks = [t for t in tracks if t.get("box") is not None]

            fraction = 0.05 + 0.9 * ((clip_i + 1) / max(num_clips, 1))
            if not seed_tracks:
                # 追跡対象が無いクリップは zero マスクで被覆を維持する。
                # OwnershipResolver は frame_masks を per_object_logits のキーのみから再構築し
                # 欠落 frame を破棄するため、空クリップでも (0,H,W) の per_object_logits を
                # 出力して frame を残し、前景 0（背景のみ）として下流に渡す。
                empty_logits = np.zeros((0, height, width), dtype=np.float32)
                if per_object_logits_max_side > 0:
                    empty_logits = _downsample_per_object_logits(empty_logits, per_object_logits_max_side)
                for local in range(clip_stop_exclusive - clip_start):
                    g = source_indices[clip_start + local]
                    global_frame_masks[g] = np.zeros((height, width), dtype=np.float32)
                    global_per_object_logits[g] = empty_logits
                propagated_at_start = {}
                _notify(progress_callback, "deva_tracker", fraction,
                        f"クリップ {clip_i + 1}/{num_clips}（対象なし）")
                continue

            # SAMURAI など single_object_only な tracker は複数同時追跡不可（ERR051）。
            if (
                len(seed_tracks) > 1
                and getattr(self._propagator, "single_object_only", False)
            ):
                raise ValueError(
                    "single_object_only な tracker（SAMURAI）では複数オブジェクトを"
                    f"同時追跡できません（seed={len(seed_tracks)}）。SAM2 系 tracker を使用してください。"
                )

            seed_boxes = [list(t["box"]) for t in seed_tracks]
            # 手動補正 point/label は第1クリップの seed にのみ渡す（対象が動くため後続クリップへ
            # 同じ座標を再投影しない）。propagator 内部の assign_points_to_boxes が各 point を
            # 最近傍 seed box に割り当てる（negative=0 含む）。
            seed_points = list(initial_points) if (clip_i == 0 and initial_points) else None
            seed_labels = (
                list(initial_labels) if (clip_i == 0 and initial_points and initial_labels is not None)
                else None
            )
            # 第1クリップの seed を逆伝播用に退避（起点フレーム>0 のときのみ後段で使用）。
            # obj_id 並び（1..N）を forward clip0 と一致させるため同じ seed_tracks から作る。
            if clip_i == 0 and detection_start_frame > 0:
                backward_seed_boxes = list(seed_boxes)
                backward_seed_points = list(initial_points) if initial_points else None
                backward_seed_labels = (
                    list(initial_labels) if (initial_points and initial_labels is not None) else None
                )
            _notify(progress_callback, "deva_tracker", fraction,
                    f"クリップ {clip_i + 1}/{num_clips} を SAM2 伝播しています")
            result = self._propagator.run(
                frames=clip_frames,
                metadata=None,
                points=seed_points,
                labels=seed_labels,
                boxes=seed_boxes,
                prompt_frame_idx=0,
                progress_callback=None,
            )
            union = result["masks"]
            clip_frame_masks = union["frame_masks"]
            per_object_logits = union.get("per_object_logits", {})

            for t in seed_tracks:
                if t["object_id"] not in seen_object_ids:
                    seen_object_ids.append(t["object_id"])

            # グローバル frame へ書き戻す（重ね frame は次クリップが再アンカー済みで上書きするため除外）。
            # 出力キーは下流が引く source_index に写像する。union(frame_masks) と
            # per_object_logits の両方を出力し、OwnershipResolver / BEN2 の union・per_object
            # 両経路に対応する。
            write_stop_local = clip_stop_exclusive - clip_start
            for local in range(write_stop_local):
                g = source_indices[clip_start + local]
                if local in clip_frame_masks:
                    global_frame_masks[g] = np.asarray(clip_frame_masks[local], dtype=np.float32)
                if local in per_object_logits:
                    logits_full = np.asarray(per_object_logits[local], dtype=np.float32)
                    if per_object_logits_max_side > 0:
                        logits_full = _downsample_per_object_logits(
                            logits_full, per_object_logits_max_side
                        )
                    global_per_object_logits[g] = logits_full

            # 次クリップ consensus 用に、重ね frame（クリップ末）の per-object マスクを保持。
            if not is_last:
                overlap_local = clip_stop_exclusive - clip_start
                propagated_at_start = self._extract_propagated(
                    per_object_logits, overlap_local, seed_tracks, height, width
                )

        # 検出起点フレーム>0 のとき、起点で確立した seed から [0, detection_start_frame) を
        # 逆伝播でカバーする（双方向）。「被写体が最大に映るフレーム」等で seed しつつ冒頭 frame も
        # 取りこぼさない。逆伝播は前段のガードで標準 SAM2 系に限定済み。
        if detection_start_frame > 0:
            if backward_seed_boxes:
                _notify(progress_callback, "deva_tracker", 0.97,
                        "起点フレームより前を逆伝播しています")
                backward_frames = frames[: detection_start_frame + 1]
                back = self._propagator.run(
                    frames=backward_frames,
                    metadata=None,
                    points=backward_seed_points,
                    labels=backward_seed_labels,
                    boxes=backward_seed_boxes,
                    prompt_frame_idx=detection_start_frame,
                    bidirectional=True,
                    progress_callback=None,
                )
                back_union = back["masks"]
                back_frame_masks = back_union["frame_masks"]
                back_per_object_logits = back_union.get("per_object_logits", {})
                # 起点フレーム自体は forward clip0 が既に書いているため [0, start) のみ書く。
                for local in range(detection_start_frame):
                    g = source_indices[local]
                    if local in back_frame_masks:
                        global_frame_masks[g] = np.asarray(back_frame_masks[local], dtype=np.float32)
                    if local in back_per_object_logits:
                        logits_full = np.asarray(back_per_object_logits[local], dtype=np.float32)
                        if per_object_logits_max_side > 0:
                            logits_full = _downsample_per_object_logits(
                                logits_full, per_object_logits_max_side
                            )
                        global_per_object_logits[g] = logits_full
            else:
                # 起点フレームで対象を確立できなかった場合も冒頭 frame を欠落させない（zero 被覆）。
                # OwnershipResolver は per_object_logits のキーから frame を復元するため空でも残す。
                empty_logits = np.zeros((0, height, width), dtype=np.float32)
                if per_object_logits_max_side > 0:
                    empty_logits = _downsample_per_object_logits(empty_logits, per_object_logits_max_side)
                for local in range(detection_start_frame):
                    g = source_indices[local]
                    global_frame_masks[g] = np.zeros((height, width), dtype=np.float32)
                    global_per_object_logits[g] = empty_logits

        masks = build_frame_mask_sequence(
            global_frame_masks,
            object_ids=sorted(seen_object_ids) if seen_object_ids else [1],
            source="deva_semi_online",
            metadata={
                "text_prompt": text_prompt,
                "detection_every": int(detection_every),
                "max_missed_detection_count": int(max_missed_detection_count),
                "iou_threshold": float(iou_threshold),
                "detection_start_frame": int(detection_start_frame),
                "detection_frame_indices": [source_indices[i] for i in detection_frame_indices],
                "new_object_ids": all_new_object_ids,
                "deleted_object_ids": all_deleted_object_ids,
                "num_objects": len(seen_object_ids),
                "source_metadata": metadata or {},
            },
        )
        # OwnershipResolver / BEN2 per_object 経路が消費する per-object logits を引き継ぐ
        # （build_frame_mask_sequence は union 契約のみ生成するため追加で付与する）。
        masks["per_object_logits"] = global_per_object_logits
        if per_object_logits_max_side > 0:
            # per_object_logits を縮小したため、OwnershipResolver が soft guard を原寸へ
            # 復元できるよう原寸 (H,W) を contract として残す（frame_masks は原寸のまま）。
            masks["frame_hw"] = (int(height), int(width))
        _notify(progress_callback, "deva_tracker", 1.0, "追跡が完了しました")
        return {"masks": masks}

    @staticmethod
    def _extract_propagated(
        per_object_logits: dict,
        overlap_local: int,
        seed_tracks: list[dict[str, Any]],
        height: int,
        width: int,
    ) -> dict[int, np.ndarray]:
        """重ね frame の per-object logit を {object_id: bool マスク} に変換する。

        SAM2VideoPropagator へ ``metadata=None`` で伝播させたため、返る
        ``per_object_logits`` のキーは local frame index（0..len(clip_frames)-1）となる
        （video_model_components: source_indices=range(len(frames))）。よって
        ``overlap_local = clip_stop_exclusive - clip_start``（クリップ末の local index）で
        重ね frame を取り出せる。per-object の並びは seed_tracks 順（obj 1..N）に一致する。
        """
        propagated: dict[int, np.ndarray] = {}
        stacked = per_object_logits.get(overlap_local)
        if stacked is None:
            return propagated
        stacked = np.asarray(stacked)
        for k, track in enumerate(seed_tracks):
            if k < stacked.shape[0]:
                # logit > 0 ⇔ sigmoid 確率 > 0.5。
                propagated[track["object_id"]] = stacked[k] > 0.0
        return propagated

    @staticmethod
    def _empty_detection(height: int, width: int) -> dict[str, Any]:
        """検出フレーム情報が無い場合の空検出エントリ。"""
        return {
            "masks": np.zeros((0, height, width), dtype=bool),
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": [],
        }
