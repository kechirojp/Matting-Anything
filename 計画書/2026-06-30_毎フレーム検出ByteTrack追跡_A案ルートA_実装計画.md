# 毎フレーム検出＋ByteTrack 追跡（A案）× ブラー誘導BEN2（ルートA）実装計画

- 作成日: 2026-06-30
- 区分: 実装計画（MVP）
- ベース: `gradio_app_sam2_ben2_route_a_for_Movie.py`（変更せず、複製ベース）
- 関連: `調査/2026-06-30_RF-DETR Nano + ByteTrack_ダブル追跡.md`（要件定義書 / A案・B案）

---

## 1. 目的

追跡対象（人物等）を動画から切り抜き、透明背景（RGBA / α）として出力する。
本MVPの主眼は **「トラッキングがはがれない」**こと（要件 F-2 ロスト復帰）。
追跡軸は **A案＝毎フレーム検出**（構造的にロストしない）、合成軸は **ルートA＝ブラー誘導BEN2** を採用する。

Haystack 2.x の Component 分割・疎結合・I/O 契約厳守で実装し、検出器を差し替え可能にする。

---

## 2. 設計方針（確定事項）

| 項目 | 決定 |
|------|------|
| 環境 | 既存 `.venv + pip`（uv 移行は対象外） |
| 進め方 | A案 × ルートA を最初の MVP |
| 検出器（ハイブリッド） | frame0 で GroundingDINO（自由テキスト→枠＋ラベル＝身元シード）。毎フレーム検出は MVP=GroundingDINO（追加インストール不要）。ByteTrack で ID 継続、シード枠と IoU 最大の track を対象に選別→枠/ラベル引き継ぎ |
| RF-DETR Nano | Phase2 で同一 `detections` 契約に差し替え（速度向上、`rfdetr` を pip 追加） |
| 対象選択 | MVP は「シードに最も合致する track を自動選択」 |
| 触らない | 既存アプリ / `segment-anything/` / `samurai/` |

---

## 3. フロー図（ASCII）

### 3.1 全体パイプライン（A案 × ルートA）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Haystack Pipeline（型付き DAG）                          │
└──────────────────────────────────────────────────────────────────────────┘

  入力動画
    │
    ▼
┌──────────────┐  frames(list[HWC uint8])
│ VideoReader  │  metadata(fps/size/…)
└──────┬───────┘
       │ frames
       ▼
┌───────────────────────────┐   detections =
│ GroundingDINOVideoDetector│   { f: { boxes(K,4)xyxy, scores, labels } }
│  毎フレーム検出（テキスト） │   ← frame0 の枠＋ラベルが「身元シード」
└──────────┬────────────────┘
           │ detections
           ▼
┌───────────────────────────┐   tracks =
│   ByteTrackAssigner        │   { f: { boxes, track_ids, scores, labels } }
│   ID 付与（検出器非依存）   │   ← フレーム間で ID 継続
└──────────┬────────────────┘
           │ tracks
           ▼
┌───────────────────────────┐   target_boxes =
│   TargetTrackSelector      │   { f: [x1,y1,x2,y2] | None }
│  シード枠と IoU 最大を選別  │   chosen_track_id / label
│  欠落フレームは hold        │
└──────────┬────────────────┘
           │ target_boxes          frames ┐
           ▼                              │
┌───────────────────────────┐            │
│  SAM2ImageMaskSequencer    │◀───────────┘
│  SAM2 画像モードで box→mask │   masks =
│  （橋渡し: BEN2契約に整形）  │   { frame_masks{f:(H,W)f32}, object_ids, metadata }
└──────────┬────────────────┘
           │ masks                  frames ┐
           ▼                              │
┌───────────────────────────┐            │
│ BEN2RouteAVideoExtractor   │◀───────────┘
│ ① mask 膨張→ゲート G        │   matte =
│ ② G 外をブラー → I'         │   { rgba/alpha/preview パス, fps, … }
│ ③ BEN2 で I' を再α化        │
│ ④ ゲート/α floor/RGBA 合成  │
└──────────┬────────────────┘
           │ matte
   ┌───────┼─────────────────┬───────────────────────┐
   ▼       ▼                 ▼                       ▼
┌─────────┐ ┌──────────────────┐         ┌──────────────────────┐
│VideoWriter│ │FrameSequenceWriter│         │ TrackingOverlayWriter │
│ 動画書出  │ │ 連番PNG書出        │         │ 追跡確認オーバーレイ    │
└─────────┘ └──────────────────┘         └──────────────────────┘
   │             │                                  │
   ▼             ▼                                  ▼
 RGBA動画     連番PNG                         追跡確認動画（はがれ検証）
```

### 3.2 「はがれない」仕組み（A案の核）

```
従来（伝播ベース）:                    A案（毎フレーム検出）:
  frame0 でプロンプト                    各フレームで再検出
        │                                    │
        ▼                                    ▼
  SAM2 が記憶で追う ──┐               検出 → ByteTrack ID → SAM2
        │            │ 記憶が外れると         │
        ▼            ▼ そのまま「はがれ」      ▼
  …伝播…          ロスト継続           「見失う」概念が薄い
                                       （毎フレーム取り直す）
```

### 3.3 検出器の差し替え（疎結合 / Phase2）

```
   detections 契約（boxes/scores/labels）は固定
        ▲                         ▲
        │                         │
┌───────┴────────┐      ┌────────┴─────────┐
│ GroundingDINO  │      │  RF-DETR Nano    │   ← 同じ socket 型で差し替え
│ Video Detector │      │  Video Detector  │      （Haystack の狙い）
│ MVP / 自由語彙  │      │ Phase2 / 高速COCO │
└────────────────┘      └──────────────────┘
        │                         │
        └──────────┬──────────────┘
                   ▼
            ByteTrackAssigner（変更不要）
```

---

## 4. 新規 Component（`pipelines/components/tracking_components.py`）

| Component | 入力 | 出力（@component.output_types） |
|-----------|------|-------------------------------|
| `GroundingDINOVideoDetector` | `frames:list`, `text_prompt:str`, 閾値 | `detections:dict` = {f:{boxes(K,4)xyxy f32, scores, labels}} |
| `ByteTrackAssigner` | `detections:dict`, `metadata:dict` | `tracks:dict` = {f:{boxes, track_ids, scores, labels}} |
| `TargetTrackSelector` | `tracks:dict`, `seed_box`, `seed_label` | `target_boxes:dict` = {f:[x1,y1,x2,y2]|None}, `chosen_track_id`, `label` |
| `SAM2ImageMaskSequencer` | `frames:list`, `target_boxes:dict` | `masks:dict` = {frame_masks{f:(H,W)f32}, object_ids, metadata} |

- `GroundingDINOVideoDetector` は既存 `GroundingDINOMultiBoxDetector` のモデルをフレームループ再利用。
- `ByteTrackAssigner` は `supervision.ByteTrack`（検出器非依存・純関数寄り）。
- `SAM2ImageMaskSequencer` は既存 `SAM2Segmenter`（画像モード）を box prompt で実行し、binary→float soft 化して BEN2 の union masks 契約へ橋渡し。

### 再利用 Component（変更なし）
`VideoReader` / `BEN2RouteAVideoExtractor` / `VideoWriter` / `FrameSequenceWriter` / `TrackingOverlayWriter`

---

## 5. TODO

### Phase 0 — 依存確認
- [ ] `supervision` の ByteTrack 対応（`sv.ByteTrack`）をバージョン確認（不足なら最小 upgrade）
- [ ] GroundingDINO / SAM2 既存重み・config の存在確認（追加 DL 不要を確認）

### Phase 1 — 新規 Component（RED→GREEN）
- [ ] `ByteTrackAssigner` の純関数テスト（合成 box で track_id 連続性）→ 実装
- [ ] `TargetTrackSelector` のテスト（IoU 選別 / 欠落フレーム hold）→ 実装
- [ ] `GroundingDINOVideoDetector` 実装（既存 DINO モデル再利用、`detections` 契約）
- [ ] `SAM2ImageMaskSequencer` の masks 契約テスト（キー / shape）→ 実装（`SAM2Segmenter` 画像モード利用）
- [ ] モデル依存 Component に `warm_up()` ＋ `@pytest.mark.integration` 骨格

### Phase 2 — Pipeline 配線
- [ ] `pipelines/route_a_tracked_video_pipeline.py` に `build_sam2_ben2_route_a_tracked_pipeline(...)`（依存注入対応）
- [ ] `VideoReader→…→BEN2RouteAVideoExtractor→Writer群` ＋ `masks→TrackingOverlayWriter` を connect
- [ ] `pipelines/__init__.py` へ export 追加

### Phase 3 — Gradio アプリ
- [ ] `gradio_app_sam2_ben2_route_a_tracked_for_Movie.py` をベース複製して作成
- [ ] pipeline build を tracked 版へ差し替え（`get_route_a_tracked_pipeline()`）
- [ ] run callback の SAM2 伝播入力（points/box/bidirectional）→ `text_prompt + 自動対象選択` に置換
- [ ] 伝播専用 UI（bidirectional 等）撤去、ルートA ブラー params と overlay 出力は維持

### Phase 4 — 検証
- [ ] `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
- [ ] `.venv\Scripts\python.exe gradio_app_sam2_ben2_route_a_tracked_for_Movie.py --help`
- [ ] Playwright 実行時検証（UI 配線、ERR035 準拠。完了記録前に実施）
- [ ] サブエージェントレビュー（全 socket 具体型 / 差し替え可能性 / `torch.load weights_only=True`）

### Phase 5 —（任意・後）RF-DETR Nano 差し替え
- [ ] `rfdetr` を pip 追加、`RFDETRVideoDetector`（`detections` 契約同一）実装
- [ ] registry に detector entry 追加、UI で検出器切替（GroundingDINO は frame0 シード専用に残す）

### 完了処理
- [ ] `WHITEBOARD.md` 更新（完了内容 / 次アクション / テスト省略理由）
- [ ] 新規エラー解決時は `ERROR_LOG.md` 更新

---

## 6. 留意・リスク

- 毎フレーム GroundingDINO は重く「10秒 → 2分」budget 超過の恐れ。MVP は `frame_step` / `max_frames` で緩和。速度は Phase5 の RF-DETR Nano で改善。
- `torch.load(..., weights_only=True)` を厳守。`try/except: pass` 禁止（`raise` / `gr.Error`）。
- 設定値は `config/*.toml` 経由。ハードコード禁止。
- 既存正常系・`segment-anything/`・`samurai/` は変更しない。
