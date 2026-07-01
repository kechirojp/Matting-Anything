# DEVA方式 再構成 実装計画（SAM2伝播＋周期再検出＋consensusマージ × ルートA）

- 作成日: 2026-06-30
- 区分: 実装計画（**本プロジェクトの中核方針。外すとプロジェクト破壊につながる**）
- ベースアプリ: `gradio_app_sam2_ben2_route_a_for_Movie.py`（ルートA：SAM2伝播＋BEN2ブラー誘導）
- 関連調査: `調査/2026-06-30_動画マッティング_追跡はがれ再検出復帰_手法調査.md`（§8 に DEVA 詳細）
- 位置づけ: `計画書/2026-06-30_毎フレーム検出ByteTrack追跡_A案ルートA_実装計画.md`（A案）を**本計画で置換**。A案は「伝播で追従しきれない素材」の代替候補に降格。

---

## 0. 最重要原則（破壊防止ライン）

> **DEVA を“採用”するのではない。DEVA の“方式（decoupled video segmentation）”を、ベースアプリ上に SAM2.1 / BEN2 / GroundingDINO で再構成する。**

- DEVA 本体は使わない（理由：SAM2 非対応／非商用ライセンス懸念／Windows WSL2＋Gurobi 依存。調査 §8-2 参照）。
- DEVA から受け継ぐのは**設計思想だけ**：
  1. **検出（image-level）と伝播（temporal）を分離**する。
  2. 検出は**周期的（detection_every）**にだけ走らせる。
  3. 検出結果と伝播結果を**consensus（IoUマッチ）でマージ**する。
  4. 消えた対象は**missedカウント→閾値超で削除**（memory掃除）。
- これにより **SAM2.1 高品質マスク / Apache ライセンス / 既存 Windows 環境**を保ったまま「はがれ→自動復帰」を構造的に獲得する。
- **この分離・consensus・周期再検出を外した実装は本方針の破壊とみなす。**

---

## 1. DEVA方式 → 本スタックへの対応表

| DEVA の構成要素 | DEVA 標準 | 本再構成での置換 |
|----------------|-----------|------------------|
| (b) task非依存 temporal propagation | XMem系 自前モデル | **SAM2 video propagation（既存 SAM2VideoPropagator）** |
| (a) image-level 検出 | GroundingDINO | **GroundingDINO（既存 GroundingDINOMultiBoxDetector）** |
| (a) image-level セグメンテーション | SAM(初代)/HQ-SAM | **SAM2 画像モード（既存 SAM2Segmenter）** |
| consensus マージ | in-clip consensus + 双方向伝播 | **新規 ConsensusMerger（IoUマッチ＋track memory）** |
| 周期検出 | `detection_every` | **新規パラメータ `detection_every`** |
| memory 掃除 | `max_missed_detection_count` | **同名パラメータで再現** |
| 最終出力 | マスク | **ルートA：OwnershipResolver → BEN2RouteAVideoExtractor**（ベース流用） |

---

## 2. フロー図（ASCII）

### 2.1 semi-online クリップ処理（中核ループ）

```
 入力動画
   │
   ▼
┌───────────┐
│VideoReader│ frames:list[(H,W,3)uint8] + metadata
└─────┬─────┘
      │
      ▼
╔══════════════════════════════════════════════════════════╗
║  DevaSemiOnlineTracker（コーディネータ：ループを内包）       ║
║                                                          ║
║  for クリップ in 動画（detection_every フレーム毎に区切る）:   ║
║    ┌────────────────────────────────────────────────┐    ║
║    │ ① 検出フレームで image-level 仮説を作る           │    ║
║    │    GroundingDINO(text→box) → SAM2画像(box→mask)  │    ║
║    │    ＝ DetectionIsland                            │    ║
║    └───────────────┬────────────────────────────────┘    ║
║                    ▼                                      ║
║    ┌────────────────────────────────────────────────┐    ║
║    │ ② SAM2 伝播でクリップ内を埋める                   │    ║
║    │    SAM2VideoPropagator（前クリップの種から伝播）   │    ║
║    └───────────────┬────────────────────────────────┘    ║
║                    ▼                                      ║
║    ┌────────────────────────────────────────────────┐    ║
║    │ ③ ConsensusMerger（IoUマッチ）                   │    ║
║    │    伝播マスク vs 検出マスク を突き合わせ:          │    ║
║    │    ・マッチ → track維持・missed=0               │    ║
║    │    ・新規検出 → 新object追加（次クリップへ再シード）│    ║
║    │    ・伝播のみ未マッチ → missed++、>max で削除     │    ║
║    └───────────────┬────────────────────────────────┘    ║
║                    ▼                                      ║
║          確定マスク（frame_masks＋object_ids）             ║
╚════════════════════╪═════════════════════════════════════╝
                     ▼
            ┌─────────────────┐
            │ OwnershipResolver│  （ベース流用：重なり所有権）
            └────────┬────────┘
                     ▼
        ┌──────────────────────────┐
        │ BEN2RouteAVideoExtractor  │  ルートA：ブラー誘導 → 再α化
        └────────────┬─────────────┘
                     ▼
        ┌──────────┐ ┌──────────────────┐ ┌──────────────────┐
        │VideoWriter│ │FrameSequenceWriter│ │TrackingOverlay   │
        └──────────┘ └──────────────────┘ └──────────────────┘
```

### 2.2 「はがれ→復帰」が効く理由

```
        伝播だけ（ベース）              DEVA方式再構成（本計画）
  f0  ●対象くっきり              f0  ●対象くっきり（検出で初期化）
  f10 ◐少しずれる                f10 ◐ずれ始める
  f20 ◔大きくずれ（はがれ）        f20 ← detection_every で再検出
  f30 ○別物に飛ぶ ✗はがれ放置       │   GroundingDINO+SAM2 が現在の対象を再取得
                                  │   ConsensusでIoUマッチ→track補正/再シード
                                 f20 ●復帰（はがれを定期的に打ち消す）
                                 f30 ●維持
```

### 2.3 疎結合の保ち方（フィードバックの隔離）

```
  前向き疎結合（Haystack DAG）        フィードバックはここだけに閉じ込める
  VideoReader ─►[DevaSemiOnlineTracker]─► OwnershipResolver ─► BEN2 ─► Writers
                      │
                      └─ 内部でのみ ①検出島 ②伝播 ③consensus を周回
                         （再シードの状態はコーディネータが保持。外へは漏らさない）
```

---

## 3. Component 設計（pipelines/components/）

> フィードバック・ループは **DevaSemiOnlineTracker に内包**し、外部は前向き疎結合・安定 I/O 契約を維持する。サブ部品は単体テスト可能な純関数寄りに切る。

### 新規

1. **DetectionIsland**（@component, image-level 仮説）
   - in: `frames`, `detection_frame_indices:list[int]`, `text_prompt:str`, 閾値類
   - out: `detections = {frame_idx: {"masks":(K,H,W)bool, "boxes":(K,4)xyxy, "scores":(K,), "labels":list[str]}}`
   - 内部: 既存 `GroundingDINOMultiBoxDetector`（box）→ 既存 `SAM2Segmenter`（box→mask）を検出フレームのみ実行。

2. **ConsensusMerger**（@component, 純関数寄り＝先にテスト）
   - in: `propagated_masks`, `detected_masks`(=DetectionIsland出力), `track_memory`, `iou_threshold`, `max_missed_detection_count`
   - out: `merged = {frame_masks:{idx:(H,W)float32}, object_ids:list[int], metadata}`, `track_memory`(更新)
   - ロジック: IoUマッチ→維持/新規追加/missed++/削除。object_id の継続管理。

3. **DevaSemiOnlineTracker**（@component, コーディネータ＝ループ内包）
   - in: `frames`, `metadata`, `text_prompt`, `detection_every:int`, `max_missed_detection_count:int`, 閾値類, `progress_callback`
   - out: `masks = {frame_masks:{idx:(H,W)float32}, object_ids:list[int], metadata}`（**BEN2 が消費する union 契約に一致**）
   - 内部で DetectionIsland / SAM2VideoPropagator / ConsensusMerger を周回。**再シードの状態を保持**し外に漏らさない。

### 再利用（変更なし）
- `VideoReader` / `SAM2VideoPropagator`（再シード対応の確認は要）/ `OwnershipResolver` / `BEN2RouteAVideoExtractor` / `VideoWriter` / `FrameSequenceWriter` / `TrackingOverlayWriter`

### I/O 契約の要
- `DevaSemiOnlineTracker.masks` は **既存 BEN2 union 契約** `{frame_masks:{idx:(H,W)float32}, object_ids:list[int], metadata}` に厳密一致させる（配線互換）。
- SAM2 画像モード（`SAM2Segmenter`）の出力 `masks:(K,H,W)bool` → float soft 化して contract に橋渡し。

---

## 4. Pipeline 配線（新 `pipelines/route_a_deva_video_pipeline.py`）

```
VideoReader → DevaSemiOnlineTracker → OwnershipResolver
  → BEN2RouteAVideoExtractor → VideoWriter / FrameSequenceWriter
  + masks → TrackingOverlayWriter（追跡確認）
```
- `build_sam2_ben2_route_a_deva_pipeline(tracker=None, extractor=None)` で依存注入（差し替え可能性）。

## 5. Gradio アプリ（新 `gradio_app_sam2_ben2_route_a_deva_for_Movie.py`）
- ベースの UI / イベント配線を流用。差分:
  - pipeline build を DEVA 版に差し替え（`get_route_a_deva_pipeline()`）。
  - 追加 UI: `text_prompt`, `detection_every`, `max_missed_detection_count`, `iou_threshold`。
  - ルートA ブラー params（dilation/blur/feather/gate/mask_floor/output_type）はそのまま。
  - 追跡確認 overlay 出力は維持。

---

## 6. TODO

### Phase 0 — 前提確認
- [ ] `SAM2VideoPropagator` が**途中再シード（追跡開始後の新オブジェクト追加）**に対応できるか実コード確認（SAM2 本体は対応済。本リポジトリ実装の確認）
- [ ] `SAM2Segmenter`（画像モード）の box プロンプト I/O 再確認
- [ ] BEN2 union 契約（frame_masks/object_ids/metadata）の最終確認
- [ ] RTX4090 実測の準備（GroundingDINO 検出時間 / ルートA matting 後処理コスト＝`outputs/measure_route_a_vram.py` 系）

### Phase 1 — 純関数 Component（RED→GREEN）
- [ ] `ConsensusMerger` テスト（IoUマッチ／新規追加／missed++／max超削除／object_id継続）→ 実装
- [ ] DetectionIsland の I/O 契約テスト（検出フレームのみ実行、masks/boxes/labels 形状）

### Phase 2 — コーディネータ
- [ ] `DevaSemiOnlineTracker` 実装（クリップ周回＋再シード状態保持）
- [ ] 出力 masks が BEN2 union 契約に一致することを検証
- [ ] `detection_every` / `max_missed_detection_count` の挙動テスト（はがれ→復帰の最小ケース）

### Phase 3 — Pipeline / アプリ
- [ ] `pipelines/route_a_deva_video_pipeline.py`（build 関数＋配線）
- [ ] `gradio_app_sam2_ben2_route_a_deva_for_Movie.py`（新ファイル、ベース流用）

### Phase 4 — 検証
- [x] `.venv\Scripts\python.exe -m pytest -m "not integration" -q`（342 passed / 3 deselected）
- [x] `.venv\Scripts\python.exe gradio_app_sam2_ben2_route_a_deva_for_Movie.py --help`
- [~] **1分動画を50分以下**で処理できるか実測（短尺 samurai_demo.mp4 は数秒で完走＝予算大幅余裕。真の1分素材での実測はユーザー要確認）
- [x] Playwright 実行時検証（UI 配線、ERR035 準拠）＝`outputs/verify_routea_deva_sync_output.py` PASS（出力動画5本 src 充足・status「完了:／処理時間」）。検証中に ERR067（検出島の複数 box 4 次元マスク `ValueError`）を `SAM2Segmenter.run` の候補軸畳み込みで根治
- [x] サブエージェントレビュー（socket 具体型／consensus 正しさ／`weights_only=True`／フィードバック隔離）＝Phase 1-3 で実施済み

### 完了処理
- [x] `WHITEBOARD.md` / `ERROR_LOG.md` / `REFERENCE.md` 更新

---

## 7. 留意・リスク

- **核心を外さない**: 検出と伝播の分離・周期再検出・consensus マージ・memory 掃除の4点はマージン無しの必須要件。
- `SAM2VideoPropagator` の途中再シードが本リポ実装で難しい場合、**クリップ毎に propagation state を作り直す**フォールバック設計を用意（速度は予算1.67秒/frameに収まる見込み＝調査 §4）。
- 既存正常系（ベースアプリ）は変更しない。新規ファイルで作る。
- `segment-anything/` `samurai/` 直接変更禁止。`torch.load(..., weights_only=True)`。`try/except: pass` 禁止。
- 複数被写体は `demo.py` 経路でなく SAM2 per-object 経路で（調査 §8-1）。
- DEVA 由来の弱点「対象の出入りが激しいと false positive 増」→ `max_missed_detection_count` を下げて積極削除で緩和。
