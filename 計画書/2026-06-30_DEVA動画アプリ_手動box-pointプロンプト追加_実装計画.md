# DEVA方式 動画アプリ 手動 box / point プロンプト追加 実装計画

- 作成日: 2026-06-30
- 区分: 実装計画（新規アプリ追加 + 既存 tracker の後方互換拡張）
- ベースアプリ（**新規ファイルの土台**）: `gradio_app_sam2_ben2_route_a_deva_for_Movie.py`（DEVA方式・text駆動・ポート7863）
- UI 移植元: `gradio_app_sam2_ben2_route_a_for_Movie.py`（手動 box / point prompt canvas を持つ・ポート7862）
- 関連skill: `.github/skills/haystack-pipeline/SKILL.md`（8ステップ・I/O契約）
- 関連: `計画書/2026-06-30_DEVA方式再構成_SAM2伝播_周期再検出_consensus_ルートA_実装計画.md`

---

## 0. 結論（実現可否）

**実現可能。** 既存資産でほぼ揃っており、追加は「新規アプリ1本」＋「`DevaSemiOnlineTracker` の後方互換な引数拡張」だけで足りる。

根拠（コード実機確認済み・ハルシネーション対策）:

1. **SAM2 伝播は既に box / point(pos/neg) / labels / prompt_frame_idx / bidirectional を受け付ける。**
   - `SAM2VideoPropagator.run(... points, labels, box, boxes, object_id, prompt_frame_idx, bidirectional ...)`
     （[pipelines/components/video_model_components.py](pipelines/components/video_model_components.py#L599-L700)）。
   - `boxes` 指定で各 box を obj_id 1..N に登録し、`assign_points_to_boxes` で各 point(pos/neg) を最近傍 box に同梱する経路が既にある。**ネガティブ point（label 0）も既存経路で機能する。**
2. **DEVA tracker はこの propagator に box を渡してクリップ伝播している。**
   - `DevaSemiOnlineTracker.run` は `seed_boxes` を作り `self._propagator.run(boxes=seed_boxes, prompt_frame_idx=0, ...)` で伝播する
     （[pipelines/components/deva_semi_online_tracker.py](pipelines/components/deva_semi_online_tracker.py#L120-L320)）。
   - つまり「手動 box / point を最初のクリップの seed に注入する」だけで手動プロンプトが乗る。
3. **手動 prompt canvas の UI 部品はベースアプリ＋`ui_helpers` に完成済みで再利用できる。**
   - `select_sam2_prompt` / `remove_selected_points` / `remove_selected_boxes` / `build_prompt_selection_choices` / `draw_prompt_overlay`
     （[pipelines/components/ui_helpers.py](pipelines/components/ui_helpers.py#L50-L260)）。
   - ベースアプリの `detect_text_boxes_for_video` / 第1フレーム抽出 / clean-base-frame state（ERR063）も流用可能。
4. **Pipeline 配線は変更不要。**
   - `build_sam2_ben2_route_a_deva_pipeline` の tracker は同一 Component。引数を後方互換に増やせば、アプリ側で `pipeline.run({"deva_semi_online_tracker": {... 手動seed ...}})` として渡せる
     （[pipelines/route_a_deva_video_pipeline.py](pipelines/route_a_deva_video_pipeline.py#L36-L80)）。

---

## 1. 設計方針（破壊防止ライン）

> **DEVA の価値（周期再検出による「はがれ→自動復帰」）を壊さず、手動 box / point で「初期 seed の精度」と「ネガティブ point による除外」を上乗せする。**

- **既存アプリ・既存挙動を一切変えない。**
  - `gradio_app_sam2_ben2_route_a_for_Movie.py`（7862）はそのまま。
  - `gradio_app_sam2_ben2_route_a_deva_for_Movie.py`（7863）はそのまま。
  - `SAM2VideoPropagator` / `segment-anything/` / `samurai/` は変更しない。
- **新規アプリは新ファイル・新ポート**（`gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py` / **7864**）。
- **`merge_consensus` は純関数のまま保つ**（box/mask/label/score のみ）。手動 point/label は consensus に混ぜず、tracker 内部に「**第1クリップ限定の point/label マップ**」として保持する。
- **設定値はハードコードしない。** 既定値は `config/route_a.toml` 経由（既存 `_ROUTE_A_DEFAULTS` / `[deva]` 系）を踏襲する。
- **`torch.load(weights_only=False)` / `try/except: pass` を使わない。** エラーは `raise` または `gr.Error`。

---

## 2. 動作モード（2モード）

| モード | text_prompt | 手動 box/point | 挙動 | 用途 |
|--------|-------------|----------------|------|------|
| **A. 手動seedのみ** | 空 | 必須 | 第1フレームの手動 seed から SAM2 伝播。周期再検出は走らせない（検出島スキップ）。実質ベースアプリ相当を DEVA tracker 経由で実行 | 検出語が決めにくい対象を正確に指定したい |
| **B. ハイブリッド（推奨）** | 入力 | 任意 | 第1フレームを手動 box/point で**正確に** seed → 以降は GroundingDINO(text) が `detection_every` 毎に再検出し consensus で復帰 | はがれ復帰を効かせつつ初期 seed を精緻化／ネガティブ point で誤検出領域を除外 |

- どちらのモードでも **point/label は第1クリップの seed にのみ適用**する（対象が動くため、後続クリップへ同じ座標 point を再投影しない＝既存設計思想に一致）。
- **両方空はエラー**（`gr.Error`）。「text_prompt または手動 box/point のいずれかは必須」。

---

## 3. I/O 契約の拡張（Haystack 8ステップ準拠）

### 3.1 `DevaSemiOnlineTracker.run` 引数追加（後方互換）

既存シグネチャ（[deva_semi_online_tracker.py](pipelines/components/deva_semi_online_tracker.py#L112-L127)）に **既定値付き** で追加:

| 新引数 | 型 | 既定 | 意味 |
|--------|----|------|------|
| `initial_boxes` | `list[list[int]] \| None` | `None` | 手動 box（xyxy）。obj_id 1..M を確定し第1クリップ seed にする |
| `initial_points` | `list[tuple[int,int]] \| None` | `None` | 手動 point 座標（pos/neg 混在） |
| `initial_labels` | `list[int] \| None` | `None` | 各 point の label（1=pos / 0=neg） |
| `prompt_frame_idx` | `int` | `0` | 手動 seed を置くフレーム（既定0） |

- 後方互換: 全て省略時は現挙動（text駆動 DEVA）と完全一致。
- `text_prompt` の必須判定を緩和: **`text_prompt` と `initial_boxes/initial_points` が両方空のときだけ** `ValueError`。
- 既存 `det_out = self._detection_island.run(...)` は **モードA（text空）ではスキップ**し、検出 dict を空にする（検出島を呼ばない）。

### 3.2 seed 注入ロジック（consensus を汚さない）

```
run():
  if initial_boxes/initial_points:
      tracks ← 手動 box から初期 track（object_id 1..M, box, missed=0）を pre-populate
      first_clip_points_by_obj ← assign_points_to_boxes(points, boxes) で obj_id→point/label を保持
  for clip_i, clip_start in enumerate(detection_frame_indices):
      detected ← (text空ならスキップで空) / それ以外は検出島
      consensus ← merge_consensus(tracks, propagated, detected, ...)   # 純関数のまま
      seed_tracks ← box を持つ track
      seed_points, seed_labels ← (clip_i == 0 のときだけ) first_clip_points_by_obj から構築、else None
      propagator.run(boxes=seed_boxes, points=seed_points, labels=seed_labels, prompt_frame_idx=...)
```

- `merge_consensus` は**無改修**。手動 track は通常 track と同じ dict 形（box/mask/label/score）で渡す。
- 第1クリップで手動 box と frame0 の text 検出が重なれば consensus が IoU で自然に統合（モードB）。
- **`per_object_logits` / `frame_masks` の source_index キー規約・空クリップ (0,H,W) 規約は不変**（OwnershipResolver/BEN2 ドロップイン契約を維持）。

### 3.3 Pipeline / builder

- **builder 無改修**（`build_sam2_ben2_route_a_deva_pipeline` をそのまま使う）。
- 新規アプリは `pipeline.run({"deva_semi_online_tracker": {"initial_boxes":..., "initial_points":..., "initial_labels":..., "prompt_frame_idx":..., "text_prompt":..., ...}})` で手動 seed を渡す。

---

## 4. 新規アプリ UI（移植 + DEVA制御）

新ファイル `gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py`（ポート **7864**）:

- **移植（ベース7862から）**: prompt canvas（box モード / point モード pos・neg）、`prompt_canvas` + clean-base-frame `gr.State`（ERR063）、`select_sam2_prompt` 配線、prompt 選択削除（points/boxes）、`detect_text_boxes_for_video`、第1フレーム抽出、`draw_prompt_overlay`。
- **移植（DEVA7863から）**: `detection_every` / `max_missed_detection_count` / `iou_threshold` / `matte_mode` / RouteA Advanced 群 / `STAGE_PROGRESS_RANGES` / 出力7タプル / `run_btn.click` 同期（ERR064）。
- **text_prompt を「任意」化**（モードA許容）。空+手動なし送信は `gr.Error`。
- gradio bool-schema patch / Proactor 抑制は DEVA アプリのものをそのまま踏襲。

---

## 5. TODO（実装チェックリスト）

> 進め方: workflow Section 0（ERROR_LOG→REFERENCE→WHITEBOARD→skill）→ RED→GREEN TDD → Section 5 サブエージェントレビュー → Section 6 記録更新。

### Phase 0: 準備・確認
- [x] `ERROR_LOG.md` / `REFERENCE.md` / `WHITEBOARD.md` を読む
- [x] `haystack-pipeline` skill の8ステップ＋I/O契約を再確認
- [x] `assign_points_to_boxes` の戻り値仕様（[common.py](pipelines/components/common.py)）を確認

### Phase 1: tracker 拡張（RED→GREEN・純関数優先）
- [x] **RED**: `initial_boxes` のみで seed → 第1クリップが手動 box を伝播することを検証する純関数/モックテストを書く
- [x] **RED**: `initial_points`(neg含む)+`initial_labels` が第1クリップ propagator 呼び出しにだけ渡る（clip_i>0 では None）テスト
- [x] **RED**: text空+手動なし → `ValueError`、text空+手動あり → 検出島を呼ばない（モックで assert 呼び出し0）テスト
- [x] **RED**: 全省略時に既存出力（source_index キー・空クリップ規約・per_object_logits）が不変な回帰テスト
- [x] **GREEN**: `DevaSemiOnlineTracker.run` に4引数追加・seed注入・text必須緩和・検出島スキップを実装
- [x] `merge_consensus` が無改修であることを差分で確認

### Phase 2: 新規アプリ作成
- [x] `gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py` を DEVA アプリ複製で作成（ポート7864）
- [x] ベース7862から prompt canvas + ui_helpers 配線 + clean-base-frame state（ERR063）を移植
- [x] text_prompt 任意化 + 「両方空は `gr.Error`」を実装
- [x] `pipeline.run` に手動 seed（initial_boxes/points/labels/prompt_frame_idx）を渡す
- [x] 出力7タプル・`run_btn.click` 同期（ERR064）を確認

### Phase 3: 検証
- [x] `.venv\Scripts\python.exe -m pytest -m "not integration" -q`（全GREEN・353 passed / 3 deselected）
- [x] `.venv\Scripts\python.exe gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py --help`（smoke）
- [x] **ERR035**: UI（canvas・配線）変更のため Playwright 実行時検証（ページ表示→point/box モード切替で Point Label 表示制御→入力なし送信で `gr.Error`）を実施し記録
- [ ] 短尺動画でモードA / モードB の end-to-end 動作を実機確認（host-RAM は ERR068 の `per_object_logits_max_side` で抑制）※ユーザーの実素材で要確認

### Phase 4: レビュー・記録
- [x] サブエージェントによる差分レビュー（正確性/パフォーマンス/セキュリティ/可読性/規約）＝承認適性あり（Critical/Major なし・Minor 3件は許容）
- [x] `WHITEBOARD.md`（完了内容・次アクション）更新
- [x] 必要なら `ERROR_LOG.md` / `REFERENCE.md`（新アプリ・新ポート7864・tracker契約拡張）更新

---

## 6. リスクと対策

| リスク | 対策 |
|--------|------|
| 既存DEVA挙動の回帰 | 新引数は全て既定値付き。全省略時の不変を回帰テストで固定（Phase1 RED） |
| consensus 純関数の汚染 | point/label は tracker 内 `first_clip_points_by_obj` に隔離。consensus へ渡さない |
| 対象移動後の point 誤適用 | point/label は **clip_i==0 のみ**適用（既存設計思想と一致） |
| host-RAM OOM（ERR068再発） | 既存 `per_object_logits_max_side`（config駆動）をそのまま適用 |
| SAMURAI（single_object_only）で複数box | 既存の fail-fast（ERR051）がそのまま機能 |
| ネガティブ point が効かない | 既存 `assign_points_to_boxes`＋`add_new_points_or_box` 経路（実機確認済み）を流用 |

---

## 7. 影響ファイル一覧

| ファイル | 区分 |
|----------|------|
| `gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py` | **新規**（ポート7864） |
| `pipelines/components/deva_semi_online_tracker.py` | **後方互換拡張**（run に4引数・seed注入） |
| `tests/unit/test_deva_semi_online_tracker*.py` | **追記**（RED→GREEN） |
| `pipelines/route_a_deva_video_pipeline.py` | 無改修（builder流用） |
| `pipelines/components/consensus.py` | **無改修** |
| `pipelines/components/video_model_components.py`（SAM2VideoPropagator） | **無改修** |
| `gradio_app_sam2_ben2_route_a_for_Movie.py` / `..._deva_for_Movie.py` | **無改修** |
| `WHITEBOARD.md` / `ERROR_LOG.md` / `REFERENCE.md` | 記録更新 |
