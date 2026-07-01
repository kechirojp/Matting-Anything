# DEVA方式 BEN2/TB ハイブリッド動画アプリに手動 box/point プロンプトを追加する実装計画

- 作成日: 2026-07-01
- 対象依頼（逐語要旨）: 「`gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py` には SAM2.1 の BBOX プロンプト・ポイントプロンプト（ネガティブ／ポジティブ）を追加できる機能がない。`gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py` を参考に、`gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py` をベースに `gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py` を新しく作り、SAM2.1 の BBOX / point（pos/neg）プロンプト機能を追加してほしい。Haystack component pipeline の機能分割・疎結合・I/O 契約を守ること。専用クイックスタートも作成。」
- ワークフロー: `.github/copilot-instructions.md` / `.github/instructions/workflow.instructions.md`
- 参照スキル: `.github/skills/haystack-pipeline/SKILL.md`
- ハルシネーション防止: `.github/prompts/prevent-hallucination.prompt.md`

---

## 1. 前提調査（裏取り済み）

| 確認事項 | 結論 | 根拠 |
|---|---|---|
| ベースアプリ | `gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py`（port 7865）。DEVA 方式で人物を追跡し、人物 mask 内は transparent-background、外側は BEN2 で alpha 生成し合成。 | 当該ファイル本体 |
| 参考アプリ | `gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py`（port 7864）。手動 box / point（pos・neg）canvas を DEVA 方式へ統合済み。 | 当該ファイル本体 |
| 手動 seed の受け口 | `DevaSemiOnlineTracker.run()` は既に `initial_boxes` / `initial_points` / `initial_labels` を受け付ける（後方互換・既定 None）。モードA（text 空＋手動 seed のみ）／モードB（text 併用の周期再検出）を引数で自動判定。 | `pipelines/components/deva_semi_online_tracker.py` L126-199, 272-350 |
| ハイブリッドパイプライン | `build_sam2_ben2_tb_deva_hybrid_pipeline()` は同じ `deva_semi_online_tracker` component を使用。tracker の inputs に seed を渡すだけで手動プロンプトが効く（パイプライン改変不要）。 | `pipelines/route_a_deva_hybrid_video_pipeline.py` |
| UI ヘルパ | `pipelines/components/ui_helpers.py` に `empty_prompt_state` / `copy_prompt_state` / `draw_prompt_overlay` / `select_sam2_prompt` / `extend_box_to_edge` / `remove_selected_points` / `remove_selected_boxes` / `build_prompt_selection_choices` が揃っている。 | 当該ファイル |
| 検出起点フレーム | 共有トラッカーに `detection_start_frame` 実装済み。手動版は canvas 張り替え方式（`extract_prompt_frame`）で対応する。 | 参考アプリ・WHITEBOARD 2026-07-01 |
| clean-base-frame 契約 | overlay 焼き込み防止のため `prompt_base_image = gr.State` を真実の源にする（ERR063）。 | 参考アプリ・ERROR_LOG ERR063 |
| run 同期直結 | ローカル実行前提で `run_btn.click` はコア関数へ同期直結（ERR064）。 | 参考アプリ・ERROR_LOG ERR064 |

**結論**: パイプライン層（tracker / hybrid extractor / writer）は改変不要。新規アプリを 1 本追加し、ベースのハイブリッド UI に参考アプリの手動 prompt canvas を統合し、`run` で `prompt_state → initial_boxes/points/labels` を構築して `deva_semi_online_tracker` へ注入するだけで要件を満たす。

## 2. 設計方針（Haystack 疎結合・機能分割・I/O 契約）

- **パイプライン無改変**: `build_sam2_ben2_tb_deva_hybrid_pipeline` の DAG・socket 契約（`masks` / `matte` の dict socket）は一切変更しない。手動 seed は tracker の入力ソケットに渡すだけ。
- **UI ロジックはアプリ層のみ**: canvas・prompt 編集・text→box 補助は `ui_helpers`（純関数中心）を再利用し、アプリ側は配線と結果整形だけにする。
- **モデル初期化は遅延**: パイプライン singleton は import 時に重いモデルを読まない（`get_*_pipeline()` 遅延構築）。Canvas 補助の GroundingDINO も遅延構築し、重い DEVA ロード前に `release_text_detector()` で VRAM を空ける。
- **例外の握り潰し禁止**: すべて `raise` / `gr.Error` で通知。`torch.load` は使わない（本アプリは呼ばない）。
- **後方互換**: ベースアプリ・参考アプリ・`segment-anything/` / `samurai/` は無改変。

## 3. 2 つの動作モード（tracker が引数で自動判定）

| モード | Text Prompt | Canvas box/point | 挙動 |
|---|---|---|---|
| A. 手動 seed のみ | 空 | 必須 | 先頭（起点）フレームの box/point から SAM2 伝播（再検出なし） |
| B. ハイブリッド（推奨） | 入力 | 任意 | 先頭を box/point で精緻 seed → 以降 text で周期再検出し「はがれ復帰」 |

- point の positive/negative で前景・背景を補正（negative で誤検出除外）。
- 手動 point は先頭クリップの seed にのみ反映（対象が動くため後続へ再投影しない）＝tracker の既存契約に準拠。

## 4. 実装 ToDo

- [ ] (T1) 新規 `gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py`（port 7866）を作成。
  - [ ] ベース hybrid の import に `ui_helpers` 一式と `GroundingDINOMultiBoxDetector` を追加。
  - [ ] canvas 系ヘルパ（placeholder / `extract_first_frame*` / `extract_prompt_frame` / `clear_prompt`）を参考アプリから移植。
  - [ ] text→box 補助（`detect_text_boxes_for_video` / `populate_candidate_choices` / `apply_selected_boxes`）を移植。
  - [ ] prompt 編集（個別削除・mode 切替・選択肢再生成）を移植。
  - [ ] `run_deva_hybrid_manual_background_removal` を新設し、`prompt_state → initial_boxes(union 優先/single fallback)/initial_points/initial_labels` を構築、text/box いずれか必須・point は box 必須を `gr.Error` 検証、`deva_semi_online_tracker` へ seed 注入。ハイブリッド alpha 段の入力（composition/tb_*/person_region_*）はベースと同一に維持。
  - [ ] UI に prompt canvas / Input Mode / Point Label / Extend / 個別削除 / Text→Box Accordion / 検出起点フレーム（canvas 張り替え）を追加。
  - [ ] `run_btn.click` は同期直結（ERR064）。inputs 順＝ハンドラ signature 完全一致。
- [ ] (T2) 専用クイックスタート `QUICKSTART_DEVA_BEN2_TB_HYBRID_MANUAL.md` を作成。
- [ ] (T3) 検証: `--help` smoke（Windows `.venv`）、`get_errors` 0、既存の非 integration テストが回帰しないこと。
- [ ] (T4) サブエージェントレビュー（正確性/パフォーマンス/セキュリティ/可読性/規約準拠）。
- [ ] (T5) `WHITEBOARD.md` 更新。UI fixed は Playwright 実機検証後に別途記録（本計画時点では実装完了までを記録し、ERR035 に従い UI fixed は未確定とする）。

## 5. I/O 契約（run → UI）

- 入力（`run_btn`）: `input_video, prompt_state, text_prompt, detection_every, max_missed_detection_count, iou_threshold, box_threshold, text_threshold, top_k, detection_start_frame, max_frames, frame_step, output_mode, rgba_codec, composition_mode, tb_mode, tb_threshold, tb_crop_padding, tb_mask_guard_feather, tb_mask_guard_dilate, person_region_dilate_px, person_region_feather_px, refine_foreground, output_type, overlay_enabled`
- 出力（7-tuple）: `rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status`
- tracker への seed 注入: `initial_boxes` / `initial_points` / `initial_labels` / `detection_start_frame`
- hybrid alpha 段: ベースと同一キー（`composition_mode` / `tb_mode` / `tb_threshold` / `tb_crop_padding` / `tb_mask_guard_dilate` / `tb_mask_guard_feather` / `person_region_dilate_px` / `person_region_feather_px` / `refine_foreground` / `output_type` / `rgba_codec`）

## 6. テスト方針

- 挙動変更はアプリ層の配線とパラメータ整形のみ（パイプライン・tracker・純関数は既存＝既に単体テスト済み）。
- `--help` smoke で import/構築エラーを検出。
- `get_errors` で静的エラー 0 を確認。
- 非 integration テスト全体で回帰が無いことを確認（新規 seed 経路は tracker 既存テストがカバー）。
- UI 実機検証（Playwright）は GPU/モデル/動画を要するため、実施後に WHITEBOARD へ「UI fixed」を記録する（ERR035）。

## 7. リスクと回避

| リスク | 回避策 |
|---|---|
| inputs 順とハンドラ signature の不一致 | 参考アプリと同じ順序規約で 25 inputs / 7 outputs を厳密一致。 |
| overlay 焼き込みで clear/削除が UI に残る | `prompt_base_image` gr.State をクリーン基準にする（ERR063）。 |
| 重い DEVA ロードと Canvas GroundingDINO の VRAM 二重確保 | run 冒頭で `release_text_detector()`。 |
| ベース/参考アプリの破壊 | 新規ファイルのみ追加。既存は無改変。 |
