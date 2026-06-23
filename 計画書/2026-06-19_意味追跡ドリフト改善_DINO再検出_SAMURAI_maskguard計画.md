# 意味追跡ドリフト改善 実装計画（DINO 再検出 / SAMURAI 切替 / mask_guard 手動調整）

作成日: 2026-06-19
関連調査: `調査/2026-06-19_DINO_sam2のトラッキング背景除去がtransparetBGに劣る理由.md`
対象: `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` 系の動画意味追跡パイプライン

---

## 0. 背景（なぜやるか）

- transparent-background 単体は高品質だが、GroundingDINO + SAM2 の意味追跡を挟むと劣化する。
- 根本原因（調査確定）:
  1. **意味アンカーが frame0 限定**: GroundingDINO は第1フレームのみ。SAM2 の点/box も frame0 のみ条件付け。
  2. **デフォルトは素の SAM2**: SAMURAI ではない（motion-aware 補正が効いていない）。
  3. **tb が SAM2 マスクに完全従属**: マスクの crop + mask_guard で、追跡がズレると正しい被写体まで 0 に潰す。
- 本計画は劣化の主因に対する 3 つの改善（優先順）を、独立に着手・検証できる単位で定義する。

---

## 改善1: GroundingDINO の周期的再検出 + SAM2 re-prompt

### 目的
frame0 限定の意味アンカーを N フレームごとに更新し、SAM2 のドリフトを意味情報で補正する。

### 現状（確認済み）
- `detect_text_boxes_for_video()` が `get_text_detector().run(image=first_frame, ...)` を **1回だけ**呼ぶ。
- `SAM2VideoPropagator.run()` は `add_new_points_or_box(frame_idx=prompt_frame_idx, ...)` を **1回**呼んだ後、`propagate_in_video()` で全フレーム伝搬。途中の再検出・再 prompt は無い。

### 方針（段階）
- **Phase 1（調査・PoC）**: 既存伝搬に手を入れず、再検出の効果を計測する小規模実験を先に行う。
  - 一定間隔フレームで GroundingDINO を再実行し、bbox の IoU 推移（frame0 prompt の伝搬 mask vs 再検出 bbox）を計測。
  - ドリフトが起きるフレーム帯と再検出で回復するかを可視化（既存 Tracking Overlay を流用）。
- **Phase 2（実装）**: `SAM2VideoPropagator.run()` に「再 prompt 間隔」を追加。
  - 新パラメータ案: `redetect_interval: int = 0`（0 = 無効 / 既存挙動を完全維持）, `redetect_text_prompt: str | None`, `redetect_box_threshold`, `redetect_text_threshold`。
  - 伝搬ループ中、`frame_idx % redetect_interval == 0`（>0 のとき）で GroundingDINO を再実行し、`add_new_points_or_box(frame_idx=該当フレーム, obj_id=既存, box=再検出bbox, clear_old_points=True)` で **同一 obj_id を refine**。
  - SAM2 制約: tracking 開始後の box 追加は warning が出る（`sam2_video_predictor.py`）。refinement 目的のため許容するが、必要なら該当フレームから `reset_state` ではなく point ベース refine も検討。
  - **疎結合維持**: GroundingDINO 呼び出しは propagator に直結させず、再検出を担う関数/Component を注入する形にして Haystack 境界を崩さない（copilot-instructions 10〜13）。

### 影響範囲
- `pipelines/components/video_model_components.py`（`SAM2VideoPropagator.run`）
- `pipelines/sam2_tb_video_pipeline.py`（配線・I/O 契約）
- `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`（再検出間隔 UI、`run_video_background_removal` の引数追加）

### リスク / 留意
- 再検出が誤検出すると逆にドリフトを誘発。`redetect_interval=0` を既定にし、明示オプトインにする。
- 計算コスト増（N フレームごとに DINO 推論）。Colab RAM ピークに注意（`release_text_detector` との両立）。
- I/O 契約変更は最小限。既存テストの mask 契約を壊さない。

### 受け入れ条件
- `redetect_interval=0` で既存出力とバイト一致（デグレ無し）。
- `redetect_interval>0` でドリフト帯の mask IoU が改善することを 1 本以上の動画で確認。
- 非 integration テスト緑、Gradio smoke（`--help`）緑。

---

## 改善2: SAMURAI バリアントへ切替

### 目的
motion-aware な SAMURAI で素の SAM2 よりドリフトを抑える。

### 現状（確認済み）
- tracker レジストリに `samurai_hiera_l` / `samurai_hiera_b_plus` が登録済み（`config/inference_models.toml`、`requires = "sam2_samurai"`）。
- UI ドロップダウン初期値は `sam2_hiera_l`（標準）。`INFERENCE_TRACKER_VARIANT` 未設定のため SAMURAI は「選べるが既定では選ばれない」。
- `samurai/sam2/sam2/configs/samurai/*.yaml` と Hydra 検索パス追加処理（`_ensure_samurai_config_searchpath`）は実装済み。

### 方針（段階）
- **Phase 1（運用確認のみ・コード変更なし）**:
  - `$env:INFERENCE_TRACKER_VARIANT = "sam2_samurai"` を設定して起動。
  - UI で `samurai_hiera_l` を選択し、同一動画・同一 prompt で標準 SAM2 と出力比較。
  - SAMURAI config が解決でき、`samurai_mode=True` が overlay metadata に出るか確認。
- **Phase 2（任意・既定切替を検討）**:
  - 比較で SAMURAI が明確に良ければ、UI 初期値や `INFERENCE_TRACKER_VARIANT` の運用既定を見直す。
  - ※ `segment-anything/` と `samurai/` は直接変更しない（Hard Rule）。切替は env + config + registry のみで行う。

### 影響範囲
- Phase 1 はコード変更なし（環境変数 + UI 選択のみ）。
- Phase 2 で UI 既定値変更時のみ `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` を1行修正。

### リスク / 留意
- Colab で facebook 版 sam2 が入っていると `configs/samurai/...` 解決が衝突しうる（既存 `_samurai_config_root` で吸収）。
- checkpoint は SAM2 と共用（`sam2.1_hiera_large.pt`）。追加 DL 不要。

### 受け入れ条件
- SAMURAI 選択時に例外なく完走し、追跡 overlay で追従改善を目視確認。
- 標準 SAM2 経路は無変更で従来どおり動作。

---

## 改善3: mask_guard の手動調整 UI（feather / dilate）★今回の主実装

### 目的
追跡が少しズレた時に mask_guard が正しい被写体を 0 に潰すのを、ユーザーが手動で緩められるようにする。

### 要件（ユーザー指定）
- **feather と dilate をチェックボックス + 範囲設定（スライダー）で手動入力できるようにする。**
- **デフォルトはオフ**（チェックを入れない限り現行挙動 = feather 0 / dilate 21 固定を維持）。

### 現状（確認済み）
- `TransparentBGExtractor.run()` は `mask_guard_dilate=21`（固定）、`mask_guard_feather=0`（既定）。
  - feather>0 のとき `soft_probability_guard` / `feather_binary_mask` を使う soft guard、=0 のとき dilate 二値ゲート。
- 動画経路 `TransparentBGVideoExtractor.run()` は `mask_guard_feather` を受けるが **`mask_guard_dilate` は受けていない**（extractor 既定 21 のまま）。
- Gradio 動画 UI に feather/dilate の露出は**無い**。`mask_guard_feather` は `bg_entry.get("mask_feather", 0)`（TOML 由来）で渡されるのみ。

### 方針（段階）
- **Phase 1（配線追加）**: `mask_guard_dilate` を動画経路へ通す。
  - `TransparentBGVideoExtractor.run()` と `_run_per_object_frame()` に `mask_guard_dilate: int = 21` を追加し、`self.extractor.run(..., mask_guard_dilate=...)` へ伝搬。
  - `pipelines/sam2_tb_video_pipeline.py` の `transparent_bg_video` 入力契約に `mask_guard_dilate` を追加（既定で現行値を維持）。
- **Phase 2（UI 追加）**: 「SAM2 追跡 + 背景除去」タブの Advanced 内に以下を追加。
  - `mask_guard_enabled = gr.Checkbox(value=False, label="mask guard を手動調整", info=...)`（既定 OFF）。
  - `mask_guard_feather = gr.Slider(0, 64, value=0, step=1, label="Mask guard feather", info=...)`。
  - `mask_guard_dilate = gr.Slider(1, 81, value=21, step=2, label="Mask guard dilate", info=...)`（奇数推奨）。
  - チェック OFF 時はスライダー値を無視し、現行既定（feather=TOML値, dilate=21）を使う。
    - 実装は `run_video_background_removal` 内で `if not mask_guard_enabled: feather, dilate = 既定` と分岐。
    - もしくは `gr.update(interactive=...)` でスライダーを有効/無効化し、無効時は既定を送る。
- **Phase 3（配線 UI → run）**: `run_video_background_removal` の引数に 3 つを追加し、`transparent_bg_video` の `mask_guard_feather` / `mask_guard_dilate` へ反映。`run_btn.click` の inputs にも追加。

### 影響範囲
- `pipelines/components/video_model_components.py`（`TransparentBGVideoExtractor.run`, `_run_per_object_frame`）
- `pipelines/sam2_tb_video_pipeline.py`（入力契約）
- `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`（UI 3 部品 + `run_video_background_removal` 引数 + `run_btn.click` inputs）
- `config/inference_models.toml`（必要なら `mask_feather` の意味と既定を注記）

### デフォルトオフの担保
- UI 既定: チェック OFF、feather=0、dilate=21。
- OFF 時は現行と同一パラメータを送るため、**既存出力とバイト一致（デグレ無し）**を受け入れ条件にする。

### リスク / 留意
- dilate は奇数 kernel が安全（`step=2` で奇数を維持）。偶数を渡した場合の正規化を extractor 側で保証するか確認。
- feather を上げすぎると輪郭が滲む。info に目安（feather 8〜16、dilate 21〜41）を明記。
- per_object 経路にも同じパラメータを通す（両経路で挙動を揃える）。

### 受け入れ条件
- チェック OFF（既定）で既存出力とバイト一致。
- チェック ON + feather/dilate 変更で guard 範囲が変わることを 1 本の動画で確認（追跡が少しズレても被写体が消えない）。
- 非 integration テスト緑、Gradio smoke（`--help`）緑。
- UI 実行時検証（Playwright）でチェック ON/OFF とスライダー連動を確認後に「fixed」記録（ERR035）。

---

## 共通: テストと検証（Hard Rule 準拠）

- 挙動変更は可能な限り RED → GREEN。
- コマンド（Windows）:
  - 非 integration 全体: `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
  - Gradio smoke（動画）: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help`
- `torch.load(..., weights_only=False)` 禁止 / `try/except: pass` 禁止 / 設定はハードコードせず config 経由。
- `segment-anything/` と `samurai/` は直接変更しない。
- UI / 配線の「fixed」は Playwright 実行時検証を通してから記録（ERR035）。
- 作業前後に `WHITEBOARD.md` / `ERROR_LOG.md` / `REFERENCE.md` を更新。

## 着手順序（推奨）
1. **改善3 Phase 1〜3**（影響局所・即効・既定オフでデグレ無し）。
2. **改善2 Phase 1**（コード変更なしで SAMURAI 効果を計測）。
3. **改善1**（最も効果が大きいが実装コスト高。PoC 計測の後に本実装）。
