# ホワイトボード — 作業引継ぎメモリ

> **ルール**: 作業開始前に必ずこのファイルを読む。作業完了後に必ず更新する。

---

## 現在の作業状況

| 項目 | 内容 |
|------|------|
| **作業中のタスク** | SAM2 / GroundingDINO 遅延 telemetry・GPU first 方針反映（完了・Colab実測待ち） |
| **最終更新日** | 2026-05-28（GPU first policy） |
| **担当者/セッション** | GitHub Copilot |

---

## 進行中タスクの詳細

<!-- 現在取り組んでいるタスクの目的・進捗・残作業を記述 -->

### タスク名: SAM2 positive point `Connection errored out` 対応（2026-05-28）
- **目的**: 静止画 Haystack 版で SAM2 mask の positive point を選択した時に Gradio 側で `Connection errored out` が出る問題を修正する
- **進捗**: 完了
- **変更ファイル**: `gradio_app_sam2_transparent_BG_haystack.py`, `tests/unit/test_jupytext_notebooks.py`, `REFERENCE.md`, `ERROR_LOG.md`, `WHITEBOARD.md`
- **原因**: `SAM2 Prompt Canvas` は `prompt_canvas.select(...)` でクリックイベントを受ける設計だが、静止画版だけ `interactive=False` になっており、プロジェクト仕様の `gr.Image(type="numpy", interactive=True)` と動画版実装から外れていた
- **対応**:
  1. 静止画版 `SAM2 Prompt Canvas` を `interactive=True` に変更
  2. アップロード先にしないため `sources=[]` は維持
  3. 回帰テストを prompt canvas block 単位で `interactive=True` / `sources=[]` / `interactive=False` 不在を確認する形に強化
  4. `ERROR_LOG.md` に ERR026 として原因・対処・再発防止を追加
- **関連 ERR 横展開**: ERR019（positive/negative Radio）、ERR021（Input Image と Prompt Canvas 分離）、ERR017（座標手入力禁止）、ERR011/ERR016（Gradio 接続系汎用表示）、ERR010/ERR025（SAM2 import / GPU preflight）
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 78 passed / 3 deselected、Gradio `--help` smoke 2本成功、`git diff --check` 成功、サブエージェントレビューで重要指摘なし

### タスク名: エラーログ_07 SAM2 package import preflight 対応（2026-05-28）
- **目的**: `エラーログ\エラーログ_07.md` の `ModuleNotFoundError: No module named 'sam2'` を起点に、SAM2 install 失敗または未実行のまま Gradio が公開される導線を起動前に止める
- **進捗**: 完了
- **変更ファイル**: `Sam2_Transparent_Background_Haystack.py`, `Sam2_Transparent_Background_Haystack.ipynb`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `tests/unit/test_jupytext_notebooks.py`, `REFERENCE.md`, `ERROR_LOG.md`, `WHITEBOARD.md`
- **原因**: Colab runtime に `sam2` package が import 可能な状態で入っていない、または `git+https://github.com/facebookresearch/sam2.git` の install 失敗後に Gradio 起動セルまで進んだ
- **対応**:
  1. 静止画 Notebook の Gradio 起動前診断セルで `import sam2`, `build_sam2`, `SAM2ImagePredictor` を確認
  2. 動画 Notebook の Gradio 起動前診断セルで `import sam2`, `build_sam2_video_predictor` を確認
  3. 未導入なら `RuntimeError` で install cell 再実行を案内し、Gradio 公開前に停止
  4. `ERROR_LOG.md` の ERR010 に Haystack 版 / エラーログ_07 の追記を追加
  5. `REFERENCE.md` に SAM2 Colab import preflight を追記
- **関連 ERR 横展開**: ERR010（SAM2 install / import）、ERR025（GPU runtime preflight）、ERR004 / ERR006（GroundingDINO CUDA ops）、ERR023 / ERR024 / ERR005（GroundingDINO 依存・transformers 互換）
- **検証**: `.venv\Scripts\python.exe -m jupytext --to ipynb` で静止画・動画 Notebook を再生成、`.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 78 passed / 3 deselected、Gradio `--help` smoke 2本成功、`git diff --check` 成功、サブエージェントレビューで重要指摘なし

### タスク名: エラーログ_06 GPU 必須 fail fast 対応（2026-05-28）
- **目的**: `エラーログ\エラーログ_06.md` の `GroundingDINOMultiBoxDetector requires a CUDA GPU... selected_device=cpu cuda_available=False torch_cuda_version=None` を起点に、Colab CPU / CPU-only torch で Gradio を公開後に失敗する導線を起動前に止め、原因・対処・再発防止を ERR として記録する
- **進捗**: 完了（原因確定: Colab ランタイムが CPU）
- **変更ファイル**: `Sam2_Transparent_Background_Haystack.py`, `Sam2_Transparent_Background_Haystack.ipynb`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `tests/unit/test_jupytext_notebooks.py`, `REFERENCE.md`, `ERROR_LOG.md`, `WHITEBOARD.md`
- **原因**: ユーザー確認により Colab ランタイムが CPU だったと確定。Gradio 実行プロセスで `torch.cuda.is_available() == False` かつ `torch.version.cuda is None` のため、GPU first 方針の `require_gpu_for_heavy_inference()` が正しく fail fast した
- **対応**:
  1. 静止画・動画 Notebook の Gradio 起動前診断セルで CUDA 不可かつ `MATTING_ANYTHING_ALLOW_CPU=1` 未設定なら `RuntimeError` を出して起動前に停止
  2. `ERROR_LOG.md` に ERR025 として原因・対処・再発防止を追加
  3. `REFERENCE.md` に Colab 起動前 GPU preflight を追記
- **関連 ERR 横展開**: ERR004（GroundingDINO CPU fallback）、ERR006（CUDA ops `_C`）、ERR010（SAM2 install log）、ERR023（GroundingDINO 依存）、ERR024（BERT 互換）、ERR005（transformers signature）
- **検証**: `.venv\Scripts\python.exe -m jupytext --to ipynb` で静止画・動画 Notebook を再生成、`.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 78 passed / 3 deselected、Gradio `--help` smoke 2本成功、`git diff --check` 成功

### タスク名: SAM2 / GroundingDINO 遅延 telemetry・安全修正（2026-05-28）
- **目的**: `Detect Text Boxes` と `Predict SAM2 Candidate Masks` が Colab で 2 回目も 5 分以上かかる原因を切り分けるため、Gradio 実行プロセス内の device / checkpoint / cache / stage timing / CUDA custom-op 状態を表示し、明らかな env 契約不一致と CPU autocast リスクを修正する
- **進捗**: 完了（Colab GPU 実測待ち）
- **変更ファイル**: `pipelines/components/model_components.py`, `gradio_app_sam2_transparent_BG_haystack.py`, `pipelines/components/video_model_components.py`, `gradio_app.py`, `Sam2_Transparent_Background_Haystack.py`, `Sam2_Transparent_Background_Haystack.ipynb`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `tests/unit/test_pipeline_wiring.py`, `tests/unit/test_video_pipeline_wiring.py`, `tests/unit/test_jupytext_notebooks.py`, `REFERENCE.md`, `ERROR_LOG.md`, `WHITEBOARD.md`
- **変更内容**:
  1. `SAM2Segmenter` が `SAM2_CKPT_PATH` / `SAM2_CONFIG_NAME` を読むようにし、Notebook launch env と Gradio process の契約を一致
  2. `SAM2Segmenter.run()` に `warm_up`, `prepare_image`, `set_image`, `predict`, `build_mask_set`, `total` の timing と runtime/checkpoint/cache 診断を追加
  3. SAM2 の autocast を CUDA の場合だけ `bfloat16` にし、CPU では autocast を無効化
  4. `GroundingDINOMultiBoxDetector` / `GroundingDINODetector` に `warm_up`, `prepare_image`, `predict_with_caption`, `nms/topk`, `total` の timing と runtime/checkpoint/cache/CUDA custom-op 診断を追加
  5. `Detect Text Boxes` / `Predict SAM2 Candidate Masks` の status に Gradio callback total と component diagnostics を表示
  6. Movie 版 `SAM2VideoPropagator` も `SAM2_CKPT_PATH` / `SAM2_CONFIG_NAME` を読むようにして、静止画版と同じ checkpoint 契約へ統一
  7. 静止画・動画 Notebook に `nvidia-smi`, torch CUDA, checkpoint path/size/Drive 判定の診断セルを追加し、Jupytext で ipynb を再生成
  8. env 読み込み、CPU-safe diagnostics、GroundingDINO custom-op diagnostics、Notebook 診断セルを unit test で固定
  9. レビュー指摘を反映し、CUDA autocast は `device` 文字列だけでなく `torch.cuda.is_available()` も満たす場合のみ有効化
  10. GroundingDINO 互換パッチの前提に合わせ、`transformers>=4.26.0` と `torch>=2.0.0` を requirements / Notebook / 参照ログへ明示
  11. 映像制作運用では CPU fallback は緊急回避専用という方針に合わせ、SAM2 / GroundingDINO / SAM2 video / legacy MAM Gradio の重い推論は CUDA 不可時に既定で fail fast。CPU を意図的に許可する場合のみ `MATTING_ANYTHING_ALLOW_CPU=1` を使う
  12. レビュー指摘を反映し、GroundingDINO の `torch.amp.autocast` は hardcoded CUDA ではなく入力 tensor の device type を使うよう修正
- **調査結果**:
  1. ローカル `.venv` では `torch` 自体が未導入で、Colab の遅延症状は実測不可
  2. ローカル checkpoint は `checkpoints/SAM2/sam2.1_hiera_large.pt`（約 856MB）と `checkpoints/groundingdino_swint_ogc.pth`（約 662MB）が存在
  3. Colab での根本原因分類は、追加した UI status / Notebook 診断の `cuda_available`, `cached_before`, `warm_up`, `set_image`, `predict_with_caption`, `groundingdino_cuda_ops` を見て判断する
- **関連 ERR 横展開**: ERR004 / ERR006（GroundingDINO CPU fallback / CUDA ops）、ERR010（SAM2 install build log）、ERR014（lazy import / warm_up）、ERR018（include_outputs_from）、ERR022（Haystack socket 型）、ERR023 / ERR024 / ERR005（GroundingDINO deps / transformers 互換）を確認対象にした。今回ローカルでは新規 ERR として確定できる再現エラーは未検出
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 78 passed / 3 deselected、Gradio `--help` smoke 2本成功、`git diff --check` 成功、Jupytext で静止画・動画 Notebook 再生成済み、サブエージェントレビュー指摘を反映済み
- **残作業**: Colab GPU runtime 上で `Detect Text Boxes` と `Predict SAM2 Candidate Masks` を 2 回ずつ実行し、表示される diagnostics を保存して原因を確定する

### タスク名: SAM2 + transparent-background Haystack UI Playwright 調査・導線改善（2026-05-28）
- **目的**: `Sam2_Transparent_Background_Haystack.ipynb` / `_for_Movie.ipynb` が起動する Gradio UI を実起動し、スクリーンショット・操作要素・想定フローを確認した上で、必須入力削減と導線改善を実装する
- **進捗**: 完了
- **変更ファイル**: `gradio_app_sam2_transparent_BG_haystack.py`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`, `tests/unit/test_jupytext_notebooks.py`, `Sam2_Transparent_Background_Haystack.ipynb`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `WHITEBOARD.md`
- **確認内容**:
  1. webapp-testing skill の `with_server.py` で静止画版 `7862`、動画版 `7861` を実起動
  2. Playwright Chromium で初期表示・入力後スクリーンショット、DOM 操作要素一覧、アップロードと prompt canvas クリックを確認
  3. 静止画版は要素数 137、動画版は要素数 79。静止画版は Text Prompt / SAM2 / Union / MatteExtractor が初期展開で、初心者には必須入力が多く見える
  4. 動画版は「第1フレームを取得」後に `Prompt Status` が更新されることを確認。出力形式と codec / sequence の関係は UI 上でやや分かりにくい
- **成果物**: セッション成果物 `files/ui_static/*.png|json`, `files/ui_movie/*.png|json`
- **変更内容**:
  1. 静止画版は Text Prompt / Mask Union / transparent-background 詳細設定を任意・Advanced 導線へ移し、最短フローを `Input Image` → bbox 2クリック → `Predict SAM2 Candidate Masks` → `Run transparent-background` に短縮
  2. `Mask Source for MatteExtractor` の既定を `Best Candidate Mask` にし、Union を作らなくても実行できる導線に変更
  3. `Prompt Mode=box` では `Point Label` を非表示にし、point prompt 時だけ positive / negative を表示
  4. 動画版はアップロード時に第1フレームを自動取得し、手動ボタンを「第1フレームを再取得」へ変更
  5. 動画版の最短フローを `Input Video` → bbox 2クリック → `動画背景除去を実行` にし、動画処理設定は Advanced に折りたたみ
  6. `output_mode=sequence` 時は動画 codec UI を非表示・無効化
  7. 上記導線を `tests/unit/test_jupytext_notebooks.py` で回帰テスト化
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 66 passed / 3 deselected、Gradio `--help` smoke 2本成功、Jupytext で両 ipynb 再生成、webapp-testing + Playwright で変更後 UI を再確認（静止画 83 要素、動画 56 要素、操作エラーなし）。モデル推論ボタンは checkpoint / GPU に依存するため実行対象外

### タスク名: copilot-instructions 診断修正（2026-05-28）
- **目的**: Chat Customizations Evaluations の診断に従い、`.github/copilot-instructions.md` の高認知負荷・曖昧さ・未定義ケースを解消する
- **進捗**: 完了
- **変更ファイル**: `.github/copilot-instructions.md`, `WHITEBOARD.md`
- **変更内容**:
  1. 詳細ルールを `REFERENCE.md` / `ERROR_LOG.md` / `.github/instructions/workflow.instructions.md` 参照へ寄せ、常時適用ルールカードとして短縮
  2. 関連ファイル確認の上限、MD ファイル優先、テスト省略条件、small / medium 判定、必要な整理の境界を明確化
  3. Windows / macOS / Linux の検証コマンドを併記
  4. Haystack callback、SAM2 UI ブラウザ確認、`segment-anything/` 変更禁止時の対応、禁止事項 override 記録を明文化
  5. ERR 参照へ一行要約を付与
- **検証**: プロンプト文書修正のみのため pytest はスキップ。`get_errors` は旧 line 208 など存在しない行を含む診断を返しており、拡張機能側の診断キャッシュと判断

### タスク名: Copilot instructions 改善（2026-05-28）
- **目的**: `/chronicle improve` により、過去セッションで繰り返し発生した Colab / GroundingDINO / Haystack UI・契約まわりの friction を `.github/copilot-instructions.md` に反映する
- **進捗**: 完了
- **変更ファイル**: `.github/copilot-instructions.md`, `WHITEBOARD.md`
- **変更内容**:
  1. 非 integration pytest、単一 pytest、Jupytext 生成、Gradio `--help` smoke の検証コマンドを追記
  2. Text Prompt / GroundingDINO 依存と Colab install cell / Jupytext 再生成ルールを追記
  3. `patch_transformers_bert_for_groundingdino()` 再利用ルールを追記
  4. Haystack Component 境界で `MaskSet` / `SelectedMask` / `MatteResult` の安定 I/O 契約を優先するルールを追記
  5. SAM2 + transparent-background UI の必須 / 任意 / 推奨フロー説明ルールを追記
  6. `/chronicle tips` の提案を反映し、/plan 判断基準、haystack-pipeline skill、webapp-testing skill、ERROR_LOG 横展開、観点つきレビューの運用ルールを追記
- **検証**: ドキュメント変更のみ

### タスク名: SAM2 + transparent-background Haystack 動画対応 `_for_Movie` 版（2026-05-27）
- **目的**: 静止画版 Haystack 構造を保ちつつ、SAM2 video predictor + transparent-background による動画背景除去 UI / Pipeline / Notebook を `_for_Movie` ファイルとして追加する。出力は動画・連番静止画・両方を選択可能にする。
- **進捗**: 実装完了（GPU / checkpoint を使う integration 実機確認は未実施）。計画書 `2026-05-27_SAM2_tb_Haystack_動画対応計画.md` は 2 回レビュー反映済み。
- **変更ファイル**: `pipelines/components/ui_helpers.py`, `pipelines/components/video_common.py`, `pipelines/components/video_model_components.py`, `pipelines/sam2_tb_video_pipeline.py`, `gradio_app_sam2_transparent_BG_haystack.py`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `tests/unit/test_ui_helpers.py`, `tests/unit/test_video_common_components.py`, `tests/unit/test_video_pipeline_wiring.py`, `tests/integration/test_sam2_tb_video_pipeline.py`, `REFERENCE.md`, `ERROR_LOG.md`, `requirements.txt`
- **決定事項**:
  1. prompt UI 純粋関数は案Bとして `pipelines/components/ui_helpers.py` に抽出し、静止画版と動画版で import 共有する
  2. Component 層は `ValueError` / `RuntimeError`、Gradio callback 層で `gr.Error` に変換する
  3. 出力形式は `video` / `sequence` / `both` を UI で選択し、連番は `outputs/<timestamp>/sequence/{rgba,alpha,preview}/` に保存する
  4. Pipeline builder は `build_video_reader_pipeline` / `build_sam2_video_propagation_pipeline` / `build_sam2_tb_video_pipeline` の 3 分割
- **検証**: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help` 成功、`.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help` 成功、`.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 62 passed / 3 deselected
- **残作業**: GPU / checkpoint ありで短尺動画を使い、`output_mode=video` / `sequence` / `both` の実機確認。SAM2 video predictor の `init_state(video_path=<frame_dir>)` API 実機確認。

### タスク名: SAM2 Haystack GroundingDINO / transformers 互換性修正（2026-05-26）
- **目的**: `Sam2_Transparent_Background_Haystack.ipynb` の Text Prompt 実行時に `'BertModel' object has no attribute 'get_head_mask'` が出る問題を解消し、ERR023 と合わせて GroundingDINO 依存周辺の潜在エラーを減らす
- **進捗**: 完了（Colab 実機での再実行確認待ち）
- **変更ファイル**: `pipelines/components/model_components.py`, `gradio_app.py`, `tests/unit/test_pipeline_wiring.py`, `REFERENCE.md`, `ERROR_LOG.md`, `WHITEBOARD.md`
- **変更内容**:
  1. `patch_transformers_bert_for_groundingdino()` を追加し、新しい `transformers` で削除された `BertModel.get_head_mask` を GroundingDINO 初期化前に補う
  2. `GroundingDINODetector` / `GroundingDINOMultiBoxDetector` の `warm_up()` で GroundingDINO import 前に互換パッチを呼ぶ
  3. 既存 `gradio_app.py` の互換パッチも `[None] * num_hidden_layers` / `is_attention_chunked` / `self.dtype` 変換を含む実装へ修正
  4. `REFERENCE.md` / `ERROR_LOG.md` に `transformers` 互換性と Colab 再起動注意を記録
  5. 回帰テストで互換パッチの存在と `warm_up()` 呼び出しを確認
- **検証**: `.venv\Scripts\python.exe -m pytest tests\unit\test_pipeline_wiring.py::test_groundingdino_transformers_bert_compat_patch_is_called_before_model_import -q` 成功
- **残作業**: Colab で `Sam2_Transparent_Background_Haystack.ipynb` を install cell から再実行し、`Detect Text Boxes` が `supervision` / `get_head_mask` で止まらないことを確認

### タスク名: SAM2 Haystack Notebook 依存修正 + 日本語フロー説明追加（2026-05-26）
- **目的**: `Sam2_Transparent_Background_Haystack.ipynb` の Text Prompt 実行時に `No module named 'supervision'` が出る問題を解消し、Gradio UI 上で「どの入力が必須か / 任意か」を日本語で案内する
- **進捗**: 完了（Colab 実機での再実行確認待ち）
- **変更ファイル**: `Sam2_Transparent_Background_Haystack.py`, `Sam2_Transparent_Background_Haystack.ipynb`, `gradio_app_sam2_transparent_BG_haystack.py`, `tests/unit/test_jupytext_notebooks.py`, `requirements.txt`, `REFERENCE.md`, `ERROR_LOG.md`, `WHITEBOARD.md`
- **変更内容**:
  1. Colab install cell に GroundingDINO runtime 依存 (`transformers>=4.25.0`, `addict`, `yapf`, `timm`, `supervision`, `pycocotools`) を追加
  2. `checkpoints/groundingdino_swint_ogc.pth` の自動ダウンロードを追加
  3. `requirements.txt` に不足していた `timm` を追加
  4. Gradio UI 冒頭に「すべてを上から順に入力する必要はありません」「必須 / 任意」「推奨フロー」を日本語で追記
  5. Notebook 依存と UI ガイドの回帰テストを追加
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 45 passed / 2 deselected、`.venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py` 成功、`.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help` 成功
- **残作業**: Colab で `Sam2_Transparent_Background_Haystack.ipynb` を再実行し、Text Prompt → Detect Text Boxes が `supervision` 依存で止まらないことを確認

### タスク名: SAM2 Haystack 複合対象 MaskSet / union / MatteExtractor 契約統合（2026-05-26）
- **目的**: 「ドラムをたたく人」「自転車に乗る人」のような複合対象を、SAM2 multimask 候補と union mask で扱えるようにし、transparent-background を `image + mask -> MatteResult` の差し替え可能 Component として使う
- **進捗**: 実装完了（GPU / checkpoint / Colab 実機確認は未実施）
- **変更ファイル**: `archive/sam2_haystack_pre_mask_contract/`, `gradio_app_sam2_transparent_BG_haystack.py`, `Sam2_Transparent_Background_Haystack.py`, `Sam2_Transparent_Background_Haystack.ipynb`, `pipelines/components/common.py`, `pipelines/components/model_components.py`, `pipelines/sam2_tb_pipeline.py`, `tests/unit/test_common_components.py`, `tests/unit/test_pipeline_wiring.py`, `tests/unit/test_jupytext_notebooks.py`, `REFERENCE.md`, `Haystack_pipeline計画表.md`, `ERROR_LOG.md`
- **変更内容**:
  1. 既存 SAM2 Haystack 3 ファイルを `archive/sam2_haystack_pre_mask_contract/` へ退避し、同名ファイルを新契約ベースで再作成
  2. `MaskSet` / `SelectedMask` / `MatteResult` の dict 契約を `common.py` の helper と unit test で固定
  3. `MaskCandidateSelector` / `MaskUnion` / `MaskPreviewComposer` を追加し、candidate 選択・OR union・preview 合成を純粋 Component 化
  4. `SAM2Segmenter` が `masks` / `scores` に加えて `mask_set` を返すよう拡張
  5. `TransparentBGExtractor` が `matte_result` を返すよう拡張し、`image + mask -> MatteResult` の MatteExtractor として扱えるようにした
  6. `GroundingDINOMultiBoxDetector` を追加し、Text Prompt から複数 bbox / phrase / confidence を返す TextToRegion Component を用意
  7. `build_sam2_maskset_pipeline()` / `build_mask_union_pipeline()` / `build_mask_to_matte_pipeline()` / `build_sam2_union_tb_pipeline()` を追加
  8. Gradio UI に Text Prompt、Detected Text Boxes、SAM2 Candidate Mask Table、Candidate Mask Indices、Union Mask Preview、Union Mask を tb に渡す導線を追加
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 43 passed / 2 deselected、`.venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py` 成功、`.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help` 成功
- **残作業**: GPU / checkpoint ありで Text Prompt → SAM2 candidate → union → tb の実機確認、Colab で新 notebook 全セル実行、動画 tracking 用 frame_id / object_id 契約の拡張検討

### タスク名: SAM2 Haystack 版の SAM 利用調査報告（2026-05-26）
- **目的**: `Sam2_Transparent_Background_Haystack.py` が `README.md` / `gradio_app.py` / `main.py` / `segment-anything` の SAM 利用方針を踏襲しているか、また「人 + 物」を segmentation したいプロジェクト目的に合うかを調査する
- **進捗**: 完了
- **変更ファイル**: `SAM2_Haystack_SAM_USAGE_REPORT.md`, `WHITEBOARD.md`
- **調査結果**:
  1. SAM2 Haystack 版は point / box prompt、multimask、best score mask 採用という SAM の操作思想は踏襲している
  2. MAM の M2M alpha matte 推定と GroundingDINO text prompt 経路は踏襲しておらず、SAM2 + transparent-background の別系統実験導線である
  3. `Sam2_Transparent_Background_Haystack.py` 自体は Jupytext 正本の Colab 起動ノートであり、実処理は `gradio_app_sam2_transparent_BG_haystack.py` と `pipelines/` に委譲される
  4. 「人 + ドラム」「人 + 自転車」は手動 bbox 実験には向くが、最高 score mask 自動採用と SAM2 guard により人だけに固定される可能性が残る
  5. 次の改善候補は mask 候補選択 UI、複数 mask union、GroundingDINO text-to-box 接続、Pipeline I/O としての mask 選択 Component 化
- **成果物**: `SAM2_Haystack_SAM_USAGE_REPORT.md`
- **残作業**: 複合対象向けに mask 選択・union を設計し、必要なら実装する

### タスク名: SAM2 Haystack 版 UI のフロー説明・パラメータ説明強化（2026-05-26）
- **目的**: ユーザーから「どこにポイントを打つのかフローがよくわからない」「各パラメータが何を意味するか不明」というフィードバックを受け、`gradio_app_sam2_transparent_BG_haystack.py` の UI にガイドと説明を追加する
- **進捗**: 完了
- **変更ファイル**: `gradio_app_sam2_transparent_BG_haystack.py`, `tests/unit/test_jupytext_notebooks.py`, `REFERENCE.md`, `ERROR_LOG.md`, `WHITEBOARD.md`
- **変更内容**:
  1. アップロード用 `Input Image` と、クリック編集専用 `SAM2 Prompt Canvas` を分離
  2. `input_image.change(sync_prompt_canvas, ...)` で入力画像を prompt canvas にコピーし、prompt state と `SAM2_STATE` をリセット
  3. SAM2 point / bbox のクリックイベントを `prompt_canvas.select(...)` に移動し、入力欄をクリック対象から外した
  4. prompt / transparent-background 各コントロールへ `info=` を追加し、point / box / positive / negative / SAM2 mask / threshold / crop padding の意味を明示
  5. 未使用になった `SAM2 Mask Preview` 欄を削除し、prompt canvas に prompt と mask overlay を集約
  6. 回帰テストとして prompt canvas 分離と sync 時の state reset を追加
- **検証**: `tests/unit/test_jupytext_notebooks.py -q` で 14 passed、`.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 25 passed / 2 deselected、`gradio_app_sam2_transparent_BG_haystack.py --help` smoke 成功、サブエージェントレビュー済み（必須修正なし）
- **残作業**: GPU / checkpoint ありの UI 実機確認、Colab 実機確認、必要なら Examples 追加
- **追加修正（2026-05-26）**:
  1. `SAM2 Prompt Canvas` を `sources=[]` / `show_download_button=False` / `show_fullscreen_button=False` / `interactive=False` にし、Canvas へのドラッグ＆ドロップ導線を削除
  2. 空 Canvas は Gradio 標準アップロード placeholder ではなく、`create_prompt_canvas_placeholder()` の説明画像を初期表示
  3. `Image Display Size` Radio を追加し、prompt canvas / RGBA / Alpha / Preview を `window`（既定）と `original`（原寸）で切り替え可能にした
  4. 回帰テストを 17 件へ拡張し、非 integration 全体は 28 passed / 2 deselected
- **参照すべきもの**: ERR019（positive/negative Radio / Extend ボタン）、`.github/skills/ui-ux-pro-max/SKILL.md`（UI ガイド作成の観点）

### タスク名: SAM2 Haystack 版 positive/negative Radio + Edge Extend ボタン（2026-05-25）
- **目的**: SAM2 の point prompt 正負を Radio で明示化し、bbox を画像端 (0 / w-1 / h-1) まで届かせる補助 UI を追加する
- **進捗**: 完了
- **変更ファイル**: `gradio_app_sam2_transparent_BG_haystack.py`, `tests/unit/test_jupytext_notebooks.py`, `ERROR_LOG.md`, `.github/copilot-instructions.md`, `.github/instructions/workflow.instructions.md`, `WHITEBOARD.md`
- **変更内容**:
  1. SAM2 prompt 入力を `gr.ImageEditor` から `gr.Image(type="numpy", interactive=True)` に変更
  2. `gr.Checkbox("Positive Point")` を `gr.Radio(["positive", "negative"], value="positive")` に変更
  3. `select_sam2_prompt` の `point_label` を str / bool 両対応に拡張（後方互換）
  4. `EDGE_SNAP_PIXELS` を 8 → 16 へ拡大
  5. `extend_box_to_edge(input_image, prompt_state, side)` と `Extend Left/Right/Top/Bottom` 4 ボタンを追加。確定済み bbox の指定辺を画像端に揃え、bbox 未確定時は `gr.Error`
  6. 回帰テスト 5 件と関連文書を更新（ERR019 を追記）
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` で 23 passed / 2 deselected、サブエージェントレビュー済み（必須修正なし）
- **残作業**: Colab / GPU / checkpoint ありの UI 実機確認

### タスク名: SAM2 Haystack 版 bbox マウス選択化（2026-05-25）
- **目的**: SAM2 の bbox / point 座標手入力を廃止し、画像上のマウス操作で端まで選択できる UI にする
- **進捗**: 完了
- **変更ファイル**: `gradio_app_sam2_transparent_BG_haystack.py`, `tests/unit/test_jupytext_notebooks.py`, `ERROR_LOG.md`, `REFERENCE.md`, `.github/copilot-instructions.md`, `.github/instructions/workflow.instructions.md`, `WHITEBOARD.md`
- **変更内容**:
  1. `Point X` / `Point Y` の `gr.Number` を削除
  2. `ImageEditor.select` で point / box prompt を蓄積する UI に変更
  3. bbox は 2 クリックで確定し、端付近クリックを画像端へ吸着
  4. Haystack `transparent_bg` 中間出力を `include_outputs_from` で取得し、ERR018 を解消
  5. 回帰テストと関連文書を更新
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -v` で 18 passed / 2 deselected、`gradio_app_sam2_transparent_BG_haystack.py --help` smoke 成功、サブエージェントレビュー済み
- **残作業**: Colab / GPU / checkpoint ありの UI 実機確認

### タスク名: SAM2 Haystack 版 Gradio `/info` schema crash 対応（2026-05-25）
- **目的**: `Sam2_Transparent_Background_Haystack.ipynb` / `.py` 起動時に発生する Gradio 5 `/info` の bool schema 例外を解消する
- **進捗**: 完了
- **変更ファイル**: `gradio_app_sam2_transparent_BG_haystack.py`, `tests/unit/test_jupytext_notebooks.py`, `ERROR_LOG.md`, `REFERENCE.md`, `WHITEBOARD.md`
- **変更内容**:
  1. Haystack 版 SAM2 Gradio アプリに ERR011 と同じ `gradio_client.utils._json_schema_to_python_type` bool schema patch を追加
  2. `demo.launch(..., show_api=False)` を API 表示の補助設定として追加
  3. 回帰テストと関連文書を更新
- **検証**: `.venv\Scripts\python.exe -m pytest -m "not integration" -v` で 14 passed / 2 deselected、`gradio_app_sam2_transparent_BG_haystack.py --help` smoke 成功
- **残作業**: Colab 実機での UI 起動確認

### タスク名: Haystack 2.x Component / Pipeline 化（2026-05-25）
- **目的**: 既存 Gradio アプリの推論処理を Haystack 2.x の型付き DAG と Component に分離し、モデル差し替えとテストを容易にする
- **進捗**: 初期実装完了
- **変更ファイル**: `pipelines/`, `gradio_app_haystack.py`, `gradio_app_sam2_transparent_BG_haystack.py`, `.github/skills/haystack-pipeline/`, `tests/`, `requirements.txt`, docs
- **変更内容**:
  1. `haystack-ai==2.29.0` を `requirements.txt` に追加
  2. 共通純粋 Component（入力正規化、スクリブル解析、bbox、mask dilate、alpha 合成）を追加
  3. 外部モデル Component（GroundingDINO、MAM、SAM2、transparent-background、背景生成、出力保存）を追加
  4. MAM text/scribble Pipeline と SAM2 prompt / tb Pipeline を追加
  5. Haystack 版 Gradio entrypoint を新規追加し、既存アプリは保持
  6. unit / integration test 骨格と Haystack 作業スキルを追加
  7. サブエージェントレビュー指摘（ScribbleParser bbox / MAM box shape）を反映
  8. unit test と Pipeline builder smoke test を実行
- **残作業**: 実機 GPU / checkpoint ありの integration 動作確認

### タスク名: Haystack 版 Notebook の Jupytext 管理化（2026-05-25）
- **目的**: `Matting_Anything.ipynb` と `Sam2_Transparent_Background.ipynb` を参考に、Haystack 版 notebook を Jupytext 正本 `.py` から生成する運用へ統一する
- **進捗**: 初期実装完了
- **変更ファイル**: `Matting_Anything_Haystack.py`, `Sam2_Transparent_Background_Haystack.py`, 生成予定 `.ipynb`, docs, tests, requirements
- **変更内容**:
  1. Haystack 版 Colab 起動ノートの Jupytext percent source を追加
  2. `gradio_app_haystack.py` / `gradio_app_sam2_transparent_BG_haystack.py` に `--share` / `--debug` / port 引数を追加
  3. `requirements.txt` に `jupytext` を追加
  4. Jupytext 正本ルールを instructions / reference に追記
  5. `Matting_Anything_Haystack.ipynb` と `Sam2_Transparent_Background_Haystack.ipynb` を Jupytext で生成
  6. unit test、CLI `--help` smoke、サブエージェントレビューを実施
- **残作業**: Colab 実機と GPU/checkpoint ありの integration 動作確認

### タスク名: SAM2 なしモード + UI 説明追加（2026-05-23）
- **目的**: tb 単体で背景除去できるよう改造・UI に用語説明・パラメータ説明を追加
- **進捗**: 完了
- **変更ファイル**: `gradio_app_sam2_transparent_BG.py`
- **変更内容**:
  1. `on_run_tb`: `selected_mask is None` チェックを削除 → SAM2 なしでも tb が画像全体を処理
  2. 完了メッセージ: SAM2 なし時は「抽出完了（SAM2 なし・全体処理）」と表示
  3. UI ヘッダー: 「SAM2 はスキップ可能」の案内を追記
  4. tb パラメータ（mode/JIT/threshold/output）に `info=` ツールチップ追加
  5. パイプラインパラメータ（crop_pad/use_guard/guard_dilate/decontam）に `info=` 追加
  6. 「📖 用語・アルゴリズム解説」アコーディオン追加（tb/SAM2/alpha matte/dilate/soft alpha/decontam を説明）
  7. 実行ボタンラベルを「背景除去を実行（SAM2 なしでも動作）」に更新
- **残作業**: 黄緑合成プレビュー追加（別要求、未着手）

### タスク名: SAM2 + transparent-background パイプラインのパス整理（2026-05-22）
- **目的**: チェックポイント・入出力をすべてプロジェクト内パスに統一（Drive 依存撤廃）
- **進捗**: 完了
- **変更点**:
  - `Sam2_Transparent_Background.ipynb` Cell 2: `google.colab.drive.mount` 撤去 → `PROJECT_ROOT` 自動判定（env / Colab / cwd）
  - `checkpoints/SAM2/sam2.1_hiera_large.pt`, `checkpoints/transparent_BG/ckpt_*.pth` に統一
  - Cell 4: `TB_CKPT_BY_MODE` 追加・ローカル ckpt が存在すれば自動使用
  - `gradio_app_sam2_transparent_BG.py`: 同様にパス変更 + `OUTPUT_DIR = ./outputs/` + `save_to_disk` チェックボックス追加
  - uv で `.venv` (Python 3.11.12) を新規作成
- **残作業**: 実機（GPU）での動作確認

### タスク名: Gradio 5 対応完了
- **目的**: gradio_app.py を Gradio 5 互換 API に移行し、Matting_Anything.ipynb のランタイムパッチセルを削除
- **進捗**: 完了
- **残作業**: なし
- **注意事項**: `gr.ImageEditor` 戻り値は `{background, layers, composite}` の dict

### タスク名: Colab GPU 対応完了
- **目的**: Matting_Anything.ipynb を Google Colab GPU ランタイムで動作させる
- **進捗**: 完了
- **残作業**: 実機での動作確認
- **注意事項**: GroundingDINO の CUDA ビルドには `CUDA_HOME` 環境変数と `--no-build-isolation` が必要

---

## 直近の決定事項・変更履歴

<!-- 重要な設計判断・変更点を記録 -->

| 日付 | 変更内容 | 理由 |
|------|----------|------|
| 2025-07-23 | `gr.Image(tool="sketch")` → `gr.ImageEditor` に変更 | Gradio 5 で `tool="sketch"` が廃止 |
| 2025-07-23 | `block = block.queue()` → `with gr.Blocks() as block:` + `block.queue()` 分離 | Gradio 5 で `queue()` が `self` を返さなくなった |
| 2025-07-23 | `input_image["image"]` → `.get('background', ...)` に変更 | Gradio 5 ImageEditor の戻り値キー変更 |
| 2025-07-23 | `torch.load(..., weights_only=True)` 追加 | セキュリティ強化（OWASP A08: Insecure Deserialization 対策） |
| 2025-07-23 | `print(...)` → `raise gr.Error(...)` に変更 | エラー時に実行継続し NameError 発生するバグを修正 |
| 2025-07-23 | Matting_Anything.ipynb の 14 個のパッチセル削除 | gradio_app.py 修正完了により不要 |
| 2025-07-24 | Matting_Anything.ipynb セル2: GPU/CUDA チェック強化、torch.version.cuda 表示追加 | GPU 未設定時の早期検出のため |
| 2025-07-24 | Matting_Anything.ipynb セル5: `!export` → `os.environ` に変更、`--no-build-isolation` 追加 | `!export` は別サブシェルで実行されるため CUDA_HOME が pip install に引き継がれなかった |
| 2025-07-24 | セル5: GroundingDINO インストール時の `-q` 削除 | CUDA ビルドエラーが隠れるため |
| 2025-07-24 | セル5: クローンロジックを `.git` 存在確認＋`--depth=1` に変更 | 不完全クローンの再利用防止・高速化 |
| 2026-05-14 | `bertwarper.py`: `get_extended_attention_mask` 呼び出しから `device` 引数を削除 | 新しい transformers で第3引数が dtype に変更され TypeError 発生 |
| 2026-05-14 | `ms_deform_attn.py`: `CUDA_OPS_AVAILABLE` フラグを追加し forward() の分岐条件に追加 | CUDA ops ビルド失敗時に `_C` が未定義のまま CUDA パスに入り NameError が発生していた |
| 2026-05-14 | `transformer.py`（2箇所）・`backbone/swin_transformer.py`（1箇所）: `checkpoint.checkpoint()` に `use_reentrant=False` 追加 | PyTorch 2.9 以降で use_reentrant 未指定が例外になるため事前対応 |
| 2026-05-14 | `gradio_app.py`: `image_ori` 取得直後に RGBA→RGB 変換チェックを追加 | gr.ImageEditor が RGBA (4ch) ndarray を返す場合に pixel_mean (3ch) とテンソル次元不一致で RuntimeError が発生していた（ERR008） |
| 2026-05-14 | `GroundingDINO/.../transformer.py`: `torch.cuda.amp.autocast` → `torch.amp.autocast('cuda', ...)` に変更 | PyTorch 2.x で deprecated API の FutureWarning 解消（ERR009） |
| 2026-05-14 | `INSTALL_ja.md` 新規作成 | INSTALL.md の日本語版。Colab セル構成・チェックポイント配置・トラブルシューティング表を含む |
| 2026-05-14 | `GETTING_STARTED_ja.md` 新規作成 | GETTING_STARTED.md の日本語版。Matting_Anything.ipynb のセル構成・実行手順・Google Drive パスを追記 |
| 2026-05-22 | `.github/copilot-instructions.md`, `whiteboard-manager.agent.md`, `project-reference.agent.md`, `error-knowledge-base.agent.md` を Matting-Anything 用に全面書き直し | 別プロジェクト（Enhanced3ModalHRM: MLflow/Prefect/Optuna/Lightning）の内容が誤って混入していたため || 2026-05-22 | `copilot-instructions.md` 実装フローにTDDを追加（RED/GREEN/REFACTOR）。`pytest.mark.integration` で推論パイプラインを CI 分離 | エージェント作業品質向上のため |
| 2026-05-22 | `Sam2_Transparent_Background.ipynb` に全7セル追加。SAM2 + transparent-background パイプライン完全実装 | `SAM2 + Transparent-Background Background Removal Pipeline.md` の内容を反映 |
| 2026-05-22 | `gradio_app_sam2_transparent_BG.py` 新規作成。ローカル実行向け Gradio 5 デモ | 同上。Colab 版 Cell 6 と同等機能をスタンドアロン化 |
| 2026-05-22 | Drive パス (`/content/drive/MyDrive/bg_removal_ckpts`) → プロジェクト内 `checkpoints/SAM2/`, `checkpoints/transparent_BG/`, `outputs/` に移行 | ローカルワークステーションでの実行を可能にするため。Cell 2 に `PROJECT_ROOT` 自動検出・Cell 4 に `TB_CKPT_BY_MODE` を追加 |
| 2026-05-22 | `gradio_app_sam2_transparent_BG.py` に `OUTPUT_DIR = ./outputs/` + `save_to_disk` チェックボックス追加 | ノートブック Cell 6 と機能を揃えるため |
| 2026-05-22 | `uv venv --python 3.11 .venv` でプロジェクト直下に仮想環境を作成（CPython 3.11.12） | Windows の `python` が MS Store スタブで動作しないため、プロジェクトローカルの Python を確保 |
| 2026-05-22 | `Sam2_Transparent_Background.ipynb` Cell 2 ・ `gradio_app_sam2_transparent_BG.py` に Google Drive 自動マウントと `PROJECT_ROOT = /content/drive/MyDrive/AI_picasso/Matting-Anything` を追加 | プロジェクト本体が Google Drive 上にあるため、Colab では Drive のパスをルートにする必要がある |
| 2026-05-22 | `Sam2_Transparent_Background.ipynb` Cell 6 ・ `gradio_app_sam2_transparent_BG.py` に `demo.launch(show_api=False)` を適用、`mask_idx` Radio を文字列 choices 化（`["0","1","2"]`）し ハンドラ側で `int(idx)` キャスト | Gradio 5.x の `/info` api_info エンドポイントが整数 choices の Radio で schema 変換クラッシュする既知バグ（ERR011）に対処 |
| 2026-05-22 | `Sam2_Transparent_Background.ipynb` を 4 セル構成にスリム化（install / Drive mount + PROJECT_ROOT / `!{sys.executable} gradio_app_sam2_transparent_BG.py --share`）。SAM2/tb/pipeline/UI セルを削除し実装本体を `.py` に一本化 | DRY 原則。ノートブックと `.py` の二重保守を解消（ERR011 の修正が両方に必要だった反省） |
| 2026-05-22 | `gradio_app_sam2_transparent_BG.py` の `if __name__ == "__main__"` に `argparse` を追加（`--share` / `--debug` / `--server-name` / `--server-port`） | Colab ではサブプロセス起動 + `--share` で公開 URL、ローカルでは引数なしで `127.0.0.1` 起動を実現するため |
| 2026-05-22 | `gradio_app_sam2_transparent_BG.py`：`gradio.routes.App.api_info` モンキーパッチ（旧）を `gradio_client.utils._json_schema_to_python_type` パッチ（新・ERR011 確定修正）に置き換え | 旧パッチは FastAPI がルート登録時に関数を参照コピーするため無効だった。クラッシュ箇所は `additionalProperties: bool` を受け取った `_json_schema_to_python_type`。新パッチは `isinstance(schema, bool)` チェックを追加し "Any" を返すことでクラッシュを根本解消（ERR011 参照） |
| 2026-05-25 | `haystack-ai==2.29.0` を固定し、`pipelines/` に Haystack 2.x Component / Pipeline 構成を追加 | Gradio callback から推論 DAG を分離し、モデル差し替え・Component 単位テストを容易にするため |
| 2026-05-25 | 既存 Gradio アプリは残し、`gradio_app_haystack.py` と `gradio_app_sam2_transparent_BG_haystack.py` を新規追加 | 既存動作を温存しながら Haystack 版を検証できるようにするため |
| 2026-05-25 | `.github/skills/haystack-pipeline/` を追加 | 今後の Component 化作業で同じ設計判断を再利用するため |
| 2026-05-25 | Haystack unit test 11 件と Pipeline builder smoke test が成功 | 重いモデルを初期化しない範囲で Component と DAG 接続を検証するため |
| 2026-05-25 | Haystack 版 notebook は Jupytext percent `.py` を正本にし、`.ipynb` を生成物として扱う方針に決定 | Notebook の JSON 差分を抑え、レビューと保守を簡単にするため |
| 2026-05-25 | `Matting_Anything_Haystack.ipynb` と `Sam2_Transparent_Background_Haystack.ipynb` を Jupytext source から生成 | 既存 notebook を参考に Haystack 版 Colab 起動導線を追加するため |
| 2026-05-25 | `gradio_app_sam2_transparent_BG_haystack.py` に `gradio_client.utils._json_schema_to_python_type` bool schema patch と `show_api=False` を追加 | Haystack 版 SAM2 notebook 起動時の ASGI 例外（ERR016）を解消するため |
| 2026-05-25 | SAM2 Haystack 版の座標手入力 UI を廃止し、マウスクリックによる point / bbox prompt と端吸着 bbox を追加 | 被写体が画面端・画面外へ続くケースでも bbox を画面端まで選択できるようにするため |
| 2026-05-25 | Haystack 版 tb 実行で `include_outputs_from` を指定 | 中間 Component 出力 `transparent_bg` が結果に含まれず `KeyError` になる問題（ERR018）を防ぐため |
| 2026-05-26 | `.github/skills/haystack-pipeline/SKILL.md` の Chat Customizations 診断を修正 | 否定形見出し配下に推奨表がある混乱、Component I/O 粒度、device / model 共有、レビュー完了条件の曖昧さを解消するため |
| 2026-05-26 | SAM2 Haystack 版 UI でアップロード用 `Input Image` と編集用 `SAM2 Prompt Canvas` を分離 | bbox / point 指定がアップロード欄に埋もれて行方不明になる UX を解消するため |
| 2026-05-26 | `SAM2 Prompt Canvas` のアップロード導線を削除し、`Image Display Size` 切替を追加 | Canvas は Input Image の同期先に限定し、予測画像は既定でウィンドウサイズ表示・必要時に原寸表示へ切替できるようにするため |
| 2026-05-26 | SAM2 Haystack 版を `MaskSet` / `SelectedMask` / `MatteResult` 契約へ更新 | SAM2 / GroundingDINO / transparent-background / 将来モデルの差し替えを mask / alpha / bbox / score / metadata 境界で疎結合化するため |
| 2026-05-26 | SAM2 candidate mask table と union mask UI を追加 | best mask 自動採用で「人だけ」になる問題を避け、人物 + 物体の複合対象をユーザーが統合できるようにするため |

---

## 既知の問題・ブロッカー

<!-- 未解決の問題や次のセッションで対応が必要なもの -->

- GroundingDINO CUDA ops（`_C`）がビルド失敗するとCPUフォールバックで動作する。GPU を活かしきれていない可能性がある（`CUDA_HOME` 設定・ビルドログで確認要）。
- SAM2 Haystack mask union 版は非 integration テストと CLI smoke は完了。GPU / checkpoint / 外部モデルを使う integration 動作確認は未実施。
- Haystack 版 notebook は Jupytext 生成済み。Colab 実機での全セル実行は未実施。

---

## 次のアクション

<!-- 次のセッションで最初にやること -->

1. 実機 GPU / checkpoint ありで `gradio_app_sam2_transparent_BG_haystack.py` を起動し、Text Prompt → Detected Boxes → SAM2 Candidate Masks → Mask Union → transparent-background の一連操作を確認
2. 実機 GPU / checkpoint ありで `gradio_app_haystack.py` を起動確認
3. Colab で `Matting_Anything_Haystack.ipynb` / `Sam2_Transparent_Background_Haystack.ipynb` を全セル実行確認
4. 必要なら `inputs/` のサンプル画像を使った `gr.Examples` とヘルプ Accordion を追加
5. integration test に最小画像・checkpoint 存在確認・skip 条件を追加

---

## プロジェクト固有メモ

<!-- Matting-Anything 固有の注意事項、モデル・データセットに関するメモ -->

- チェックポイントはプロジェクト内で以下に統一:
  - MAM: `checkpoints/mam_vit{b,l,h}.pth`
  - SAM v1: `segment-anything/checkpoints/sam_vit_b_01ec64.pth`
  - SAM v2: `checkpoints/SAM2/sam2.1_hiera_large.pt`
  - transparent-background: `checkpoints/transparent_BG/ckpt_{base,fast,base_nightly}.pth`
  - GroundingDINO: `checkpoints/groundingdino_swint_ogc.pth`
- 設定ファイル: `config/MAM-ViT*.toml`
- 出力先: `outputs/<YYYYMMDD_HHMMSS>/` (`rgba.png`, `alpha.png`, `preview.png`)
- Python 環境: `.venv/` (uv 作成 / Python 3.11.12)、キックは `.venv\Scripts\python.exe`
- `.venv` は当初 pip なしだったため、必要時は `.venv\Scripts\python.exe -m ensurepip --upgrade` で復旧済み
- PROJECT_ROOT の解釈:
  - Windows ローカル: `J:\マイドライブ\AI_picasso\Matting-Anything`
  - Google Colab: `/content/drive/MyDrive/AI_picasso/Matting-Anything`（Drive 自動マウント）
  - 環境変数 `PROJECT_ROOT` を設定すればどちらでも手動上書き可
