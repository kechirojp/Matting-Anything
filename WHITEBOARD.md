# ホワイトボード — 作業引継ぎメモリ

> **ルール**: 作業開始前に必ずこのファイルを読む。作業完了後に必ず更新する。

---

## 現在の作業状況

| 項目 | 内容 |
|------|------|
| **Windows 非ASCII（日本語）パスで `cv2.imwrite`/`cv2.imread` が無言失敗し overlay PNG 保存エラー（`PNG 保存に失敗しました: ...frame_000000.png`）を根治（ERR061・2026-06-26）** | **✅ 完了**。依頼（逐語要旨）: 「エラーログ\エラーログ_28.md / `gradio_app_sam2_ben2_route_a_for_Movie.py` / エラー発生 / `.github\copilot-instructions.md` に従い対処」。**真因**: ワークスペースが `J:\マイドライブ\...`（「マイドライブ」が非ASCII）にあり、OpenCV の `cv2.imwrite`/`cv2.imread` は Windows で ANSI codepage を使うため日本語パスを開けず、`cv2.imwrite` は **`False`**（無言失敗・`TrackingOverlayWriter` が `RuntimeError("PNG 保存に失敗しました")` 送出）、`cv2.imread` は **`None`** を返す。パイプライン本体は ~298.8s で成功するが最終 overlay PNG の 1 枚目で失敗。実機検証で確定（`cv2.imwrite`→False/未生成、`cv2.imencode`+`write_bytes`→成功、`cv2.imread`→None）。`cv2.VideoWriter` は FFmpeg backend で日本語パスでも成功（検証済み・据え置き）、SAM2 一時 JPEG は ASCII の `%TEMP%` 配下で無影響。**対処**: ①`pipelines/components/common.py` に Unicode 安全な `imwrite_unicode`（`cv2.imencode`→`Path.write_bytes`、bool 互換）/`imread_unicode`（`np.fromfile`→`cv2.imdecode`、失敗時 None）を追加。②`video_common.py::write_png_frame` を `imwrite_unicode` 使用に変更（`RuntimeError` 契約維持）。③`model_components.py` の背景読み込み `cv2.imread`→`imread_unicode`+None チェック（`ValueError`）。**TDD**: RED（非ASCIIパスで write/read/overlay 失敗）→ GREEN。新規テスト3件（`test_common_components.py` 2 + `test_video_common_components.py` 1）。**検証**: 対象3テスト **3 passed**（Drive のため295s）、変更2ファイル全体 **32 passed**、`get_errors`=0。サブエージェントレビュー **APPROVED**（軽微指摘=docstring に欠損パス時 `FileNotFoundError` 挙動を追記して反映済み）。**教訓**: 出力/入力パスへ読み書きする `cv2.imwrite`/`cv2.imread` は `imwrite_unicode`/`imread_unicode` を使う。根本的にはローカルの ASCII パス（例 `C:\dev\...`）へ移すと本クラス問題と Drive の遅い I/O を同時回避（ユーザーは移行を自分で後日実施予定）。詳細は ERROR_LOG ERR061。 |
| **BEN2 ロードで `loadcheckpoints` にディレクトリを渡し `[Errno 13] Permission denied: 'checkpoints\BEN2'`（DL成功・ロード失敗）を根治（ERR060・2026-06-25）** | **✅ 完了**。依頼（逐語要旨）: 「以下のエラーに対処お願い このエラーから類推されるエラーがあるならそれにも対処して 対処フローは `.github\copilot-instructions.md`」。**真因**: `BEN_Base.loadcheckpoints` は内部で `torch.load(path, weights_only=True)`＝**`.pth` ファイルパス**を要求するが、`BEN2Extractor.warm_up` がローカル既存経路でも DL 経路でも **ディレクトリ** `checkpoints/BEN2`（config 既定 `ben2_checkpoint_path`）をそのまま渡していた。Windows / Google Drive FUSE では `torch.load(<dir>)`＝`open(<dir>)` が `[Errno 13] Permission denied` になる（DL の成否と無関係）。ログの再 DL は初回でディレクトリ空だった正常動作。**対処**: ①`pipelines/components/ben2_components.py` に `BEN2Extractor._resolve_loadable_checkpoint(target)` を追加（`.pth` ファイルはそのまま、ディレクトリは直下 `glob('*.pth')`→無ければ `.cache` 除外 `rglob`、未発見は `RuntimeError`）。②`warm_up` で DL/既存判定後に `.pth` へ解決してから `loadcheckpoints(str(checkpoint_file))` を呼ぶよう変更。**TDD**: 誤った契約（`load_path == str(ckpt_dir)`）を固定化していた `test_ben2_extractor_uses_local_checkpoint_without_download` を `str(ckpt_dir / 'ckpt_base.pth')` 期待へ修正（RED→GREEN）。**検証**: `pytest tests\unit\test_ben2_components.py -q` **9 passed**、非 integration **281 passed / 3 deselected**、対象2ファイル `get_errors`=0。**教訓**: `torch.load` 系へ渡すパスは必ずファイルへ解決（ディレクトリ禁止）。「DL 成功」と「ロード可能」は別事象＝DL 後は `.pth` 存在をファイル単位で検証。配布物のファイル名（`BEN2_Base.pth`）は固定名ではなく `*.pth` 探索で解決。詳細は ERROR_LOG ERR060。 |
| **GroundingDINO top_k=20 + checkpoint永続化ポリシー明文化 + 動画アプリ起動bat追加（2026-06-25）** | **✅ 完了**。依頼（逐語要旨）: 「groundDINOのtop kは20にしてくれ」「今後の各種チェックポイントもローカル永続化をリファレンスに書く」「起動バッチを2アプリ分、`.venv\\Scripts\\python.exe` で」。実装: ①`pipelines/components/model_components.py` の `GroundingDINOMultiBoxDetector.run(..., top_k)` 既定を **5→20**。②動画UI 2本（`gradio_app_sam2_ben2_route_a_for_Movie.py` / `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`）の「候補数 top-k」スライダを **1..20 / default 20** に更新（説明文も 20 へ）。③`REFERENCE.md` に「チェックポイント永続化ポリシー（ローカル優先→未存在時download→同一ローカルへ保存→次回再利用、保存先は `checkpoints/` 配下、config 駆動維持）」を追記。④起動batを追加: `run_routea_movie.bat` / `run_transparent_movie.bat`（どちらも `.venv\\Scripts\\python.exe ... %*` で引数透過）＋補助 `run_movie_apps.bat`（2アプリ同時起動）。検証: 両 app `--help` 正常、両 bat 経由 `--help` 正常、`pytest tests/unit/test_movie_app_ui_wiring.py tests/unit/test_route_a_video_pipeline_wiring.py -q` **10 passed**、対象3ファイル `get_errors`=0。サブエージェントレビュー実施（重大指摘なし、top_k増による計算量増は意図変更として許容）。 |
| **BEN2 チェックポイント運用をローカル優先+永続化へ統一（2026-06-25）** | **✅ 完了**。依頼（逐語要旨）: 「チェックポイントはローカルから使いたい / ローカルになければ download / そのチェックポイントは永続化」。実装: `BEN2Extractor` を改修し、`checkpoint_path`（既定 `checkpoints/BEN2`）をローカル保存先として解決、`.pth` が存在すれば `loadcheckpoints` でローカル即時ロード、無ければ `huggingface_hub.snapshot_download` で同ディレクトリへ取得して永続化後に同パスからロード。download 後に `.pth` 検証を追加し、見つからない場合は明確な `RuntimeError` を送出（握り潰しなし）。`route_a_common` 既定値と `config/route_a.toml` を `checkpoints/BEN2` に揃え、`QUICKSTART_uv_local.md` に運用説明を追記。テスト: `tests/unit/test_ben2_components.py` に local優先/未存在時download+永続化/warm_up冪等性 を追加し、`test_ben2_components.py`+`test_route_a_common.py` **32 passed**。サブエージェントレビュー指摘（download後検証不足・冪等性テスト欠落）を反映済み。 |
| **ローカル RTX 4090 直結運用へ向け .venv+pip → uv へ全面移行（transparent-background 同梱・ERR059・2026-06-24）** | **✅ 環境構築完了（実機モデル推論はユーザー要確認）**。依頼（逐語要旨）: 「google cloud run にデプロイで解決できるか調べて。だめならローカルだけで UV 環境作る、シンプルに」→「B にするよ（=フル uv 移行）、transparent background もいずれ使うので考慮した環境に」。**経緯**: ERR058 で確定した根治＝Colab/gradio.live トンネルをやめローカル 4090 で `--share` なし `127.0.0.1` 直結（トンネル無し＝SSE 切断クラスが原理的に消滅）。Cloud Run は技術的には 60 分タイムアウト等で接続切れを直せるが、GroundingDINO CUDA ビルド・GPU L4 quota/課金・大容量コンテナ/コールドスタート・コストで本アプリには非シンプル → ローカル直結を採用。**実施**: ①`pyproject.toml`（`package=false`・`requires-python ">=3.11,<3.12"`・torch/torchvision を index `pytorch-cu124` へ・ben2 を git・transparent-background 同梱）＋`.python-version`="3.11" 作成。②`uv sync` で torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / transparent-background 1.3.4 / ben2 / 他を導入。③**SAM-2 のみ別途** `$env:SAM2_BUILD_CUDA="0"; uv pip install --python .venv\Scripts\python.exe --no-build-isolation -e samurai/sam2`（externally managed 回避＋torch import 解決＋optional nvcc 拡張スキップ）。④`uv add --dev pytest`。**検証（全 PASS）**: `torch.cuda.is_available()=True` / "NVIDIA GeForce RTX 4090"、`import sam2/transparent_background/ben2/gradio(5.9.1)/haystack` OK、RouteA・tb 両動画アプリ `--help` 正常、**非 integration 278 passed / 3 deselected**、get_errors=0。**ローカル起動**: `.venv\Scripts\python.exe gradio_app_sam2_ben2_route_a_for_Movie.py`（`--share` を付けない＝127.0.0.1 直結）。**留意**: GroundingDINO custom CUDA ops は未導入（テキスト検出時のみ必要・optional、手動 bbox/point の RouteA には不要）。`flet` の UserWarning は無害。実機での end-to-end モデル推論・GPU 速度・ERR058 の SSE 切断解消の最終確認はユーザーが 4090 上で実施（ここでは配線・import・CUDA 検出・テストまで）。手順は repo memory `/memories/repo/env-uv-local.md`。 |
| **RouteA 動画: 全工程を 1 本の同期リクエスト＝長時間 SSE として gradio.live 越しに保持し続け SAM2 伝搬完了後（BEN2 抽出/書き出し）で切断→全出力「Error」を非同期ジョブ化で根治（ERR058・ERR048-057 の対症療法の限界・2026-06-24）** | **✅ 実装完了（実機 Playwright 検証は ERR035 によりユーザー/GPT-5.5 へ委譲）**。依頼（逐語要旨）: 「非同期にしてくれ」「クライアントがすぐ試す環境が欲しい」「スコープはルートA」「テキスト化して計画書に書くこと」「`.github\copilot-instructions.md` にしたがって実装、agent モード」。レビューは GPT-5.5 に委譲（自前 subagent レビューはしない）。**真因**: Gradio 単体バグではなく 3 層相互作用＝①Gradio が 1 予測=1 本の長寿命 SSE を全処理時間（数分）占有 ②無料 gradio.live FRP トンネルの総接続時間上限が長時間接続を切る（真因の核心・localhost 直結なら切れない）③Colab の追加 proxy。ERR048→055→056→057 の keep-alive/prewarm は対症療法で停止点を 1 段ずつ動かすだけ（最新 `エラーログ/エラーログ_26.md` では prewarm 完走→SAM2 伝搬 116s 完走→その後の BEN2 抽出/writer で停止）。**根治（非同期ジョブ化でリクエストを <1s に短命化）**: ①新規 `pipelines/job_manager.py`（stdlib のみ・torch/gradio 非依存）= `JobState`/`JobManager.submit(work)→job_id`（daemon スレッド・進捗を JobState へ・**例外は握り潰さず error 保持**）/`snapshot`/`cleanup`。②`gradio_app_sam2_ben2_route_a_for_Movie.py` に `_ProgressBridge`（`gr.Progress` 互換で既存 `build_video_progress_callback` を無改変再利用）・`start_route_a_job`/`start_route_a_only_job`（fail-fast `gr.Error`→submit→即 `(job_id, 進捗テキスト, gr.Timer(active=True), btn 無効)` 返却）・`poll_route_a_job`/`poll_route_a_only_job`（`gr.Timer.tick` 束縛: running=進捗テキスト更新/done=出力返却・Timer 停止・btn 復帰/error=初回 tick で `gr.Error` 通知→`_REPORTED_JOB_ERRORS` で 2 回目以降の多重トースト抑止し UI 復帰）を追加し両タブを再配線。進捗は `gr.Progress` バーをやめ `run_status` Markdown のテキスト更新に変更（トンネル安全）。keep-alive/prewarm は二重防御として温存。**検証**: RED→GREEN `tests/unit/test_job_manager.py`(6)＋`tests/unit/test_route_a_async_wiring.py`(7)=14 passed、**非 integration 277 passed / 1 skipped / 3 deselected**（回帰なし）、RouteA app `--help` smoke 正常、get_errors=0。**ERR035 留意**: UI/配線「fixed」確定には実機（ローカル RTX 4090 もしくは Colab）での Playwright 実行時検証（run→Timer ポーリングで進捗更新→完了で出力／失敗で 1 度だけ赤トースト＋UI 復帰）が必要＝本記録時点では「実装完了」止まりで「UI fixed」とは断定しない。接続層の別解（範囲外）: 4090 で `--share` なし `127.0.0.1` 直結ならトンネルが無くなり ERR048-058 の切断クラスが原理的に消滅。transparent_BG 動画版への `job_manager` 流用は fast-follow（範囲外）。 |
| **RouteA 動画: BEN2 約380MB DL が HF レート制限で低速化しリクエスト中の長時間 DL で SSE 切断→全出力「Error」を根治（ERR057・ERR055/ERR056 follow-up・2026-06-24）** | **✅ 完了**（起動前事前ロードで根治・keep-alive はフォールバック併存）。依頼（逐語）: 「前のエラーより一つだけ進んで とまった 処理は走ってるけど gradioでエラー表示 同じ系統のエラーがずーーーっとつづいてる」。**真因（`エラーログ/エラーログ_25.md` のタイミング精査）**: SAM2 伝搬は 30 frame を 135s（≈4.5s/frame）で完走＝SSE idle 許容は 4.5s 超なので keep-alive 間隔(2s)は十分。bert(440M) はリクエスト中でも 286MB/s で一瞬→無事だが、BEN2(`PramaLLC/BEN2`) 約380MB は **HF 未認証レート制限**で `?B/s`(0%) と極端に低速。核心＝**レート制限で分単位に伸びた DL がリクエスト処理中（SSE ストリーム中）に走る**こと自体。keep-alive(ERR055/056)は無通信ギャップは埋めるが**超低速 DL の総所要時間は短縮できない**。**対処**: ①`pipelines/route_a_video_pipeline.py` に純関数 `warm_up_ben2_in_pipelines(pipelines, *, log=print)->int`（各 Pipeline の `get_component("ben2_route_a_video").extractor.warm_up()`、例外は握り潰さず log し続行、成功件数返却、component 名は定数 `BEN2_COMPONENT_NAME`）を追加。②`gradio_app_sam2_ben2_route_a_for_Movie.py` に `prewarm_ben2_models()` を追加し `__main__` で `demo.launch()` の**前**に呼ぶ→BEN2 DL は gradio.live URL 印字前（SSE 接続が無い段階）にセル内で完結、リクエスト時はキャッシュ済み重みを即時ロード。`get_route_a_pipeline()`/`get_route_a_only_pipeline()` の**キャッシュ済み実インスタンス**を warm_up するため再 DL 無し（`warm_up` は `if self._model is not None: return` で冪等）。GPU/ben2 無い環境で raise しても起動継続（keep-alive フォールバック）、`--help` は argparse が prewarm 前に exit。**検証**: RED→GREEN `test_route_a_video_pipeline_wiring.py::test_warm_up_ben2_in_pipelines_{prewarms_all_extractors,continues_and_logs_on_failure}` 追加、**非 integration 263 passed / 1 skipped / 3 deselected**、RouteA app `--help` smoke 正常、get_errors=0、サブエージェント(Explore)レビュー **PASS（重大問題なし）**。**教訓**: keep-alive(通信の隙間)と prewarm(長時間 DL を SSE 外へ追い出す)は別レイヤー。レート制限され得る重い HF DL は SSE ストリーム中に走らせない。**ERR035 留意**: 実機 Colab で起動前 DL 完了後にリクエストがエラーなく完走するかはユーザー要確認（ローカル .venv は torch/sam2/BEN2/GPU 無し）。 |
| **RouteA 動画: keep-alive 固定ペイロードが Gradio に coalesce され BEN2 DL 中に再び SSE idle 切断→全出力「Error」を修正（ERR056・ERR055/ERR048 follow-up・2026-06-24）** | **✅ 完了**（既存ヘルパ内で完結・新機構なし）。依頼（逐語）: 「gradioUIの方は…エラーを吐くが…まだ処理が続いているログが出ている 対処お願い」「似たエラーがつづいているので 発生しそうなエラーも類推して確認 対処が必要なら対処」。**経緯**: ERR055 で `run_with_progress_keepalive`（warm_up をスレッドポンプ化）を入れた**後も** Error 継続（新ログ `エラーログ/エラーログ_24.md`、BEN2 `config.json 124/124`＝約380MB DL 直前で途切れ）。**真因**: ERR048 のループ版 keep-alive が効くのは frame 番号で `(fraction, description)` ペイロードが**毎回変わる**から。Gradio/gradio.live は**同一内容の進捗更新を coalesce** し実ワイヤ通信が起きない。ERR055 の `run_with_progress_keepalive` は**毎回固定ペイロード**を送っていたため coalesce され、約380MB DL の長時間ブロッキング中に idle 切断が再発（スレッドは回っていたがワイヤ上は無通信）。**対処**: `pipelines/components/video_model_components.py::run_with_progress_keepalive` のループで keep-alive を**毎回ユニーク化**—①説明文に経過秒付加（本番 interval=2.0s で各 tick +2s）②fraction を `min(base+min(tick,9)*1e-4, base+9e-4, 1.0)` で微小単調増加（バー実質不動・上限 9e-4 で stage 範囲不侵食）。併せて `join` 後 `if not thread.is_alive(): break` で work 完了後の余分通知を抑止、`clock` 引数（テスト注入用）追加。例外再送出・`progress_callback is None` 直接実行は不変。**検証**: RED→GREEN `tests/unit/test_video_pipeline_wiring.py::test_run_with_progress_keepalive_sends_unique_payload_each_tick` 追加、既存3 keep-alive テスト維持、**非 integration 261 passed / 1 skipped / 3 deselected**、RouteA movie app `--help` smoke 正常、get_errors=0。サブエージェント(Explore)レビュー **PASS（重大問題なし・スレッド安全/規約準拠・fraction 上限が stage 範囲を侵食しないことを確認）**。**教訓**: keep-alive は「スレッドが回ること」でなく「ワイヤ上で内容が毎回変わること」で初めて有効。単一ブロッキング版は明示的にペイロードを変化させる必要がある。**ERR035 留意**: 実機 Colab での SSE 維持はユーザー要確認（ローカル .venv は torch/sam2/BEN2/GPU 無し）。 |
| **ルートA: SAM2 マスクで α を底上げ合成（mask_floor_mode / screen・比較明）でちらつき抑制（2026-06-23）** | **✅ 完了**（新機能・既定 OFF で従来挙動不変）。依頼（逐語）: 「ムチャクチャ難しい動画…前景にブラー/服装が背景と同系色/ドラム金具とたたく面が背景と同系色/スティックが素早い動き」「sam2.1 のマスクが最後まで追跡できてるなら、そのマスクを BEN2 のマスクにスクリーンもしくは比較(明)で合成すればちらつきなくなるはず」。実装: SAM2 の安定 soft マスク M を最終 α の「床(floor)」として加算合成する `mask_floor_mode` を新設。`gate_alpha`(α を G 内へ絞る＝乗算/減算的)とは**逆向き**で、α を底上げ(加算的)＝BEN2 の取りこぼしを SAM2 が補い時間方向ちらつきを低減。①`pipelines/components/route_a_common.py` に純関数 `combine_alpha_with_mask(alpha, mask, mode)`（`none`=passthrough / `screen`=1-(1-a)(1-m) / `lighten`=`max`=比較明、uint8/float/bool/shape resize 対応、未知 mode は `ValueError`）を追加し `_DEFAULT_ROUTE_A_CONFIG["composite"]["mask_floor_mode"]="none"`。②`config/route_a.toml [composite]` に `mask_floor_mode = "none"` を追記。③`pipelines/components/ben2_components.py`: import 追加・`_process_union_frame`(raw soft mask を床に)・`_process_per_object_frame`(全対象 `max` の union soft mask を床に)・`run()` に `mask_floor_mode` 引数追加し両経路へ伝搬。床合成は **gate_alpha 適用後・膨張前 raw マスク使用**（SAM2 が安定追跡している領域だけ底上げ）。④`gradio_app_sam2_ben2_route_a_for_Movie.py`: `MASK_FLOOR_MODE_CHOICES`/`_normalize_mask_floor_mode`/`_mask_floor_label_from_value` と gr.Radio「SAM2マスクでα底上げ（合成）」を gate_alpha 隣に追加、`run_route_a_background_removal` に `mask_floor_mode_label` 引数・pipeline dict `"mask_floor_mode"`・run button inputs（gate_alpha と output_type の間）へ配線、`mask_floor_mode == "none"` 時に flicker hint 追加。**検証**: `tests/unit/test_route_a_common.py` に 11 件追加 **23 passed**、wiring **4 passed**、`--help` smoke 正常、`get_errors`=0。サブエージェント(Explore)レビュー **PASS（クリティカル/重要 なし、配線順序・dict キー・設計意図[raw soft mask 床]・既定 none passthrough を確認）**。RED→GREEN 実施。**ERR035 留意**: UI 配線は Playwright 実行時検証を GPU/モデル/動画要のため未実施＝「UI fixed」とは断定せず。実機 Colab での効果（ちらつき低減）はユーザー要確認。低リスク改善候補: cv2.resize の発生頻度実測、UI ラベル "lighten / 比較明" の括弧表記統一。 |

| **ルートA: 「prompt が伝わらない/overlay が MOT 経路を通らない」確認＋診断改修（2026-06-23）** | **✅ 確認完了・診断改修済み（配線 fix ではない）**。依頼: 「tracking overlay が 検出(RF-DETR)→ID追跡(ByteTrack/BoT-SORT)→SAM2.1 経路を通っていないのでは」「ちらつき箇所にポジ点→変わらず＝明らかに prompt が伝わっていない」を確認し間違いがあれば改修。**確認結果(コード追跡)**: ①現実装に **RF-DETR / ByteTrack / BoT-SORT(MOT) は存在しない**。RouteA pipeline は `VideoReader→SAM2VideoPropagator→OwnershipResolver→BEN2RouteAVideoExtractor→Writer/Overlay`、テキスト検出は GroundingDINO。仕様書 line10 の「RF-DETR→ByteTrack/BoT-SORT→SAM2.1」は**設計意図(要件定義 §10.1)で未実装**＝spec と impl の乖離(バグではない)。②**point は SAM2 へ確実に渡っている**（`video_model_components.py` L547-562: boxes経路は最近傍boxへ同梱、単一経路は points/labels 追加）。`TrackingOverlayWriter` は point を反映した SAM2 union soft mask(`frame_masks`)を描画する。③**ちらつきが point で変わらない真因**: RouteA では SAM2 マスクは「背景ブラーのゲート G」生成にのみ使われ、**gate_alpha=OFF(既定)では最終 α を BEN2 が単独生成**（BEN2 はマスク入力ポート無し＝仕様 A-2）。よって point→union マスク微修正→24px 膨張で吸収→BEN2 入力ほぼ不変→α/ちらつき不変＝「伝わっていない」ように見える。これは設計通りで配線バグではない。**改修**: `run_route_a_background_removal` の実行 status に、`points and not gate_alpha` のとき「point/SAM2 マスクは現在ブラー範囲のみに使われ最終αはBEN2単独。point で直接ちらつきを抑えるには gate_alpha=ON」＋「overlay マスクが point で変われば SAM2 へは伝達済み＝ちらつきは BEN2 側要因」の診断行を追加（[gradio_app_sam2_ben2_route_a_for_Movie.py](gradio_app_sam2_ben2_route_a_for_Movie.py#L455-L477)）。**検証**: `get_errors`=0、`--help` smoke 正常、サブエージェントレビュー **PASS**（gate_alpha スコープ内・条件論理妥当・例外なし・既存ヒント様式と整合）。**RED 省略理由**: 追加は実行 status の診断テキスト追記のみでモデル/配線挙動は不変（純粋に UI メッセージ）。**ERR035 留意**: 配線は変更していない（diagnostic 文字列のみ）。診断文は full pipeline 実行後にのみ表示されるため Playwright での文言確認は GPU/モデル/動画を要し未実施＝「UI/配線 fixed」とは記録しない。RF-DETR/ByteTrack/BoT-SORT の MOT 層導入は大規模・別タスク（スコープ外）。 |
| **ルートA UI: 選択 prompt 個別削除 + ちらつき調査（2026-06-23）** | **✅ 完了**。依頼: 「選択した BBOX と選択したポイント（ネガ/ポジ）を削除できる機能」「ポイントが伝わっていない/ポジ入力でもちらつく」を `gradio_app_sam2_ben2_route_a_for_Movie.py` と `Sam2_BEN2_RouteA_for_Movie.ipynb` で調査。実装: ①`pipelines/components/ui_helpers.py` に prompt 個別削除 API（`build_prompt_selection_choices` / `remove_selected_points` / `remove_selected_boxes`）を追加。②RouteA Gradio に「Prompt 編集（個別削除）」UI（point/bbox 選択 + 削除ボタン）を追加し、prompt 更新イベントごとに候補リストを自動同期。③実行 status に `points(pos/neg)`・`manual/union box`・`point assignment(obj別)` を出力し、標準 SAM2 かつ双方向OFF時/union時に flicker ヒントを表示。④Notebook 正本 `Sam2_BEN2_RouteA_for_Movie.py` の手順へ個別削除とちらつき対策（双方向ON/per_object）を追記し `.ipynb` 再生成。検証: `tests/unit/test_ui_helpers.py` **12 passed**（新規3テスト含む）、`gradio_app_sam2_ben2_route_a_for_Movie.py --help` 正常、`get_errors` 対象4ファイル 0件。調査結果: point/label は SAM2 へ渡っており、複数box時は最近傍 box へ割当済み。ちらつきは prompt未伝達よりも伝播方向/合成モード設定影響が大きい。RED→GREEN 実施。実機Colab挙動はユーザー要確認。 |
| **ルートA案 動画αマット Gradio UI 新規実装（2026-06-23）** | **✅ 完了**（新規ファイルのみ・既存改変なし）。依頼: ルートA案（ブラー誘導 → BEN2 再α化）を既存動画版（`gradio_app_sam2_transparent_BG_haystack_for_Movie.py` 等）を参考に**新規ファイル**として Gradio UI 実装、Colab で `.py` 起動方式、`.github/copilot-instructions.md` 準拠。確定事項: ①α生成= BEN2 base 新規導入（MIT・商用可、HF `PramaLLC/BEN2` から重み自動取得）②下地マスク M = 既存 SAM2 マスク方式を踏襲 ③複数対象を最初から対応。新規作成: `config/route_a.toml`／`pipelines/components/route_a_common.py`（純関数・12 tests）／`pipelines/components/ben2_components.py`（`BEN2Extractor` plain class + `BEN2RouteAVideoExtractor` @component・4 tests）／`pipelines/route_a_video_pipeline.py`（3 ビルダー・4 tests）／`gradio_app_sam2_ben2_route_a_for_Movie.py`（port 7862・2 タブ）／`Sam2_BEN2_RouteA_for_Movie.py`（Jupytext 正本）＋ `.ipynb` 生成／`requirements.txt` に BEN2 注記。Haystack 疎結合: BEN2(α)・SAM2(マスク) 差し替え可、ルートA合成は `BEN2RouteAVideoExtractor` に封じ込め。既存 matte dict 契約を厳密再現し `VideoWriter`/`FrameSequenceWriter`/`TrackingOverlayWriter` を再利用。**非 integration 241 passed / 1 skipped / 3 deselected**、両 `--help` smoke 正常、get_errors=0。サブエージェントレビューで M-1（`refine_foreground` を `[composite]` から誤読→ `[alpha]` に修正・`_alpha_defaults()` 追加、BEN2 のみタブの既定も config 化）を修正済み。残 Low/M-2（private API 結合・既定値二重定義）は既存 TB 版と同パターンのため現状維持。RED→GREEN 実施（純関数・配線・component）。実機 Colab GPU 動作はユーザー要確認（ローカル .venv は torch/sam2/BEN2/GPU 無しで配線・契約検証に留まる）。 |
| **追跡B案ロスト復帰方式（決定事項・2026-06-23）** | 要件定義書 §8 に「追跡B案のロスト復帰方式（決定事項）」を追記。①ロスト復帰 = id BBOX（MOT の現在位置で空間再アンカー）＋直前の良好マスク（形/中身の手がかり）で再生成し forward 再伝播（SAM2 の box 再注入／`add_new_mask` を使用）。点ではなく位置と形で作り直す。②ポジ/ネガ点の自動再注入は無し（対象移動後の BBOX 相対座標は信用できず、ズレ点は誤り注入になるため設計に入れない）。③侵入物が復帰枠に残る稀ケースはインライン手動補正（伝播途中の再クリック=複雑・副作用大）を採らず、second-pass（ユーザーがマスク適用済み結果動画を確認→ネガ点指定→単純再実行）で割り切る。MVP は「単純処理（各実行が独立1本のパイプライン）」を優先。markdownlint=0。RED 省略理由: 文書のみ・コード挙動不変。 |
| **設計討議の確定事項（点プロンプト追跡・2026-06-23）** | ユーザーとの討議で確定: SAM2.1 動画モードでは先頭フレームでポジ/ネガ点を打つ→SAM2 がマスクに反映→**条件付けフレーム（conditioning frame）のメモリは追い出されず最終フレームまで保持**され、除外意図（ネガ点）は対象マスク伝播に暗黙的に維持される。点そのものは追跡されず、運ばれるのは「対象マスクのメモリ特徴」。よって毎フレームのネガ点再投影は不要（私の当初案=過剰設計として撤回）。例外はロスト時のみで上記 §8 決定方式に従う。 |
| **双方向伝播・任意起点（決定事項・2026-06-23）** | 要件定義書 §8 に追記。①双方向伝播（`propagate_in_video(reverse=True/False)`）は**標準 SAM2 tracker（`sam2_facebook`）限定**、SAMURAI は forward-only（KF 逆走で崩壊、`supports_bidirectional` 既存・ERR050 と整合）。②双方向を活かすため**任意起点フレーム**が前提（標準 SAM2 は任意 `frame_idx`、SAMURAI は 0 固定）。③MOT（ByteTrack/BoT-SORT）は online/forward-only だが、本用途は**オフラインのバッチ処理**なので**クリップ全体を先に1回 forward 実行**して全フレーム `track_id`+BBOX テーブルを作れば、SAM2 は任意起点から双方向伝播でき前方/後方どちらの再アンカーもテーブル参照で成立（MOT に逆走能力は不要）。④本質的制約: 後方伝播は物体が映っているフレーム範囲に限られる（初出現前に id は存在しない＝因果挙動として正しい）。markdownlint=0、RED 省略=文書のみ。 |
| **MOT 内蔵復帰・tracklet stitching（決定事項・2026-06-23）** | 要件定義書 §8 に追記。復帰は**MOT 層（id 連続性）＋ SAM2 層（マスク再アンカー）の2段で有界**。①ByteTrack=`track_buffer`(~30f)+低スコア2段関連付け（短遮蔽に強い、appearance re-id 無し→長遮蔽で id スイッチ）。②BoT-SORT=GMC+（変種で）appearance Re-ID（長め遮蔽に強い、Roboflow `trackers` の Re-ID 配線は要実測確認）。③MOT 復帰失敗（gap/id スイッチ）時は second-pass で割り切り。④**tracklet stitching は任意・MVP 外**として手法を6ステップで詳述（tracklet 抽出→候補列挙[正ギャップ・非重複]→親和度[appearance Re-ID/motion 外挿/scale-aspect/temporal の重み合成]→大域最適化[Hungarian / min-cost flow / 閾値貪欲マージ]→マージ+補間 or SAM2 双方向へ橋渡し→全フレーム参照テーブル再生成）。注意: Re-ID 抽出器のライセンス商用可確認、検証コスト、貪欲マージから段階強化。markdownlint=0、RED 省略=文書のみ。 |
| **Gradio UI 点プロンプト注意＋Haystack 設計方針（追記・2026-06-23）** | **✅ 完了**（文書のみ・コード変更なし）。①要件定義書 §5.1 に「★最重要: box＋点の結合は対象1つにつき逐次で行う」を追記。要点: box+点が結合不可という UI ガイドの記述は **UI 実装側の制約でモデル本体の制約ではない**／SAM2 は box を内部で角・中心の点トークン＋内外ラベルに変換するため box も点トークン扱い／正しいフロー=同じ `obj_id` に box を `add_new_points_or_box` で投入→同 obj_id に points+labels 追加→propagate／UI 側で「box と点は排他」と誤制限しない。出典 2 件（sam2-playground PROMPT_GUIDE / SAM3 video tracking negative box）。②要件定義書 §6.1「設計方針（Haystack による疎結合アーキテクチャ）」を新設、ルートA/B 仕様書にも簡潔注記。要点: `.github/skills/haystack-pipeline/SKILL.md` に従い検出・MOT・SAM2.1・BEN2・合成を独立 Component に機能分割・単一責任・疎結合・安定 I/O 契約・遅延初期化で実装し差し替え容易性と保守性を確保。markdownlint=0（編集後 MD058/MD022/MD032 を1巡で修正済み）。RED 省略理由: 文書のみ・コード挙動不変。セルフレビュー: 既存 §5.1 の SKILL 参照方針・REMINDER #10-14 と整合、用語ブレ無し。 |
| **直近の文書タスク（2026-06-22）** | `計画書/2026-06-22_動画αマット_要件定義書.md`／`..._ルートA案_ブラー誘導_仕様書.md`／`..._ルートB案_領域ゲート_仕様書.md` を新規作成。出典は `調査/2026-06-22_トラッキング許可方法調査とBEN2採用計画.md`。要点: ①「追跡軸（毎フレーム検出 vs propagation+再追跡）」と「合成軸（ブラー誘導 vs 領域ゲート刈り）」は直交する別レイヤーで命名衝突を明示分離。②MVP 受け入れ基準は主観評価一次＋2分以内。③ブラー誘導は未検証の設計仮説として隔離。④ライセンス確定表（RF-DETR/ByteTrack/SAM2.1=Apache、BEN2=MIT、MatAnyone/YOLO/SAM3 は除外理由付き）。RED テスト省略理由: 文書のみでコード挙動変更なし（markdownlint MD040 を1件修正済み、get_errors=0）。セルフレビュー実施（下記）。 |
| **最終更新日** | 2026-06-26（ERR061: Windows 非ASCII[日本語]パスで `cv2.imwrite`/`cv2.imread` が無言失敗→overlay PNG 保存エラーを `imwrite_unicode`/`imread_unicode`[imencode/imdecode+ファイルI/O]で根治。検証: 新規3 passed・対象2ファイル32 passed・get_errors=0・レビュー APPROVED。ユーザーは Drive→ローカル ASCII パス移行を後日自分で実施予定。先行: ERR060 BEN2 `.pth` 解決。） |
---

## 直近完了タスク: SAMURAI + 複数オブジェクトで伝搬が `Boolean value of Tensor ... ambiguous`（ERR051）（2026-06-22, high）
- **依頼（ユーザー命令、逐語）**: 「Sam2_Transparent_Background_Haystack_for_Movie.ipynb」「エラーログ\エラーログ_21.md」「エラーが出た　対処お願いします」。改修フローは `.github\copilot-instructions.md` に従う。
- **真因（ERR051）**: SAMURAI fork の `_forward_sam_heads`（`samurai/sam2/sam2/modeling/sam2_base.py:451`）が **単一オブジェクト(B=1)前提**。`best_iou_inds = torch.argmax(ious, dim=-1)` は形状 `[B]`、`ious[0][best_iou_inds]` は B≥2 で多要素テンソルになり `if tensor > threshold` の boolean 評価が曖昧 → `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。SAMURAI は KF 状態（`kf_mean`/`stable_frames`）をモデルインスタンスで共有し複数オブジェクト同時追跡を想定しない。本リポジトリの `boxes`→obj_id 1..N 複合対象 union 配線と組み合わさると B≥2 に到達。
- **変更点（`samurai/`・`segment-anything/` は不変、標準 SAM2 経路も不変）**:
  - `pipelines/components/video_model_components.py`: `SAM2VideoPropagator.__init__` に `single_object_only: bool = False`（既定 False=後方互換）を追加・保持。`run` の冒頭バリデーション直後（`warm_up` より前＝fail-fast）で `requested_object_count = len(boxes) if boxes else 1` を計算し、`self.single_object_only and requested_object_count > 1` のとき actionable な `ValueError`（「単一オブジェクト専用。box を 1 つに減らすか標準 SAM2 へ」）を raise。
  - `config/inference_models.toml`: 全 tracker entry に `single_object_only` を追加。SAMURAI（`samurai_hiera_l`/`samurai_hiera_b_plus`）=true、標準 SAM2（`sam2_hiera_l`/`sam2_hiera_b_plus`）=false。
  - `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`: `get_video_pipeline` が `tracker_entry.get("single_object_only", False)` を propagator へ渡す。冒頭 SAMURAI 推奨設定 Markdown 表に「対象オブジェクト数＝1 個のみ（複数は標準 SAM2 へ）」行を追加。
  - `Sam2_Transparent_Background_Haystack_for_Movie.py`（notebook 正本）: 先頭セルの表に同じ行を追加。`.ipynb` を jupytext 再生成。
- **テスト（RED→GREEN）**: `tests/unit/test_video_pipeline_wiring.py`（4 件追加）: 複数 box で `ValueError` かつ `warm_up` 未到達（`_video_predictor is None`）/ 単一 box は `single_object_only=True` でも正常（後方互換）/ registry の `single_object_only`（SAMURAI=True・標準=False）/ app の配線文字列。**非 integration 221 passed / 1 skipped / 3 deselected**、movie `--help` smoke 正常、get_errors=0（既存 `import torch` 解決不可のみ・本変更起因でない）。
- **レビュー**: サブエージェント（Explore）**APPROVE（指摘なし）**。正確性（B≥2 を確実に伝搬前に止める・`requested_object_count` の数え方が points のみ/box 単数/boxes 複数で妥当）、後方互換（既定 False・標準 SAM2 不変）、config 駆動の一貫性（`supports_bidirectional`/`autocast_dtype` と同パターン）、fail-fast 位置（warm_up 前）を PASS 判定。任意改善として Playwright で UI 実行時のエラー表示検証を提案（重大度低）。
- **未検証メモ**: 実機 Colab での単一 obj 正常動作・複数 obj の明示エラー表示は**ユーザー要確認**（ローカル .venv は torch/sam2/GPU 無しのため fail-fast 配線・契約検証に留まる）。

---

## 直近完了タスク: SAMURAI 動画伝搬 VRAM 枯渇 follow-up（autocast fp16 / 双方向自動 OFF / 起点先頭 / 推奨設定明記）（ERR050）（2026-06-22, high）
- **依頼（ユーザー命令、逐語）**: 「123すべてやろう」「あとsamurai用の設定をgradioやipynbにもがっつり書いておいてくれ」「どちらも目立つ一番上や最初のセルに書いておいてください」。改修フローは `.github\copilot-instructions.md` に従う。
- **背景（ERR049 の follow-up）**: ERR049 の CPU offload 後も SAMURAI 伝搬の VRAM が逼迫しうる。SAMURAI 本家 `scripts/main_inference.py` は autocast fp16・起点 0・forward-only 前提だが、本リポジトリ propagator は autocast 未適用（fp32）で、UI は SAMURAI でも双方向 ON を許容していた（逆走は KF 破綻 + per-frame memory 2 倍 = ERR049 stall 誘発）。
- **変更点（`samurai/`・`segment-anything/` は不変、標準 SAM2 経路の数値挙動も不変）**:
  - `pipelines/components/video_model_components.py`: `import contextlib` 追加。`SAM2VideoPropagator.__init__` に `autocast_dtype: str | None = "float16"`、helper `_autocast_context(torch)`（`device=="cuda"` かつ dtype が `None/""/"none"` 以外のとき `torch.autocast("cuda", dtype=float16|bfloat16)`、他は `contextlib.nullcontext()`）。`run` を `with torch.inference_mode(), self._autocast_context(torch):` に変更。
  - `config/inference_models.toml`: 全 tracker entry に `autocast_dtype` と `supports_bidirectional` を追加。SAMURAI（`samurai_hiera_l`/`samurai_hiera_b_plus`）= `autocast_dtype="float16"` / `supports_bidirectional=false`。標準 SAM2（`sam2_hiera_l`/`sam2_hiera_b_plus`）= `autocast_dtype="none"`（fp32 維持）/ `supports_bidirectional=true`。
  - `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`: `get_video_pipeline` が `tracker_entry.get("autocast_dtype", "none")` を propagator に渡す。`update_bidirectional_for_tracker`（registry の `supports_bidirectional` を見て SAMURAI は `gr.update(value=False, interactive=False)`、標準は `interactive=True`、未知 id は `KeyError` 捕捉で安全側 `interactive=True`）を追加し `tracker_model.change(..., outputs=[bidirectional])` で配線。タイトル直後の最上部に SAMURAI 推奨設定 Markdown を追加。`prompt_frame_idx`（既定 0 のまま）と双方向 checkbox の info に SAMURAI forward-only / 自動 OFF を明記。tracker 既定値は `sam2_hiera_l` のまま。
  - `Sam2_Transparent_Background_Haystack_for_Movie.py`（notebook 正本）: 先頭の markdown セルに SAMURAI 推奨設定の表を追加。`.ipynb` を jupytext で再生成。
- **テスト（RED→GREEN）**: `tests/unit/test_video_pipeline_wiring.py`（10 件追加）: autocast 既定 float16 / CPU は nullcontext / `none` で cuda でも無効 / cuda+float16 で `torch.autocast("cuda", dtype=float16)` / registry の autocast・supports_bidirectional フラグ（標準 SAM2=autocast none・双方向 true、SAMURAI=float16・双方向 false） / app の autocast 配線文字列 / SAMURAI 双方向自動 OFF 挙動（module import で `update_bidirectional_for_tracker` を直接検証） / change 配線 / SAMURAI 推奨設定 doc が app・notebook 双方に存在。**非 integration 217 passed / 1 skipped / 3 deselected**、movie `--help` smoke 正常、get_errors=0（既存 `import torch` 解決不可のみ・本変更起因でない）。
- **レビュー**: サブエージェント（Explore）**APPROVE**。唯一の実質指摘「標準 SAM2 entry の fp16 autocast 化は実績ある経路の数値挙動変更」を反映し、標準 SAM2 を `autocast_dtype="none"`（fp32 維持）に修正、app の get-default も `"none"` に、回帰テストで標準 SAM2=none を固定。自己レビュー: autocast が CPU で発火しない / config 駆動でハードコード無 / try-except は明示フォールバック（握り潰し無）/ `samurai/`・`segment-anything/` 不変 / TOML 構文健全 / 標準経路保全 を確認。
- **未検証メモ**: 実機 Colab T4 での stall 解消・出力品質は**ユーザー要確認**（ローカル .venv は torch/sam2/GPU 無しのため配線・契約検証に留まり、伝搬の実挙動・VRAM 実測・fp16 出力差は再現不能）。

---

## 直近完了タスク: SAMURAI 動画伝搬の GPU メモリ枯渇 stall（`propagate 1/N` 凍結）を config 駆動の CPU offload で対処（ERR049）（2026-06-22, high）
- **依頼（ユーザー命令、逐語）**: 「エラーログ\エラーログ_20.md」「同じところでとまってるね」「samuraiのチェックポイントを使ってるところだけ違う」。改修フローは `.github\copilot-instructions.md` に従う。
- **真因（ERR049）**: SAMURAI は motion-aware memory（Kalman filter 状態）で GPU 常駐メモリが標準 SAM2 より大きい。T4(Turing 7.5) は非 Ampere で Flash Attention 無効＝attention が重い。双方向伝搬は forward+reverse の 2 pass 分 per-frame memory が積み上がり、伝搬の最初の重い frame で VRAM 枯渇 → CUDA アロケータ待ち/スラッシングで stall（OOM 例外を投げ切らず stdout が `propagate 1/67` で凍結）。ERR048（SSE 切断・処理は継続）と異なり **stdout 自体が進まない hang/stall**。`samurai/` の `init_state`/`propagate_in_video` は bounded ループで無限ループではない（コード確認済み）。
- **変更点（`samurai/`・`segment-anything/` は不変、標準 SAM2 経路も不変）**:
  - `config/inference_models.toml`: SAMURAI tracker entry（`samurai_hiera_l` / `samurai_hiera_b_plus`）にのみ `offload_video_to_cpu = true` / `offload_state_to_cpu = true` を追加（ERR049 コメント付き）。標準 SAM2 entry（`sam2_hiera_l` / `sam2_hiera_b_plus`）は無変更。
  - `pipelines/components/video_model_components.py`: `SAM2VideoPropagator.__init__` に `offload_video_to_cpu: bool = False` / `offload_state_to_cpu: bool = False`（既定 False=現状維持）を追加、`self.offload_*` に保持。`run` の `init_state(video_path=..., offload_video_to_cpu=self.offload_video_to_cpu, offload_state_to_cpu=self.offload_state_to_cpu)` へ転送。
  - `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`: `get_video_pipeline` が `tracker_entry.get("offload_video_to_cpu", False)` / `offload_state_to_cpu` を読んで propagator に渡す（非 SAMURAI entry は既定 False で無影響）。
  - SAMURAI fork の `init_state` は両 kwarg を受け取る（`samurai/sam2/sam2/sam2_video_predictor.py`、レビューで署名確認）。ハードコードせず registry 経由＝差し替え容易。
- **テスト（RED→GREEN）**: `tests/unit/test_video_pipeline_wiring.py`（4 件追加）: 既定で offload 無効 / offload 有効時に `init_state` へ kwargs が届く / SAMURAI registry entry のみ offload 有効・標準 SAM2 は無効 / `get_video_pipeline` が offload 設定を読む。fake predictor 2 つの `init_state` を `**kwargs` 受け入れに更新（実装が offload を渡すため）。**非 integration 208 passed / 1 skipped / 3 deselected**、movie `--help` smoke 正常。
- **レビュー**: サブエージェント（Explore）**APPROVE**（SAMURAI `init_state` 署名が `offload_video_to_cpu=False, offload_state_to_cpu=False` を含むことを確認＝Colab で TypeError なし、標準 SAM2 経路保全・registry passthrough・Hard Rules 準拠を PASS 判定）。自己レビュー: get_errors=0（既存 `import torch` の解決不可のみ・本変更起因でない）、既定 False で後方互換、config 駆動でハードコード無、torch.load/try-except pass/samurai 直接変更が無いことを確認。
- **未検証メモ**: 実機 Colab T4 での stall 解消は**ユーザー要確認**（ローカル .venv は torch/sam2/GPU 無しで freeze 再現不能、配線のみ検証）。offload で解消しない場合の follow-up 候補（本タスク未実施）: ① SAMURAI KF 状態が cache モデルに残り pass/run 間でリセットされない（正確性懸念）, ② 双方向+SAMURAI は reverse が forward の KF 状態を流用する点が意味的に疑問。

---

## 直近完了タスク: 動画処理中の SSE idle 切断（UI 全出力 Error / Connection errored out）を時間ベース keep-alive で対処（ERR048）（2026-06-22, high）
- **依頼（ユーザー命令、逐語）**: 「添付画像にあるようにエラーが出た / ログにはエラーが出ていない / 処理はつづいているようだ / エラーコネクションというメッセージが出た気がする / 調査後 対処お願い / Sam2_Transparent_Background_Haystack_for_Movie.ipynb / エラーログ\エラーログ_19.md / 改修フローは .github\copilot-instructions.md を参照」。
- **真因（ERR048）**: Colab/gradio.live の共有トンネルは event SSE に無通信が続くと idle 切断する。ブラウザは pending 中の全出力を "Error" にするが、サーバ側 Python は処理を継続（stdout に例外なし、`propagate in video` が継続）。進捗通知が frame 数ベース間引き（伝搬ループ `propagated_count % 10`）だったため、実測 3.81s/it で**最大約38秒の無通信ギャップ**が生じ SSE が切れていた。
- **変更点（`pipelines/components/video_model_components.py`、`samurai/` は不変）**:
  - `import time` 追加、モジュール定数 `_PROGRESS_KEEPALIVE_SEC = 2.0` 追加。
  - 新クラス `_ProgressKeepAlive`（`progress_callback`, `stage`, `min_interval_sec`, `clock` 注入可）。`maybe(index, total, fraction, description, force=False)` が「境界 frame（`index<=0 or index+1>=total`）」「`force`」「前回送信から `min_interval` 経過」のいずれかで `_notify_progress` を呼ぶ。
  - 配線: `SAM2VideoPropagator.run` の frame 準備ループ・伝搬ループ、`TransparentBGVideoExtractor.run`(streaming) の tb ループ、tracking overlay ループの進捗通知を `_ProgressKeepAlive.maybe` に置換。frame 速度によらず無通信ギャップを最大 2.0s に抑える。legacy `VideoWriter._write_*`（cv2・高速）は `% 20` 維持で対象外。
  - notebook 変更不要（コードのみ・依存追加なし）。
- **テスト（RED→GREEN）**: `tests/unit/test_progress_keepalive.py`（5 件、FakeClock 注入）。旧 frame 数ベースなら落ちる「非境界 frame でも経過時間で発火」回帰テストを含む。**非 integration 204 passed / 1 skipped / 3 deselected**、movie `--help` smoke 正常。
- **レビュー**: サブエージェント（Explore）BLOCKER なし。指摘の「伝搬ループ最終 frame が `gathered_any=False` だと進捗漏れ」は**旧実装と同等挙動で接続維持（keep-alive）は達成済み**のため任意改善として見送り。自己レビュー: コンパイル（get_errors=0）・index 計算（prep=frame_index / 伝搬=propagated_count-1 / tb・overlay=local_index）・fraction 範囲(0..1)・Hard Rules（torch.load/try-except pass/samurai・segment-anything 不変/keepalive 定数は内部チューニング値）を確認。
- **未検証メモ**: 実機 Colab での長時間処理中の接続維持はユーザー要確認（ローカル .venv に torch/sam2/GPU/gradio.live なし）。

---

## 直近完了タスク: RGBA(透過)動画の cv2 書き出し失敗を imageio+ffmpeg で恒久対処（ERR047）（2026-06-21, high）
- **依頼（ユーザー命令、逐語）**: 「背景除去作業中にエラー / エラーログ\エラーログ_18.md / Sam2_Transparent_Background_Haystack_for_Movie.ipynb」「処理フローは .github\copilot-instructions.md に従うこと」「サブエージェントのレビューに加えて 自己レビューを一度いれること」。
- **真因（ERR047）**: `cv2.VideoWriter` は 4ch(RGBA/BGRA) を書けず（`isColor` は 1ch/3ch のみ）、FFmpeg が "expected 3 channels but got 4" で**毎 frame skip**→透過 webm が空。OpenCV の VP9/webm 経路は `VP90` fourcc を webm で拒否。旧 `_select_rgba_codec` は `cv2.VideoWriter.isOpened()` だけで判定し、VP90 では **open 成功（偽陽性）**のため失敗を検知できなかった。OpenCV は本質的に alpha 動画を書けない。
- **変更点（`pipelines/components/video_model_components.py`、`samurai/` は不変）**:
  - `_require_imageio()`（`imageio.v2`+`imageio_ffmpeg` を遅延 import、無ければ握り潰さず連番(PNG)を促す `RuntimeError`）、`_RgbaCodecSpec`（NamedTuple）、`_ImageioAlphaVideoWriter`（`append_data` で RGB order RGBA を書く）を追加。
  - `_select_rgba_codec` を cv2 fourcc → alpha 対応 imageio spec 返却に再設計: `webm_vp9`=`libvpx-vp9`/`yuva420p`/`-auto-alt-ref 0`/`macro_block_size=2`、`mov_png`=`png`/`rgba`/`macro_block_size=1`。
  - 配線: streaming `TransparentBGVideoExtractor.run` の rgba_stream を `_ImageioAlphaVideoWriter` に、legacy `VideoWriter.run` を `_write_rgba_video` に。alpha(1ch)/preview(3ch) は cv2 のまま（問題なし）。偽陽性の `_test_codec`/`_codec_cache` を削除。
  - `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`: RGBA codec radio info 2 箇所から誤解を招く「自動で他方式に fallback」を削除し「imageio+ffmpeg で alpha 保持・書けない環境は連番(PNG)が確実」へ更新。
  - 依存: Cell 1 は既に `imageio[ffmpeg]` を install 済みのため **notebook 変更不要**（`.ipynb` 再生成も不要）。
- **テスト（RED→GREEN）**: `tests/unit/test_movie_runtime_bugs.py`（Bug D）に 3 テスト追加（spec が alpha 対応 imageio パラメータを返す / imageio 欠如時に明確エラー / 動画モードの RGBA stream が 4ch を imageio へ append し cv2 へ渡さない）。RED 3 失敗 → 実装 → **非 integration 199 passed / 1 skipped / 3 deselected**、movie `--help` smoke 正常。
- **レビュー**: サブエージェント（Explore）実施。`macro_block_size` を「無効パラメータ」とする 2 件の BLOCKER 指摘は **公式 imageio ドキュメントで反証**（`macro_block_size` は正規の書き込みパラメータで奇数解像度を偶数へ自動スケール＝yuva420p 要件を満たす）→ 偽陽性として却下。自己レビュー: RGB order 保持・fail-loud・hard rules（torch.load/try-except pass/samurai 不変/ハードコード無）・dead code 削除を確認。
- **未検証メモ**: 真の alpha-webm 実エンコードは Colab（imageio+ffmpeg+GPU）でのみ確認可能。ローカルは mock + 全テスト + smoke の論理検証に留まる（.venv に imageio/ffmpeg/torch なし）。

---

## 直近完了タスク: 動画 任意フレーム実行（範囲外バリデーション撤廃）+ loguru 依存対処（ERR046）（2026-06-21, high）
- **依頼（ユーザー命令、逐語）**: 「Error: 'プロンプト起点フレーム位置 63 は処理フレーム数 30 以上です。…' この処理はなくしてください … 任意のフレームで実行できるようにしなさい」「つぎに エラーログ\エラーログ_17.md に対処をねがい」「処理フローは .github\copilot-instructions.md に従うこと」「サブエージェントのレビューに加えて 自己レビューを一度いれること」。
- **Task1 真因**: `prompt_frame_idx` はサンプリング後シーケンスの index で、読み込み窓は `max_frames` で頭打ち。GUI の fail-fast（`prompt_frame_idx >= processed_frames` で gr.Error）に加え、`SAM2VideoPropagator.run` も `prompt_frame_idx >= len(frames)` で ValueError を出すため、GUI 検証を消すだけでは propagator で落ちる。
- **Task1 変更点（`gradio_app_sam2_transparent_BG_haystack_for_Movie.py`）**:
  - 新ヘルパ `_effective_read_frames(max_frames, prompt_frame_idx) -> max(1, max_frames, prompt_frame_idx+1)` を追加。
  - `run_video_background_removal`: 範囲外 fail-fast ブロックを撤廃し、`effective_max_frames` を算出 → `video_reader` の `max_frames` へ渡す（prompt フレームを必ず窓に含め propagator の ValueError を回避）。進捗文言は `processed_frames`（effective 反映）を参照。
  - prompt_frame_idx スライダー info から「最大処理フレーム数より小さい値にする」を削除し「超える値でも自動読込（範囲外エラーなし。その分読込時間が増える）」へ更新。
  - `run_tb_only_background_removal`（prompt_frame_idx 不使用）は未変更。`samurai/` は読み取りのみ。
- **Task2 真因（ERR046）**: SAMURAI fork の `sam2/modeling/sam2_base.py` が `from loguru import logger` するが fork の `setup.py` の `install_requires` に `loguru` 無し → fork を install しても入らず `build_sam2_video_predictor` で `ModuleNotFoundError`。
- **Task2 変更点**: `Sam2_Transparent_Background_Haystack_for_Movie.py`（正本）Cell 1 に `!{sys.executable} -m pip install loguru` を追加（理由コメント ERR046）。stale な「editable 導入」コメントを ERR045 反映へ修正。`.ipynb` 再生成。`samurai/` は読み取りのみ。
- **テスト（RED→GREEN）**: `tests/unit/test_movie_runtime_bugs.py` の旧 `test_run_video_validates_prompt_frame_idx_before_pipeline` を撤廃し、`_effective_read_frames` 単体テスト + 「範囲外エラーを出さない」回帰テストへ置換。`tests/unit/test_jupytext_notebooks.py` に `pip install loguru` 検証テストを追加。**非 integration 196 passed / 1 skipped / 3 deselected**、movie `--help` smoke 正常。
- **レビュー**: サブエージェント（Explore）合格（指摘の「窓拡張時の読込時間増」を info に追記）+ 自己レビュー（編集領域の整合確認、torch.load/try-except pass/ハードコード/samurai 直接変更が無いことを確認）。
- **未検証メモ**: 実機 Colab での loguru install 後の伝搬動作・任意フレーム実行はユーザー要確認（.venv に torch/sam2/loguru なし）。

---

## 直近完了タスク: SAMURAI fork の Colab install 失敗を恒久対処（ERR045）（2026-06-20, high）
- **依頼（ユーザー命令、逐語）**: 「Sam2_Transparent_Background_Haystack_for_Movie.ipynb / エラー対処お願い / エラーログ\エラーログ_16.md / .github\copilot-instructions.md / これにしたがって改修しなさい / ERROR_LOG.md ＜＝再発が無いように書き込みなさい」。
- **真因（2 つの独立問題）**: (1) Drive(FUSE) 上の `pip install -e samurai/sam2`（editable）が `.pth`/egg-info 書き込みに失敗し sam2 が入らない（`!pip` は沈黙失敗 → Cell 2.5 で cryptic な ModuleNotFoundError）。(2) `build_sam.py` が `fill_hole_area=8` を強制し伝搬で `sam2._C`（connected_components、CPU fallback 無し）が必須だが、通常の `pip install`（build isolation 有効）は torch が見えず `_C` がビルドされない。
- **変更点**:
  - `Sam2_Transparent_Background_Haystack_for_Movie.py`（正本）: Cell 2 の install を `!{sys.executable} -m pip install --no-build-isolation "{SAMURAI_SAM2_DIR_POSIX}"`（非 editable + no-build-isolation）へ変更。非 editable で Drive 書き込み問題回避（configs は MANIFEST で wheel 同梱）、no-build-isolation で torch を見せ nvcc で `_C` ビルド。install 直後に `importlib.invalidate_caches()` + `find_spec("sam2")` の fail-loud 検証を追加。Cell 2.5 診断メッセージを非 editable 参照へ更新。`.ipynb` 再生成。
  - コード変更なし（`_require_samurai_capable_sam2` は ERR041 のまま有効。非 editable install でも installed sam2 に configs/samurai が同梱されるため検査は pass）。`samurai/` は読み取りのみ。
- **テスト**: notebook/env のみの変更（GPU 非依存テスト対象外）。回帰確認として非 integration **194 passed / 1 skipped / 3 deselected**、movie `--help` smoke 正常。RED テストは「.venv に torch/sam2/nvcc 無くインストール挙動を再現不可」のため省略（fail-loud 検証は実機 Colab でのみ意味を持つ）。
- **未検証メモ**: 実機 Colab での `_C` ビルド成否・SAMURAI 追跡動作はユーザー要確認（.venv に torch/sam2/CUDA toolkit なし）。

---

## 直近完了タスク: SAMURAI checkpoint で DINO を動かす（SAMURAI Hydra 再発 / ERR041）（2026-06-19, high）
- **依頼（ユーザー命令、逐語）**: 「samuraiのチェックポイントをつかって DINO」（添付 `エラーログ/エラーログ_14.md`）。
- **真因**: Colab notebook が facebook 版 sam2（`pip install git+facebookresearch/sam2.git`）を導入。SAMURAI は (1) `configs/samurai/` が無い → MissingConfigException、(2) `SAM2Base` が `samurai_mode` 等を受け付けない → 解決できても TypeError。ERR038 の Hydra 検索パス append は症状対処に過ぎず不十分。SAMURAI は訓練不要で標準 `sam2.1_hiera_large.pt` を再利用（追加 DL 不要）。
- **変更点**:
  - `Sam2_Transparent_Background_Haystack_for_Movie.py`（正本）: Cell 1 から facebook sam2 install 削除、Cell 2（Drive マウント後）で同梱 fork を editable 導入 `pip install -e "{PROJECT_ROOT}/samurai/sam2"`（`.as_posix()` でパス安全化）。fork は `configs/sam2.1/` と `configs/samurai/` 両方を含み facebook/SAMURAI 両 tracker を 1 つの sam2 で賄う。Cell 2.5 診断メッセージを fork 参照へ更新。`.ipynb` 再生成。
  - `pipelines/components/video_model_components.py`: `_require_samurai_capable_sam2(config_name)` を新設し `warm_up()` の build 直前で呼ぶ。samurai config 時のみ installed sam2 の `configs/samurai` を検査、無ければ `pip install -e samurai/sam2` を促す actionable な RuntimeError（非 samurai は `import sam2` せず no-op）。`samurai/` は読み取りのみ。
- **テスト（RED→GREEN）**: `tests/unit/test_video_pipeline_wiring.py` に `_install_fake_sam2` ヘルパ + 3 テスト（facebook で raise / fork で pass / 非 samurai で no-op、`sys.modules` の fake sam2 で GPU 非依存）。**非 integration 194 passed / 1 skipped / 3 deselected**。`--help` smoke 正常。
- **レビュー**: サブエージェント（Explore）実施。ブロッカー指摘（Windows パス安全性）を `.as_posix()` で対処。sys.modules キャッシュ懸念は「新カーネルで上から実行」する通常フローで Cell 2.5 が初回 import のため非該当（再実行時は診断メッセージで再起動を案内）。
- **未検証メモ**: 実機 Colab での SAMURAI 追跡品質・GPU 動作はユーザー要確認（.venv に torch/sam2 なし）。

---

## 直近完了タスク: 意味追跡ドリフト改善・改善3（mask_guard 手動調整 UI）（2026-06-19, high）
- **依頼（ユーザー命令、逐語）**: 「1と2と3の計画表を計画書フォルダに作成」「3はフェザーとdilateのチェックボタンと範囲設定を手動で入れるようにする　デフォルトはオフ」「書き込みおねがい その後 .github\copilot-instructions.md に従い実装お願い」。
- **目的**: 動画背景除去（SAM2 追跡 + tb）経路で mask guard（SAM2 mask の外側ゲート）の feather/dilate をユーザーが手動調整できるようにする。既定 OFF のとき従来挙動を完全維持。
- **変更点**:
  - `pipelines/components/video_model_components.py`: `TransparentBGVideoExtractor.run()` と `_run_per_object_frame()` に `mask_guard_dilate: int = 21` を追加し、union/per_object 両経路から `self.extractor.run(..., mask_guard_dilate=...)` へ配線（既存 `TransparentBGExtractor.run` は既に受理）。docstring 更新。
  - `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`:
    - `run_video_background_removal()` に `mask_guard_enabled / mask_guard_feather_ui / mask_guard_dilate_ui` を追加（`crop_padding` の後、`overlay_enabled` の前）。OFF 時は config feather + dilate=21（従来）を、ON 時は UI 値を `transparent_bg_video` dict に渡す。
    - Advanced 内に `mask_guard_enabled = gr.Checkbox(value=False)`、`mask_guard_feather = gr.Slider(0,64,value=0,step=1)`、`mask_guard_dilate = gr.Slider(1,81,value=21,step=2)` を追加（各 info 付き）。
    - `run_btn.click` の inputs リストに3部品を signature と同順で追加。
- **テスト（RED→GREEN）**: `tests/unit/test_video_per_object_frame.py` に per_object/union 配線テスト2件、`tests/unit/test_video_pipeline_wiring.py` に UI source テスト1件を追加。`tests/unit/test_movie_runtime_bugs.py` 既存テストに新3引数を追加。**非 integration 191 passed / 1 skipped / 3 deselected**。`--help` smoke 正常。
- **レビュー**: サブエージェント（Explore）APPROVE（Blocker なし）。inputs 順序一致・OFF 後方互換・規約遵守を確認済み。
- **UI 実行時検証（ERR035）**: Movie アプリを起動（port 7866）し Playwright で「Mask guard を手動調整 / Mask guard feather / Mask guard dilate」の3部品描画と checkbox 既定 OFF（unchecked）を確認。実モデル出力品質は torch/checkpoints の GPU 環境でユーザー要確認。
---

## 直近完了タスク: 背景除去のみ (tb only) 動画経路を追加（2026-06-18, high）
- **依頼（ユーザー命令、逐語）**: 「Transparent bgオンリーの経路作っておく … 背景除去モデルのみの運用で事足りることも多分ある グリーンバックとかさ … 除去モデルのみの経路作ったほうがいいよな」。
- **目的**: SAM2/GroundingDINO を使わず transparent-background モデルのみで動画を全画面処理する軽量経路。グリーンバック・単一 salient 対象など追跡不要なケース向け。
- **変更点**:
  - `pipelines/sam2_tb_video_pipeline.py`: `build_tb_only_video_pipeline()` を新設。配線は `video_reader → transparent_bg_video → (video_writer, frame_sequence_writer)`。masks ソケット未接続のため `mask=None` → 全フレーム全画面 tb（crop/guard/所有権合成/overlay なし）。既存 `build_sam2_tb_video_pipeline` は不変。
  - `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`:
    - 新コールバック `run_tb_only_background_removal`（prompt/SAM2 入力なし、6-tuple 返却、`gr.Error` で通知）。
    - `get_tb_only_pipeline()` lazy 初期化（global `_TB_ONLY_PIPELINE`）。
    - UI を `with gr.Tabs():` で「SAM2 追跡 + 背景除去」「背景除去のみ (tb only)」の2タブ化。tb-only タブのコンポーネントは全て `tb_only_*` プレフィックス（SAM2 タブと衝突なし）。tb-only には crop_padding / overlay なし。
- **テスト（RED→GREEN）**: `tests/unit/test_video_pipeline_wiring.py` に `test_tb_only_video_pipeline_builds_without_sam2`（パイプライン構造）と `test_movie_app_exposes_tb_only_tab_and_callback`（UI source 文字列）を追加。**非 integration 188 passed / 1 skipped / 3 deselected**。`--help` smoke 正常（Blocks 構築成功）。
- **レビュー**: サブエージェント（Explore）APPROVE（Blocker なし）。crop_padding/mask_guard_feather 省略は mask=None で未使用のため機能影響なし、意図コメント付与済み。
- **未検証メモ**: 実機での tb-only 出力品質は GPU 環境で要確認（.venv に torch/transparent-background なし）。Playwright によるタブ切替実行時検証は未実施（必要なら別途）。

---

## 過去タスク: 合成方式を比較明（lighten / max）に変更（2026-06-18, high）
- **依頼（ユーザー命令、逐語）**: 「乗算じゃダメだ 手前のオブジェクトを重ねたとき マスクに黒があれば背後のオブジェクトも黒になる たとえ残しておきたい人物でも黒で塗りつぶされる 四の五の言わず比較明でやれ」→「合成方法 比較明にしなさい」。
- **対象**: `pipelines/components/video_common.py` の `composite_alpha_by_ownership`（per_object モードの最終アルファ合成）。
- **変更**: 旧 所有権加重和 `alpha_final = Σ_o ownership_o × alpha_o` → 比較明 `alpha_final = max_o alpha_o`（画素ごと max）。`ownership` 引数は合成に乗じず、前景チャネル数 N と per_object_alphas 数の一致検証にのみ使用。docstring 全面刷新。
- **理由**: 加重和は対象が重なる画素で手前対象のアルファが 0（黒）のとき背後の残したい対象まで減衰し黒く潰す。max ならどれか 1 対象でも前景なら最終アルファに残り黒抜けしない。
- **テスト（RED→GREEN）**: `tests/unit/test_per_object_composite.py` を max 仕様へ書換（`test_composite_lighten_keeps_object_even_when_other_is_zero` 等 6 件）。RED 3 失敗→実装後 GREEN。`test_video_per_object_frame.py` も pass。**非 integration 187 passed / 1 skipped / 3 deselected**。
- **影響範囲**: per_object 経路のみ（既定 video_matte_mode）。union 経路（model_components.py L687 `full_alpha * guard`）は別機構（形状ゲート）で、本変更の対象外。ユーザーの不満は「重なる対象の黒抜け」＝per_object 合成なので対象は本関数に限定。
- **レビュー**: サブエージェント（Explore）APPROVE。指摘は REFERENCE.md L415 の旧式（Σ加重和）記載の陳腐化のみ → **修正済み**。
- **重要な切り分け（再掲・ハルシネーション防止）**: 比較明化は合成品質の正当な改善だが、**実機の壊れた出力（overlay 背景青／alpha 輪郭のみ）の根本原因ではない**。根本原因は「全画面 box のみ・text/point prompt なし」で SAM2 に人物位置の手掛かりが無く背景形状 mask を返したこと（prompt/入力問題、SAM2 ラッパーのバグではない／反転バグも否定済み）。実機修正には適切な prompt（DINO テキスト検出 or 人物 box）が必要。

---

## 過去タスク: Phase 1 — guard 半透明化（S2）修正（2026-06-18, high）

- **症状（ユーザー報告 S2）**: 人体が得意なはずの transparent-background の人物アルファが半透明。トラッキング領域全体の信頼度をアルファに変えている懸念（本来は境界の信頼度であるべき）。
- **根本原因**: `model_components.py TransparentBGExtractor.run` の `full_alpha = full_alpha * guard`。union モードで `OwnershipResolver` が frame_masks を「前景 soft = 1 − 背景所有権」という領域全体の連続確率に差し替え、その float mask が `soft_probability_guard` 経由で内部も 1.0 未満になり tb の人物アルファ内部を減衰。
- **対処**: guard 分岐を「float/binary 型」ではなく `mask_guard_feather` の有無で分岐。既定（feather=0）は float/binary 問わず `dilate_binary_mask`（内部 1.0・外部 0 の二値ゲート、float は 0.5 閾値で二値化）。feather>0 のときのみ soft guard をオプトイン。guard は形状外ゲートに徹し内部を削らない。
- **テスト**: `tests/unit/test_transparent_bg_mask_guard.py` に `test_float_soft_mask_guard_keeps_interior_alpha_unscaled` / `test_float_soft_mask_guard_feather_opt_in_softens_edge` を追加。非 integration 180 passed/1 skipped。レビュー APPROVE。
- **次フェーズ未着手メモ**: Phase 2〜4 は計測優先（ハルシネーション禁止）。config 既定は Phase 2 が GREEN になるまで union を維持。

---

## 過去タスク（参考・前セッション、※ per_object 既定化は revert 済み）: 実動画バグ調査＋エッジ実験（mask_feather=0 / per_object 既定化）（2026-06-18, medium）
- **依頼（ユーザー報告の実バグ）**: グリーンバックのドラマー動画で「プロンプトした箇所がマスク/合成に反映されない箇所が多い」「素早いドラムスティック・前ボケのシンバルのエッジに半透明がかかりすぎ人物に被る」「一旦エッジ処理を切るのが良くないか」「処理フローを資料(box+pos/neg point per obj_id)と照合して再確認」「ハルシネーション対策もして」。
- **フロー再確認の結論**:
  - point 所有権は**バグではない**。`SAM2VideoPropagator` は `assign_points_to_boxes(points, boxes)` で各 point を最近傍 box に割当て、その box の `add_new_points_or_box(box=, points=, labels=)` に統合（資料の box→同一 obj_id 要件を満たす）。
  - 正直な制約: ある negative point が意図と別の box に幾何的に近いとそちらへ付く。ユーザーは「BBOX への所属は重なってもよい／どの BBOX 所属かという要件は無いに等しい」と明言 → point-obj_id 明示割当機構は**不要**（追加実装しない）。
- **根本原因と是正**:
  - (A) 欠落: union モードは大きな union bbox に対し tb を 1 回呼ぶ。tb は salient-object matting のため、SAM2 が追跡できていても細い/前ボケのスティック等を under-matte する。→ **per_object 既定化**（対象ごとに crop して tb の salient 前景として扱う）。
  - (B) 過ソフトエッジ: tb の連続アルファは motion blur/前ボケを半透明として正しく matte する（=本質的なソフトさ）。これに `mask_feather=8` と union のソフトが加算。→ `mask_feather=0` ＋ **Alpha threshold スライダ手動調整**（ユーザーが自分で試す）で硬化。
- **変更**:
  - `config/inference_models.toml`: 全 background entry（tb_base/tb_fast/tb_base_nightly）の `mask_feather` 8→0、`video_matte_mode` union→per_object。コメント刷新（mask_feather=0 の意味、per_object（現行既定）、誤字「张実→忠実」修正）。
  - `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`: UI 手順テキストを per_object 既定＋union は軽量モード＋エッジ硬化は Alpha threshold へ更新。mask_feather=0 は guard 境界の Gaussian feather オフだが per_object の sigmoid 由来ソフトは残る旨を明記（ハルシネーション防止＝過大表現回避）。
  - `run_video_matting_headless.py`: `--matte-mode` help の古い「既定 union」→「未指定時は config の video_matte_mode に従う。config 既定は per_object」。
- **サブエージェントレビュー（Explore）指摘と対応**:
  - HIGH（app/CLI の `.get(..., "union")` フォールバックが config 既定 per_object と矛盾）→ **据え置き判断**。全 entry が `video_matte_mode` を明示するためフォールバックは到達しないデッドパス。union はキー欠落/不正 config 時の**軽量・安全側**フォールバックとして意図的に維持。代わりに誤解を招く CLI help の「既定 union」表記を修正。
  - HIGH（CLI help の古い「既定 union」）→ **修正済み**。
  - MEDIUM（UI の mask_feather=0＝エッジ処理オフ が不正確）→ **修正済み**（Gaussian feather オフだが sigmoid ソフトは残ると明記、硬化は Alpha threshold）。
  - LOW（フォールバックパスのテスト無し）→ デッドパスのため据え置き。
- **テスト**: config 変更後 非 integration **178 passed, 1 skipped, 3 deselected**。doc 修正後 `test_jupytext_notebooks.py`+`test_headless_cli.py` **39 passed**。movie app `--help` / CLI `--help` exit 0。
- **Playwright 実行時検証（ERR035）**: UI 変更は Markdown 手順テキストのみ（レイアウト/Canvas/イベント配線の変更なし）→ doc-only UI text のため Playwright 実行時検証は必須でない。実画質（per_object の欠落是正・Alpha threshold 硬化の効果）は GPU+動画でユーザー確認待ち。
- **次アクション**: ユーザーが GPU 環境で per_object 既定＋Alpha threshold（≒0.5 目安）を手動調整し画質確認。Alpha threshold スライダ既定 0.0 は変更しない（手動チューニング前提）。
---


## 直近完了タスク: 動画アルファ処理フロー τ config化 + ヘッドレスCLI + レビュー修正（2026-06-17, medium）
- **依頼**: Phase1 完了後の「進んでください」— τ を config 化し movie app から配線、ヘッドレス CLI 実行経路を追加、必須サブエージェントレビューと記録更新。
- **実装**:
  - τ config化: `config/inference_models.toml` の全 background entry（tb_base / tb_fast / tb_base_nightly）に `ownership_temperature = 1.0` を定義（τ 意味コメント付き）。ハードコード禁止に準拠。
  - movie app 配線: `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` が `bg_entry.get("ownership_temperature", 1.0)` を読み `pipeline.run` の `"ownership_resolver": {"temperature": ...}` へ渡す。
  - ヘッドレス CLI: `run_video_matting_headless.py`（新規）— Gradio 非起動の end-to-end 検証経路。`--video`/`--box`/`--point`/`--tracker`/`--background`/`--temperature`(未指定で config)/`--output-mode` 等。`_parse_box`/`_parse_point`/`build_arg_parser`。GroundingDINO テキストは扱わず box/point 直接指定。`--video` 存在を fail-fast 検証。
  - テスト: `tests/unit/test_headless_cli.py`（新規）— 引数解析、movie app の τ 配線 grep、全 background entry の τ 定義（>0）を検証。
- **サブエージェントレビュー指摘と対応**:
  - 中-1（根治）: `SAM2VideoPropagator` の forward/reverse 2pass マージが位置ベースで、pass 間で obj 数が変わると別 obj の logit が混入する潜在バグ。中間構造を `source_index → {obj_id → (H,W)}` に変更し、`target_object_ids` 順で整列して (N,H,W) を構築（欠損 obj は -1e6 埋め）。チャネル位置と obj_id を固定対応。
  - 低-1: `OwnershipResolver.run` の未使用 `frames` 引数と docstring 不一致を削除・修正。
  - 低-2: CLI の `--video` パス存在を `run` 冒頭で fail-fast。
  - 中-2（overlay は生トラッキング可視化の仕様意図）/低-3/低-4 は仕様確認・任意改善として保留。
- **テスト**: 非 integration **168 passed, 1 skipped, 3 deselected**。CLI `--help` exit 0、movie app `--help` exit 0、pipeline import OK。
- **Playwright 実行時検証（ERR035）**: 変更は config 値・パイプライン内部パラメータ・CLI で UI レイアウト/Canvas/イベントの変更なし。movie app callback は τ 読込・配線のみ追加。実画質（τ 効果）は GPU+動画でユーザー確認が必要。
---

## 直近完了タスク: 動画アルファ処理フロー Phase2（対象ごと crop tb 合成）（2026-06-17, medium）
- **依頼**: 「進めてください 背景除去精度が低ければ 再度 per_object で回すわ」「gradioUI にある背景除去手順がアップデートされてたらそれも変更お願いね」。ハルシネーション防止徹底。
- **承認済み設計**: 合成式 `alpha_final(p) = Σ_{o=0..N-1} ownership_o(p) × alpha_o(p)`（RGB は元フレームのまま、アルファのみ合成）。モードは config `video_matte_mode`: `"union"`（既定・フレームあたり tb 1 回）/ `"per_object"`（オプトイン・フレームあたり tb N 回 = 忠実だが重い）。high-end GPU でも tb 呼び出し回数は減らないため既定 union。
- **実装（RED→GREEN）**:
  - 純粋ヘルパー: `pipelines/components/video_common.py` に `composite_alpha_by_ownership(per_object_alphas, ownership)` を追加。形状検証・[0,1] clip・長さ不一致 ValueError。GPU 非依存でテスト可能。
  - per_object フレーム処理: `TransparentBGVideoExtractor._run_per_object_frame` を追加。各対象 logit を `stable_sigmoid` → soft mask で既存 `TransparentBGExtractor.run` を呼び（bbox 導出・crop・tb・full frame 配置・soft guard を再利用）対象ごとアルファを得て所有権合成。RGB は元フレーム保持。
  - `run()` に引数 `video_matte_mode`（既定 "union"）追加。フレームループで logits/ownership が揃い per_object 指定時のみ per_object 経路、それ以外は従来 union 経路へフォールバック（後方互換）。matte メタデータに `video_matte_mode` 記録。
  - config: 全 background entry に `video_matte_mode = "union"` + 意味コメント追加。
  - 配線: movie app が `bg_entry.get("video_matte_mode", "union")` を読み `transparent_bg_video` へ渡す。CLI に `--matte-mode`（未指定で config 既定）追加。
  - UI 手順テキスト更新: movie app の「処理順の考え方」を `フレーム取得 → DINO で候補生成 → SAM2 で対象ごとに prompt/追跡（logit 保持・2値化しない）→ 所有権解決（ピクセル softmax で重なりを各対象へ排他割当）→ 背景透過（連続アルファ）→ 所有権でアルファ合成` に刷新。union/per_object の意味も追記。
  - テスト: `tests/unit/test_per_object_composite.py`（合成数学 5 件）、`tests/unit/test_video_per_object_frame.py`（extractor をモックし N 回呼び出し・合成一致・RGB 保持を検証）、`tests/unit/test_headless_cli.py` に `--matte-mode`・全 entry の `video_matte_mode`・movie app 配線 grep を追記、`tests/unit/test_jupytext_notebooks.py` の UI 手順 assert を更新。
- **サブエージェントレビュー（Explore, thorough）**: 総評 APPROVE。高/中の指摘なし。低-1（`use_per_object` 判定の `np.asarray` 二重呼び出し）を `logits_array` 事前計算で修正。低-2（composite の二重 clip）は浮動小数丸め対策の防御的実装として据え置き。
- **テスト**: 非 integration **178 passed, 1 skipped, 3 deselected**。CLI `--help` exit 0、movie app `--help` exit 0。
- **Playwright 実行時検証（ERR035）**: UI 変更は Markdown 手順テキストのみ（レイアウト/Canvas/イベント配線の変更なし）。per_object 経路は config・パイプライン内部パラメータの追加で UI 構造に影響しない。実画質（per_object の精度向上）は GPU+動画でユーザー確認が必要。
- **対象外/残**: パイプライン統合テスト + Playwright スモーク（GPU 実行を伴う検証）はユーザー環境での確認待ち。

---

## 直近完了タスク: 動画アルファ処理フロー logit保持リファクタ Phase1（2026-06-17, large）
- **依頼**: DINO系(BBOX) → SAM2系 → 背景除去のアルファ処理フローを深く理解し、最重要点（box+補正点を同一 obj_id の1回呼び出しで結合）を確実化した上でフローをリファクタ。計画書を `計画書/` に timestamp 付きで保存。深度推定・静止画版は対象外。
- **設計判断**:
  - per-object logit 保持契約: `SAM2VideoPropagator.run` で `propagate_in_video` の per-object logit を `per_object_logits[frame_idx]=(N,H,W)` として収集。`np.maximum` union は廃止せず overlay/後方互換用の union soft `frame_masks`（`stable_sigmoid`→画素 max）として派生し、FrameMaskSequence に `per_object_logits` を同梱。
  - `OwnershipResolver`（新規 Component）: `masks` を受け per-object logits に背景 logit=0 チャネルを加えた温度 τ softmax で画素ごと所有権（和=1）を算出。前景 soft = 1-背景所有権 を `frame_masks` に差し替え、`ownership` も同梱して下流へ渡す。単一 obj 画素は sigmoid 相当、重なり画素のみ softmax 分配。
  - transparent-background は従来の soft `frame_masks`（H,W float[0,1]）を guard として受ける契約を維持（`Remover.process` はマスク入力不可のため extractor 側で guard 乗算）。
- **実装ファイル**:
  - `pipelines/components/ownership_resolver.py`（新規）: `_softmax_across_objects`（temperature>0 検証, axis=0 安定 softmax）, `OwnershipResolver.run(masks, frames, temperature)`。
  - `pipelines/components/video_model_components.py`: `SAM2VideoPropagator.run` を per-object logit 収集 + union soft 派生 + `masks["per_object_logits"]` 同梱へ変更。box+点の同一 obj 結合（`assign_points_to_boxes`）は維持。
  - `pipelines/sam2_tb_video_pipeline.py`: `OwnershipResolver` を propagator↔extractor 間に挿入、`sam2_video_propagator.masks→ownership_resolver.masks→transparent_bg_video.masks` で配線。overlay は propagator.masks を継続使用。
  - `pipelines/components/model_components.py`: 一時導入した ownership-dict 受理ハックを撤去し、soft frame_masks 受理契約に戻す。
  - `tests/unit/test_ownership_resolver.py`（新規）: softmax 安定性（和=1）、N+1 チャネル、前景 soft=1-背景、metadata 引継ぎを検証。
- **テスト**: 非 integration **161 passed, 1 skipped, 3 deselected**。`test_video_pipeline_wiring.py` の既存契約（`metadata`/`object_ids`/soft union `frame_masks`）は union 派生維持で GREEN。movie app `--help` exit 0、pipeline import OK。
- **Playwright 実行時検証（ERR035）**: 変更はバックエンド Component 層（logit 保持・所有権 softmax・配線）で UI 配線・Canvas・イベント変更なし。実画質（所有権ゲートの継ぎ目線消滅・τ の効果）は GPU+動画でユーザー確認が必要。
- **対象外/残**: Phase2（対象ごと crop tb 合成）、τ の config 化と movie app 配線、ヘッドレス CLI 実行経路は未着手。計画書: `計画書/2026-06-17_動画アルファ処理フロー_logit保持リファクタ実装計画.md`。



## 直近完了タスク: 継ぎ目線+点未反映の根治（修正1 最近傍box割当 + 修正2 soft合成）（2026-06-16, large）
- **依頼**: `報告書\...16_old.md` と `調査\2026-06-16_背景除去マスク継ぎ目線_根本原因調査.md` を読み、§2.5「今回の修正案を実施すること」を実装。ユーザー確定スコープ: 修正1=「最近傍box割当（方針1）のみ」、修正2=「ソフト合成＋末端feather（根治）」、修正3（奥行き）は対象外。
- **根本原因**:
  - 点未反映: `SAM2VideoPropagator` が全 point を1つの追加 obj にまとめていた。SAM2 は複数インスタンスを1 mask で表現できず point が union から落ちる（前タスク方針A の限界）。
  - 継ぎ目線（消える線）: 各 obj を早期に二値化（`logits>0.0`）し binary OR で union → 境界がずれた継ぎ目に黒線。さらに二値 guard を tb alpha に乗算して継ぎ目線を出力に焼き込んでいた。
- **設計判断**:
  - 修正1（最近傍box割当）: `assign_points_to_boxes(points, boxes)` で各点を矩形距離最小の box に割当て、その box の `add_new_points_or_box(box=..., points=..., labels=...)` に同梱。positive 点は最寄り box を補強、negative 点は box 内部をくり抜く。追加 obj を作らない。
  - 修正2（soft合成＋末端feather, 根治）: 各 obj を二値化せず `stable_sigmoid(logits)` で確率化し `np.maximum` で union（forward/reverse も max 統合）。契約を float32[0,1] のまま `build_frame_mask_sequence`→extractor へ疎通。最終 guard は `soft_probability_guard`（grayscale closing で継ぎ目谷を橋渡し + GaussianBlur で末端 feather、二値化なし）。`max(probA,probB)>=0.5 ⟺ binaryA OR binaryB` のため閾値0.5の後方互換は保たれる。
- **実装ファイル**:
  - `pipelines/components/common.py`: 新規 `stable_sigmoid`（overflow 回避の数値安定 sigmoid）, `assign_points_to_boxes`（最近傍box割当, 空入力で各obj空リスト）, `soft_probability_guard`（closing+gaussian feather, [0,1] float32）。`render_tracking_overlay_frame` を float mask は `>=0.5` 閾値、bool は従来通りに分岐。
  - `pipelines/components/video_common.py`: `build_frame_mask_sequence` を float 入力時 `clip(0,1).astype(float32)` 保持、bool 入力時は従来 bool（後方互換維持）。
  - `pipelines/components/video_model_components.py`: `SAM2VideoPropagator.run` 修正1（`point_group_obj_id` 廃止→`assign_points_to_boxes` で box ごとに点同梱）・修正2（binary OR→`stable_sigmoid`+`np.maximum` soft union）。
  - `pipelines/components/model_components.py`: `TransparentBGExtractor.run` を float(soft確率)/bool 両対応。float は `mask_soft=clip(0,1)`, `mask_binary=soft>=0.5`（has_mask/bbox 判定用）、guard は `soft_probability_guard`。bool は従来パス（feather>0 で `feather_binary_mask`、それ以外 `dilate_binary_mask`）。
- **テスト**: RED→GREEN。`test_common_components.py` に5件（最近傍box割当/空入力/stable_sigmoid/soft guard中間値保持/継ぎ目谷橋渡し）、`test_video_pipeline_wiring.py` の propagator テスト2件を soft float・最近傍割当へ更新、`test_transparent_bg_mask_guard.py` に2件（float mask の soft guard・bbox 閾値0.5）。全非 integration **159 passed, 1 skipped, 3 deselected**。両アプリ smoke `--help` exit 0。
- **レビュー**: サブエージェント Explore レビュー「PASS（軽微改善推奨）」。assign_points_to_boxes の矩形距離・空入力、soft union/guard の値域[0,1]、float/bool 分岐の完全性、build_frame_mask_sequence の bool 後方互換すべて正確と確認。`guard` 分岐は `if apply_mask_guard and has_mask:` でガードされ全パスで定義されるため NameError なし。指摘の `soft_probability_guard` 冗長 clip は防御目的で許容（負荷無視できる）。
- **Playwright 実行時検証（ERR035）**: 本タスクの変更はバックエンド Component 層（点割当・union・guard）で、UI 配線・Canvas・イベントの変更なし。前タスクで UI 描画/Point mode 登録は検証済み。propagator/extractor は monkeypatch stub の単体テストで検証。**soft guard/feather の視覚品質（継ぎ目線の消滅・末端の自然さ）は実画像/動画のモデル実行（checkpoints+GPU）が必要なため、UI 描画＋単体テストでの検証に留める。実素材での見た目確認はユーザーの GPU 実行で要確認。**
- **対象外**: 修正3（奥行き推定による前後関係制御）はユーザー指示により本タスク対象外（事前計測が必要）。
- **ERROR_LOG**: ERR043（点未反映の根治: 最近傍box割当）, ERR044（継ぎ目線の根治: soft合成＋末端feather）を追記。



## 直近完了タスク: 2値エッジ解消（union マスク feather + tb alpha 乗算）（タスク2, 2026-06-15, medium）
- **依頼**: 「マスクのエッジがグラデーションと2値（ブラック/ホワイト）の2種類ある＝transparent-background と SAM2/SAMURAI mask の合成を意味する。2値エッジは避けるべき。union マスク+point 領域を tb に渡すのが良いか」。ユーザー確定方針: 「sam2/samurai の統合された二値化マスクを feather、そのマスクを transparent-background に入力」。強度は config/*.toml で制御。
- **根本原因（ERR042）**: `TransparentBGExtractor.run` の最終 alpha = tb の連続 gradient alpha × **二値 guard**（`dilate_binary_mask`, 0/1）。二値 guard が mask 境界で tb の gradient を硬く切断し、黒/白の2値エッジを生む。guard を外すと ERR039（横一直線切れ）が再発するため除去不可。
- **技術的制約**: `transparent_background.Remover.process` は画像のみ受け取りマスク入力（ヒント）を受け付けない。よって「feather マスクを tb に入力」を文字通り画像前処理で行うと tb の salient 検出を阻害する。代わりに **feather した union マスクを tb 出力 alpha に乗算**（＝二値 guard の feather 版）することでユーザー意図（feather で領域制限・2値縁解消）を tb 精度を落とさず実現。
- **設計判断**: 新規 `feather_binary_mask(mask, dilate_size, feather_radius)`（`pipelines/components/common.py`）を追加。`feather_radius<1` で従来二値（後方互換）、`>=1` で `effective_dilate=max(1,min(dilate_size,feather_radius))` で軽く dilate した base 境界を中心に符号付き距離変換で ±feather_radius を 0↔1 遷移させた float32 soft guard を返す。遷移帯が mask 境界（=tb 前景 alpha 境界）に重なり中間 alpha を生むのがポイント（dilate を大きくし過ぎると遷移帯が前景外に出て中間値が消えるため effective_dilate で抑制）。
- **実装**:
  - `TransparentBGExtractor.run`: 引数 `mask_guard_feather:int=0` 追加。`>0` で feather guard、`==0` で従来二値。metadata に `mask_guard_feather` 記録。
  - `SAM2GuardFilter.run`: 引数 `feather:int=0` 追加。同分岐。
  - `TransparentBGVideoExtractor.run`: `mask_guard_feather:int=0` 追加し extractor へ伝播（動画版は extractor が最終段のため soft guard ×1 回で完結）。
  - `config/inference_models.toml`: `[[background]]` 各 entry に `mask_feather=8`。
  - 動画 UI（`...for_Movie.py`）: `bg_entry.get("mask_feather",0)` を `transparent_bg_video` へ渡す。
  - 静止画 UI（`...haystack.py`）: `entry.get("mask_feather",0)` を extractor へ渡し、**feather>0 のとき `sam2_guard` を enabled=False** にして二重 guard（soft×二値で2値エッジ再発）を回避。最終 alpha/rgba/preview は extractor の feather 済み出力で整合。
- **テスト**: RED→GREEN。`tests/unit/test_transparent_bg_mask_guard.py` に6件追加（extractor feather で境界中間値・metadata、feather=0 で二値後方互換、SAM2GuardFilter feather、helper の連続値/二値、極端 feather_radius+極小 mask の範囲保持、空 mask で全0）。全非 integration **150 passed, 1 skipped**。両アプリ smoke `--help` exit 0。
- **レビュー**: サブエージェント Explore レビュー「承認（軽微改善推奨）」。正確性/後方互換/二重guard回避/動画整合/規約すべて良好。指摘の docstring（feather_radius 推奨値ガイドライン）と極端ケーステストを反映。distanceTransform 飽和は既定値8で安全と確認。
- **Playwright 実行時検証（ERR035）**: 静止画版（port 7862）を起動し UI 描画（タイトル / Input Image / Prompt Canvas / Run transparent-background ボタン / model dropdown / 各 accordion）を確認、配線変更で UI 破綻なしを検証。font 404 は Gradio static の既知無害警告。**feather の視覚的品質は実画像/動画のモデル実行（checkpoints+GPU）が必要なため、UI 描画＋単体テストでの検証に留める。実素材での見た目確認はユーザーの GPU 実行で要確認。**
- **ERROR_LOG**: ERR042 を追記。


## 直近完了タスク: 動画 SAM2 box+point 併用で point 無視を修正（タスク1, 2026-06-15, medium）
- **依頼**: 「文字プロンプト→BBOX→BBOX融合まで完了確認。しかし point prompt（ネガティブ/ポジティブ）の追跡ができていない。まずタスク1（point追跡）を完了させる。その後タスク2（2値エッジ/マスク合成見直し）に着手」。
- **根本原因（ERR041）**: `SAM2VideoPropagator.run`（`pipelines/components/video_model_components.py`）の `if boxes:` 分岐が box のみ登録し points/labels を渡していなかった。`apply_selected_boxes` は `state["boxes"]` 設定時に `state["points"]` をクリアしないため box と point が共存するが、propagator が point を黙殺。UI 層（`select_sam2_prompt`）は正常で欠陥は propagator のみ。
- **設計判断（方針A）**: box 群を obj 1..N、point 群を追加 obj N+1 として登録し全 obj を OR 統合（負点は point 群 obj 内で除外）。union ロジックは既に `target_object_ids` を走査するため追加 obj も自動 union。`else`（point のみ/単一 box）分岐は未変更で後方互換維持。
- **実装**: `target_object_ids` に `point_group_obj_id = len(boxes)+1`（boxes と points 両方ある時）を追加。`if boxes:` 登録後に point 群を `add_new_points_or_box(obj_id=point_group_obj_id, points=..., labels=...)` で登録。
- **テスト**: RED→GREEN。新規 `test_sam2_video_propagator_registers_point_group_with_boxes`（torch 未導入のため `monkeypatch` で `inference_mode` のみ stub 注入）。box2つ+point群1つの計3登録、point 群 obj_id=3・labels=[1,0]、metadata の points/labels 保持、排他領域 OR で union 全面 True を検証。全非 integration **146 passed, 1 skipped**。動画版 smoke `--help` exit 0。
- **レビュー**: サブエージェント Explore レビュー実施。正確性/後方互換/union guard/Hard Rules/テスト品質すべて「APPROVED」。指摘の nit「union 検証が all-1s で弱い」を受け、各 obj を排他領域（左/中/右 1/3）に分割し point 群 obj が穴を埋めて全面 True になることまで検証するよう強化。
- **Playwright 実行時検証（ERR035）**: 動画版（port 7861）で Point mode 切替→positive/negative ラジオ表示、positive 点登録（`Point selected: (319,209), label=positive`）、negative 点登録（`label=negative`）を確認。propagator 修正はバックエンドで単体テスト検証済み（実伝搬は GPU/動画が必要なため UI 登録経路の検証に留める）。
- **既知の無害警告**: Canvas select 時に Gradio が `Too many arguments provided for the endpoint` を出すが point 登録は正常動作。本タスクのバックエンド変更とは無関係の既存警告。
- **ERROR_LOG**: ERR041 を追記。
- **残課題（タスク2）**: マスクエッジの2種類（グラデーション + 2値ブラック/ホワイト）問題。transparent-background と SAM2/SAMURAI mask の合成が2値エッジを生む仮説。union マスク+point prompt 領域のみを transparent-background へ渡す案を調査・実装予定。ERR039 の mask guard と関連。


## 直近完了タスク: 動画/静止画 UI 巻き戻り復旧 + git stash 事故回収（2026-06-15, large）
- **依頼**: ユーザー報告の動画 UI 退行3件「1: フレームのシーク機能 / 2: 文字プロンプトで複数 bbox が反映されず1つだけ / 3: mask union 統合未済」を原因説明の上で修正。「一ヶ月前ほどに巻き戻ったことは大問題」。承認タスク: (1)静止画/notebook 退行修正, (3)冗長stash削除, (4)Playwright検証。**(2)再出現した11個の.md削除は未承認のため据置**。
- **根本原因**: 動画版・静止画版 UI と jupytext notebook が**未コミットの作業ツリー変更**で全機能版→旧版へロールバック。Component 層（`model_components.py` / `video_model_components.py` / `model_registry.py`）は無傷。HEAD `2702d6b` より新しい UI 実装は commit / stash / branch / reflog のいずれにも無く**復元不能**だった。詳細は `調査/2026-06-15_130117_動画UIファイル巻き戻り_根本原因調査.md`。
- **復元方針（テストを正本に再実装）**:
  - 動画版: `git checkout HEAD` で復元後、HEAD より新しい RED テスト（`test_movie_app_ui_wiring` / `test_movie_runtime_bugs` / `test_video_pipeline_wiring` / `test_jupytext_notebooks`）の要求を**再実装**（HEAD 比 +76/-22）。主な復元: `_normalize_detected_rows`（ERR036 型判別→`.values.tolist()`）, `populate_candidate_choices`（top1 既定 ON）, `run_video_background_removal`（`prompt_frame_idx >= processed_frames` で fail-fast `gr.Error`, ERR037）, `get_video_pipeline`（`entry_by_id("tracker",...)`→`SAM2VideoPropagator(checkpoint_path, config_name)`→`build_sam2_tb_video_pipeline`）, `extract_first_frame_outputs`（4戻り値・`prompt_frame_idx` リセット）, セクションコメント, `prompt_canvas sources=[]`, スライダー集約＋`.change` 配線。
  - 動画版 jupytext 追加3点: スライダー label を `プロンプト起点フレーム位置（ドラッグで Canvas 更新）` に, `frame_step` に `elem_id="movie-frame-step"`, 処理順表示を `フレーム取得 → DINO で候補生成 → SAM で prompt / tracking → 背景透過` に更新。
  - 静止画版: HEAD が全機能版（`background_model` Dropdown + registry）だったため `git checkout HEAD -- gradio_app_sam2_transparent_BG_haystack.py` のみで復旧。
- **git stash 事故と回収**: 復元途中に `git stash`（引数なし）で全作業を退避。`git stash pop` が EOL 正規化で2回失敗。`git checkout 'stash@{0}' -- .`（マージ無しでファイル内容展開）で全データ回収。検証後、冗長 stash を `git stash drop 'stash@{0}'` で削除。副作用: ユーザーが整理削除した11個の.md（root直下）が HEAD 版として再出現したが、**削除は未承認（タスク2）のため touch せず据置**。
- **テスト**: 全非 integration **145 passed, 1 skipped**（修正前 3 failed → 解消）。両アプリ smoke `--help` exit 0。
- **Playwright 実行時検証（ERR035）**: 動画版（port 7861）で `prompt-frame-idx` シークスライダー / 複数 bbox 候補 CheckboxGroup / 処理順表示 / `movie-frame-step` / Prompt Canvas / Text Prompt accordion / tracker dropdown のレンダリングを全 OK 確認。静止画版（port 7862）で `background_model` Dropdown レンダリング OK 確認。検証スクリプト・スクショは検証後削除。
- **レビュー**: サブエージェント Explore レビュー実施。観点1〜5（シーク配線 / 複数box配線 / mask union+registry / Hard Rules: weights_only・try/except pass・ハードコード・ERR036 / `movie-frame-step`）すべて「問題なし」。
- **ERROR_LOG**: ERR040（UI巻き戻り＋git stash事故の根本原因・安全回収手順・再発防止）を追記。
- **残課題（別タスク）**: 再出現した11個の.md の扱い（ユーザー承認待ち=タスク2）。複合対象 union の実素材での品質確認は動画アップロード＋モデル実行が必要なため UI レンダリング検証に留めた。


## 直近完了タスク: マスク横一直線切れ修正（SAM2 mask 形状を最終 alpha に反映）（2026-06-12, medium）
- **依頼**: 6/4 報告書 2.2 と調査 `調査/2026-06-04_222747_現行動画パイプライン_フロー調査_MAM比較.md` の「マスクが画面下で横一直線に切れる」不具合の最優先修正を進める。
- **根本原因**: `TransparentBGExtractor.run`（`pipelines/components/model_components.py`）が SAM2 mask の形状を使わず外接矩形でクロップし、矩形範囲だけに alpha を貼り戻していた（矩形内・mask 外に alpha が残り横一直線切れ）。`SAM2GuardFilter` は実装済みだが動画パイプラインに未接続。
- **実装**: `TransparentBGExtractor.run` に `apply_mask_guard: bool = True` / `mask_guard_dilate: int = 21` を追加。`full_alpha` 算出後に mask があれば `dilate_binary_mask` の guard を乗算し mask 形状外の alpha を 0 に。metadata に `mask_guard_applied` を追加。extractor 内適用のため動画版 `TransparentBGVideoExtractor`（frame ごとに同 run を呼ぶ）にも自動波及。preview/rgba も guard 後 `full_alpha` から生成。
- **設計判断**: 静止画 `build_sam2_union_tb_pipeline` は後段 SAM2GuardFilter を同一 mask で接続するが二値 guard 乗算は冪等で二重適用しても不変。`mask_guard_dilate=21` は SAM2GuardFilter 既定値と一致。既存の `crop_padding=40` 等と同じデフォルト引数パターンに合わせ、別パイプラインの mask 未接続配線（既存 no-op）は依頼外のため変更せず。
- **テスト**: RED→GREEN。新規 `tests/unit/test_transparent_bg_mask_guard.py` 4件（mask 形状反映 / mask 未指定後方互換 / SAM2GuardFilter 二重適用の冪等 / guard 無効化で従来挙動）。動画版 smoke `--help` exit 0。
- **既存失敗テストについて**: 全体実行で 18 件失敗するが、すべて UI/callback 配線層（`test_movie_app_ui_wiring` / `test_video_pipeline_wiring` / `test_movie_runtime_bugs` / `test_pipeline_wiring` / `test_jupytext_notebooks`）の UI ソーステキスト assert。調査報告が指摘した「計画と実装の乖離（現行 UI ファイル巻き戻り）」による既存失敗で、今回の `model_components.py` 変更とは無関係（変更前から失敗）。
- **レビュー**: サブエージェント Explore レビュー実施。A正確性/C動画波及/D規約/E可読性は OK。CRITICAL「二重適用テストなし」を受け冪等性テストを追加。HIGH「他パイプラインの sam2_guard mask 未接続」は既存挙動で依頼外のため据置。MEDIUM「config化」は既存デフォルト引数パターン踏襲のため過剰設計回避で据置（判断を WHITEBOARD に記録）。
- **残課題（別タスク）**: 複合対象 union UI 復旧（1 box しか使わない配線の根本対処）、frame選択/双方向/dropdown 復旧、UI 配線テスト群 GREEN 化。guard は「直線切れ」を「mask 形状に沿った切れ」に改善するが、SAM2 が囲みきれない未検出領域そのものは復元しない。
- **ERROR_LOG**: ERR039 を追記。


## 直近完了タスク: copilot-instructions REMINDER の Haystack 記述増補（2026-06-05, small, コード変更なし）
- **依頼**: REMINDER（ユーザー表記: remainder）に Haystack 記述を追加し、理由と設計意図を明確化する。
- **反映内容**:
  1. Haystack 採用理由（機能分割・疎結合）を明記。
  2. 機能分割による可読性・保守性の担保を明記。
  3. 画面解釈/トラッキング/背景透過モデルの頻繁な差し替え前提を明記。
  4. 安定 I/O 契約を破ると配線不整合が連鎖しバグ温床化する点を明記。
- **レビュー結果**: 変更差分のセルフレビューを実施し、既存ルールとの矛盾なし（番号繰り上げ整合を確認）。
- **テスト省略理由**: 指示文書（.md）の更新のみで挙動変更なし。RED テスト対象なし（規約 step7 に基づき省略）。


## 直近完了タスク: Copilot指示体系 多層化リファクタ実装（2026-06-05, medium, コード変更なし）
- **依頼**: 資料「データサイエンティストのためのAGENTS.mdとSkills.md」に沿って、プロジェクトの指示体系を妥当な範囲で再構成し、3回レビュー後に計画書へ保存する。
- **実装内容**:
  1. ルート `AGENTS.md` を新設し、Hard Rules / Routing Table / Project Context / 既存agents参照 / 検証コマンドを集約。
  2. `.github/copilot-instructions.md` を薄い常時ルールカードへ再構成（詳細は AGENTS + skills へ委譲、REMINDER 12項維持）。
  3. `.github/instructions/workflow.instructions.md` から領域重複を削減し、実行フローとレビュー・記録更新に集中。
  4. 新規 skill 追加: `.github/skills/gradio5-sam2-ui/SKILL.md`, `.github/skills/sam2-tracking-dino/SKILL.md`。
  5. 新規 prompt 追加: `.github/prompts/plan-change.prompt.md`, `.github/prompts/verify-ui-playwright.prompt.md`。
  6. 計画書保存: `計画書/2026-06-05_Copilot指示体系_多層化リファクタ計画.md`。
- **レビュー結果**: Explore サブエージェントレビューで重大指摘なし。計画書の ERR028 割当のみ整合修正済み。
- **テスト省略理由**: 指示文書（.md/.prompt）の更新のみで挙動変更なし。RED テスト対象なし（規約 step7 に基づき省略）。


## 直近完了タスク: /chronicle improve — ERR035〜038 再発防止ルールを instructions へ反映（2026-06-05, medium, コード変更なし）
- **背景**: ローカル session store はメタデータ（15 セッションの日時）のみで turns/checkpoints/files が空（reindex/force でも 0）のため、セッション本文からの friction 抽出不可。代わりに ERROR_LOG.md（ERR001〜038）を一次情報として分析。
- **検出 friction**: 直近の新規エラーは動画版 UI/配線（ERR031〜038）に集中。最も高コストなメタ friction は「fixed/完了 判定がソーステキスト一致止まりで実行時検証されていない」（ERR033→ERR035 の誤 fixed、WHITEBOARD 6/4 調査の「完了」記録不一致）。
- **反映した5点（ユーザー承認 A=全反映）**:
  1. `### Gradio 5 / SAM2 UI`: UI/配線の fixed/完了 記録は Playwright 実起動での実行時検証を必須化（ERR035）。
  2. 同: Gradio 5/Svelte で DOM 直接 value 代入＋native event dispatch の JS ブリッジは実行時に機能しない。ネイティブイベントで構成・冗長コントロール集約（ERR035）。
  3. 同: `gr.Dataframe` の値は pandas DataFrame で渡る。真偽評価禁止・型判別して `.values.tolist()`（ERR036）。
  4. `### モデル・評価・学習`: スライダー上限と実処理レンジ乖離 UI は重い処理前に fail-fast 範囲検証し `gr.Error`（ERR037）。
  5. `## 1. 常時適用ルール`: SAMURAI fork 同梱 config は installed sam2 の Hydra 検索パスに自動で載らない。`Path.as_uri()` で GlobalHydra へ append、`samurai/` は変更せず `MissingConfigException` 伝搬（ERR038）。REMINDER に項目12（UI/配線 fixed は Playwright 実行時検証必須）を追加。
- **レビュー**: サブエージェント（Explore）が観点 A〜E（ERROR_LOG 一致 / 既存ルール矛盾 / セクション名・ERR番号 / REMINDER 整合 / Markdown 構造）全合格「準拠」。CRITICAL/HIGH なし。
- **テスト省略理由**: instructions（.md）のルール追記のみで振舞変更なし。RED テスト対象なし（規約 step7 に基づき省略）。

---
- **依頼**: ①6/4 報告書が前回 5/19 報告書からの差分（進化）を反映できていない→反映。②処理フローを ASCII アートで記載し、`report-for-leader-denshi` スキルへもその規約を反映。③改修後サブエージェントでレビュー、④WHITEBOARD/REFERENCE 更新。
- **進化の反映（5/19→6/4）**: Gradio UI 搭載／MAM から DINO 意味理解＋プロンプト粒度向上／SAM→SAM2 差替／プルダウンで SAM2→SAMURAI 切替枠組み／意味理解モデルもプルダウン枠組み（※実際はまだ切替不可）／背景透過モデルも切替可／Haystack で機能分割・疎結合 Component パイプライン。
- **報告書の変更**: セクション0に進化1行＋狙いを追記。セクション1進捗表に新規行（操作画面と言葉で選ぶ機能✅／追跡 SAM→SAM2✅／モデル選べる仕組み📋一部のみ稼働／マスク横切れ調査✅原因確定）。セクション2.1「5/19からの進化—全体の仕組み」を新設（ASCII フロー図①言葉→②DINO→③SAM2/SAMURAI→④transparent-background→出力＋▲プルダウン差替注記＋Haystack 基盤＋5行進化表＋部品分割効果）。2.2=不具合原因／2.3=乖離／2.4=素材限界に番号繰下げ（grep で 2.1〜2.4 整合確認済み）。
- **スキルへの ASCII 規約反映（3ファイル）**: `SKILL.md`（書き方ルール7「処理の流れはアスキーアート図を1つ」）／`references/writing_guide.md`（目次5＋チェックリスト2項＋新節5「アスキーアートフロー図の作り方」：左→右・四角枠・矢印に渡るもの注記・始点終点明示・図下1行補足・3〜5部品・未完成は▲注記＋雛形）／`assets/report_template.md`（2.1 に ASCII 雛形プレースホルダ）。
- **サブエージェント(Explore)スキルベースレビュー反映**: SAMURAI に説明追加「動きを予測して見失いにくい高機能版」(high)／ASCII図の背景透過モデル表記が "transp-arency"＝透明性に誤読される→枠内を「背景消し用モデル」に変え実モデル名 transparent-background を図下補足へ移動(high)／2.2 マスク初出に「マスク（切り抜きの型紙）」注釈追加(medium)。LOW 指摘（〔〕括弧／節番号参照）は据置。CRITICAL なし・準拠度高。
- **テスト省略理由**: 報告書・スキル文書（.md）のみで挙動変更なし。RED テスト対象なし（規約 step7 に基づき省略）。

---

## 直近完了タスク: 電子さん向け報告書スキル作成（2026-06-04, medium, コード変更なし）
- **依頼**: 5/19 リーダー報告書を見本に、電子さん向け報告書作成スキルを作る。
- **電子さんの読み手特性（スキルに encode 済み）**: ①AI/CGに疎い→専門用語にやさしい言い換え＋たとえ話 ②長文は読まれない→過剰説明しない（1項目2〜4行） ③中学生偏差値50の読みやすさ ④会議で無発言→報告書単体で完結。
- **成果物**: `.github/skills/report-for-leader-denshi/`（SKILL.md ＋ `assets/report_template.md` ＋ `references/writing_guide.md`）。
- **テスト省略理由**: スキル文書（.md）のみで挙動変更なし。RED テスト対象なし（規約 step7 に基づき省略）。
- **レビュー**: サブエージェント Explore で6観点（frontmatter規約／トリガー妥当性／読み手要件反映／書式一致／progressive disclosure／矛盾誤り）全て合格＝問題なし。

---

## 進行中タスクの詳細

### タスク名: マスク横切れ不具合 原因調査・報告書作成（2026-06-04, large, コード変更なし）
- **背景**: ユーザー観察「DINO の BBOX は出るが SAM2 連携・マスク統合に不備。画面下のマスクが横一直線で切れる。BBOX そのものをユニオンマスク範囲に使っている疑い。パイプラインは壊れている」。報告書3本の作成依頼。
- **確定した根本原因（コードリードで裏付け、サブエージェント Explore がA〜E全て妥当と判定）**:
  - **横一直線切れの主因**: `TransparentBGExtractor.run`（`pipelines/components/model_components.py` 609行付近）が SAM2 mask の **形を使わず外接矩形（`mask_to_bbox`+`crop_padding`）で画像をクロップ**し、その矩形内に transparent-background を適用、`full_alpha[y_min:y_max, x_min:x_max] = alpha_crop` で矩形範囲だけ貼り戻す。SAM2 mask が対象を囲みきれず矩形が途中で切れると **矩形下端＝横一直線で alpha が切れる**。ユーザー推測は本質的に正しい。
  - **複合対象が取れない**: `detect_text_boxes_for_video` が DINO 候補の `boxes[0]`（最上位1個）だけを `state["box"]` にコピー。
  - **配線欠落（事実上の機能後退）**: `run_video_background_removal` に `boxes`/`prompt_frame_idx`/`bidirectional`/`tracker_model`/`background_model`/`overlay_enabled` が無く、`get_video_pipeline` は `build_sam2_tb_video_pipeline()` を引数なしで呼ぶ。`SAM2VideoPropagator.run` は multi-box union（mask logits>0、BBOX直接ではない）/双方向に**対応済みだが UI/callback が呼んでいない**。
  - **WHITEBOARD とコードの乖離**: 過去記録は複合対象 union・frame slider・dropdown・overlay を「完了」とするが、現行ファイルに未配線。`tests/unit/test_video_pipeline_wiring.py` が **7件失敗**（GroundingDINOMultiBoxDetector / STAGE_PROGRESS_RANGES / prompt-frame-idx / tracker_model / background_model / tracking overlay / 見出し系を要求）。
  - **未接続部品**: `SAM2GuardFilter`（mask 外 alpha 削り）は `model_components.py` に実装済みだが `sam2_tb_video_pipeline.py` に未接続。透過抽出に組み込めば横切れを大幅改善可能。
- **成果物**:
  - `調査/2026-06-04_222747_MattingAnything_BBOX-マスク-トラッキング-統合フロー調査.md`（オリジナル MAM のフロー）
  - `調査/2026-06-04_222747_現行動画パイプライン_フロー調査_MAM比較.md`（現行動画版フロー＋MAM比較＋根本原因＋推奨対応）
  - `報告書/リーダー電子様：PBR連番画像動画・ 動画背景除去 報告書 2026/6/4.md`（テンプレ書式に準拠）
- **レビュー**: サブエージェント Explore が A〜E をコード行番号付きで検証し全て妥当と判定。指摘（SAM2GuardFilter 未接続の明記）を2文書に反映済み。
- **素材限界（チーム周知）**: 被写界深度ボケ・前景/背景同色は DINO/SAM2 が原理的に苦手（不具合とは別問題）。
- **次タスク候補（未着手・別タスク）**: ①TransparentBGExtractor へ SAM2 mask AND（SAM2GuardFilter 組込）、②複合対象 union UI 復旧、③frame選択/双方向/dropdown 復旧、④test_video_pipeline_wiring 7件 GREEN化。

---

## 過去タスク詳細（参考）

<!-- 現在取り組んでいるタスクの目的・進捗・残作業を記述 -->

### タスク名: 動画版 Colab 実機ランタイムエラー 3 件修正（2026-06-06 着手, large）
- **背景**: ユーザーが Colab（`エラーログ/エラーログ_11.md`）で動画版アプリの 3 フローのエラーを報告。(1) Text Prompt → 検出ボタン → エラー、(2) SAM系プルダウンを SAMURAI に切替 → 実行ボタン → エラー（スキーマ違い？）、(3) 背景透過系チェックポイント変更でもエラー。
- **確定した根本原因**:
  - **Bug A（DataFrame 真偽値曖昧）**: `populate_candidate_choices(detected_rows)` の `rows = list(detected_rows or [])`。Gradio 5 の `gr.Dataframe` は値を pandas DataFrame で渡すため `detected_rows or []` が `ValueError: The truth value of a DataFrame is ambiguous` を送出。検出ボタンの `.then(populate_candidate_choices)` が失敗 → CheckboxGroup が空 → `apply_selected_boxes` が「少なくとも 1 つの候補 bbox を選択してください」を送出（連鎖）。
  - **Bug B（prompt_frame_idx 範囲外）**: スライダー max=1999 だが処理は `max_frames`（既定 30）frame のみサンプリング。スライダー 75・max_frames 30 で実行すると propagator が `prompt_frame_idx が範囲外です: 75（許容 0〜29）` を 18s 後に送出。`run_video_background_removal` に事前検証なし。
  - **Bug C（SAMURAI config 未発見）**: SAMURAI tracker 選択時 `hydra.errors.MissingConfigException: Cannot find primary config 'configs/samurai/sam2.1_hiera_l.yaml'`。Colab の installed sam2 が facebook 版で `configs/samurai/` を含まないため。samurai/ 同梱の configs が Hydra 検索パス未登録。
- **方針**: (A) `populate_candidate_choices` を pandas DataFrame / list / None 安全に正規化。(B) `run_video_background_removal` で pipeline.run 前に `prompt_frame_idx >= _estimate_processed_frames` を `gr.Error` で fail-fast（18s 待たせない）。(C) `SAM2VideoPropagator.warm_up` で samurai config 利用時にローカル `samurai/sam2/sam2`（configs/ を含む package root）を Hydra 検索パスへ append（samurai/ は変更しない）。解決不能なら元 MissingConfigException を握り潰さず伝搬。
- **手順**: WHITEBOARD(済) → RED テスト3件 → 実装(GREEN) → pytest + smoke → Playwright(検出→候補選択→適用) → サブエージェントレビュー → ERROR_LOG(ERR036-038)/WHITEBOARD 更新。
- **完了状況（2026-06-06）**: ✅ 全手順完了。
  - **実装**: (A) `_normalize_dataframe_rows` ヘルパ追加し `populate_candidate_choices` を pandas DataFrame / list / None 安全化（真偽評価を排除）。(B) `run_video_background_removal` の `processed_frames` 算出直後・pipeline.run 前に `prompt_frame_idx >= processed_frames` を `gr.Error` で fail-fast。(C) `video_model_components.py` に `_samurai_config_root` / `_ensure_samurai_config_searchpath` を追加し `SAM2VideoPropagator.warm_up` の build 直前で呼出（samurai/ は不変更、Hydra 検索パスへ `as_uri()` で append、解決不能は MissingConfigException を伝搬）。
  - **テスト**: RED→GREEN（`tests/unit/test_movie_runtime_bugs.py` 7件）。全体 `.venv\Scripts\python.exe -m pytest -m "not integration" -q` → `141 passed, 1 skipped, 3 deselected`。smoke `--help` OK。
  - **UI 確認**: UI 要素の追加・削除なし（ハンドラ/バックエンド検証のロジック修正のみ）のため Playwright 必須対象外。
  - **サブエージェントレビュー（Explore）**: A/B は CORRECT 判定（DataFrame 判別・行アクセス・early fail-fast の順序と except gr.Error 再 raise を確認）。C は URI 形式の改善指摘（`f"file://{as_posix()}"`→`as_uri()`）を受け修正適用、ほか samurai/ 不変更・エラー非握り潰し・冪等性を確認。
  - **ERROR_LOG**: ERR036（gr.Dataframe 真偽値曖昧）・ERR037（prompt_frame_idx 範囲外の遅延発覚）・ERR038（SAMURAI config 検索パス未登録）を追記。

### タスク名: 動画版 シーク連動UI簡素化（2026-06-06 着手, large）
- **背景**: ユーザーから「プロンプト起点フレーム位置（シーク連動）/ 表示中フレーム再取得 / シーク位置をsam2に反映」の3つが実機で機能しないと報告（ERR033 で「修正済」としたが実機 NG）。ui-ux-pro-max での要否調査を依頼。
- **根本原因**: `VIDEO_SEEK_SYNC_JS`（gr.Blocks(js=...)）が動画プレイヤーの seeked を拾い、生 DOM でスライダー value を書換え `input`/`change` を dispatch する方式。Gradio 5（Svelte）では DOM 直書きが内部 state に伝わらず `.change` がバックエンドに届かない既知の不安定パターン。3つとも `prompt_frame_idx` 値に依存するため連鎖して機能せず。ボタン2つ（表示中フレーム再取得 / シーク位置を SAM2 に反映）は同一ハンドラ `extract_prompt_frame` の重複。
- **ユーザー決定**: 選択肢 A（スライダー1本へ集約・確実）を採用。video 純正シークバー→backend は Gradio で信頼性が出せないため断念。
- **方針**: (1) `VIDEO_SEEK_SYNC_JS` / `build_video_seek_sync_js` と `gr.Blocks(js=...)` を削除。(2) 「表示中フレームを再取得」「シーク位置を SAM2 に反映」ボタンと配線を削除。(3) `prompt_frame_idx` スライダー（ラベルから「シーク連動」除去・「ドラッグで Canvas 更新」明示）の `.change(extract_prompt_frame)` を**唯一の確実な操作元**として残す。(4) JS 専用だった hidden `video_fps`（elem_id=movie-video-fps）を削除し `extract_first_frame*` の fps 出力も除去。(5) 使い方 Markdown を「シーク自動同期」→「スライダーでフレーム選択」へ修正。
- **手順**: WHITEBOARD(済) → 既存テスト更新(RED) → 実装(GREEN) → smoke → Playwright(スライダー→Canvas 同期確認) → サブエージェントレビュー → ERROR_LOG(ERR035)/WHITEBOARD 更新。
- **影響テスト**: `test_jupytext_notebooks.py::test_sam2_movie_ui_auto_loads_first_frame_and_simplifies_settings`、`test_video_pipeline_wiring.py::test_movie_app_exposes_text_prompt_to_box_flow`、`test_movie_app_ui_wiring.py::test_movie_redisplay_frame_button_follows_seek_position`（build_video_seek_sync_js / 表示中フレーム / movie-video-fps / seeked / currentTime / load_first_frame_btn 系 assert を新仕様へ更新）。
- **完了状況（2026-06-06）**: ✅ 全手順完了。
  - **実装**: `VIDEO_SEEK_SYNC_JS` / `build_video_seek_sync_js` / `gr.Blocks(js=...)` 削除。`load_first_frame_btn`（表示中フレームを再取得）・`show_frame_btn`（シーク位置を SAM2 に反映）とその `.click` 配線を削除。hidden `video_fps`（movie-video-fps）削除。`extract_first_frame` / `extract_first_frame_outputs` を 4-tuple（fps 除外）へ、`input_video.change` outputs から `video_fps` 除外。`prompt_frame_idx` スライダー label を「ドラッグで Canvas 更新」へ、`.change(extract_prompt_frame)` を唯一の操作元として維持。使い方 Markdown と双方向伝播横の補足文もスライダー基調へ修正。
  - **テスト**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` → `134 passed, 1 skipped, 3 deselected`。`test_video_pipeline_wiring.py` の fps 期待テストを 4-tuple 新仕様（`test_extract_first_frame_outputs_resets_prompt_slider`）へ更新。smoke `--help` OK。
  - **Playwright UI 確認**: 7862（CUDA_VISIBLE_DEVICES=-1）で起動し `tests/manual/verify_movie_slider_ui.py` で検証 — 新スライダー label「プロンプト起点フレーム位置（ドラッグで Canvas 更新）」表示、冗長ボタン2つ不在、`#prompt-frame-idx` range スライダー存在を確認（outputs/movie_slider_ui.png）。
  - **サブエージェントレビュー（Explore）**: 問題なし。stale 参照なし、配線整合（extract_first_frame_outputs 4-tuple↔input_video.change outputs）、後方互換（Text Prompt→box / 双方向 / 複数 bbox）維持を確認。
  - **ERROR_LOG**: ERR035（Gradio 5/Svelte の DOM ブリッジ不安定でシーク連動3コントロールが実行時無反応、ERR033 の「fixed」記述を訂正）を追記。

### タスク名: 動画版 実機 4 ギャップ修正（2026-06-06 着手, large）
- **背景**: ユーザーから3度目の不具合報告。計画書・WHITEBOARD は「完了」記載だが実機で動作しない。ERROR_LOG にも未記載だったとの指摘。精読で UI 実装と配線の 4 ギャップを特定。
- **確定した根本原因**:
  - **Gap A（tracker 切替が no-op / SAMURAI 試せない）**: `get_video_pipeline(tracker_model, ...)` が `build_sam2_tb_video_pipeline()` を**引数なし**で呼び、registry の config_name/checkpoint_path が propagator に伝搬しない。→ `build_sam2_tb_video_pipeline(propagator=...)` 注入対応 + `get_video_pipeline` で `entry_by_id("tracker", id)` から `SAM2VideoPropagator` を構築。
  - **Gap B（dropdown が見えない）**: tracker/background Dropdown が閉じた `Accordion("Advanced: 動画処理設定", open=False)` 内に埋没。→ 可視セクション（## 3. SAM系 / ## 4. 背景透過系）へ移動。
  - **Gap C（「表示中フレーム再取得」と「シーク位置反映」が非連動）**: 再取得ボタンが `extract_first_frame`（常に frame 0）を呼びシーク位置を無視。→ `extract_prompt_frame`（prompt_frame_idx 連動）へ再配線。
  - **Gap D（prompt canvas 空白, ERR026 違反）**: 動画版 prompt_canvas に `sources=[]` 欠落。静止画版にはあり（divergence）。→ `sources=[]` + download/fullscreen 無効を付与。
- **手順**: WHITEBOARD(済) → RED テスト → Gap D→A→B→C → GREEN+smoke → Playwright → サブエージェントレビュー → ERROR_LOG(ERR031-034)/WHITEBOARD 更新。
- **共通処理影響**: pipeline builder 改修は後方互換（`propagator=None` で従来通り）。動画版 notebook は薄いランチャー（gradio app を import 起動）のため .ipynb 再生成不要。
- **完了状況（2026-06-06）**: ✅ 全 4 ギャップ GREEN。
  - Gap A/ERR034: `build_sam2_tb_video_pipeline(propagator=None)` 依存注入 + `get_video_pipeline` で `entry_by_id`→`SAM2VideoPropagator` 構築・注入（checkpoint は `_resolve_project_path` で絶対化）。
  - Gap B/ERR032: tracker/background Dropdown を可視セクション（## 3 SAM系 / ## 4 背景透過系）へ移動。Advanced には tb_jit/tb_threshold/crop_padding のみ。
  - Gap C/ERR033: 「表示中フレームを再取得」を `extract_prompt_frame(input_video, prompt_frame_idx, frame_step)` へ再配線（シーク連動、prompt_frame_idx は出力で上書きしない）。
  - Gap D/ERR031: prompt_canvas に `sources=[]` + download/fullscreen 無効を付与（静止画版と一致、ERR026 解消）。
  - **テスト**: `.venv\Scripts\python.exe -m pytest -m "not integration" -q` → `134 passed, 1 skipped, 3 deselected`。smoke `--help` OK。途中 prompt_mode 行の reformat による IndentationError と既存テスト退行を発生させたが単一行形式へ復元して解消。
  - **Playwright UI 確認**: 7861 (CUDA_VISIBLE_DEVICES=-1) で起動し目視確認 — Prompt Canvas がプレースホルダー表示・アップロード UI なし（Gap D✓）、トラッカー Dropdown が ## 3 SAM系 に可視（Gap B✓）、背景除去 Dropdown が ## 4 背景透過系・実行ボタン前に可視（Gap B✓）、「表示中フレームを再取得」「シーク位置を SAM2 に反映」ボタン存在（Gap C✓）、Advanced 折りたたみにモデル Dropdown なし。
  - **サブエージェントレビュー（Explore）**: APPROVE。Critical/High なし。Low 指摘（`_resolve_project_path` が registry の絶対 checkpoint_path を無検査で返す）は registry が管理者制御ファイルで外部入力でないため許容・据え置き。torch.load weights_only 違反なし、try/except: pass なし、segment-anything/samurai 直接変更なし、後方互換維持を確認。
  - **ERROR_LOG**: ERR031（canvas空白/sources=[]欠落）・ERR032（Dropdown埋没）・ERR033（再取得ボタン非連動）・ERR034（tracker選択がpipeline非反映）を追記。

### タスク名: 動画版 Phase A〜E 実装（2026-06-05 着手, large）
- **目的**: `2026-06-02_SAM2動画_複合対象トラッキング_フレーム選択_双方向伝播_計画.md` の Phase A〜E（§11.3 で次フェーズに先送りされていた部分）を実装する。複合対象（例 `person playing drums`）を「複数 bbox → 各 obj_id 登録 → frame ごと union → 双方向伝播」で確実に追跡する。
- **ギャップ確認（精読済み）**:
  - `ui_helpers.empty_prompt_state()` に `"boxes"` キーなし → Phase A
  - `copy_prompt_state()` / `draw_prompt_overlay()` が `boxes` を扱わない → Phase A
  - `SAM2VideoPropagator.run` に `boxes` / `prompt_frame_idx` / `bidirectional` なし、`frame_idx` が 0 ハードコード、forward only、単一 obj → Phase B
  - movie app にフレーム選択 Slider / 候補 CheckboxGroup / `apply_selected_boxes` / bidirectional 無し → Phase D
- **設計判断**:
  - `prompt_frame_idx` = propagator が受け取る **frames リスト内の 0-based index**。UI のフレーム Slider は「サンプリング後シーケンス index（0〜max_frames-1）」を指し、preview は raw_index = slider * frame_step で抽出（VideoReader のサンプリングと一致）させ index 整合を担保。
  - union は propagator 内で `union_masks(mode="or")` 相当（np.any）で frame ごとに 1 枚へ統合。下流契約 `frame_masks: source_index→1枚` は不変 → TransparentBG/Writer 改修不要（Phase C は結線確認のみ）。
  - 後方互換: `boxes=None` のとき従来の単一 box/point・frame_idx=0・forward only パスを完全維持。
- **手順**: WHITEBOARD(済) → RED テスト → Phase A→B→C→D → GREEN → サブエージェントレビュー → Playwright UI 確認 → 記録更新。ユーザーは「Phase A〜E 全部実装」を承認済み。
- **完了状況（2026-06-05）**: ✅ Phase A（`ui_helpers` boxes 対応）/ B（`SAM2VideoPropagator` multi-box・prompt_frame_idx・bidirectional union）/ C（pipeline 結線は run kwargs auto-socket で改修不要と確認）/ D（movie app に フレーム Slider・このフレームを表示・双方向 Checkbox・候補 CheckboxGroup・`apply_selected_boxes` 追加）/ E（GREEN）すべて完了。
  - **テスト**: `127 passed, 1 skipped, 3 deselected`（union 伝播テストは torch 不在環境で `importorskip` により skip、GPU 環境で実行可）。
  - **サブエージェントレビュー（Explore）**: 重大バグなし。中程度指摘 #1（候補ラベルの bbox 抽出が phrase 内 `[]` で誤マッチ）を rsplit 方式へ修正、未使用となった `import re` を削除。#2（frame_step 整合）はコメント追記で補強、#3（候補 parse の握りつぶし）は軽微につき据え置き。
  - **Playwright UI 確認**: `gradio_app_..._for_Movie.py` を 7861 で起動、5 つの新コントロール（起点フレーム Slider / このフレームを表示 / 双方向伝播 Checkbox / 候補 bbox CheckboxGroup / 反映ボタン）の描画と info= 完備を確認。Prompt Canvas プレースホルダーの「Only first-frame prompt」表記を frame 選択対応の文言へ更新。スクショ: `outputs/movie_ui_phase_d.png`。
  - **追補（2026-06-06）**: `gr.Video` の seek / pause / loadedmetadata を `Blocks(js=...)` で拾う seek-sync ブリッジを追加し、入力動画のシーク位置と `prompt_frame_idx` Slider を自動同期する UX に更新。画面の見出し順も `フレーム取得 → DINO → SAM → 背景透過` に整理し、`frame_step` をフレーム取得系へ移動した。
  - **共通処理影響**: `ui_helpers.py` 変更は加算的・後方互換（静止画版も同 GREEN スイートで担保）。

### タスク名: 役割別モデル差し替え（レジストリ + Gradio プルダウン）計画策定（2026-06-03 設計のみ）
- **目的**: detector(GroundingDINO) / tracker(SAM2,SAMURAI) / background(transparent-background) を config 駆動で差し替え、Gradio に役割別プルダウンを設ける設計を策定。コード実装はまだ行わない（large・設計先行）
- **成果物**: `2026-06-03_モデル差し替え_レジストリ_Gradioプルダウン_計画.md`
- **ユーザー指示3点を反映**:
  1. SAM2 ⇄ SAMURAI の切替は**環境・インストール分離**を前提（同名 `sam2` パッケージのため 1 プロセス内ランタイム切替はしない）。`is_available()` は各環境にインストールされた変種を検出し `requires` 一致 entry のみ Dropdown に出す（§5.1）
  2. **GPU 必須**（プロ用途）。`require_gpu_for_heavy_inference` 踏襲。現状 TB/MAM は未呼び出しのため実装フェーズで追加（§2.0）
  3. transparent-background をベースに開発。背景透過モデルも将来切替可能にするため **`MAMAlphaPredictor` の I/O 契約と background ロール統一契約（rgba/alpha/preview/matte_result）+ `MAMBackgroundAdapter` 要件**を §3.2 に定義（実装は次フェーズ）
- **サブエージェントレビュー（Explore）**: 判定 **REVISE**。指摘を反映: (a) metadata の `model_id` は現状 TB 未実装→実装フェーズで constructor `registry_entry_id` 追加と明記、(b) GPU チェック未呼び出しの現状ギャップを §2.0 に明記、(c) `model_params` は dict ではなく run() 個別引数を registry entry から callback 展開と明確化、(d) アダプタの entry_id は constructor 経由、(e) is_available の変種検出は import + 環境変数（同名フォークのため import 単独では判別不可）で行うと明示
- **残作業**: 実装フェーズ（Phase R1 レジストリローダ〜R4 docs）はユーザー承認後に着手
- **Phase R1 完了記録（2026-06-03）**:
  - 実装: `config/inference_models.toml`（detector 1件・tracker 4件・background 3件）
  - 実装: `pipelines/components/model_registry.py`（`load_model_registry` / `entries_for` / `entry_by_id` / `is_available` / `build_dropdown_choices` / `clear_registry_cache`、`threading.Lock` によるスレッドセーフキャッシュ）
  - `pipelines/components/__init__.py` にエクスポート追加（`clear_registry_cache` 含む）
  - テスト: `tests/unit/test_model_registry.py` 21件 PASSED
  - サブエージェントレビュー（Explore）: デプロイブロッカー3件（`clear_registry_cache` 欠落 / `pytest.raises(ValueError, KeyError)` 曖昧 / スレッドセーフ未対応）→ 全修正完了。中程度指摘（`is_available` docstring / 未知 requires への `warnings.warn`）→ 対応完了。
- **次のアクション**: Phase R3（tracker Dropdown 結線 or MAMBackgroundAdapter）
- **Phase R2 完了記録（2026-06-04）**:
  - 実装: 3アプリ（静止画・動画・MAM）に `gr.Dropdown(build_dropdown_choices("background"))` 追加、`_PIPELINE_CACHE` で pipeline 再利用、`entry_by_id` で `tb_mode` 取得
  - 動画版: `run_video_background_removal` の `tracker_model` / `background_model` / `tb_jit` を位置引数（デフォルトなし）にして SyntaxError を回避（後続の `tb_threshold` 等もデフォルトなしのため）
  - MAM版: `background_model` パラメータ受け取りのみ・パイプライン未接続（暫定）。Dropdown `info=` に「現バージョンでは MAM パイプラインに未接続」と明記
  - テスト: 7件 RED→GREEN。全 122 テスト PASS（3 deselected: integration）
  - サブエージェントレビュー（Explore）: 全観点 PASS。MEDIUM指摘「動画版 `tb_jit` デフォルト `= False` 明示化」は後続位置引数の制約上変更不要と判断（現状正しい）
- **関連 ERR 横展開**: ERR018（include_outputs_from 明示）、ERR030（RAM 非保持）

### タスク名: トラッキング可視化 UI + SAMURAI config切替（2026-06-03 完了）
- **目的**: (1) SAM2/SAMURAI のマスクが動画フレームを正しく追従しているか目視確認できる「Tracking Overlay」動画を生成し Gradio に表示する。(2) 今後より良いトラッキングモデルへ柔軟・安定に差し替えられるよう、config/env 駆動でモデル切替し、使用 config / samurai_mode を mask metadata に記録する
- **規模**: large（新 Component + pipeline 結線 + Gradio UI + docs）。TDD フロー 1〜11 全適用、サブエージェントレビュー必須、UI 変更は Playwright 確認必須
- **設計方針（疎結合 / YAGNI）**:
  - 純粋描画関数 `render_tracking_overlay_frame()` を `pipelines/components/common.py` に追加（単一責務・テスト容易）
  - 副作用を持つ `TrackingOverlayWriter` Component を `video_model_components.py` に追加。入力 frames/masks/metadata/enabled/progress_callback、出力 `overlay_video_path`。frame ごとに輪郭+半透明塗りを描き mp4 へ逐次書き出し、RAM に全 frame を保持しない（ERR030）
  - SAMURAI は専用抽象化を作らず、既存 `SAM2_CONFIG_NAME` / `SAM2_CKPT_PATH` env 切替を正式機構とし、`SAM2VideoPropagator` が masks metadata に `tracker_config` / `tracker_checkpoint` / `samurai_mode` を記録
- **残作業**: なし（完了）
- **完了記録（2026-06-03）**:
  - 実装: `render_tracking_overlay_frame()`（`pipelines/components/common.py`）、`TrackingOverlayWriter`（`pipelines/components/video_model_components.py`、frame ごとに mp4+PNG 逐次書き出し・RAM 非保持）、pipeline 結線（`pipelines/sam2_tb_video_pipeline.py`）、Gradio UI（`overlay_enabled` Checkbox + `Tracking Overlay (追跡確認用)` Video 出力、`include_outputs_from` に `tracking_overlay`、stage progress range 追加）
  - SAMURAI: `SAM2VideoPropagator.tracker_metadata()` が `tracker_config` / `tracker_checkpoint` / `samurai_mode` を返し overlay metadata へ伝搬
  - テスト: `tests/unit/test_tracking_overlay.py`（新規 5件）、`tests/unit/test_video_pipeline_wiring.py`（overlay/samurai metadata 追加）。非 integration 全体 **94 passed, 3 deselected**
  - サブエージェントレビュー（reviewing-code）: 判定 **APPROVE**。minor 4件中 #2（overlay metadata に `tracker_checkpoint` 欠落）を修正。#1 progress range 表示順は表示のみ・#3 output_mode 無視は YAGNI・#4 webapp 確認は実施で対応不要
  - UI 検証（ui-ux-pro-max / webapp-testing + Playwright `tests/manual/verify_tracking_overlay_ui.py`）: Checkbox / Video 出力 / info text を画面で確認（3/3 OK + スクショ確認）
  - 回帰修正: 自己混入した jupytext 単一行 substring 回帰（`prompt_mode` / `max_frames` / `frame_step` の Radio/Slider 開始行）を単一行へ戻して解消
  - docs: `.github/copilot-instructions.md`（samurai/ 直接変更禁止＋config切替＋Tracking Overlay 要件＋REMINDER）、`REFERENCE.md`（SAMURAI / Tracking Overlay セクション追加）
  - テスト省略記録: docs（markdown）更新は挙動変更なしのため RED テスト省略（フロー step 7）
- **関連 ERR 横展開**: ERR018（Haystack Component 契約 / include_outputs_from）、ERR029（進捗）、ERR030（全 frame RAM 保持禁止・逐次保存）

### タスク名: エラーログ09 動画進捗/UX改善（2026-05-29）
- **目的**: `Sam2_Transparent_Background_Haystack_for_Movie.ipynb` の初回実行が 5% 付近で長時間止まって見える問題を解消し、使い方・フロー順・パラメーター説明を改善する
- **進捗**: 完了
- **変更ファイル**: `pipelines/components/video_model_components.py`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `tests/unit/test_video_pipeline_wiring.py`, `tests/unit/test_jupytext_notebooks.py`, `.github/copilot-instructions.md`, `REFERENCE.md`, `ERROR_LOG.md`, `エラーログ/エラーログ_09.md`, `WHITEBOARD.md`
- **原因**: エラーログ09には final traceback が欠けていたが、240 frames の動画読込と SAM2 propagation は完走していた。UI は Pipeline 開始時の 5% 表示後に内部 stage 進捗を出しておらず、初回のモデル読込・SAM2伝搬・transparent-background 処理が停止に見えていた
- **対応**:
  1. end-to-end Pipeline は維持し、`VideoReader` / `SAM2VideoPropagator` / `TransparentBGVideoExtractor` / `VideoWriter` / `FrameSequenceWriter` に任意の `progress_callback` を追加
  2. Gradio 側で stage progress を `video_reader` / `sam2_video` / `transparent_bg` / writer にマッピングし、エラー時は最後の stage と elapsed 秒を表示
  3. 初回既定を `max_frames=60`, `frame_step=1` のクイックプレビューへ変更
  4. UI と notebook に箇条書きフロー、`SAM2 Prompt Canvas` の説明、bbox の「対角 2 点」説明、Text Prompt→SAM2→transparent-background の順序説明、パラメーター説明表を追加
  5. `ERROR_LOG.md` に ERR029、`エラーログ/エラーログ_09.md` に原因・修正・再発防止を追記
- **関連 ERR 横展開**: ERR004/ERR006（GroundingDINO CUDA ops）、ERR010/ERR025（SAM2/GPU preflight）、ERR018（Haystack Component 契約）、ERR028（VideoWriter warm_up 契約）、ERR029（動画進捗/長時間処理）
- **UI 確認**: `.github/skills/webapp-testing/scripts/with_server.py` で Gradio を `127.0.0.1:7861` に実起動し、Playwright で `movie_ui_error09.png` と `movie_ui_error09_elements.txt` を取得。説明文、Text Prompt、Advanced パラメーター、操作要素を確認
- **UX 判断**: 必須入力は「動画 + SAM2 prompt」のまま増やさず、Text Prompt は任意導線。初回は短尺 preview を既定にし、最終出力時だけ Advanced で frame 数を増やす
- **検証**: `tests/unit/test_video_pipeline_wiring.py tests/unit/test_jupytext_notebooks.py` 38 passed。`gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help` 成功。`git diff --check -- . ':!*.ipynb'` 成功。サブエージェント code-review の進捗非単調指摘を反映済み

### タスク名: 動画版 Text Prompt / GroundingDINO 導線復旧と VideoWriter warm_up 修正（2026-05-29）
- **目的**: `VideoWriter.warm_up() missing 1 required positional argument: 'frame_shape'` を解消し、動画版でも `person playing drums` / `person riding bicycle` のような複合対象を Text Prompt から選べる導線を復旧する
- **進捗**: 完了
- **変更ファイル**: `pipelines/components/video_model_components.py`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `tests/unit/test_video_pipeline_wiring.py`, `tests/unit/test_jupytext_notebooks.py`, `.github/copilot-instructions.md`, `REFERENCE.md`, `ERROR_LOG.md`, `エラーログ/エラーログ_08.md`, `WHITEBOARD.md`
- **原因**:
  1. Haystack は `warm_up()` を no-arg で呼ぶが、`VideoWriter.warm_up(frame_shape, ...)` が runtime frame shape に依存していた
  2. 動画版は静止画版にある Text Prompt / GroundingDINO bbox 作成導線を持たず、複合対象を第 1 フレームで意味的に指定するプロジェクト目的に合っていなかった
- **対応**:
  1. `VideoWriter.warm_up()` を no-op / no-arg にし、codec 選択は `_select_rgba_codec(...)` として `run()` 内へ移動
  2. 動画版 UI に `Optional: Text Prompt to Box (GroundingDINO)` を追加し、検出 top bbox を `prompt_state["box"]` にコピー
  3. Movie Notebook で GroundingDINO checkpoint 取得と `GROUNDING_DINO_CKPT_PATH` 設定を追加し、Jupytext で `.ipynb` を再生成
  4. `.github/copilot-instructions.md` / `REFERENCE.md` に、静止画 / 動画の両方で Text Prompt / GroundingDINO 導線を維持する規約を追記
  5. `ERROR_LOG.md` に ERR028 を追加
- **関連 ERR 横展開**: ERR018（Haystack 中間出力 / Component 契約）、ERR019（SAM2 prompt UI）、ERR021/ERR026（Prompt Canvas）、ERR023/ERR024/ERR005（GroundingDINO 依存・互換）、ERR025（GPU policy）、ERR027（Colab public URL）
- **UI 確認**: `.github/skills/webapp-testing/scripts/with_server.py` で Gradio を `127.0.0.1:7861` に実起動し、Playwright で `movie_ui.png` スクリーンショットと `movie_ui_elements.txt` 操作要素一覧を取得。Text Prompt textbox、`Text Prompt から bbox を検出`、Detected boxes、既存の bbox / Extend / Run 導線を確認
- **UX 判断**: Text Prompt は任意 accordion に置き、既存の最短手動 bbox フローは維持。必須入力は「動画 + bbox/point」のまま増やさず、複合対象時だけ Text Prompt を使える設計にした
- **検証**: `tests/unit/test_video_pipeline_wiring.py tests/unit/test_jupytext_notebooks.py` 37 passed。`gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help` 成功。Playwright UI 確認成功。サブエージェント code-review で重要指摘なし

### タスク名: Colab Gradio share frpc 欠落 / 127.0.0.1 接続拒否対応（2026-05-29）
- **目的**: Colab Gradio 起動時に public share URL が生成されず、`http://127.0.0.1:7861` を開いて `ERR_CONNECTION_REFUSED` になる問題を修正する
- **進捗**: 完了
- **変更ファイル**: `Matting_Anything_Haystack.py`, `Matting_Anything_Haystack.ipynb`, `Sam2_Transparent_Background_Haystack.py`, `Sam2_Transparent_Background_Haystack.ipynb`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `Sam2_Transparent_Background_Haystack_for_Movie.ipynb`, `tests/unit/test_jupytext_notebooks.py`, `ERROR_LOG.md`, `REFERENCE.md`, `WHITEBOARD.md`
- **原因**: Colab の `127.0.0.1` は Colab VM 内部の local URL で手元ブラウザから開けない。Gradio share tunnel 用 `frpc_linux_amd64_v0.3` が package 配下に欠落し、public URL が生成されていなかった
- **対応**:
  1. 前回追加した `ensure_gradio_share_binary_for_colab()` / frpc 手動取得 / checksum fail-fast を撤回
  2. Haystack 版 Colab Notebook 3本の Gradio 起動セルで、`google.colab` の import spec による Colab 判定へ変更
  3. Colab では Gradio 5 の既定 share 機能に任せて `--share` を渡し、`Running on public URL` の `gradio.live` を開くよう案内
  4. Jupytext 正本 `.py` から `.ipynb` を再生成
  5. `ERROR_LOG.md` に ERR027、`REFERENCE.md` に Colab Gradio share URL ルールを追加
- **関連 ERR 横展開**: ERR010（SAM2 import preflight）、ERR011/ERR016（Gradio 接続系汎用表示）、ERR023（GroundingDINO runtime 依存）、ERR025（Colab GPU preflight）、ERR026（Gradio `Connection errored out` はサーバーログを一次情報にする）
- **レビュー**: `code-review` サブエージェントでレビュー実施後、ユーザー指摘により checksum fail-fast は Colab UX を悪化させる過剰介入と判断して撤回
- **検証**: `tests/unit/test_jupytext_notebooks.py` 29 passed。`gradio_app_haystack.py --help`, `gradio_app_sam2_transparent_BG_haystack.py --help`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help` 成功。`git diff --check -- . ':!*.ipynb'` 成功

### タスク名: レビュー必須化・共通処理影響確認ルール反映（2026-05-29）
- **目的**: コードレビュー、設計 / 実装プランレビュー、仕様影響レビューを差分の大小・行数・ファイル数に関わらず必須化し、共通処理変更時に静止画版 / 動画版 / notebook / pipeline など差分ファイルの代表経路も挙動確認する運用を明文化する
- **進捗**: 完了
- **変更ファイル**: `.github/copilot-instructions.md`, `.github/instructions/workflow.instructions.md`, `WHITEBOARD.md`
- **変更内容**:
  1. レビュー step を、コード / 設計 / 実装プラン / 仕様影響レビューは作成・変更・提示した場合にサブエージェントレビュー必須となるよう更新
  2. `.github/skills/reviewing-code/` はサブエージェントレビューを起動・整理する入口として扱うことを明記
  3. 共通処理変更時の確認対象として、静止画 Gradio、動画版 Gradio、Haystack Notebook 正本 `.py`、MAM Haystack Gradio、関連 pipeline / unit test を明記
  4. 共通処理例として `pipelines/components/common.py`, `model_components.py`, `video_model_components.py`, `ui_helpers.py`, 設定 / 評価 / 前処理モジュールを明記
  5. 検証コマンドに MAM Haystack Gradio smoke を追加
  6. サブエージェント利用不可時のセルフレビューは完全な代替ではなく、利用不可理由をユーザーへ明示するルールを追記
- **レビュー**: `Explore` サブエージェントでレビュー実施。frontmatter 追加提案は `copilot-instructions.md` では不要と判断して不採用。レビュー手段統一、共通処理確認対象具体化、MAM smoke 追加、セルフレビュー注意書きは反映済み
- **検証**: プロンプト文書修正のみのため pytest はスキップ。`git diff --check -- .github/copilot-instructions.md .github/instructions/workflow.instructions.md WHITEBOARD.md` 成功。`get_errors` は修正前の同一ファイルアンカー診断を引き続き返したが、現ファイルには `](#...)` 形式の markdown link がないため拡張機能側の診断キャッシュと判断

### タスク名: copilot-instructions 診断修正（2026-05-29）
- **目的**: Chat Customizations Evaluations の指定診断に従い、`.github/copilot-instructions.md` の曖昧さ、矛盾、優先順位、検証 OS 判定、領域横断チェック、REMINDER の正本関係を明確化する
- **進捗**: 完了
- **変更ファイル**: `.github/copilot-instructions.md`, `WHITEBOARD.md`
- **変更内容**:
  1. 代表呼び出し元の選定基準を「変更した関数 / クラスを直接呼ぶ箇所、または同じ public API を共有するファイル」に定義
  2. MD 間の優先順位を `ERROR_LOG.md` > `REFERENCE.md` > `WHITEBOARD.md` に明文化
  3. medium / large の実装前 WHITEBOARD 更新、small のテスト省略記録、判別困難時の RED テスト既定を整理
  4. 大きな差分のレビュー閾値を 100 行超または 3 ファイル超に定義
  5. 変更規模の境界例、検証コマンドの OS 判定、禁止事項衝突時の暫定対応禁止を追記
  6. 領域横断変更の ERR 照合表、重いモデル初期化の定量基準、GroundingDINO `--no-build-isolation` ルールを追記
  7. REMINDER は本文を正とする要約であることを明記し、壊れた同一ファイルアンカーを避けて本文見出し参照に整理
- **検証**: プロンプト文書修正のみのため pytest はスキップ。`git diff --check -- .github/copilot-instructions.md WHITEBOARD.md` 成功。`get_errors` は修正前の同一ファイルアンカー診断を残して返したが、現ファイルには該当 markdown link がないため拡張機能側の診断キャッシュと判断

### タスク名: copilot-instructions /chronicle improve 反映（2026-05-29）
- **目的**: 過去セッションの friction（Preflight 繰り返し指示、SAM2/GroundingDINO 遅延調査の計測不足、Colab Gradio 起動後エラー、SAM2 Prompt Canvas 接続エラー）を `.github/copilot-instructions.md` に反映する
- **進捗**: 完了
- **変更ファイル**: `.github/copilot-instructions.md`, `WHITEBOARD.md`
- **変更内容**:
  1. エラー対応開始時に instruction 読み込み状態・標準 Preflight・参照エラーログ・関連 ERR 横展開を明示するルールを追加
  2. SAM2 Prompt Canvas は `sources=[]` かつ `interactive=True` を維持する top-level ルールを追加
  3. SAM2 / GroundingDINO Colab Gradio 起動前に `sam2` import、CUDA ポリシー、GroundingDINO runtime 依存を確認し fail fast するルールを追加
  4. SAM2 / GroundingDINO 遅延調査では stage timing / CUDA / device / checkpoint / cache を計測してから判断するルールを追加
- **検証**: ドキュメント変更のみ。`git diff --check` 成功

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

## 2026-05-29 13:02 作業開始: 動画版 Text Prompt / VideoWriter warm_up
- 対象: gradio_app_sam2_transparent_BG_haystack_for_Movie.py, pipelines/components/video_model_components.py, tests.
- 方針: Haystack warm_up は無引数化し、動画版には任意の GroundingDINO Text Prompt -> bbox -> SAM2 video prompt 導線を追加する。
- UI 変更のため Gradio 実起動と Playwright 確認を実施する。

## 2026-05-29 15:28 作業開始: エラーログ09 動画進捗/UX改善
- instruction 読み込み確認済み: .github/copilot-instructions.md と .github/instructions/workflow.instructions.md を確認。
- Preflight: WHITEBOARD/REFERENCE/ERROR_LOG/エラーログ_09 と project-reference/error-knowledge-base を確認。
- 方針: end-to-end Pipeline は維持し、Component progress callback で 5% 固定表示を解消。初回クイックプレビュー既定と使い方説明/パラメータ説明を改善する。

## 2026-05-29 19:18 作業開始: エラーログ10 Colab RAM クラッシュ対策
- instruction 読み込み確認済み: .github/copilot-instructions.md と .github/instructions/workflow.instructions.md を確認。
- Preflight: WHITEBOARD/REFERENCE/ERROR_LOG/エラーログ_10 と project-reference/error-knowledge-base を確認。
- 方針: transparent-background の RGBA/alpha/preview frame 全保持をやめ、処理中に動画/連番へ書き出す。Gradio へ巨大中間出力を返さない。

## 2026-05-29 19:28 完了: エラーログ10 Colab RAM クラッシュ対策
- 原因: transparent-background 動画処理で RGB/mask に加えて RGBA/alpha/preview frame list を全保持し、Colab RAM 枯渇で traceback 前に runtime kill。
- 修正: TransparentBGVideoExtractor を streaming 保存へ変更し、compact matte を下流へ渡す。writer は保存済み path/dir を pass-through。Gradio は巨大中間出力を include_outputs_from に含めない。
- UX: 初回 max_frames 既定を 30 にして短尺プレビュー優先。Notebook 正本と ipynb を更新。
- 検証: pytest -m 'not integration' -q は 85 passed / 3 deselected。Movie Gradio --help と git diff --check は成功。code-review 指摘の writer progress callback も修正済み。

## 2026-05-29 19:31 追記: Text Prompt cache 解放
- Text Prompt 使用後の GroundingDINO/BERT 常駐を副因として確認。動画実行直前に release_text_detector() で cache を解放する。
- 検証: targeted tests / Movie --help / diff check 成功。非 integration 全体を再実行中。
