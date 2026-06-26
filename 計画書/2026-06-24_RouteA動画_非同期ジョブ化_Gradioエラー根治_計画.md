# RouteA 動画 非同期ジョブ化 / Gradio「Error」根治 計画書（ERR058）

- 作成日: 2026-06-24
- 対象: `gradio_app_sam2_ben2_route_a_for_Movie.py` / `Sam2_BEN2_RouteA_for_Movie.ipynb`
- スコープ: **RouteA 動画アプリのみ**（transparent_BG 動画は本計画の対象外、後続で共通基盤を流用）
- レビュー: **GPT-5.5 に委譲**（自前 subagent レビューは行わない。差分サマリ＋リスク箇所のレビューパケットを用意する）

---

## 1. ユーザー要件（逐語）

- 「非同期にしてくれ」
- 「クライアントがすぐ試す環境が欲しいので」
- 「スコープはるーとA」
- 「テキスト化して計画書に書くこと」

→ 確定事項:
1. 方針A（非同期ジョブ化）を採用する。
2. 進捗表示は **テキスト化**（`gr.Progress` バー → `gr.Timer` 毎 tick 更新のテキスト）。
3. スコープは RouteA のみ。
4. クライアントが Colab / 共有リンクでも安定して「すぐ試せる」ことを最優先する。

---

## 2. 背景・確定事実

- `エラーログ_26.md` の事実:
  - prewarm 成功（BEN2 が起動前に 36.5MB/s で完了 = ERR057 修正は有効）。
  - public URL 出力後、SAM2 propagation `100% 30/30 [01:56]`（116 秒）で完走。
  - ログはそこで途切れる = 停止点が propagation の**後**（BEN2 per-frame 抽出 / writer）へ一歩前進。
- ERR048 → 055 → 056 → 057 は全て keep-alive の対症療法で、毎回**停止点が 1 段ずつ動くだけ**。
- 現状アーキ: read → SAM2(116s) → BEN2 → write の全工程が **1 つの Gradio イベントハンドラ内で同期実行**され、壊れやすい無料トンネル越しに**単一 SSE を数分占有**する。

---

## 3. 根本原因（ERR058 の核）

これは **Gradio 単体のバグではない**。3 層の相互作用:

1. **Gradio**: 1 予測 = 1 本の長寿命 SSE 接続を全処理時間（数分）占有する。
   - Gradio 公式: SSE は POST と違い「タイムアウトしない」= ただし **localhost 直結が前提**。
2. **無料 gradio.live FRP トンネル（真犯人）**: Colab サーバを公開 URL へ中継する無料トンネル。**総接続時間 / ライフタイム上限**があり長時間接続を切る。keep-alive は無通信の隙間は埋めても**総処理時間そのものは縮められない**（ERR057 で実証）。
3. **Colab**: さらにもう 1 段の proxy + リソース制約。

結論: **長時間の単一リクエストが壊れやすいトンネル越しに存在し続ける限り、対症療法では根治しない。** リクエスト自体を短くする（非同期化）ことで、トンネルの長時間切断クラスのバグを原理的に消す。

---

## 4. 方針（非同期ジョブ化 = 方針A）

各 HTTP リクエストを **<1 秒**にし、重い処理を裏のデーモンスレッドで実行、UI は `gr.Timer` でジョブ状態をポーリングする。

- submit（実行ボタン）→ 入力検証 → ジョブ起動 → **即座に job_id を返す**（リクエスト終了）。
- 裏のワーカーがパイプラインを実行し、進捗を共有 `JobState` に書き込む。
- `gr.Timer`（約 1 秒間隔）が `JobState` をポーリングし、**テキストで進捗表示**を更新。
- 完了 → Timer が出力を返し自身を停止。エラー → `gr.Error` を送出。

これにより「長時間 SSE」が存在しなくなり、Colab / 共有リンク経由でも安定する（クライアントがすぐ試せる）。

---

## 5. 実装ステップ

### Phase 1: ジョブ基盤（GPU 不要・テスト容易）

1. 新規 `pipelines/job_manager.py`:
   - `JobState`（dataclass）: `status`("running"/"done"/"error"), `fraction`(float), `description`(str), `result`(任意), `error`(str|None), `created_at`, `updated_at`。
   - `JobManager`: スレッドセーフな `dict[job_id -> JobState]`（`threading.Lock`）。
   - `submit(work: Callable[[report], Any]) -> job_id`:
     - daemon thread を起動。`work` は `report(fraction, description)` コールバックを受け取る。
     - 例外は **握り潰さず** `JobState.error` に保持（`raise` を捨てない）。poll 側で `gr.Error` 化する。
     - 成功時は `result` を格納し `status="done"`。
   - `snapshot(job_id) -> JobState`（イミュータブルなコピーを返す）。
   - 古いジョブの簡易掃除（任意・TTL）。
2. RED → GREEN: `tests/unit/test_job_manager.py`
   - submit が裏で `work` を実行し進捗を反映する。
   - `running` → `done` の遷移。
   - 例外が捕捉され `error` 化される（**握り潰さない**ことを検証）。
   - `snapshot` の不変性（呼び出し側の変更が内部状態に波及しない）。

### Phase 2: RouteA アプリの非同期配線（*Phase 1 依存*）

3. `run_route_a_background_removal` のコアを **`progress_report(fraction, description)` を受け取る関数**に分離（`gr.Progress` 依存を除去）。入力検証（prompt 未指定の `gr.Error` 等）は submit 側で即時実行（瞬時なので同期で OK / ERR037 の fail-fast を維持）。
4. `start_route_a_job(...)`:
   - 入力検証 → `JobManager.submit` でパイプライン実行（`progress_report` が `JobState` を更新）。
   - 戻り値: `job_id`(`gr.State`), status テキスト, `gr.Timer(active=True)`, `run_btn` 無効化(`gr.update(interactive=False)`)。
5. `poll_route_a_job(job_id)`（`gr.Timer.tick` 束縛）:
   - `snapshot` 取得。
   - `running` → 進捗テキスト更新 + Timer 継続 + outputs は `gr.update()` 据置。
   - `done` → outputs 返却 + status テキスト + `gr.Timer(active=False)` + `run_btn` 再有効化。
   - `error` → `gr.Error`（+ Timer 停止 + `run_btn` 再有効化）。
6. 配線:
   - `run_btn.click(start_route_a_job, inputs=[...既存...], outputs=[job_id_state, run_status, timer, run_btn])`。
   - `timer.tick(poll_route_a_job, inputs=[job_id_state], outputs=[rgba_video, alpha_video, preview_video, tracking_overlay_video, sequence_files, sequence_dirs, run_status, timer, run_btn])`。
7. 進捗表示の **テキスト化**:
   - `gr.Progress` バー依存を撤去し、`run_status`（Textbox / Markdown）を Timer 毎 tick で更新。
   - 既存 keep-alive（`_ProgressKeepAlive` / `run_with_progress_keepalive`）は**撤去せず温存**し、`progress_callback` → `JobState` 更新の内部生存表示として再利用（無害・二重防御）。
8. RED → GREEN: `tests/unit/test_route_a_async_wiring.py`（`JobManager` を fake / 実物で）
   - `start_route_a_job` が `job_id` + Timer 活性 + ボタン無効化を返す。
   - `poll_route_a_job` が `running` / `done` / `error` 各状態で正しい更新・`gr.Error` を返す。

### Phase 3: ドキュメント / 検証

9. `ERROR_LOG.md` に **ERR058** を詳細追記（症状 / `エラーログ_26.md` 根拠 / 3 層 root-cause / 「Gradio 単体バグでない」/ 非同期ジョブ化による根治 / ERR035 留意）。
10. `WHITEBOARD.md` 更新（ERR058 行 + 最終更新日 + 次アクション + テスト省略理由が無いこと）。
11. `Sam2_BEN2_RouteA_for_Movie.ipynb` は `__main__` 経由で起動するため追加改修は基本不要（必要時のみ Jupytext `.py` 正本で同期）。

---

## 6. 検証

1. `.venv\Scripts\python.exe -m pytest -m "not integration" -q`（job_manager + wiring テスト GREEN）
2. `.venv\Scripts\python.exe gradio_app_sam2_ben2_route_a_for_Movie.py --help`（smoke）
3. `get_errors` = 0
4. **Playwright 実行時検証（ERR035）**: 実起動 → run → Timer ポーリングで進捗テキスト更新 → 完了で出力表示、を確認してから fixed 記録。
   - エージェント `.venv` に torch 未導入の場合、end-to-end はユーザー / GPT-5.5 の実機（または Colab）確認とする。
5. **レビュー: GPT-5.5 に委譲**。差分サマリ + リスク箇所のレビューパケットを用意（自前 subagent レビューはしない）。

---

## 7. 影響ファイル

| ファイル | 変更内容 |
|----------|----------|
| `pipelines/job_manager.py` | 新規。スレッドセーフなジョブ登録・進捗・結果/例外保持 |
| `gradio_app_sam2_ben2_route_a_for_Movie.py` | コア分離 / `start_route_a_job`・`poll_route_a_job` 追加 / `run_btn.click` + `gr.Timer` 配線 / 進捗テキスト化 |
| `pipelines/components/ben2_components.py` | 既存 keep-alive を `progress_report` 経由に温存（無害） |
| `pipelines/components/video_model_components.py` | 同上（keep-alive 温存） |
| `ERROR_LOG.md` / `WHITEBOARD.md` | ERR058 記録・更新 |
| `tests/unit/test_job_manager.py` | 新規テスト |
| `tests/unit/test_route_a_async_wiring.py` | 新規テスト |

---

## 8. Hard Rules 順守

- `torch.load(..., weights_only=False)` 禁止。
- `try/except: pass` 禁止（例外は `JobState.error` に保持し `gr.Error` で通知）。
- 設定値は `config/route_a.toml` 等から取得しハードコードしない。
- `segment-anything/` と `samurai/` は直接変更しない。
- `gr.Blocks()` の `queue()` は分離呼び出し（ERR001）。
- UI「fixed」記録は Playwright 実行時検証後（ERR035）。
- keep-alive / prewarm は撤去せず温存（二重防御）。

---

## 9. 留意点 / 今後

- transparent_BG 動画アプリ（`gradio_app_sam2_transparent_BG_haystack_for_Movie.py`）は同 `job_manager.py` を流用する fast-follow（本計画の対象外）。
- ローカル GPU 直結（`--share` なし）も接続層の根治として有効だが、本要件は「クライアントがすぐ試せる」= 共有リンク前提のため、**非同期ジョブ化を主対策**とする。

---

## 10. 実装進捗ログ（テキスト化）

> ユーザー要件「テキスト化して計画書に書くこと」に対応する実装の進捗記録。

### Phase 1 — ジョブ基盤（完了 ✅）

- `pipelines/job_manager.py` を新規作成（stdlib のみ・torch/gradio/GPU 非依存）。
  - `JobState`（`job_id` / `status`=running·done·error / `fraction` / `description` / `result` / `error` / `created_at` / `updated_at`）。
  - `JobManager.submit(work)`：daemon スレッドで `work(report)` を実行。`report(fraction, desc)` が JobState を更新。
  - 例外は `except Exception`（`# noqa: BLE001`）で捕捉し **握り潰さず** `error` に `f"{type(exc).__name__}: {exc}"` を保持（Hard Rule 順守）。
  - `snapshot(job_id)`：独立コピーを返す（未知 id は `KeyError`）。`cleanup(ttl_sec)`：TTL 超過ジョブを削除。
- `tests/unit/test_job_manager.py`（6 件・GREEN）：進捗+結果 / 例外を握り潰さない（"boom"+"ValueError" を確認）/ snapshot 独立性 / fraction クランプ（5.0→1.0）/ 未知 job KeyError / fake clock の cleanup。

### Phase 2 — RouteA 両タブの非同期配線（完了 ✅）

- `gradio_app_sam2_ben2_route_a_for_Movie.py`：
  - `_ProgressBridge`：`gr.Progress` 互換 `__call__(value, desc="")` を JobState 進捗へ橋渡し。既存 `build_video_progress_callback` を **無改変で再利用**。
  - `start_route_a_job` / `start_route_a_only_job`：fail-fast（動画・prompt 未指定で即 `gr.Error`）→ `JobManager.submit` → **即座に** `(job_id, 進捗テキスト, gr.Timer(active=True), btn 無効化)` を返却（リクエスト <1s）。
  - `poll_route_a_job` / `poll_route_a_only_job`（`gr.Timer.tick` 束縛）：running=進捗テキスト更新・出力据置・Timer 継続 / done=出力返却・Timer 停止・btn 復帰 / error=初回 tick で `gr.Error` 通知し `_REPORTED_JOB_ERRORS` で 2 回目以降の多重トースト抑止・UI 復帰。
  - 進捗表示は `gr.Progress` バーをやめ `run_status`（Markdown）の **テキスト更新**（`処理中… N%　<stage>`）に変更（トンネル安全）。
  - 追加コンポーネント：両タブに `gr.State("")`（job_id）＋ `gr.Timer(1.0, active=False)`。`run_btn.click(start_*) → outputs=[job_id, status, timer, btn]` ＋ `timer.tick(poll_*) → outputs=[...出力 + timer + btn]` で再配線。
  - keep-alive（ERR055/056）/ prewarm（ERR057）は撤去せず温存（二重防御）。
- `tests/unit/test_route_a_async_wiring.py`（7 件・GREEN）：importlib で app を import / 検証 `gr.Error` / job_id 返却 / 完了出力一致 / error 初回 `gr.Error`→2 回目リセット / running 進捗テキスト / BEN2 のみタブ。

### Phase 3 — 検証・記録（完了 ✅ / 実機検証のみ委譲）

- 非 integration 全体：**277 passed / 1 skipped / 3 deselected**（回帰なし。新規 14 件含む）。
- `gradio_app_sam2_ben2_route_a_for_Movie.py --help` smoke 正常、`get_errors` = 0。
- `ERROR_LOG.md` に ERR058、`WHITEBOARD.md` に完了行＋最終更新日を追記。
- **未了（委譲）**：Playwright 実行時検証（ERR035）＝実機（ローカル RTX 4090 もしくは Colab）で run→Timer ポーリング進捗更新→完了出力／失敗で 1 度だけ赤トースト＋UI 復帰、を確認してから「UI fixed」確定。レビューは GPT-5.5。

### GPT-5.5 レビュー用パケット（差分サマリ＋リスク箇所）

- 差分：新規 `pipelines/job_manager.py`・`tests/unit/test_job_manager.py`・`tests/unit/test_route_a_async_wiring.py`、`gradio_app_sam2_ben2_route_a_for_Movie.py`（ハンドラ追加・コンポーネント追加・両タブ再配線）。
- 重点レビュー観点（リスク箇所）：
  1. `gr.Timer` 活性化パターン（ハンドラから `gr.Timer(active=True/False)` を outputs として返す）が Gradio 5.9.1 で意図通りか。
  2. error 通知の「初回 tick で `gr.Error`→`_REPORTED_JOB_ERRORS` で以降抑止」設計（raise と Timer/btn リセットを同一 return で両立できないための分割）が UX/規約上妥当か。
  3. スコープ＝RouteA 両タブのみで、共通核（`run_route_a_background_removal` / `run_route_a_only_background_removal`）の戻り値契約（7-tuple / 6-tuple）を崩していないか。
  4. `JobManager` のスレッド安全性（daemon スレッド・snapshot の独立コピー・`_REPORTED_JOB_ERRORS` の参照）。
  5. ERR035：実機 Playwright 検証前は「実装完了」止まりで「UI fixed」未確定である点の確認。
