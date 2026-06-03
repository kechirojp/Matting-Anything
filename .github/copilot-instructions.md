# Matting-Anything — プロジェクト統合ルール

このプロジェクトの目的は
ドラムをたたいている人＝＞ドラム＋人
自転車に乗っている人＝＞自転車＋人
と
セグメンテイションできることを目指す
現状は
ドラムをたたいている人＝＞ドラム＋人を選びたいにもかかわらず人しか選ばないことがおきてしまうため
Matting anythingリポジトリを選んで実験している※静止画動画含めて

つまり
画像の意味解釈ができるモデル　そのオブジェクトが何なのか　もしくは発展的にそのオブジェクトが何をしているのか　意味をプロンプトなどでユーザーがテキスト入力などができるモデル

SAMに限らずオブジェクトをトラックできる機能をもったモデルで　動画に対応する
GroundingDINOなどの画像の意味解釈ができるモデルと組み合わせて　ユーザーがテキスト入力で意味的にオブジェクトを選べるようにする
プラス
SAM（Segment Anything Model）などオブジェクトトラッキング機能をバックボーンに用いた背景除去システム。
GroundingDINO、SAM / SAM2、transparent-background、Gradio 5、Haystack 2.x を組み合わせたデモと Colab を含む。

> **読み込み方**: このファイルは常時適用の短いルールカード。詳細は必要なときだけ `REFERENCE.md`、`ERROR_LOG.md`、`.github/instructions/workflow.instructions.md`、該当 skill を読む。

---

## 0. 作業手順（全エージェント共通）

### 実装フロー（テスト駆動）

1. **要件分析**: 変更対象、変更対象が import するモジュール、変更対象を import している直接の呼び出し元を確認する。直接の呼び出し元が 10 ファイルを超える場合は、変更した関数 / クラスを直接呼ぶ箇所、または同じ public API を共有するファイルから代表 3〜5 ファイルに限定し、その判断理由を `WHITEBOARD.md` に記録する。
2. **作業前確認**:最初に `.github/copilot-instructions.md` と該当する `.github/instructions/*.instructions.md` の読み込み状態、標準 Preflight 実施。次に `ERROR_LOG.md`、`REFERENCE.md`、`WHITEBOARD.md` を読む。`@whiteboard-manager` / `@error-knowledge-base` が使える場合は併用する。次にエラー対応の場合は参照した エラーログ / 関連 ERR 横展開を明示する。
3. **情報の優先順位**: エージェント出力と MD ファイルが食い違う場合は MD ファイルを正とし、差分を `WHITEBOARD.md` に記録する。MD 間の優先順位は `ERROR_LOG.md` > `REFERENCE.md` > `WHITEBOARD.md` とし、食い違いを発見した場合は `WHITEBOARD.md` に記録のうえユーザーへ確認する。
4. **設計提案**: medium / large の変更では、実装前に方針と影響範囲をユーザーへ示す。
5. **進捗記録**: medium / large の変更では、step 7 のテスト省略記録とは別に、実装開始前の `WHITEBOARD.md` 更新を必須とする。
6. **テスト記述（RED）**: 挙動変更は先に失敗する `pytest` を書く。GPU・外部モデル依存は `pytest.mark.integration` の骨格でもよい。
7. **テスト省略条件**: TOML 設定追加、コメント変更、プロンプト文書修正などテスト対象がない場合は、RED テストのみをスキップし、その旨を `WHITEBOARD.md` に記録する。small 変更では step 11 の完了記録に省略理由を併記し、実装前の WHITEBOARD 更新は不要とする。判別困難な場合は RED テストを書くことを既定とし、判断保留の理由を `WHITEBOARD.md` に記録する。
8. **実装（GREEN）**: 期待動作を満たす最小限の変更を行う。
9. **必要な整理**: 許可される整理は、今回変更した関数・クラス・テストと直近の重複除去に限る。別機能、広域整形、命名統一、設計変更は、ユーザーから明示依頼がある場合のみ行う。
10. **レビュー**: 実装後に差分と意図を確認する。コードレビュー、設計 / 実装プランのレビュー、仕様影響レビューは、それらを作成・変更・提示した場合、差分の大小・行数・ファイル数に関わらずサブエージェントで必ずレビューする。`.github/skills/reviewing-code/` はサブエージェントレビューを起動・整理する入口として使ってよい。
11. **記録更新**: 作業完了後に `WHITEBOARD.md` を更新する。新しいエラーを解決した場合は `ERROR_LOG.md` も更新する。

> **変更規模**
> - **small**: 誤字・コメント・docstring・文書のみで挙動変更なし。必須: 2, 7, 8, 11。step 7 の記録は 11 の完了記録に省略理由を併記して満たす。
> - **medium**: 1 関数〜1 ファイル内の機能追加・修正、または 1 つの TOML 閾値変更で推論結果が変わる変更。必須: 1〜11（4 は方針提示、6 は必要時）。
> - **large**: 複数ファイル、API 変更、新機能、複数 TOML の連動変更、公開 API / 設定キーの追加。必須: 1〜11。
> - 判断に迷う場合は上位カテゴリ（large 寄り）を選択する。例: docstring のサンプルコード変更で実行例に影響する可能性がある場合は medium、README の表記修正のみは small、Gradio UI と Pipeline 契約を同時に変える場合は large。

### 検証コマンド

実行 OS を `platform.system()` 相当で判定し、対応する側のコマンドのみを実行する。判定不能時は Windows コマンドを試行し、失敗したら macOS/Linux コマンドに切り替える。

- 非 integration 全体（Windows）: `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
- 非 integration 全体（macOS/Linux）: `.venv/bin/python -m pytest -m "not integration" -q`
- 単一テスト（Windows）: `.venv\Scripts\python.exe -m pytest tests\unit\test_xxx.py::test_name -q`
- 単一テスト（macOS/Linux）: `.venv/bin/python -m pytest tests/unit/test_xxx.py::test_name -q`
- Jupytext 生成（Windows）: `.venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py`
- Jupytext 生成（macOS/Linux）: `.venv/bin/python -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py`
- Gradio smoke（Windows）: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help`
- Gradio smoke（macOS/Linux）: `.venv/bin/python gradio_app_sam2_transparent_BG_haystack.py --help`
- 動画版 Gradio smoke（Windows）: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help`
- 動画版 Gradio smoke（macOS/Linux）: `.venv/bin/python gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help`
- MAM Haystack Gradio smoke（Windows）: `.venv\Scripts\python.exe gradio_app_haystack.py --help`
- MAM Haystack Gradio smoke（macOS/Linux）: `.venv/bin/python gradio_app_haystack.py --help`

---

## 1. 常時適用ルール

-  **SOLID / DRY / KISS / YAGNI** を徹底する。過剰な抽象化、過度な最適化、将来の機能を見越した実装は避ける。
- 単一の関数・クラス・テストに複数の機能を持たせない。1 つの関数・クラス・テストは 1 つの責任を持つべき。
- 疎結合を保ち、モジュール間の依存関係を最小限にする。共通処理は適切なユーティリティモジュールにまとめる。
- 疎結合ルール　単一責務ルールに基づき　本プロジェクトの場合　haystack Component は、Component 内で完結する処理を実装し、Gradio callback ではモデル推論やファイル保存などの重い処理を直接実装せず Component に委譲する。
- 正常に動いている環境・コードを、依頼外で変更しない。
- 既存 API、インターフェース、設定キー名を無断で破壊しない。
- `torch.load(..., weights_only=False)` は禁止。`weights_only=True` を使う。
- `try/except: pass` などでエラーを握りつぶさず、必ず raise または `gr.Error` で通知する。
- 設定値は `config/*.toml` または `utils/CONFIG` 経由で取得し、ハードコードしない。
- 評価指標は `evaluation/metrics.py` の定義を使い、重複実装しない。
- `outputs/` に git 管理対象ファイルを追加しない。
- `segment-anything/` は直接変更しない。変更が必要と判断した場合は、ラッパーモジュールで対応可能か検討し、不可能ならユーザーに確認する。
- `samurai/`（SAM2 fork）は直接変更しない。SAMURAI の有効化は `samurai/sam2/sam2/configs/samurai/sam2.1_hiera_*.yaml` の `samurai_mode: true` と `SAM2_CONFIG_NAME` / `SAM2_CKPT_PATH` の切替（config / 環境変数）で行い、SAM2 と同一 API を前提に独自抽象を増やさない（YAGNI）。Gradio Dropdown の tracker 選択切替は `config/inference_models.toml` の `requires` フィールドと環境変数 `INFERENCE_TRACKER_VARIANT`（`sam2_facebook` = SAM2 エントリのみ表示 / `sam2_samurai` = SAMURAI エントリのみ表示 / 未設定 = 全表示）で制御する。tracker 切替の痕跡は masks metadata（`tracker_config` / `tracker_checkpoint` / `samurai_mode`）に記録する。変更が必要と判断した場合はラッパーで対応可能か検討し、不可能ならユーザーに確認する。
- 禁止事項とユーザー依頼が衝突した場合は実装せず、衝突点を提示して確認を取る。ユーザー応答待ち中の暫定対応は禁止し、代替案を最大 3 件提示する。ユーザー承認で禁止事項を破った場合は、`ERROR_LOG.md` に承認日時・理由・該当 commit（未 commit なら作業ブランチと差分要約）を記録する。

---

## 2. 領域別トリガー

領域横断変更では、影響する領域の ERR を先に照合する。

| 変更領域 | 必ず確認する ERR |
|----------|------------------|
| Haystack + Gradio + GroundingDINO に影響する SAM2 UI / Text Prompt | ERR018 / ERR021 / ERR024 / ERR026 |
| 上記に加えて Notebook も変更 | ERR004 / ERR010 / ERR023 / ERR025 |
| SAM2 / GroundingDINO の遅延・CUDA・checkpoint 調査 | ERR004 / ERR006 / ERR010 / ERR023 / ERR024 / ERR025 |

### Haystack 2.x Pipeline

- Haystack 系の新規実装・改修では `.github/skills/haystack-pipeline/` を参照し、最初に Component I/O 契約と Pipeline 結線を決める。
- Gradio callback は UI 入力整形、`pipeline.run(...)`、出力整形のみ担当する。モデル推論・ファイル保存は Haystack Component に委譲する。
- 重いモデル初期化（100MB 超のファイル読み込み、または 1 秒超を要する初期化）は `warm_up()` に置き、import 時に checkpoint を読まない。
- 中間 Component 出力を読む場合は `pipeline.run(..., include_outputs_from={...})` を指定する（ERR018: leaf 以外の Component 出力は明示指定しないと返らない）。
- Component 境界では `MaskSet` / `SelectedMask` / `MatteResult` の安定 dict 契約を優先する。

### Gradio 5 / SAM2 UI

- `gr.Image(tool="sketch")` は廃止。`gr.ImageEditor` を使う（ERR002: Gradio 5 で sketch tool 廃止）。
- `block = block.queue()` は `None` を返す。`with gr.Blocks() as block:` の後に `block.queue()` を呼ぶ（ERR001: queue は in-place 操作）。
- `ImageEditor` 戻り値は `background`, `layers`, `composite` を使う（ERR003: `image` キー廃止）。
- RGBA (4ch) はモデル入力前に RGB (3ch) へ変換する（ERR008: channel 不一致回避）。
- SAM2 bbox / point 座標は `gr.Number` や `Textbox` で手入力させない（ERR017: bbox / point 座標手入力禁止）。
- SAM2 prompt 入力は `gr.Image(type="numpy", interactive=True)` を使う。Point 正負は `gr.Radio(["positive", "negative"], value="positive")` で明示する（ERR019: positive/negative Radio 仕様）。
- Haystack 版 SAM2 UI はアップロード用 `Input Image` とクリック用 `SAM2 Prompt Canvas` を分離する。Prompt Canvas はアップロード先にしないため `sources=[]` を維持し、クリックイベントを受けるため `interactive=True` を維持する（ERR021 / ERR026）。
- SAM2 bbox は確定後に `Extend Left/Right/Top/Bottom` の 4 ボタンで画像端へ揃えられる UI を提供する（ERR019: bbox Extend UI 仕様）。
- Haystack 版 SAM2 の静止画 / 動画 UI は、複合対象（例: `person playing drums`, `person riding bicycle`）を選べるよう、Text Prompt / GroundingDINO から bbox 候補を作る任意導線を維持する。動画版は複数 bbox を `prompt_state["boxes"]`（`CheckboxGroup` で選択）に保持し、`SAM2VideoPropagator` が各 bbox を obj_id 1..N として登録・frame ごとに OR union する。起点フレームは `prompt_frame_idx`（フレーム選択 Slider）で指定でき、`bidirectional=True` で前後双方向に伝搬する。これらは run kwargs auto-socket 経由で渡し pipeline 結線は変更しない。`boxes=None` のとき従来の単一 box/point・`prompt_frame_idx=0`・forward only パスを後方互換で維持する。
- SAM2 prompt / Text Prompt / Union UI のいずれかに UI 要素を追加・削除した場合は  `.github\skills\ui-ux-pro-max`,`.github/skills/webapp-testing/` によるブラウザ確認を必須とする。Gradio を実起動し Playwright でスクショ・操作要素一覧・想定フローを確認した上で、必須入力の削減と導線改善を提案する。

### GroundingDINO / transformers

- `GroundingDINO/` 配下の変更は必要最小限にとどめ、理由を `ERROR_LOG.md` または `WHITEBOARD.md` に記録する。
- GroundingDINO を使う Component は、`groundingdino.util.inference.Model` を import する前に `patch_transformers_bert_for_groundingdino()` を呼ぶ（ERR024: `BertModel.get_head_mask` 削除互換パッチ）。
- GroundingDINO の CUDA ビルドでは Python から `os.environ` を設定し、`pip install -e GroundingDINO --no-build-isolation` を使う（ERR004: CUDA build isolation 回避）。
- `bertwarper.py` は `get_extended_attention_mask(attention_mask, input_shape)` の新シグネチャで呼ぶ（ERR005: transformers 新旧シグネチャ差分）。
- `ms_deform_attn.py` は `CUDA_OPS_AVAILABLE` フラグを参照する（ERR006: CUDA ops import 失敗時の `_C` 未定義回避）。
- `checkpoint.checkpoint()` には `use_reentrant=False` を付ける（ERR007: PyTorch checkpoint 警告対策）。

### Colab / Notebook

- Haystack 版 notebook は `.py` の Jupytext percent 形式を正本にし、`.ipynb` は Jupytext で生成する。
- `.ipynb` を直接編集しない。内容変更は対応する `.py` に行う。
- Colab install cell では `-q` を使わず、CUDA / build エラーを隠さない。
- `!export VAR=...` は後続 `!pip install` に引き継がれない。`os.environ['VAR'] = ...` を使う（ERR004: `!` は別サブシェル）。
- Text Prompt / GroundingDINO を含む notebook では `transformers`, `addict`, `yapf`, `timm`, `supervision`, `pycocotools` と GroundingDINO checkpoint 取得を確認する。
- SAM2 / GroundingDINO を含む Colab Gradio 起動セルは、Gradio 公開前に `sam2` import、CUDA ポリシー、GroundingDINO runtime 依存を確認し、不足時は起動せず fail fast する（ERR010 / ERR023 / ERR025）。

### UI / UX
- UI 変更は `.github/skills/ui-ux-pro-max/`,`.github/skills/webapp-testing/` によるブラウザ確認を必須とする。Gradio を実起動し Playwright でスクショ・操作要素一覧・想定フローを確認した上で、必須入力の削減と導線改善を提案する。
- Gradio UI の全パラメーター（`gr.Slider` / `gr.Radio` / `gr.Checkbox` / `gr.Textbox` 等）には、誤解のない解説を `info=` で完備する。解説には (1) 数値の単位（px, frame 数, 0.0〜1.0 の正規化アルファ, 個数, 真偽値, 単位なしの選択値 など）、(2) 各値が具体的に何を変えるか（低い値 / 高い値 / ON / OFF / 各選択肢の意味）、(3) 推奨の目安値、を必ず含める。選択肢型は各 choice の意味を列挙する。UI パラメーターを追加・変更した場合は `info=` の完備を同時に行う。

### モデル・評価・学習

- 推論前に `eval()` と `torch.no_grad()` を使う。
- device は `torch.device("cuda" if torch.cuda.is_available() else "cpu")` で自動選択する。
- 正規化パラメータはモデル定義または TOML 設定から取得する。
- 背景除去モデル、SAM / SAM2、動画版、静止画版は細かい差分ファイルを持つため、共通処理を変更した場合は直接変更したファイルだけでなく、その共通処理を使う静止画版 / 動画版 / notebook / pipeline の代表経路も挙動確認する。共通処理の例は `pipelines/components/common.py`、`pipelines/components/model_components.py`、`pipelines/components/video_model_components.py`、`pipelines/components/ui_helpers.py`、設定 / 評価 / 前処理の共通モジュール。影響確認対象の例は静止画 Gradio、動画版 Gradio、Haystack Notebook 正本 `.py`、MAM Haystack Gradio、関連 pipeline / unit test。
- SAM2 / GroundingDINO の遅延・5分超過など性能問題は、修正前に `warm_up` / `set_image` / `predict` / Gradio callback total、`torch.cuda.is_available()`、モデル device、checkpoint / cache 状態を記録してから原因を判断する。
- 動画版の長時間処理は固定 progress 表示で放置しない。`VideoReader` を重複実行する Pipeline 分割は避け、Component に `progress_callback` を渡して動画読込 / SAM2 伝搬 / transparent-background / 書き出しの stage と frame 進捗を表示する（ERR029）。
- 動画版の transparent-background 出力は RAM に全 frame を保持しない。RGBA / alpha / preview は frame ごとに動画または PNG へ逐次保存し、Gradio callback の `include_outputs_from` に巨大な frame list を含む中間 Component を入れない（ERR030）。
- SAM2 / SAMURAI の追跡が対象へ正しく追従しているか目視確認できるよう、動画版には追跡 mask の輪郭+半透明塗りを元動画に重ねた Tracking Overlay 出力を提供する。描画は純粋関数（`pipelines/components/common.py` の `render_tracking_overlay_frame`）に分離し、書き出しは専用 Component（`TrackingOverlayWriter`）に委譲する。overlay も frame ごとに逐次書き出し（ERR030）、`progress_callback` で stage 進捗を出す（ERR029）。overlay metadata には tracker 種別（`tracker_config` / `tracker_checkpoint` / `samurai_mode`）を残す。
- Text Prompt / GroundingDINO 使用後に動画処理へ進む場合は、GroundingDINO / BERT cache を解放してから SAM2 / transparent-background を実行し、Colab RAM の同時常駐ピークを抑える（ERR030）。
- `networks/m2ms/conv_sam.py` を変更する場合は SAM I/O 整合性を確認する。
- `config/*.toml` を変更する場合は `trainer.py` との対応を確認する。

---

## 3. エラー報告フォーマット

エラー発生時は以下の形式で報告する。

```text
【エラー内容】
TypeError: 'NoneType' object is not subscriptable

【原因分析】
user_data が None の場合にアクセスしようとしている。

【対処方法】
user_data が None でないことを確認してからアクセスする。
```

---

## REMINDER — 最重要ルール再掲（Lost in the Middle 対策）

本文を唯一の正とし、この REMINDER は要約として扱う。差分がある場合は本文の該当セクションを優先する。

1. 正常に動いている環境・コードを依頼外で変更しない（本文: `## 1. 常時適用ルール`）。
2. `torch.load(..., weights_only=False)` を使わない（本文: `## 1. 常時適用ルール`）。
3. エラーは raise / `gr.Error` で明示し、握りつぶさない（本文: `## 1. 常時適用ルール`）。
4. 作業前後に `WHITEBOARD.md` と `ERROR_LOG.md` を確認・更新する（本文: `## 0. 作業手順`）。
5. Gradio 5 API を使用する（本文: `### Gradio 5 / SAM2 UI`）。
6. GroundingDINO の CUDA ビルドには `os.environ` + `--no-build-isolation` を使う（本文: `### GroundingDINO / transformers`）。
7. `segment-anything/` は直接変更しない（本文: `## 1. 常時適用ルール`）。`samurai/`（SAM2 fork）も直接変更せず、`samurai_mode` / config / 環境変数で切替える。Gradio Dropdown の tracker 切替は `INFERENCE_TRACKER_VARIANT` 環境変数（`sam2_facebook` / `sam2_samurai`）で行う（本文: `## 1. 常時適用ルール`）。
8. Haystack 版は import 時に重いモデルを初期化しない（本文: `### Haystack 2.x Pipeline`）。
9. Haystack 版 notebook は Jupytext 正本 `.py` から `.ipynb` を生成する（本文: `### Colab / Notebook`）。
10. Haystack Component 境界は `MaskSet` / `SelectedMask` / `MatteResult` の安定 I/O 契約で接続する（本文: `### Haystack 2.x Pipeline`）。
11. SAM2 / GroundingDINO の遅延調査は計測と CUDA / package preflight を先に行い、推測だけで最適化しない（本文: `### モデル・評価・学習`）。
