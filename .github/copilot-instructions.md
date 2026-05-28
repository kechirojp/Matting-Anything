# Matting-Anything — プロジェクト統合ルール

SAM（Segment Anything Model）をバックボーンに用いた汎用画像マッティングシステム。
GroundingDINO、SAM / SAM2、transparent-background、Gradio 5、Haystack 2.x を組み合わせたデモと Colab を含む。

> **読み込み方**: このファイルは常時適用の短いルールカード。詳細は必要なときだけ `REFERENCE.md`、`ERROR_LOG.md`、`.github/instructions/workflow.instructions.md`、該当 skill を読む。

---

## 0. 作業手順（全エージェント共通）

### 実装フロー（テスト駆動）

1. **要件分析**: 変更対象、変更対象が import するモジュール、変更対象を import している直接の呼び出し元を確認する。直接の呼び出し元が 10 ファイルを超える場合は、変更面に関連する代表 3〜5 ファイルに限定し、その判断理由を `WHITEBOARD.md` に記録する。
2. **作業前確認**: `ERROR_LOG.md`、`REFERENCE.md`、`WHITEBOARD.md` を読む。`@whiteboard-manager` / `@error-knowledge-base` が使える場合は併用する。
3. **情報の優先順位**: エージェント出力と MD ファイルが食い違う場合は MD ファイルを正とし、差分を `WHITEBOARD.md` に記録する。
4. **設計提案**: medium / large の変更では、実装前に方針と影響範囲をユーザーへ示す。
5. **進捗記録**: medium / large の変更では、実装開始前に `WHITEBOARD.md` を更新する。
6. **テスト記述（RED）**: 挙動変更は先に失敗する `pytest` を書く。GPU・外部モデル依存は `pytest.mark.integration` の骨格でもよい。
7. **テスト省略条件**: TOML 設定追加、コメント変更、プロンプト文書修正などテスト対象がない場合は、RED テストのみをスキップし、その旨を `WHITEBOARD.md` に記録する。
8. **実装（GREEN）**: 期待動作を満たす最小限の変更を行う。
9. **必要な整理**: 許可される整理は、今回変更した関数・クラス・テストと直近の重複除去に限る。別機能、広域整形、命名統一、設計変更は、ユーザーから明示依頼がある場合のみ行う。
10. **レビュー**: 実装後に差分と意図を確認する。大きな差分は `.github/skills/reviewing-code/` またはサブエージェントでレビューする。
11. **記録更新**: 作業完了後に `WHITEBOARD.md` を更新する。新しいエラーを解決した場合は `ERROR_LOG.md` も更新する。

> **変更規模**
> - **small**: 誤字・コメント・docstring・文書のみで挙動変更なし。1 行でも挙動が変わる修正は medium。必須: 2, 8, 11。
> - **medium**: 1 関数〜1 ファイル内の機能追加・修正。必須: 1〜11（4 は方針提示、6 は必要時）。
> - **large**: 複数ファイル、API 変更、新機能。必須: 1〜11。
> - 判断に迷う場合は上位カテゴリ（large 寄り）を選択する。

### 検証コマンド

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

---

## 1. 常時適用ルール

- 正常に動いている環境・コードを、依頼外で変更しない。
- 既存 API、インターフェース、設定キー名を無断で破壊しない。
- `torch.load(..., weights_only=False)` は禁止。`weights_only=True` を使う。
- `try/except: pass` などでエラーを握りつぶさず、必ず raise または `gr.Error` で通知する。
- 設定値は `config/*.toml` または `utils/CONFIG` 経由で取得し、ハードコードしない。
- 評価指標は `evaluation/metrics.py` の定義を使い、重複実装しない。
- `outputs/` に git 管理対象ファイルを追加しない。
- `segment-anything/` は直接変更しない。変更が必要と判断した場合は、ラッパーモジュールで対応可能か検討し、不可能ならユーザーに確認する。
- 禁止事項とユーザー依頼が衝突した場合は実装せず、衝突点を提示して確認を取る。ユーザー承認で禁止事項を破った場合は、`ERROR_LOG.md` に承認日時・理由・該当 commit（未 commit なら作業ブランチと差分要約）を記録する。

---

## 2. 領域別トリガー

### Haystack 2.x Pipeline

- Haystack 系の新規実装・改修では `.github/skills/haystack-pipeline/` を参照し、最初に Component I/O 契約と Pipeline 結線を決める。
- Gradio callback は UI 入力整形、`pipeline.run(...)`、出力整形のみ担当する。モデル推論・ファイル保存は Haystack Component に委譲する。
- 重いモデル初期化は `warm_up()` に置き、import 時に checkpoint を読まない。
- 中間 Component 出力を読む場合は `pipeline.run(..., include_outputs_from={...})` を指定する（ERR018: leaf 以外の Component 出力は明示指定しないと返らない）。
- Component 境界では `MaskSet` / `SelectedMask` / `MatteResult` の安定 dict 契約を優先する。

### Gradio 5 / SAM2 UI

- `gr.Image(tool="sketch")` は廃止。`gr.ImageEditor` を使う（ERR002: Gradio 5 で sketch tool 廃止）。
- `block = block.queue()` は `None` を返す。`with gr.Blocks() as block:` の後に `block.queue()` を呼ぶ（ERR001: queue は in-place 操作）。
- `ImageEditor` 戻り値は `background`, `layers`, `composite` を使う（ERR003: `image` キー廃止）。
- RGBA (4ch) はモデル入力前に RGB (3ch) へ変換する（ERR008: channel 不一致回避）。
- SAM2 bbox / point 座標は `gr.Number` や `Textbox` で手入力させない（ERR017: bbox / point 座標手入力禁止）。
- SAM2 prompt 入力は `gr.Image(type="numpy", interactive=True)` を使う。Point 正負は `gr.Radio(["positive", "negative"], value="positive")` で明示する（ERR019: positive/negative Radio 仕様）。
- SAM2 bbox は確定後に `Extend Left/Right/Top/Bottom` の 4 ボタンで画像端へ揃えられる UI を提供する（ERR019: bbox Extend UI 仕様）。
- SAM2 prompt / Text Prompt / Union UI のいずれかに UI 要素を追加・削除した場合は `.github/skills/webapp-testing/` によるブラウザ確認を必須とする。

### GroundingDINO / transformers

- `GroundingDINO/` 配下の変更は必要最小限にとどめ、理由を `ERROR_LOG.md` または `WHITEBOARD.md` に記録する。
- GroundingDINO を使う Component は、`groundingdino.util.inference.Model` を import する前に `patch_transformers_bert_for_groundingdino()` を呼ぶ（ERR024: `BertModel.get_head_mask` 削除互換パッチ）。
- `bertwarper.py` は `get_extended_attention_mask(attention_mask, input_shape)` の新シグネチャで呼ぶ（ERR005: transformers 新旧シグネチャ差分）。
- `ms_deform_attn.py` は `CUDA_OPS_AVAILABLE` フラグを参照する（ERR006: CUDA ops import 失敗時の `_C` 未定義回避）。
- `checkpoint.checkpoint()` には `use_reentrant=False` を付ける（ERR007: PyTorch checkpoint 警告対策）。

### Colab / Notebook

- Haystack 版 notebook は `.py` の Jupytext percent 形式を正本にし、`.ipynb` は Jupytext で生成する。
- `.ipynb` を直接編集しない。内容変更は対応する `.py` に行う。
- Colab install cell では `-q` を使わず、CUDA / build エラーを隠さない。
- `!export VAR=...` は後続 `!pip install` に引き継がれない。`os.environ['VAR'] = ...` を使う（ERR004: `!` は別サブシェル）。
- Text Prompt / GroundingDINO を含む notebook では `transformers`, `addict`, `yapf`, `timm`, `supervision`, `pycocotools` と GroundingDINO checkpoint 取得を確認する。

### モデル・評価・学習

- 推論前に `eval()` と `torch.no_grad()` を使う。
- device は `torch.device("cuda" if torch.cuda.is_available() else "cpu")` で自動選択する。
- 正規化パラメータはモデル定義または TOML 設定から取得する。
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

1. 正常に動いている環境・コードを依頼外で変更しない。
2. `torch.load(..., weights_only=False)` を使わない。
3. エラーは raise / `gr.Error` で明示し、握りつぶさない。
4. 作業前後に `WHITEBOARD.md` と `ERROR_LOG.md` を確認・更新する。
5. Gradio 5 API を使用する。
6. GroundingDINO の CUDA ビルドには `os.environ` + `--no-build-isolation` を使う。
7. `segment-anything/` は直接変更しない。
8. Haystack 版は import 時に重いモデルを初期化しない。
9. Haystack 版 notebook は Jupytext 正本 `.py` から `.ipynb` を生成する。
10. Haystack Component 境界は `MaskSet` / `SelectedMask` / `MatteResult` の安定 I/O 契約で接続する。
