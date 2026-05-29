---
description: "Use when starting any coding task, fixing bugs, reviewing code, or making changes to this project. Enforces the whiteboard/error-log workflow, sub-agent code review, and project conventions for Matting-Anything (Python, PyTorch, SAM-based matting)."
---

# Matting-Anything 作業ワークフロー

> **実行順序**: セクション 0 → コード作業（セクション 1〜4）→ セクション 5（レビュー）→ セクション 6（記録更新）。スキップ可能な条件はない。矛盾が生じた場合はこの順序を最優先とする。

## 0. 作業開始前

作業を開始する前に、以下の順序で確認する。

1. `ERROR_LOG.md` を読む — 過去のエラーを把握し、同じ失敗を繰り返さない
2. `REFERENCE.md` を読む — 重要なコード構造・設計のリファレンスを把握する
3. `WHITEBOARD.md` を読む — 進行中タスク・決定事項・ブロッカーを把握する

---

## 1. コーディング規約（Python / PyTorch）

### 基本スタイル
- インデント: スペース 4 つ
- 1 行の最大長: 120 文字
- 型ヒントを新規関数には付与する（`def foo(x: torch.Tensor) -> torch.Tensor:`）
- docstring は Google スタイル

### インポート順序
```python
# 1. 標準ライブラリ
# 2. サードパーティ（torch, numpy, cv2 等）
# 3. プロジェクト内モジュール
```

### 命名規則
- 変数・関数: `snake_case`
- クラス: `PascalCase`
- 定数: `UPPER_SNAKE_CASE`
- モデル重みテンソル変数: 末尾に `_feat`, `_attn`, `_pred` 等の意味的サフィックスを付ける

---

## 2. プロジェクト固有構造

| ディレクトリ | 役割 |
|-------------|------|
| `networks/` | モデルアーキテクチャ定義 |
| `dataloader/` | データ読み込み・前処理 |
| `utils/` | 汎用ユーティリティ |
| `evaluation/` | 評価スクリプト（変更時は評価結果との整合性を確認） |
| `config/` | TOML 形式の学習設定ファイル |
| `outputs/` | 推論・評価結果の出力先（git 管理外） |
| `checkpoints/` | モデル重みファイル（git 管理外） |
| `pipelines/` | Haystack 2.x Component / Pipeline 定義 |

### Haystack Pipeline 変更時の注意

以下のサブセクションは該当タスクのときのみ参照する。

#### (a) Pipeline 構造規約（Pipeline / Component を追加・変更するとき）

- 純粋 Component は `pipelines/components/common.py` に配置する
- モデル初期化やファイル保存など副作用を持つ Component は `pipelines/components/model_components.py` に配置する
- `Pipeline` の組み立ては `pipelines/*_pipeline.py` に置き、Gradio callback に DAG 構築を混在させない
- 重いモデルは `warm_up()` で遅延初期化し、import 時に checkpoint を読まない
- Gradio callback が Haystack の中間 Component 出力を読む場合は `Pipeline.run(..., include_outputs_from={...})` を指定する

#### (b) UI 規約（SAM2 prompt 入力 UI を変更するとき）

- SAM2 bbox prompt は数値手入力 UI 禁止。画像上のマウス選択で作り、端付近クリックを画像端に吸着させる
- SAM2 prompt 入力は `gr.Image(type="numpy", interactive=True)` を使う（bbox/point 用途で `gr.ImageEditor` を使わない）。Point 正負は `gr.Radio(["positive","negative"])` で明示する（ERR019）
- SAM2 bbox は確定後に `Extend Left/Right/Top/Bottom` 4 ボタンで画像端 (0 / w-1 / h-1) に揃えられる UI を提供する（ERR019）

#### (c) テスト規約（新規 Component 追加時）

- 新規 Component には unit test または `@pytest.mark.integration` の骨格テストを追加する

### Jupyter Notebook / Jupytext 変更時の注意

- Haystack 版 notebook は `.py` の Jupytext percent 形式を**単一の正本（source of truth）**とする。`.py` と `.ipynb` が乖離した場合は常に `.py` を優先する
- `.ipynb` を直接編集せず、対応する `.py` を編集してから `python -m jupytext --to ipynb <file>.py` で生成する
- Notebook 生成後は `.ipynb` の差分が `.py` の意図と一致しているか確認する（編集対象はあくまで `.py`）
- Colab 用 install cell では `-q` を使わず、CUDA / build エラーが見える状態にする
- ノートブックは起動・環境設定に留め、推論実装本体は `gradio_app_*_haystack.py` と `pipelines/` に集約する

### モデル変更時の注意
- `networks/m2ms/conv_sam.py` を変更する場合は SAM の入出力インターフェースとの整合性を必ず確認する
- `config/*.toml` のパラメータ変更は `trainer.py` の該当箇所と対応を確認する

---

## 3. AI モデル・推論コードの作成規約

- GPU/CPU の自動フォールバックを実装する: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- モデルの `eval()` モードは推論前に必ず呼ぶ
- `torch.no_grad()` コンテキストを推論ループに必ず付ける
- バッチ処理では各バッチ処理の完了後、または推論ループの一定イテレーション（例: 100 iter）ごとに `torch.cuda.empty_cache()` を呼ぶ。学習ループ内では epoch 終了時に呼ぶ
- 画像の正規化パラメータ（mean/std）はハードコードせず設定ファイルから取得する

---

## 4. 評価コード作成規約

- 評価指標は `evaluation/metrics.py` に定義されたものを使用する（重複実装しない）
- 評価スクリプトは再現性のためシードを固定する: `torch.manual_seed(42)`, `np.random.seed(42)`
- 結果は `outputs/` 以下に日付つきフォルダで保存する
- 評価完了後に `WHITEBOARD.md` へ結果のサマリーを記録する

---

## 5. コード作成・改修後のレビュー

コードを作成または変更したら、差分の大小・行数・ファイル数に関わらず **サブエージェントにレビューを依頼**する。設計 / 実装プラン、仕様影響レビューを作成・変更・提示した場合も必ずレビュー対象にする。ここで言う「サブエージェント」とは `runSubagent` ツールで起動する別の Copilot エージェントインスタンス（`Explore` agent や `code-reviewer` 相当のエージェント）を指す。

背景除去モデル、SAM / SAM2、動画版、静止画版は細かい差分ファイルを持つため、共通処理を変更した場合は直接変更したファイルだけでなく、その共通処理を使う静止画版 / 動画版 / notebook / pipeline の代表経路も挙動確認する。共通処理の例は `pipelines/components/common.py`、`pipelines/components/model_components.py`、`pipelines/components/video_model_components.py`、`pipelines/components/ui_helpers.py`、設定 / 評価 / 前処理の共通モジュール。影響確認対象の例は静止画 Gradio、動画版 Gradio、Haystack Notebook 正本 `.py`、MAM Haystack Gradio、関連 pipeline / unit test。

```
以下のコードをレビューしてください。
観点: 正確性、パフォーマンス、セキュリティ、可読性、プロジェクト規約への準拠
[コードを貼り付け]
```

- 重要な指摘（バグ・セキュリティ）は無条件で修正する
- 中程度の指摘（パフォーマンス・可読性）は妥当性を判断して対応する
- 軽微な指摘（スタイル等）は裁量で対応する

### サブエージェントが利用できない場合のフォールバック

環境制約・コスト・オフライン等でサブエージェントが利用できない場合は、以下のセルフレビューチェックリストを実行し、その旨と結果を `WHITEBOARD.md` に記載する。セルフレビューはサブエージェントレビューの完全な代替ではないため、利用不可だった理由をユーザーへ明示する。

- [ ] 正確性: 期待動作とテスト結果が一致する
- [ ] パフォーマンス: GPU メモリ・計算量に問題がない
- [ ] セキュリティ: `torch.load(..., weights_only=True)`、入力検証、機密情報の漏洩なし
- [ ] 可読性: 命名・コメント・構造が規約通り
- [ ] 規約準拠: `copilot-instructions.md` の禁止事項に抵触しない

---

## 6. 作業完了後

作業が完了したら以下を更新する。エラーが発生しなかった場合でも `WHITEBOARD.md` は更新する。

### WHITEBOARD.md の更新内容
- 完了したタスクを消込み、次のアクションを更新
- 重要な設計判断があれば「決定事項・変更履歴」に追記

### ERROR_LOG.md エントリのフォーマット

作業中に遭遇したエラーと解決法を以下のテンプレートで追記する。

```markdown
## ERR0XX: <タイトル>

**深刻度**: critical | high | medium | low
**頻度**: 1回 | 散発 | 頻発
**発生日**: YYYY-MM-DD
**関連ファイル**: `path/to/file.py`

**原因**:
<根本原因の説明>

**解決**:
<実施した対処と修正内容>

**再発防止**:
<同種エラーを避けるためのチェックポイント>
```
