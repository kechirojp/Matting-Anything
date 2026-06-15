---
description: "Use when starting any coding task, fixing bugs, reviewing code, or making changes to this project."
---

# Matting-Anything 作業ワークフロー

> 実行順序: セクション 0 -> 実装 -> セクション 5（レビュー）-> セクション 6（記録更新）

## 0. 作業開始前

1. `ERROR_LOG.md` を読む
2. `REFERENCE.md` を読む
3. `WHITEBOARD.md` を読む
4. 変更タスクに対応する skill を `AGENTS.md` から選ぶ

---

## 1. 基本規約（Python / PyTorch）

- インデント: スペース 4 つ
- 1 行最大: 120 文字
- 新規関数に型ヒント
- docstring は Google スタイル
- 変数/関数: `snake_case`、クラス: `PascalCase`、定数: `UPPER_SNAKE_CASE`

---

## 2. プロジェクト構造と役割

| ディレクトリ | 役割 |
|-------------|------|
| `networks/` | モデル定義 |
| `dataloader/` | データ入出力・前処理 |
| `evaluation/` | 評価コード |
| `config/` | TOML 設定 |
| `pipelines/` | Haystack Component / Pipeline |
| `outputs/` | 出力（git 管理外） |
| `checkpoints/` | 重み（git 管理外） |

---

## 3. タスク別の詳細参照

詳細ルールはこのファイルに重複記載せず、以下の skill を参照する。

- Haystack Pipeline: `.github/skills/haystack-pipeline/SKILL.md`
- Gradio 5 / SAM2 UI: `.github/skills/gradio5-sam2-ui/SKILL.md`
- SAM2 / DINO / 追跡: `.github/skills/sam2-tracking-dino/SKILL.md`
- UI改善: `.github/skills/ui-ux-pro-max/SKILL.md`
- Playwright検証: `.github/skills/webapp-testing/SKILL.md`

補足:

- notebook は `.py`（Jupytext）を正本にし `.ipynb` は生成物として扱う。
- `segment-anything/` と `samurai/` は直接変更しない。
- `config/*.toml` 変更時は `trainer.py` 対応を確認する。

---

## 4. テストと検証

- 挙動変更は可能な限り RED -> GREEN で進める。
- 文書のみ変更の場合は RED テスト省略理由を `WHITEBOARD.md` に記録する。
- 推奨コマンド（Windows）:
  - `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
  - `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help`
  - `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help`

---

## 5. レビュー（必須）

- コード変更、設計変更、計画変更は差分の大小に関わらずサブエージェントレビューを実施する。
- 観点: 正確性 / パフォーマンス / セキュリティ / 可読性 / 規約準拠
- サブエージェントが使えない場合はセルフレビューを実施し、理由と結果を `WHITEBOARD.md` に記録する。

---

## 6. 作業完了後

- `WHITEBOARD.md` を更新する（完了内容・次アクション・テスト省略理由）。
- 新しいエラーを解決した場合は `ERROR_LOG.md` も更新する。
- fixed 記録前に必要な実行時検証（特に UI/配線）を完了してから記録する。
