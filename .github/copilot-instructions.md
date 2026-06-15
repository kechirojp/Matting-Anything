# Matting-Anything — Repository-Wide Copilot Instructions

このファイルは常時適用の短いルールカードです。
詳細手順は `AGENTS.md` と各 skill を参照してください。

## プロジェクト目的（要約）

- 意味的対象（例: `person playing drums`, `person riding bicycle`）を text prompt + tracking で切り抜けること。
- GroundingDINO / SAM2 / SAMURAI / transparent-background / Gradio 5 / Haystack 2.x を組み合わせること。

## 常時適用ルール

- 正常に動いている環境・コードを依頼外で変更しない。
- `torch.load(..., weights_only=False)` は禁止。必ず `weights_only=True`。
- `try/except: pass` でエラーを握り潰さない。`raise` または `gr.Error` で通知する。
- 設定値は `config/*.toml` または既存設定経由で取得し、ハードコードしない。
- `evaluation/metrics.py` の指標定義を優先し、重複実装しない。
- `outputs/` に git 管理対象ファイルを追加しない。
- `segment-anything/` と `samurai/` は直接変更しない。
- `samurai/` 利用時は `INFERENCE_TRACKER_VARIANT` と config 切替で運用する。

## 作業フロー

- 実装時は `.github/instructions/workflow.instructions.md` を読む。
- タスク別の詳細は `AGENTS.md` の Routing Table から対応 skill を読む。
- 作業開始時は `ERROR_LOG.md` / `REFERENCE.md` / `WHITEBOARD.md` を確認する。
- コード改修を行った場合は、テスト駆動（RED -> GREEN）でテストを実施する。
- 何らかの作業を行った場合は、差分の大小に関わらず必ずレビューを入れる。
- 作業完了時は `ERROR_LOG.md` / `REFERENCE.md` / `WHITEBOARD.md` を必ず更新する。

## 検証コマンド（Windows）

- 非 integration 全体: `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
- 単一テスト: `.venv\Scripts\python.exe -m pytest tests\unit\test_xxx.py::test_name -q`
- Jupytext 生成: `.venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py`
- Gradio smoke（静止画）: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help`
- Gradio smoke（動画）: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help`
- MAM Haystack smoke: `.venv\Scripts\python.exe gradio_app_haystack.py --help`

## 環境メモ

- 本リポジトリは `.venv + pip` を維持する（uv 移行は本タスク対象外）。

## REMINDER（最重要）

1. 依頼外で正常環境を変更しない。
2. `torch.load(..., weights_only=False)` を使わない。
3. エラーを握り潰さず `raise` / `gr.Error` で通知する。
4. 作業前後に `WHITEBOARD.md` と `ERROR_LOG.md` を更新する。
5. Gradio 5 API を使う（`queue` in-place、ImageEditor 戻り値契約など）。
6. GroundingDINO CUDA build は `os.environ` + `--no-build-isolation` を使う。
7. `segment-anything/` と `samurai/` は直接変更しない。
8. Haystack 版は import 時に重いモデルを初期化しない。
9. Haystack 版 notebook は Jupytext `.py` を正本とする。
10. Haystack を使う主目的は、画面解釈・トラッキング・背景透過をパイプラインとして機能分割し、疎結合で接続すること。
11. 機能分割と疎結合により、可読性と保守性を高く保つ（責務を明確化し、修正時の影響範囲を局所化する）。
12. 今後は画面解釈モデル・トラッキングモデル・背景透過モデルの差し替えが頻繁に起こる前提で設計する。
13. Haystack Component 境界は安定 I/O 契約で接続する。契約を崩すと配線不整合が連鎖し、バグの温床になる。
14. SAM2 / GroundingDINO 遅延調査は計測と preflight を先に行う。
15. UI / 配線の fixed は Playwright 実行時検証後に記録する。

## 詳細参照

- ルーター: `AGENTS.md`
- 実装ワークフロー: `.github/instructions/workflow.instructions.md`
- UI詳細: `.github/skills/gradio5-sam2-ui/SKILL.md`
- SAM2/DINO詳細: `.github/skills/sam2-tracking-dino/SKILL.md`
- Haystack詳細: `.github/skills/haystack-pipeline/SKILL.md`
