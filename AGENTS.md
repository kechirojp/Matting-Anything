# AGENTS.md

このファイルは Matting-Anything のエージェントルーターです。
詳細手順は skill を参照し、常時ルールは `.github/copilot-instructions.md` を参照します。

## Hard Rules (Always Apply)

- 正常に動いている環境・コードを依頼外で変更しない。
- `torch.load(..., weights_only=False)` は禁止。必ず `weights_only=True` を使う。
- `try/except: pass` でエラーを握り潰さない。`raise` または `gr.Error` で通知する。
- `segment-anything/` と `samurai/` は直接変更しない。
- 設定値は `config/*.toml` または既存設定経由で取得し、ハードコードしない。
- UI / 配線の「fixed / 完了」は Playwright 実行時検証を通してから記録する（ERR035）。

## Routing Table

| Task | Read First |
|------|------------|
| Python / PyTorch 実装、テスト、レビュー運用 | `.github/instructions/workflow.instructions.md` |
| Haystack Pipeline の設計・配線・I/O 契約 | `.github/skills/haystack-pipeline/SKILL.md` |
| Gradio 5 / SAM2 UI 改修、Canvas、配線、UI 実行時検証 | `.github/skills/gradio5-sam2-ui/SKILL.md` |
| SAM2 / SAMURAI / GroundingDINO / 動画追跡 / CUDA・Hydra 問題 | `.github/skills/sam2-tracking-dino/SKILL.md` |
| UI/UX デザイン改善 | `.github/skills/ui-ux-pro-max/SKILL.md` |
| Playwright を使った UI 確認 | `.github/skills/webapp-testing/SKILL.md` |
| 差分レビュー運用 | `.github/skills/reviewing-code/SKILL.md` |

## Project Context

| Document | Purpose |
|----------|---------|
| `ERROR_LOG.md` | 既知エラーと再発防止（ERR001-ERR038） |
| `WHITEBOARD.md` | 進行中タスク、完了履歴、次アクション |
| `REFERENCE.md` | モデル仕様、ファイル配置、運用上の正解集 |
| `DESIGN.md` | 設計判断の背景 |

## Existing Agents

- `.github/agents/whiteboard-manager.agent.md`
- `.github/agents/project-reference.agent.md`
- `.github/agents/error-knowledge-base.agent.md`

## Verification Commands

- 非 integration 全体（Windows）: `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
- 単一テスト（Windows）: `.venv\Scripts\python.exe -m pytest tests\unit\test_xxx.py::test_name -q`
- Gradio smoke（静止画）: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help`
- Gradio smoke（動画）: `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py --help`
- MAM Haystack smoke: `.venv\Scripts\python.exe gradio_app_haystack.py --help`

## Notes

- 本リポジトリは `.venv + pip` 運用を維持する。uv 移行は本タスクの対象外。
- 最重要ルールは `.github/copilot-instructions.md` の REMINDER と同期する。
