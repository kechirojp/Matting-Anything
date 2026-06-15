---
mode: agent
description: "変更実装前に影響範囲・テスト方針・記録更新方針を整理する"
---

対象変更の実装前に、以下を日本語で簡潔に整理してください。

1. 変更対象ファイルと依存先（import 先 / 呼び出し元）
2. 影響範囲（静止画/動画/notebook/pipeline/test）
3. 先に書くべき RED テスト（または省略理由）
4. 実装順序（最小差分）
5. 検証コマンド（Windows優先）
6. `WHITEBOARD.md` / `ERROR_LOG.md` の更新方針

制約:
- 依頼外のリファクタはしない
- `segment-anything/` / `samurai/` は直接変更しない
- UI 配線変更は Playwright 実行時検証を含める
