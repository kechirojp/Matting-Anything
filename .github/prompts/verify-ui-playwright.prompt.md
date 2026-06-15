---
mode: agent
description: "Gradio UI 変更後の Playwright 実行時検証チェックを行う"
---

Gradio UI 変更後の実行時検証を行ってください。日本語で結果を返してください。

チェック項目:
1. アプリ起動確認（`--help` ではなく実起動）
2. 変更した操作の実動作確認（例: slider drag -> canvas update）
3. 想定外エラーの有無（ブラウザ・バックエンド双方）
4. 必須 UI 要素の存在/非存在確認
5. スクリーンショット取得
6. fixed 判定可否（OK/NG）と根拠

出力形式:
- 実施手順
- 確認結果（PASS/FAIL）
- 残課題
- `WHITEBOARD.md` へ記録すべき要点
