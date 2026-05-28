# Haystack パイプライン化計画

## 方針

- 既存 Gradio アプリは残し、Haystack 版アプリを追加する。
- Gradio は UI、Haystack は推論 DAG とする。
- 外部モデルは Component の `warm_up()` で遅延初期化する。
- 純粋 Component は unit test、checkpoint/GPU 依存は integration test 骨格に分ける。

## 進捗

- [x] `haystack-ai==2.29.0` の固定を決定
- [x] `.github/skills/haystack-pipeline/` を追加
- [x] `pipelines/components/` を追加
- [x] `pipelines/mam_pipeline.py` を追加
- [x] `pipelines/sam2_tb_pipeline.py` を追加
- [x] Haystack 版 MAM Gradio entrypoint を追加
- [x] unit / integration test 骨格を追加
- [x] Haystack 版 SAM2 + tb Gradio entrypoint を追加
- [x] サブエージェントレビュー指摘の初期修正
- [x] 単体テスト実行
- [x] Pipeline builder smoke test 実行
- [x] サブエージェントレビュー

## 検証結果

- `.venv\Scripts\python.exe -m pytest -m "not integration" -v`: 11 passed, 2 deselected
- Pipeline builder smoke test: `pipeline builders ok`
- VS Code diagnostics: Haystack 版 Gradio entrypoint はエラーなし