---
name: haystack-pipeline
description: Haystack 2.x で推論パイプラインを型付き DAG として構築する。MAM/SAM/SAM2/GroundingDINO/transparent-background などのモデルを @component でラップし疎結合化・差し替え容易にする。使用タイミング: 新規推論パイプライン追加、既存スクリプトの Component 化リファクタ、モデル差し替え、Gradio コールバックの整理。RL 内層ループ・学習ループには適用しない。
---

# Haystack Pipeline

Matting-Anything の推論処理を Haystack 2.x の `Pipeline` と `@component` に分割するための作業ガイド。

## 推奨する場面

- Gradio コールバック内にモデル呼び出し・前処理・保存処理が混在しているとき
- SAM / SAM2 / MAM / transparent-background / GroundingDINO を差し替え可能にしたいとき
- 推論 DAG の型契約を明示し、Component 単位でテストしたいとき
- 入力正規化、マスク加工、アルファ合成、出力保存を再利用したいとき

## 領域別の適用判断

| 領域 | 適合度 | 判断 |
|------|--------|------|
| 推論 DAG | 5/5 | 推奨 |
| データ前処理 DAG | 5/5 | 推奨 |
| Gradio コールバック整理 | 5/5 | 推奨 |
| 学習エポック単位の外層制御 | 3/5 | 必要時のみ |
| PyTorch 学習内層ループ | 2/5 | 非推奨 |
| RL 環境ステップ単位ループ | 1/5 | 使用しない |

## コア原則

1. Component 1 個は 1 モデル、または 1 種類の I/O 操作（ファイル保存・ネットワーク送信・DB 書き込みなど）にする。I/O は専用 Component の `run()` に置き、モデル推論 Component に混在させない。
2. 外層は Haystack Pipeline、内層は既存の PyTorch / SAM / tb の素のコードを保つ。
3. 重いモデル初期化は `warm_up()` に置き、import 時には checkpoint を読まない。
4. GPU デバイス指定は Component の `__init__(device: str)` で受け取り、`warm_up()` 内で `.to(device)` する。
5. 複数 Component で同一モデルを共有する場合は外部で 1 回ロードし、依存注入する。
6. `run()` の入力と `@component.output_types` は具体型を明示する。
7. `torch.load()` は必ず `weights_only=True` を使う。
8. Gradio では Component 例外を握りつぶさず `raise gr.Error(...)` に変換する。

## 8 ステップ作成ワークフロー

1. 既存関数を読み、入出力境界を列挙する。
2. Component の入力・出力 socket 名と型を決める。
3. 純粋関数 Component の pytest を先に書く。
4. `@component` クラスを `pipelines/components/` に実装する。
5. モデル依存 Component は `warm_up()` と `@pytest.mark.integration` 骨格テストを作る。
6. `Pipeline.add_component()` と `connect()` で DAG を組む。
7. Gradio コールバックは `pipeline.run(...)` と結果整形だけにする。
8. サブエージェントレビューに差分を提示し、全 socket に具体型があるか、同じ socket 型で別実装に差し替え可能か、`torch.load()` に `weights_only=True` があるかを確認する。重要指摘を修正してから完了にする。

## 詳細リファレンス

- Component 実装: [references/component_patterns.md](references/component_patterns.md)
- Pipeline 結線: [references/pipeline_assembly.md](references/pipeline_assembly.md)
- Matting Component カタログ: [references/matting_components.md](references/matting_components.md)
- テスト方針: [references/testing_strategy.md](references/testing_strategy.md)
- Gradio 統合: [references/gradio_integration.md](references/gradio_integration.md)