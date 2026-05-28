## Plan: Matting-Anything の Haystack パイプライン化

### 2026-05-26 更新: MaskSet / union / MatteResult 契約を優先

初期計画では「SAM v1 → SAM2 差し替え」を Pipeline 接続の主目的としていたが、MAM は SAM v1 の内部 feature 分布に依存するため、feature-level の直接差し替えは別フェーズとする。現行方針では Haystack Component 境界を `MaskSet` / `SelectedMask` / `MatteResult` の標準 I/O 契約に寄せ、SAM2・GroundingDINO・transparent-background・将来モデルを mask / alpha / bbox / score / metadata で疎結合化する。

優先実装は以下に変更した。

1. `pipelines/components/common.py` に `build_mask_set()` / `select_candidate_masks()` / `union_masks()` / `compose_mask_preview()` と `MaskCandidateSelector` / `MaskUnion` / `MaskPreviewComposer` を置く。
2. `SAM2Segmenter` は既存の `masks` / `scores` に加えて `mask_set` を返す。
3. `TransparentBGExtractor` は `rgba` / `alpha` / `preview` に加えて `matte_result` を返し、`image + mask -> MatteResult` の MatteExtractor として扱う。
4. `gradio_app_sam2_transparent_BG_haystack.py` は Text Prompt、candidate mask table、union preview、Union Mask を tb に渡す UI を持つ。
5. 既存の SAM2 Haystack 3 ファイルは `archive/sam2_haystack_pre_mask_contract/` に退避し、同名ファイルを新契約ベースで再作成する。

両 Gradio アプリ（gradio_app.py / gradio_app_sam2_transparent_BG.py）を **Haystack 2.x の型付き DAG + Component プロトコル**へ載せ替え、推論パイプラインを疎結合化してモデル差し替えを容易にする。参考資料の指針通り **「外層=Haystack / 内層=既存ロジック」** の二層設計を厳守し、学習・GPU 密ループには適用しない。

### Steps（フェーズ構成）

**A. スキル作成（先行・他フェーズの前提）**
1. `.github/skills/haystack-pipeline/SKILL.md` を skill-creator のガイドに沿って新設（コアワークフロー + 適合度マトリクス）
2. `references/` 配下に 5 ファイル分割：`component_patterns.md` / `pipeline_assembly.md` / `matting_components.md` / `testing_strategy.md` / `gradio_integration.md`

**B. ドキュメント更新（A と並列可）**
3. REFERENCE.md に「8. Haystack パイプライン構成」追加（Component 一覧表）
4. GETTING_STARTED_ja.md に「Haystack パイプラインでの推論」セクション追加
5. copilot-instructions.md にディレクトリ規約・モデル差し替え規約を追記
6. workflow.instructions.md に Component 配置規約を追記

**C. Component 実装（TDD・*depends on A*）**
7. `pipelines/`、`pipelines/components/`、`tests/unit/`、`tests/integration/` を作成し pytest marker 定義
8. 純粋関数 Component を **RED→GREEN→REFACTOR** で実装：`ImageNormalizer` / `ScribbleParser` / `BBoxFromMask` / `MaskDilator` / `AlphaCompositor`
9. モデル系 Component を骨格＋integration テストで実装：`GroundingDINODetector` / `SAMSegmenter` / `SAM2Segmenter` / `MAMAlphaPredictor` / `TransparentBGExtractor` / `SAM2GuardFilter` / `ColorDecontaminator` / `BackgroundGenerator` / `OutputSaver`
10. Pipeline 組み立て関数：`pipelines/mam_pipeline.py:build_mam_pipeline()` と `pipelines/sam2_tb_pipeline.py:build_sam2_tb_pipeline()`

**D. Gradio アプリ改修（*depends on C*）**
11. `gradio_app_haystack.py` を新規作成（旧 gradio_app.py は残置）
12. `gradio_app_sam2_transparent_BG_haystack.py` を新規作成（ERR011 パッチ・PROJECT_ROOT 検出はそのまま流用）

**E. 仕上げ**
13. ERROR_LOG.md / WHITEBOARD.md 更新、`PLAN.md` をプロジェクト直下にコピー
14. サブエージェント `Explore` で型契約・差し替え容易性・OWASP A08・TDD カバレッジをレビュー

### Relevant files
- SKILL.md — 作成テンプレ
- Haystack_パイプライン参考資料.md — 設計根拠（外層/内層分離、Component カタログ）
- gradio_app.py — `run_grounded_sam()` を Pipeline 呼び出しに置換するベース
- gradio_app_sam2_transparent_BG.py — `run_pipeline()` を Pipeline に置換するベース
- REFERENCE.md / GETTING_STARTED_ja.md / copilot-instructions.md / workflow.instructions.md

### Verification
1. `uv pip install haystack-ai` 成功・PyTorch と衝突なし
2. `pytest -m "not integration" -v` 全 GREEN
3. `python -c "from pipelines.mam_pipeline import build_mam_pipeline; build_mam_pipeline()"` がトポロジ型検証 OK（両 Pipeline）
4. `gradio_app_haystack.py` 起動 → demo.jpg で旧版と同等の alpha matte
5. SAM v1 → SAM2 差し替えが `Pipeline.connect()` 1 行のみで完結（疎結合性の証明）
6. サブエージェント `Explore` レビュー実施

### Decisions
- **既存 2 ファイルは残置**（破壊的変更禁止ルール準拠、Haystack 版を別名で新設）
- **学習 / 評価コードは今回スコープ外**（Haystack DAG の密ループ非適合領域）
- **segment-anything / GroundingDINO サブモジュールは Component から呼ぶだけ**で直接変更しない
- TDD: 純粋 Component は厳格 TDD、モデル依存 Component は `@pytest.mark.integration`

### Further Considerations
1. **Haystack バージョン固定方針**
   - A（推奨）: `haystack-ai==2.x.y` ピン留め — 再現性最優先
   - B: `haystack-ai>=2.0,<3.0` — 追従性重視
2. **既存 Gradio アプリの撤廃タイミング**
   - A（推奨）: 1 ヶ月併存、README で Haystack 版を主推奨、旧版に DEPRECATED コメント
   - B: 即座削除（破壊的変更禁止ルールに抵触するため非推奨）
3. **MLflow Component の同時導入**
   - A（推奨）: 今回は見送り（YAGNI、推論用途）
   - B: 推論メトリクス記録のため `OutputSaver` 隣に追加

---

計画書を `/memories/session/plan.md` に保存しました。承認いただければ実装フェーズに移ります。**「Further Considerations」の 3 項目**について方針をいただけると、実装時の迷いがなくなります（推奨は A / A / A）。

「Haystack 版」と「Gradio」は**競合しません。完全に併存します**。誤解されているかもしれない点を整理します。

## 役割が違うので「どちらか」ではない

| レイヤー | 役割 | このプロジェクトでの担当 |
|---------|------|------------------------|
| **UI レイヤー** | ブラウザに描画・ユーザー入力受付 | **Gradio**（変わらず） |
| **オーケストレーション** | モデル呼び出しの DAG 制御 | 旧版=直書き関数 / **新版=Haystack Pipeline** |
| **モデル本体** | SAM2 / MAM / tb 等の推論 | PyTorch（変わらず） |

つまり Haystack 版も **中身は Gradio アプリ**です。`with gr.Blocks() as demo:` の中で `pipeline.run(...)` を呼ぶだけ。Gradio が消えるわけではありません。

## 「Option 2」で議論していた "併存" の意味

私が言っていた併存は **ファイル単位の併存**でした：

```
gradio_app.py```
                              ← 旧（関数直書き）
gradio_app_haystack.py                     ← 新（中で Haystack Pipeline を呼ぶ）
gradio_app_sam2_transparent_BG.py          ← 旧
gradio_app_sam2_transparent_BG_haystack.py ← 新
```

両方とも `python ...py` で起動できる Gradio アプリ。違いは内部実装だけ。

## 整理した決定

- **Q1（バージョン固定）**: A — `haystack-ai==2.x.y` ピン留め ✅
- **Q2（旧版の扱い）**: A — **1ヶ月併存**。両ファイルとも起動可能な Gradio アプリとして残し、README で Haystack 版を主推奨、旧版に DEPRECATED コメント ✅
- **Q3（MLflow）**: A — 今回は見送り ✅

この方針で `/memories/session/plan.md` を更新し、計画は確定とします。実装フェーズに進んでよろしいですか？両方とも `python ...py` で起動できる Gradio アプリ。違いは内部実装だけ。

## 整理した決定

- **Q1（バージョン固定）**: A — `haystack-ai==2.x.y` ピン留め ✅
- **Q2（旧版の扱い）**: A — **1ヶ月併存**。両ファイルとも起動可能な Gradio アプリとして残し、README で Haystack 版を主推奨、旧版に DEPRECATED コメント ✅
- **Q3（MLflow）**: A — 今回は見送り ✅

この方針で `/memories/session/plan.md` を更新し、計画は確定とします。実装フェーズに進んでよろしいですか？



requirements.txtの改修も必要だからそれもお願いね

では実装開始してください
