# SAM2 Haystack 複合対象マスク強化 実装計画

## 1. 背景と目的

`SAM2_Haystack_SAM_USAGE_REPORT.md` の推奨アクションを実行し、`gradio_app_sam2_transparent_BG_haystack.py` 系へ text prompt / multimask / mask union を導入する。

今回の優先方針は、SAM2 や MAM の内部 feature に結合することではなく、**Haystack 的な疎結合 Component 設計として、mask / alpha / bbox / score / metadata の標準 I/O 契約を先に作る**ことに変更する。

これにより、現在の transparent-background、将来のより高性能な背景除去モデル、SAM2、GroundingDINO、将来の text-to-region / segmenter / matting model を交換しやすくする。

## 2. 目標

- 「ドラムをたたく人」「自転車に乗る人」のような人物 + 物体の複合対象を、人物単体ではなく複合 mask として扱えるようにする
- SAM2 の multimask 候補をユーザーが比較・選択できるようにする
- 複数候補 mask / 複数プロンプト結果を union できるようにする
- GroundingDINO text prompt を bbox / label / confidence として SAM2 prompt に接続する
- transparent-background は `image + mask -> alpha / rgba / preview` の `MatteExtractor` として扱う
- MAM も将来的には `image + mask -> alpha / rgba / preview` に寄せられるか検証対象にする
- 静止画を先に安定させ、動画 tracking は標準 I/O 契約を拡張できる形で残す

## 3. 現状把握

- `Sam2_Transparent_Background_Haystack.py` は Jupytext 正本の Colab 起動ノートで、実処理は `gradio_app_sam2_transparent_BG_haystack.py` と `pipelines/` に委譲している
- 現 UI は本日更新済みで、`Input Image` / `SAM2 Prompt Canvas` 分離、upload 導線削除、point / box prompt、positive / negative、bbox edge extend、display size toggle を持つ
- `SAM2Segmenter` は `masks` / `scores` を返すが、現 UI は最高 score の 1 mask のみを `SAM2_STATE["mask"]` に保存する
- `TransparentBGExtractor` / `SAM2GuardFilter` は単一 mask を前提にしている
- `GroundingDINODetector` は単一 best bbox を返すため、複数候補 bbox / phrase / confidence の扱いが不足している
- MAM は SAM v1 の内部 feature に依存しているため、SAM2 feature へ直接差し替えるより、まず `image + mask -> matte` の外部契約へ寄せる方が保守性が高い

## 4. 最重要設計判断

### 4.1 内部 feature ではなく標準 I/O 契約を優先する

Segmenter / TextToRegion / MaskOperator / MatteExtractor / Composer を分離し、Component 間は以下のような安定したデータで受け渡す。

```python
MaskSet:
    masks: np.ndarray        # (N, H, W), bool or uint8
    scores: np.ndarray       # (N,)
    boxes: np.ndarray | None # (N, 4)
    labels: list[str]
    source: str
    metadata: dict

SelectedMask:
    mask: np.ndarray         # (H, W), bool
    source_indices: list[int]
    label: str
    metadata: dict

MatteResult:
    rgba: np.ndarray         # (H, W, 4)
    alpha: np.ndarray        # (H, W)
    preview: np.ndarray
    metadata: dict
```

### 4.2 MAM / SAM2 の完全 feature 統合は別フェーズ

今回の実装では MAM と SAM2 の feature-level 統合はしない。理由は、MAM の M2M は SAM v1 image embedding / low-res mask の形状・分布を前提にしており、SAM2 内部 feature と直接接続するとモデル互換性検証が必要になるため。

ただし、MAM を将来 `MatteExtractor` として扱えるように、`image + mask -> MatteResult` の adapter 検証項目を残す。これにより、SAM2 / future segmenter の mask を MAM / transparent-background / future matte model へ渡す疎結合構造を優先する。

### 4.3 ファイル運用

ユーザー指定に従い、既存3ファイルは `archive/` フォルダへ移動し、新ファイルを元の名前で作る。

- archive 対象
  - `gradio_app_sam2_transparent_BG_haystack.py`
  - `Sam2_Transparent_Background_Haystack.py`
  - `Sam2_Transparent_Background_Haystack.ipynb`
- 新規作成
  - `gradio_app_sam2_transparent_BG_haystack.py`
  - `Sam2_Transparent_Background_Haystack.py`
  - `Sam2_Transparent_Background_Haystack.ipynb`

新 UI は本日更新済み UI をベースにし、既存改善を失わない。

## 5. 新しい優先順位

1. **標準 I/O 契約を定義する**
   - `MaskSet`, `SelectedMask`, `MatteResult`, `RegionProposal` 相当の型 / dict 仕様を決める
   - numpy array shape、dtype、空データ時の扱い、metadata を明文化する
2. **純粋 mask 操作 Component を実装する**
   - candidate selection
   - mask union
   - mask preview composition
   - bbox extraction / dilation / min area filtering
3. **TextToRegion Component を拡張する**
   - GroundingDINO から複数 bbox / phrase / confidence を返す
   - 将来の text-to-region model へ差し替え可能にする
4. **SAM2 を Segmenter Component として標準化する**
   - SAM2 の出力を `MaskSet` 契約へ変換する
   - point / box / text-derived boxes を同じ入力契約へ寄せる
5. **transparent-background を MatteExtractor Component として標準化する**
   - `image + SelectedMask/UnionMask -> MatteResult`
   - 将来の背景除去 / matting model と差し替え可能にする
6. **Pipeline builder を標準契約ベースへ更新する**
   - `TextToRegion -> Segmenter -> MaskSelector -> MaskUnion -> MatteExtractor -> OutputSaver`
7. **Gradio UI を新契約に合わせて構成する**
   - text prompt
   - candidate mask selection
   - union mask preview
   - selected/union mask を tb に渡す導線
8. **Notebook / docs / 計画表を更新する**
   - Jupytext 正本から `.ipynb` 生成
   - `Haystack_pipeline計画表.md`, `REFERENCE.md`, `WHITEBOARD.md` を現行方針へ更新

## 6. Component 設計

### 6.1 common.py に追加する純粋処理

- `normalize_masks(masks) -> np.ndarray`
- `select_candidate_masks(mask_set, indices=None, score_threshold=None, top_k=None) -> SelectedMask / MaskSet`
- `union_masks(masks, mode="or", dilate_kernel=0, min_area=0) -> np.ndarray`
- `compose_mask_preview(image, masks, labels=None, selected_indices=None, union_mask=None) -> np.ndarray`
- `mask_set_to_status(mask_set) -> str`

### 6.2 common.py に追加する Component

- `MaskCandidateSelector`
- `MaskUnion`
- `MaskPreviewComposer`
- `BBoxFromMask` は既存を維持し、必要なら `MaskSet` 対応を追加

### 6.3 model_components.py に追加 / 拡張する Component

- `GroundingDINOMultiBoxDetector`
  - 入力: image, text_prompt, thresholds, top_k
  - 出力: boxes, phrases, confidences, metadata
- `SAM2Segmenter`
  - 既存 signature は維持
  - 出力を `masks` / `scores` だけでなく、必要なら boxes / labels / source metadata も返せるよう拡張
- `TransparentBGExtractor`
  - 入力を標準 mask 契約に寄せる
  - 出力は `rgba`, `alpha`, `preview`, `metadata`
- 将来検証用
  - `MAMMatteExtractor` adapter 案を docs に残す
  - 実装は今回スコープ外または integration skeleton

## 7. Pipeline 設計

`pipelines/sam2_tb_pipeline.py` に以下を追加する。

- `build_sam2_maskset_pipeline()`
  - image normalizer
  - optional GroundingDINO multi-box
  - SAM2 segmenter
  - mask preview composer
- `build_mask_union_pipeline()`
  - mask candidate selector
  - mask union
  - preview composer
- `build_mask_to_matte_pipeline()`
  - image normalizer
  - transparent-background matte extractor
  - SAM2 guard
  - output saver
- `build_sam2_union_tb_pipeline()`
  - 上記を Gradio から使いやすい形で接続した統合 builder

## 8. UI/UX 方針

`ui-ux-pro-max` の調査結果を反映し、分析ダッシュボード型 UI として設計する。

- 既存の `Input Image` / `SAM2 Prompt Canvas` 分離は維持
- UI は「入力 → 画像解釈 → 候補 mask → union → matte 抽出」の段階表示にする
- 追加 UI
  - Text Prompt 入力欄
  - Detect Boxes
  - SAM2 Candidate Masks
  - Candidate index / score / label 表示
  - Add to Union / Remove from Union / Clear Union
  - Union Preview
  - Run transparent-background with Union Mask
- 色だけに頼らず、status textbox と label で状態を説明する
- error は `raise gr.Error(...)`
- 既存の `Image Display Size` を candidate preview / union preview にも適用する

## 9. 実装ステップ

### Phase 1: ベースライン保護とアーカイブ

1. `git status` で作業ツリーを確認する
2. アーカイブ先 `archive/sam2_haystack_pre_mask_contract/` を作る
3. 対象3ファイルを archive へ移動する
4. 移動前の最新 UI 内容を新 `gradio_app_sam2_transparent_BG_haystack.py` に引き継ぐ

### Phase 2: RED テスト作成

1. `tests/unit/test_common_components.py`
   - `MaskSet` / `SelectedMask` / `MatteResult` 契約
   - candidate selection
   - mask union
   - preview composer
   - invalid input / empty mask
2. `tests/unit/test_pipeline_wiring.py`
   - new maskset / union / matte pipeline builder
   - import 時に重いモデルを初期化しないこと
3. `tests/unit/test_jupytext_notebooks.py`
   - 新 notebook source が新 Gradio app を起動すること
   - UI に text prompt / candidate selection / union controls があること
4. `tests/integration/test_model_components.py`
   - GroundingDINO / SAM2 / transparent-background は checkpoint 依存 skeleton
   - MAM mask-level adapter 検証 skeleton を追加してもよいが、本実装は別フェーズ

### Phase 3: 標準 I/O と純粋 Component 実装

1. `pipelines/components/common.py` に標準 I/O helper と mask 操作 Component を追加
2. `MaskSet` 相当の dict 仕様を docstring / tests で固定
3. `MaskUnion` は OR union を第一実装にし、dilation / min_area は optional とする

### Phase 4: モデル Component 拡張

1. `GroundingDINOMultiBoxDetector` を追加
2. `SAM2Segmenter` 出力を標準 mask 契約へ寄せる
3. `TransparentBGExtractor` を `image + mask -> MatteResult` 契約として扱えるよう整理
4. 既存 API は可能な限り維持する

### Phase 5: Pipeline builder 更新

1. `build_sam2_maskset_pipeline()`
2. `build_mask_union_pipeline()`
3. `build_mask_to_matte_pipeline()`
4. `build_sam2_union_tb_pipeline()`

### Phase 6: Gradio UI 実装

1. 新 `gradio_app_sam2_transparent_BG_haystack.py` を作成
2. 本日版 UI の prompt canvas 分離、upload 導線削除、edge extend、display size toggle を保持
3. text prompt / detected boxes / candidate masks / union controls を追加
4. union mask を matte 抽出へ渡す導線を明示する

### Phase 7: Notebook / docs 更新

1. `Sam2_Transparent_Background_Haystack.py` を新 app 起動用に更新
2. `Sam2_Transparent_Background_Haystack.ipynb` を Jupytext 生成
3. `Haystack_pipeline計画表.md` を新方針へ更新
4. `REFERENCE.md` / `WHITEBOARD.md` を更新
5. 新規エラーが発生した場合のみ `ERROR_LOG.md` に追記

### Phase 8: 検証とレビュー

1. `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
2. `.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py --help`
3. `.venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py`
4. サブエージェント code review
5. GPU / checkpoint / 動画 tracking は残課題として WHITEBOARD に記録

## 10. リスクと対策

| リスク | 対策 |
|---|---|
| best mask 自動採用で人だけになる | 候補 mask 選択と union を追加 |
| union mask が過剰に広がる | score threshold / min area / remove UI を追加 |
| GroundingDINO が複数 bbox を返しすぎる | NMS / confidence threshold / top-k UI を追加 |
| model 固有 feature に結合して差し替えにくくなる | mask / alpha / bbox / metadata 契約を優先 |
| MAM 統合が曖昧になる | MAM は `MatteExtractor` adapter 検証として別フェーズ化 |
| UI が複雑になる | 入力→解釈→候補→統合→抽出の段階表示にする |
| import 時に重いモデルが初期化される | `warm_up()` / `run()` 内 import を維持 |

## 11. 受け入れ条件

- 標準 I/O 契約が docs / tests / Component docstring に反映されている
- 新 UI に text prompt、SAM2 candidate mask selection、mask union 操作が存在する
- `Input Image` と `SAM2 Prompt Canvas` の分離 UI は維持される
- bbox / point 座標の手入力 UI は追加しない
- selected mask / union mask のどちらを matte extractor に渡しているか UI で分かる
- transparent-background は `MatteExtractor` として `image + mask -> MatteResult` 契約に寄せる
- Pipeline Component は入出力型で接続され、モデル差し替えの境界が明確である
- unit test が非 integration で通る
- notebook は `.py` から `.ipynb` が生成される
- `REFERENCE.md`, `WHITEBOARD.md`, `Haystack_pipeline計画表.md` が更新される

## 12. 今回は本実装しないが設計上残すもの

- SAM2 の動画 tracking 本実装
- 動画 UI / frame scrubber / temporal mask editing
- transparent-background 以外の背景除去モデル実装
- GroundingDINO 以外の text-to-region モデル実装
- MAM の feature-level SAM2 統合

ただし、MAM は将来の `MatteExtractor` として `image + mask -> MatteResult` adapter を検証できるよう、標準 I/O 契約に含めておく。
