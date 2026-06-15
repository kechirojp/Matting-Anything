# SAM2 Haystack 版の SAM 利用調査報告

## 結論

`Sam2_Transparent_Background_Haystack.py` は、`README.md` / `gradio_app.py` / `main.py` / `segment-anything` に見られる SAM の基本的な使い方のうち、**画像に対して point / box prompt を与え、複数候補 mask から 1 つを選ぶ**という操作思想は踏襲している。

ただし、Matting-Anything 本来の中核である **SAM v1 + MAM(M2M) による alpha matte 推定**、および `gradio_app.py` の **GroundingDINO による text prompt → bbox → SAM/MAM** の流れは踏襲していない。SAM2 Haystack 版は、**SAM2 mask + transparent-background による別系統の背景除去実験導線**として見るのが正確である。

## 調査対象

- `README.md`
- `gradio_app.py`
- `main.py`
- `segment-anything/`
- `Sam2_Transparent_Background_Haystack.py`
- 実処理参照先:
  - `gradio_app_sam2_transparent_BG_haystack.py`
  - `pipelines/sam2_tb_pipeline.py`
  - `pipelines/components/model_components.py`
  - `pipelines/components/common.py`
- 現状把握:
  - `REFERENCE.md`
  - `WHITEBOARD.md`
  - `ERROR_LOG.md`

## 既存 Matting-Anything / SAM v1 側の使い方

`README.md` では、MAM は pre-trained SAM と M2M module で構成され、SAM が box / point / text prompt に基づく target instance mask を生成し、M2M が alpha matte を精緻化する設計として説明されている。

`gradio_app.py` では、主に次の流れになっている。

1. `networks.get_generator_m2m(seg='sam_vit_b', m2m='sam_decoder_deep')` で MAM を構築する。
2. `torch.load(..., weights_only=True)` で MAM checkpoint を読み込む。
3. text mode では GroundingDINO が `caption=text_prompt` から bbox を検出する。
4. scribble point / scribble box / text bbox を SAM prompt として MAM に渡す。
5. `mam_model.forward_inference(sample)` で SAM mask と M2M alpha matte を得る。

`segment-anything` 側では、`SamPredictor` / `Sam` が point / box / mask input を受け、`multimask_output=True` で複数 mask を返す。MAM 側の `networks/generator_m2m.py` と `segment-anything/segment_anything/modeling/sam.py` では、複数候補のうち予測 quality が高い mask を採用して M2M に渡す実装になっている。

`main.py` は学習 entrypoint であり、SAM の直接利用コードではない。設定ファイルを読み、`DataGenerator` / `Trainer` を起動する役割に限定される。

## SAM2 Haystack 版の実態

`Sam2_Transparent_Background_Haystack.py` は Jupytext 管理の Colab 起動ノートであり、SAM2 推論本体は持たない。実処理は `gradio_app_sam2_transparent_BG_haystack.py` と `pipelines/` に委譲される。

SAM2 Haystack 版の処理は次の流れである。

1. Colab で依存関係をインストールする。
2. `PROJECT_ROOT`、`SAM2_CKPT_PATH`、`SAM2_CONFIG_NAME` を設定する。
3. `gradio_app_sam2_transparent_BG_haystack.py` を起動する。
4. Gradio UI で `Input Image` と `SAM2 Prompt Canvas` を分離する。
5. Prompt Canvas 上の point / box クリックから SAM2 prompt を作る。
6. `SAM2Segmenter` が `SAM2ImagePredictor.predict(...)` を呼び、候補 masks / scores を返す。
7. UI 側が最高 score の mask を `SAM2_STATE["mask"]` に保存する。
8. `TransparentBGExtractor` が SAM2 mask の bbox crop を transparent-background に渡す。
9. `SAM2GuardFilter` が mask 外 alpha を削る。
10. `OutputSaver` が RGBA / alpha / preview を保存する。

Haystack Pipeline としては、`SAM2Segmenter`、`TransparentBGExtractor`、`SAM2GuardFilter`、`OutputSaver` に分かれており、Component の入出力型を接続する疎結合方針には概ね合っている。

## 踏襲している点

| 観点 | 既存 SAM/MAM | SAM2 Haystack | 判定 |
|---|---|---|---|
| 画像入力 | RGB ndarray / tensor に正規化 | `ensure_rgb_array()` で RGB ndarray 化 | 踏襲 |
| point prompt | positive / negative point | positive / negative Radio + click | 踏襲 |
| box prompt | bbox prompt | 2クリック bbox + edge extend | 踏襲、UI は改善 |
| multimask | 複数候補 mask を生成 | `multimask_output` / `multimask` 対応 | 踏襲 |
| best mask 採用 | quality / IoU 最大を採用 | score 最大を採用 | 踏襲 |
| Gradio 5 対応 | `ImageEditor` / queue pattern | `gr.Image` canvas / queue pattern | 概ね踏襲 |
| 重いモデル初期化 | 旧アプリは import 時初期化 | Haystack Component で遅延初期化 | Haystack 方針として改善 |

## 踏襲していない点・差分

| 観点 | 既存 SAM/MAM | SAM2 Haystack | 影響 |
|---|---|---|---|
| text prompt | GroundingDINO で bbox 検出 | text prompt なし | 「ドラムをたたいている人」など自然言語指定はできない |
| alpha matte 推定 | MAM M2M が SAM mask を精緻化 | transparent-background が alpha 抽出 | Matting-Anything 本来の方式とは別 |
| SAM バージョン | SAM v1 submodule | external SAM2 package | API と挙動が異なる |
| 候補 mask 選択 | 内部で best 採用 | UI 側で best score 自動採用 | 複合対象より単体対象が選ばれる可能性 |
| 複数対象統合 | 基本は単一 target instance | 基本は単一 selected mask | 人 + 物の union が未実装 |
| Pipeline 状態 | 旧アプリは単一関数中心 | Gradio global `SAM2_STATE` 使用 | 完全な入出力型だけの疎結合ではない |

## プロジェクト目的への適合性

目的は「ドラムをたたいている人 → ドラム + 人」「自転車に乗っている人 → 自転車 + 人」のように、人物だけでなく接触・使用している物体も含めて segmentation / matting することである。

この目的に対して、SAM2 Haystack 版は **手動 bbox で人 + 物をまとめて囲む実験**には使える。特に現在の UI は Prompt Canvas 分離、2クリック bbox、端吸着、Extend Left/Right/Top/Bottom を持つため、対象全体を明示的に囲む操作はしやすい。

一方で、現状の仕様では **自動的に「人 + 物」を理解して union mask にする機能はない**。SAM2 の最高 score mask が「人」単体を返すと、`Use SAM2 Mask` が guard として働き、transparent-background の結果も人だけに制限される。したがって、現在の「人しか選ばない」問題は SAM2 Haystack 版でも起こり得る。

## 「人だけ選ばれる」問題の原因候補

1. **best score mask の自動採用**
   - `predict_masks()` は SAM2 の `scores` から `np.argmax(scores)` を選ぶ。
   - 複合対象より人物単体の mask が高スコアになると、人だけが選ばれる。

2. **SAM2 guard が強く効く**
   - `Use SAM2 Mask` が ON の場合、`SAM2GuardFilter` が mask 外の alpha を削る。
   - 先に人だけ mask を選ぶと、ドラムや自転車は後段で復元されない。

3. **候補 mask の手動選択がない**
   - SAM2 は複数候補を返すが、UI は最高 score だけを採用する。
   - ユーザーが「人 + 物」に近い候補を選ぶ導線がない。

4. **複数 point / mask union の設計不足**
   - positive point は複数置けるが、最終的には 1 回の predictor 出力から best mask を選ぶ。
   - 「人」と「ドラム」を別々に選んで union する機能はない。

5. **text prompt 不在**
   - `gradio_app.py` のように `"person with drum"` などの言語で bbox を作る経路がない。
   - SAM2 Haystack 版では対象関係の指定は手動 prompt に依存する。

## Haystack 疎結合方針との整合性

Haystack 化の方針には概ね合っている。`pipelines/sam2_tb_pipeline.py` は Component を接続するだけで、外部モデル処理は `pipelines/components/model_components.py` に分離されている。`@component.output_types(...)` も使われており、入出力の型で接続する方針は守られている。

ただし、次の点は改善余地がある。

- `SAM2_STATE` が Gradio global state として mask を保持しており、Pipeline の明示的な入出力だけでは完結していない。
- `SAM2Segmenter` の出力候補 masks / scores を、後段の `TransparentBGExtractor` に直接接続していない。
- mask index、複数 mask union、複数 prompt セッションなどの意思決定が Component 化されていない。

## 推奨アクション

1. **SAM2 候補 mask の選択 UI を追加する**
   - `masks` / `scores` を保存し、mask 0 / 1 / 2 をユーザーが選べるようにする。
   - 「人だけ」ではなく「人 + 物」に近い候補を選べるようにする。

2. **複数 mask の union 機能を追加する**
   - 人 mask とドラム mask / 自転車 mask を別々に選択し、OR 合成できるようにする。
   - プロジェクト目的には best 1枚より union の方が合う。

3. **SAM2 mask guard の ON/OFF を目的別に説明する**
   - 人だけ mask の時に guard ON だと物体が消えることを UI に明記する。
   - 複合対象が落ちる場合は guard OFF または union mask を使う導線にする。

4. **text prompt 経路を検討する**
   - `gradio_app.py` と同じ GroundingDINO bbox を SAM2 prompt として使う。
   - 例: `"person playing drums"` や `"person riding bicycle"` の bbox を SAM2 に渡す。

5. **Pipeline I/O をさらに明示する**
   - `SelectedMaskComponent` や `MaskUnionComponent` を追加し、Gradio global state ではなく Pipeline 入出力として mask 選択を表現する。

## 最終判定

`Sam2_Transparent_Background_Haystack.py` は、SAM の prompt ベース segmentation という考え方と Haystack による疎結合化方針には合っている。しかし、Matting-Anything 本来の MAM alpha matte 推定や GroundingDINO text-guided segmentation は踏襲していない。

そのため、現在のプロジェクト目的に対しては **「手動 prompt で複合対象を試す実験導線としては有効。ただし、人 + 物を安定して選ぶには mask 候補選択・mask union・text-to-box 接続の追加が必要」** と判断する。
