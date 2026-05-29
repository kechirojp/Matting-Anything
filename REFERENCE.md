# リファレンスボード — プロジェクト参照ガイド
## 0. プロジェクトの目的
このプロジェクトの目的は
ドラムをたたいている人＝＞ドラム＋人
自転車に乗っている人＝＞自転車＋人
と
セグメンテイションできることを目指す
現状は
ドラムをたたいている人＝＞ドラム＋人を選びたいにもかかわらず人しか選ばないことがおきてしまうため
Matting anythingリポジトリを選んで実験している※静止画動画含めて

つまり
画像の意味解釈ができるモデル　そのオブジェクトが何なのか　もしくは発展的にそのオブジェクトが何をしているのか　意味をプロンプトなどでユーザーがテキスト入力などができるモデル

SAMに限らずオブジェクトをトラックできる機能をもったモデルで　動画に対応する
GroundingDINOなどの画像の意味解釈ができるモデルと組み合わせて　ユーザーがテキスト入力で意味的にオブジェクトを選べるようにする
プラス
SAM（Segment Anything Model）などオブジェクトトラッキング機能をバックボーンに用いた背景除去システム。
GroundingDINO、SAM / SAM2、transparent-background、Gradio 5、Haystack 2.x を組み合わせたデモと Colab を含む。





> **用途**: コーディング中に迷ったら必ずここを参照する。
> 設定値・API 仕様・ファイル配置・モデル仕様の「正解」を集約したドキュメント。

---

## 1. ファイル配置マップ

| ファイル / ディレクトリ | 役割 | 変更時の注意 |
|------------------------|------|------------|
| `gradio_app.py` | Matting-Anything（MAM）Gradio 5 デモ | Gradio 5 規約を遵守 |
| `gradio_app_sam2_transparent_BG.py` | SAM2 + transparent-background ローカルデモ | SAM2_CKPT_PATH 環境変数で設定 |
| `gradio_app_haystack.py` | Haystack Pipeline 版 MAM Gradio 5 デモ | import 時に重いモデルを初期化しない |
| `gradio_app_sam2_transparent_BG_haystack.py` | Haystack Pipeline 版 SAM2 + tb デモ | SAM2 prompt と tb 実行を分離 |
| `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` | Haystack Pipeline 版 SAM2 + tb 動画デモ | 第 1 フレーム prompt から SAM2 video predictor で伝搬。出力は動画 / PNG 連番 / 両方 |
| `Matting_Anything_Haystack.py` | Jupytext 正本: MAM Haystack 版 Colab 起動ノート | `.ipynb` は Jupytext で生成 |
| `Matting_Anything_Haystack.ipynb` | MAM Haystack 版 Colab 起動ノート | 直接編集禁止 |
| `Sam2_Transparent_Background_Haystack.py` | Jupytext 正本: SAM2 + tb Haystack 版 Colab 起動ノート | `.ipynb` は Jupytext で生成 |
| `Sam2_Transparent_Background_Haystack.ipynb` | SAM2 + tb Haystack 版 Colab 起動ノート | 直接編集禁止 |
| `Sam2_Transparent_Background_Haystack_for_Movie.py` | Jupytext 正本: SAM2 + tb Haystack 動画版 Colab 起動ノート | `.ipynb` は Jupytext で生成 |
| `Sam2_Transparent_Background_Haystack_for_Movie.ipynb` | SAM2 + tb Haystack 動画版 Colab 起動ノート | 直接編集禁止 |
| `pipelines/` | Haystack 2.x Component / Pipeline 定義 | Component 単位でテストする |
| `Sam2_Transparent_Background.ipynb` | SAM2 + tb パイプライン Colab ノートブック | Cell 1 は `-q` フラグ禁止 |
| `Matting_Anything.ipynb` | MAM + GroundingDINO Colab ノートブック | `!export` 禁止→`os.environ` を使う |
| `networks/m2ms/conv_sam.py` | SAM デコーダ統合コア | SAM I/O 整合性必須 |
| `networks/generator_m2m.py` | MAM モデル生成・ロード | `torch.load(weights_only=True)` 必須 |
| `evaluation/metrics.py` | 評価指標定義（重複実装禁止） | ここを使うこと |
| `config/*.toml` | 学習設定（MAM-ViTB/L/H） | `trainer.py` との対応確認 |
| `checkpoints/` | MAM/GroundingDINO 重みファイル（git 管理外） | - |
| `checkpoints/SAM2/` | SAM2 重みファイル（git 管理外） | - |
| `checkpoints/transparent_BG/` | transparent-background 重みファイル（git 管理外） | - |
| `inputs/` | 手動配置入力画像置き場（git 管理外） | - |
| `.venv/` | uv で作成した Python 3.11.12 仮想環境（git 管理外） | - |
| `segment-anything/` | SAM サブモジュール | **直接変更禁止** |
| `GroundingDINO/` | GroundingDINO サブモジュール | 変更は最小限・理由を ERROR_LOG に記録 |
| `outputs/` | 推論・評価結果（git 管理外） | - |
| `ERROR_LOG.md` | エラー知見ログ | 作業前後に必ず確認・更新 |
| `WHITEBOARD.md` | 作業状況・変更履歴 | 作業前後に必ず確認・更新 |
| `REFERENCE.md` | このファイル。API・設定・モデル仕様の正解集 | - |

---

## 2. モデル仕様

### 2-1. MAM（Matting Anything Model）

| 項目 | 値 |
|------|-----|
| チェックポイント | `checkpoints/mam_vitb.pth` / `mam_vitl.pth` / `mam_vith.pth` |
| 入力 | RGB 画像 + SAM マスク |
| 出力 | アルファマット（0〜1 float） |
| 設定ファイル | `config/MAM-ViTB-8gpu.toml` 等 |

### 2-2. SAM（Segment Anything Model v1）

| 項目 | 値 |
|------|-----|
| チェックポイント | `segment-anything/checkpoints/sam_vit_b_01ec64.pth` |
| バックボーン | ViT-B（デモデフォルト） |
| 入力プロンプト | Point / Box / Mask |
| サブモジュール | `segment-anything/`（直接変更禁止） |

### 2-3. SAM2（Segment Anything Model v2）

| 項目 | 値 |
|------|-----|
| チェックポイント | `checkpoints/SAM2/sam2.1_hiera_large.pt` |
| 設定名 | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| インストール | `pip install git+https://github.com/facebookresearch/sam2.git`（`-q` 禁止） |
| DL URL | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` |
| 使用ファイル | `Sam2_Transparent_Background.ipynb`, `gradio_app_sam2_transparent_BG.py` |

#### SAM2 Colab import preflight

- Haystack 版 Colab Notebook は Gradio 起動前に `import sam2`, `from sam2.build_sam import build_sam2`, `from sam2.sam2_image_predictor import SAM2ImagePredictor` を確認する。
- 動画版は `from sam2.build_sam import build_sam2_video_predictor` も確認する。
- `ModuleNotFoundError: No module named 'sam2'` が出た場合は ERR010 として扱い、install cell を `-q` なしで再実行してから診断セルと Gradio 起動セルを再実行する。

#### SAM2 bbox UI 仕様

- SAM2 の bbox 座標をユーザーに数値手入力させない。
- Gradio UI では画像上のマウス操作で bbox を指定する。
- Haystack 版 SAM2 アプリでは、アップロード用 `Input Image` と prompt 編集用 `SAM2 Prompt Canvas` を分離する。`SAM2 Prompt Canvas` は `sources=[]` の編集面にし、画像のドラッグ＆ドロップ先にしない。クリックイベントは `prompt_canvas.select(...)` に紐づけ、推論には原本の `input_image` を渡す。
- `SAM2 Prompt Canvas` はアップロード先にしないため `sources=[]` を維持しつつ、point / bbox click を受けるため `interactive=True` にする（ERR026）。
- bbox は 2 点クリックで確定し、クリック順序に依存せず `[x_min, y_min, x_max, y_max]` へ正規化する。
- 画像端付近のクリックは端座標へ吸着し、被写体が画面外へ続くケースでも画面端まで選択できるようにする。
- Haystack 版 SAM2 アプリでは `gr.Image(type="numpy", interactive=True)` と `normalize_box_from_points()` / `clamp_prompt_point()` を使う。bbox / point 用途で `gr.ImageEditor` を使わない。
- 予測画像と prompt canvas は `Image Display Size` で `window`（既定・固定高さ）/ `original`（原寸）を切り替えられるようにする。

### 2-4. GroundingDINO

| 項目 | 値 |
|------|-----|
| チェックポイント | `checkpoints/groundingdino_swint_ogc.pth` |
| 設定ファイル | `GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| CUDA ビルド条件 | `CUDA_HOME` 環境変数 + `--no-build-isolation` |
| 追加依存 | `transformers`, `addict`, `yapf`, `timm`, `supervision`, `pycocotools` |
| 使用ファイル | `gradio_app.py`, `Matting_Anything.ipynb`, `gradio_app_sam2_transparent_BG_haystack.py`, `Sam2_Transparent_Background_Haystack.py` |

#### GroundingDINO + transformers 互換性

- 新しい `transformers` では `BertModel.get_head_mask` が削除されている場合があるため、Haystack Component では `patch_transformers_bert_for_groundingdino()` を GroundingDINO model import 前に呼ぶ。
- `bertwarper.py` は `get_extended_attention_mask(attention_mask, input_shape)` の新シグネチャで呼ぶ（`device` 引数を渡さない）。
- Colab install cell は `transformers>=4.26.0` と GroundingDINO runtime 依存を入れる。Text Prompt でエラーが出た場合は Notebook の install cell から実行し直し、起動済み Gradio プロセスを再起動する。

### 2-4-1. GPU first / CPU 緊急回避ポリシー

- 映像制作用途では SAM2 / GroundingDINO / MAM などの重い推論は GPU 実行を前提にする。
- `SAM2Segmenter`, `GroundingDINODetector`, `GroundingDINOMultiBoxDetector`, `SAM2VideoPropagator` と legacy `gradio_app.py` は CUDA が使えない場合、既定で fail fast する。
- CPU 実行は緊急回避専用。意図的に非常に遅い CPU 推論を許可する場合だけ `MATTING_ANYTHING_ALLOW_CPU=1` を設定する。
- Gradio status の `cuda_available=False` は正常運用ではなく環境修正対象として扱う。
- Colab Notebook は Gradio 起動前に `nvidia-smi`, `torch.cuda.is_available()`, `torch.version.cuda` を確認し、CUDA 不可かつ `MATTING_ANYTHING_ALLOW_CPU=1` 未設定なら起動前に停止する（ERR025）。

### 2-4-2. Colab Gradio share URL

- Colab では Notebook 出力の `http://127.0.0.1:<port>` は Colab VM 内部の local URL であり、手元ブラウザから直接開かない。
- Haystack 版 Colab Notebook は Gradio の `--share` デフォルト動作に任せ、`Running on public URL: https://...gradio.live` を表示させる（ERR027）。
- Colab 判定は `sys.modules` だけに依存せず、`google.colab` の import spec で判定する。`--share` が渡らないと public URL は表示されない。
- frpc 取得や checksum 検証を Notebook 側で過剰に先取りすると、Gradio の public URL 生成前に停止して UX を悪化させるため避ける。share link 生成失敗時は Colab stdout の Gradio エラーを一次情報にする。

### 2-5. transparent-background

| 項目 | 値 |
|------|-----|
| パッケージ | `pip install transparent-background`（`-q` 禁止） |
| モード | `base` / `fast` / `base-nightly` |
| Remover 引数 | `mode`, `jit`, `device`, `ckpt`（オプション） |
| process 引数 | `type`: `rgba`/`map`/`green`/`white`/`blur`/`overlay`, `threshold`: None or 0〜1 |
| チェックポイント | 初回呼び出し時に自動DL。`checkpoints/transparent_BG/ckpt_{base,fast,base_nightly}.pth` に手動配置するとローカルロードされる |
| モード→ckpt 対応表 | `TB_CKPT_BY_MODE` で参照（Cell 4 / `gradio_app_sam2_transparent_BG.py`） |
| 使用ファイル | `Sam2_Transparent_Background.ipynb`, `gradio_app_sam2_transparent_BG.py` |

---

## 3. Gradio 5 API クイックリファレンス

### 3-1. 使用可能なコンポーネント

| Gradio 4 (廃止) | Gradio 5 (正解) | 備考 |
|----------------|-----------------|------|
| `gr.Image(tool="sketch")` | `gr.ImageEditor` | ERR002 |
| `block = block.queue()` | `demo.queue(); demo.launch()` | ERR001 |
| `input["image"]` | `input["background"]` / `input["composite"]` | ERR003 |
| `/info` schema crash | `gradio_client.utils._json_schema_to_python_type` bool schema patch | ERR011 / ERR016 |

### 3-1-1. SAM2 bbox 入力禁止パターン

```python
# NG: bbox / point 座標をユーザーに数値手入力させる
x_min = gr.Number(label="X Min")
point_x = gr.Number(label="Point X")

# OK: 画像上のマウス選択で prompt を蓄積する
input_image.select(select_sam2_prompt, inputs=[...], outputs=[...])
```

SAM2 の bbox は `gr.Number` / `Textbox` による座標手入力ではなく、画像上の選択イベントから作る。端付近のクリックは 0 / `width - 1` / `height - 1` へ吸着させる。

### 3-2. ImageEditor 戻り値キー

```python
# gr.ImageEditor の戻り値は dict
result = {
    "background": np.ndarray,   # 背景レイヤー（元画像）
    "layers": [np.ndarray],     # 描画レイヤー（RGBA, 4ch）
    "composite": np.ndarray,    # 合成済み（RGBA, 4ch）
}
# RGBA (4ch) → RGB (3ch) 変換が必須
image_rgb = result["background"][..., :3]  # または composite
```

### 3-3. Gradio 5 標準パターン（ERR001 対策）

```python
# OK パターン
with gr.Blocks() as demo:
    # ... UI 定義 ...
    pass

demo.queue()
demo.launch(share=True)
```

### 3-4. エラー通知

```python
# OK: ユーザーに表示される
raise gr.Error("処理に失敗しました")

# NG: エラーが握りつぶされる
print("エラー発生")
return None
```

### 3-5. Gradio 5 `/info` schema crash 対策

Gradio 5.9.x では JSON Schema の `additionalProperties: true/false` が `bool` として渡ると、`gradio_client.utils._json_schema_to_python_type` の `/info` 生成で `TypeError: argument of type 'bool' is not iterable` が発生する場合がある。SAM2 系 Gradio アプリでは `gradio_client.utils._json_schema_to_python_type` に bool schema patch を当てる。`demo.launch(..., show_api=False)` は API 表示を隠す補助設定で、`/info` 例外の本対策は schema patch 側で行う。

---

## 4. セキュリティ規約

| 規約 | 理由 |
|------|------|
| `torch.load(weights_only=True)` | OWASP A08: Insecure Deserialization 対策 |
| `torch.load(weights_only=False)` 禁止 | 任意コード実行リスク |
| ユーザー入力のパス検証 | Path Traversal 対策 |
| 環境変数による設定外部化 | ハードコード禁止 |

---

## 5. SAM2 + transparent-background パイプライン構成

```text
入力画像
  └─→ SAM2 推論（Point/Box プロンプト）
        └─→ 候補マスク (N=3) → ユーザー選択
              └─→ マスク bbox + padding でクロップ
                    └─→ transparent-background でα抽出
                          └─→ 元画像サイズに貼り戻し
                                └─→ SAM2 guard（誤検出除去）
                                      └─→ Color decontamination（pymatting）
                                            └─→ RGBA 出力
```

### パラメータ早見表

| パラメータ | デフォルト | 用途 |
|-----------|-----------|------|
| `tb_mode` | `base` | transparent-background モデル切替 |
| `tb_jit` | `False` | TorchScript 高速化 |
| `tb_threshold` | `0.0` | ソフトα(0) vs 二値化(>0) |
| `crop_padding` | `40` | 細毛が切れる → 80〜120 に上げる |
| `use_sam2_as_guard` | `True` | tb の誤検出を SAM2 マスクで殴り消す |
| `sam2_guard_dilate` | `21` | ガード膨張量（細毛許容 → 31〜51 に上げる） |
| `apply_decontam` | `True` | 緑被り除去 |

### 5-1. Haystack 中間出力の取得

Haystack Pipeline で `transparent_bg` や `sam2_guard` の出力を Gradio callback 側で読む場合は、`Pipeline.run(..., include_outputs_from={"transparent_bg", "sam2_guard", "output_saver"})` を指定する。指定しないと leaf 以外の Component 出力が返らず、`KeyError: 'transparent_bg'` になる場合がある（ERR018）。

### 5-2. 標準 mask / matte I/O 契約

SAM2 Haystack 版の Component 境界では、SAM2 / GroundingDINO / transparent-background / 将来モデルの内部 feature を直接渡さず、以下の dict 契約で接続する。

```python
MaskSet = {
    "masks": np.ndarray,      # (N,H,W) bool
    "scores": np.ndarray,     # (N,)
    "boxes": np.ndarray,      # (N,4) or empty
    "labels": list[str],
    "source": str,
    "metadata": dict,
}

SelectedMask = {
    "mask": np.ndarray,       # (H,W) bool
    "source_indices": list[int],
    "label": str,
    "metadata": dict,
}

MatteResult = {
    "rgba": np.ndarray,       # (H,W,4)
    "alpha": np.ndarray,      # (H,W)
    "preview": np.ndarray,
    "metadata": dict,
}
```

- 純粋処理は `pipelines/components/common.py` の `build_mask_set()` / `select_candidate_masks()` / `union_masks()` / `compose_mask_preview()` を使う。
- SAM2 は `SAM2Segmenter` から `mask_set` を返す。UI は best score の単一 mask を固定採用せず、candidate index を選んで union できる。
- transparent-background は `image + mask -> MatteResult` の `MatteExtractor` として扱う。MAM も将来は同じ `image + mask -> MatteResult` adapter として検証する。

### 5-3. 動画版 Haystack I/O 契約

動画版は `pipelines/sam2_tb_video_pipeline.py` に 3 つの builder を持つ。

| builder | 用途 |
|---------|------|
| `build_video_reader_pipeline()` | 動画から第 1 フレームと metadata を取得する軽量 Pipeline |
| `build_sam2_video_propagation_pipeline()` | `VideoReader` + `SAM2VideoPropagator` の mask 伝搬確認 |
| `build_sam2_tb_video_pipeline()` | SAM2 video propagation → transparent-background → 動画/連番出力 |

Component 境界では詳細な Python 型ヒントではなく、Haystack 接続互換性のため `list` / `dict` socket を使う。詳細仕様は以下の dict 契約で扱う。

```python
VideoSource = {
    "path": str,
    "fps": float,
    "width": int,
    "height": int,
    "frame_count": int,
    "codec": str,
    "metadata": dict,
}

FrameMaskSequence = {
    "frame_masks": dict[int, np.ndarray],
    "object_ids": list[int],  # 初版は [1] 固定
    "frame_indices": list[int],
    "source": str,
    "metadata": dict,
}

VideoMatteResult = {
    "rgba_video_path": str | None,
    "alpha_video_path": str | None,
    "preview_video_path": str | None,
    "rgba_sequence_dir": str | None,
    "alpha_sequence_dir": str | None,
    "preview_sequence_dir": str | None,
    "sequence_pattern": str | None,
    "fps": float,
    "frame_count": int,
    "output_mode": "video" | "sequence" | "both",
    "metadata": dict,
}
```

出力ディレクトリは `outputs/<timestamp>/video/` と `outputs/<timestamp>/sequence/{rgba,alpha,preview}/`。PNG 連番は `frame_000000.png` 形式で保存する。

動画版の `TransparentBGVideoExtractor` は RAM 安全性のため、RGBA / alpha / preview frame list を全保持しない。frame ごとに動画または PNG 連番へ逐次保存し、`VideoMatteResult` の `rgba_frames` / `alpha_frames` / `preview_frames` は空 list の compact matte として扱う。Gradio callback も `video_reader` / `sam2_video_propagator` / `transparent_bg_video` の巨大中間出力を `include_outputs_from` に含めない。

Text Prompt / GroundingDINO を使った後に動画処理へ進む場合、GroundingDINO / BERT の cache が SAM2 / transparent-background と同時常駐して Colab RAM を圧迫する。動画実行直前に `release_text_detector()` で semantic detector cache を解放する。

### 5-4. SAM2 prompt UI helper 共通化

SAM2 prompt の端吸着・bbox 正規化・overlay 描画は `pipelines/components/ui_helpers.py` を共通利用する。対象関数は `clamp_prompt_point`, `normalize_box_from_points`, `draw_prompt_overlay`, `select_sam2_prompt`, `extend_box_to_edge`。静止画版と動画版 UI は同じ helper を import する。

### 5-5. 動画版 Text Prompt / GroundingDINO 導線

動画版 Haystack UI でも静止画版と同じく、複合対象（例: `person playing drums`, `person riding bicycle`）を意味プロンプトで選ぶ導線を維持する。`gradio_app_sam2_transparent_BG_haystack_for_Movie.py` は GroundingDINO をボタン押下時まで遅延構築し、第 1 フレームに対して Text Prompt 検出を実行する。検出結果の top bbox は `prompt_state["box"]` にコピーされ、以後の `SAM2VideoPropagator` がその bbox を動画全体へ伝搬する。

Movie Notebook 正本 `Sam2_Transparent_Background_Haystack_for_Movie.py` では `GROUNDING_DINO_CKPT_PATH` を取得・環境変数へ設定し、Gradio 起動前診断で SAM2 と GroundingDINO の GPU / checkpoint 状態を確認する。

### 5-6. 動画版の進捗表示と初回既定

動画版 end-to-end Pipeline は `VideoReader` を 1 回だけ実行し、その出力を SAM2 / transparent-background に接続する。進捗を細分化したい場合も Pipeline を複数に分けて動画読込を重複させず、`progress_callback` を `VideoReader` / `SAM2VideoPropagator` / `TransparentBGVideoExtractor` / writer Components に渡して Component 内部から stage / frame 進捗を通知する。

初回 UI 既定はクイックプレビューを優先し、`max_frames=30`, `frame_step=1` とする。長尺・全 frame の最終出力は Advanced で明示的に増やす。例外時は Gradio error に最後の stage と elapsed 秒を含め、Colab ログに final traceback と stage を残す。

---

## 6. 環境変数一覧

| 変数名 | デフォルト値 | 使用ファイル | 用途 |
|--------|------------|------------|------|
| `SAM2_CKPT_PATH` | `checkpoints/SAM2/sam2.1_hiera_large.pt` | `gradio_app_sam2_transparent_BG.py` | SAM2 チェックポイントパス |
| `SAM2_CONFIG_NAME` | `configs/sam2.1/sam2.1_hiera_l.yaml` | 同上 | SAM2 設定名 |
| `PROJECT_ROOT` | ローカルは cwd / `__file__` の親 / Colab は `/content/drive/MyDrive/AI_picasso/Matting-Anything` | `Sam2_Transparent_Background.ipynb` Cell 2 、`gradio_app_sam2_transparent_BG.py` | プロジェクトルートを手動上書き。Colab では Drive を自動マウントし Drive 上のパスを採用 |
| `CUDA_HOME` | - | GroundingDINO ビルド時 | CUDA ツールキットパス |

---

### Colab とローカルのパス対応表

| 環境 | PROJECT_ROOT の値 |
|------|--------------------|
| Windows ローカル | `J:\マイドライブ\AI_picasso\Matting-Anything`（`__file__.parent` / cwd）|
| Google Colab | `/content/drive/MyDrive/AI_picasso/Matting-Anything` |

- Colab では Cell 2 （または `gradio_app_sam2_transparent_BG.py` 起動時）に `google.colab.drive.mount('/content/drive')` を自動実行する。
- `/content/drive/MyDrive` が既に存在していればマウントをスキップ。
- 以下のいずれかの状態では `PROJECT_ROOT` 環境変数で手動上書きさせる: テスト用に別パスを使いたい / Drive マウント位置を変えたい。

---

## 7. よく使うコマンド

```powershell
# uv で仮想環境作成（初回のみ）
uv venv --python 3.11 .venv

# venv アクティベート（PowerShell）
# 注: PowerShell の実行ポリシーが Restricted の場合は Activate.ps1 が走らないので
#     プロセススコープでだけポリシーを RemoteSigned に緩めてからアクティベートする
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
もしくは
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& j:\マイドライブ\AI_picasso\Matting-Anything\.venv\Scripts\Activate.ps1)

# 依存パッケージをインストール
uv pip install -r requirements.txt

# ローカルで gradio アプリを起動（MAM 版）
.venv\Scripts\python.exe gradio_app.py

# ローカルで SAM2 + tb アプリを起動
.venv\Scripts\python.exe gradio_app_sam2_transparent_BG.py

# Haystack Pipeline 版 MAM アプリを起動
.venv\Scripts\python.exe gradio_app_haystack.py

# Haystack Pipeline 版 SAM2 + tb アプリを起動
.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py

# Haystack Component の単体テスト
.venv\Scripts\python.exe -m pytest -m "not integration" -v

# Jupytext 正本から Haystack 版 notebook を生成
.venv\Scripts\python.exe -m jupytext --to ipynb Matting_Anything_Haystack.py
.venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py

# MAM 学習
.venv\Scripts\python.exe main.py --config config/MAM-ViTB-8gpu.toml

# 評価（AM2K）
.venv\Scripts\python.exe evaluation/evaluation_am2k.py
```

> Windows では `python` が Microsoft Store スタブに跡んでいるため、必ず `.venv\Scripts\python.exe` を明示的に呼ぶこと。

### PowerShell 実行ポリシーについて

`.venv\Scripts\Activate.ps1` を実行しようとして `「このシステムではスクリプトの実行が無効になっているため...」` というエラーが出る場合は、PowerShell の実行ポリシーが `Restricted` になっている。次のいずれかで解消する。

| コマンド | スコープ | 効果 |
|---------|---------|------|
| `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` | 現在のターミナルプロセスのみ | ウィンドウを閉じれば消える。一時利用向け。管理者権限不要 |
| `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` | 現在のユーザー恒久 | 以降このユーザーでは Activate.ps1 がそのまま動く。管理者権限不要 |

- `RemoteSigned`: ローカルで作成した `.ps1` は実行可。インターネット由来のものは署名必須。
- `-Scope Process` はレジストリやシステム全体を変更しないため、最も安全。
- セキュリティ要件が厳しい環境では恒久設定（`CurrentUser`）を避け、ターミナル起動時に `Process` スコープで都度緩めるのが推奨。

---

---

## 8. Haystack パイプライン構成

Haystack 版は既存 Gradio アプリを置き換えず、新規 entrypoint として追加する。Gradio は UI、Haystack は推論 DAG、PyTorch/SAM/tb は Component 内部実装として扱う。

| パス | 役割 |
|------|------|
| `pipelines/components/common.py` | 入力正規化、スクリブル解析、bbox、mask dilate、alpha 合成の純粋 Component |
| `pipelines/components/model_components.py` | GroundingDINO、MAM、SAM2、transparent-background、背景生成、出力保存 Component |
| `pipelines/mam_pipeline.py` | MAM + GroundingDINO / scribble の Haystack Pipeline builder |
| `pipelines/sam2_tb_pipeline.py` | SAM2 prompt / transparent-background の Haystack Pipeline builder |
| `gradio_app_haystack.py` | MAM Haystack 版 Gradio アプリ |
| `gradio_app_sam2_transparent_BG_haystack.py` | SAM2 + transparent-background Haystack 版 Gradio アプリ |
| `Matting_Anything_Haystack.py` / `.ipynb` | MAM Haystack 版 Colab 起動ノート |
| `Sam2_Transparent_Background_Haystack.py` / `.ipynb` | SAM2 + transparent-background Haystack 版 Colab 起動ノート |
| `.github/skills/haystack-pipeline/` | Haystack Component 化の作業スキル |
| `tests/unit/` | 純粋 Component と wiring の単体テスト |
| `tests/integration/` | GPU/checkpoint/外部モデル依存テスト骨格 |

### セットアップ

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### テスト

```powershell
.venv\Scripts\python.exe -m pytest -m "not integration" -v
```

---

## 9. Jupytext Notebook 運用

Notebook は `.py` の Jupytext percent 形式を正本とし、`.ipynb` は生成物として扱う。Notebook の内容を変更する場合は `.ipynb` を直接編集せず、対応する `.py` を編集してから Jupytext で再生成する。

### 生成コマンド

```powershell
.venv\Scripts\python.exe -m jupytext --to ipynb Matting_Anything_Haystack.py
.venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py
```

macOS / Linux / Colab では次のように実行する。

```bash
python -m jupytext --to ipynb Matting_Anything_Haystack.py
python -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py
```

### 対応表

| 正本 | 生成物 |
|------|--------|
| `Matting_Anything_Haystack.py` | `Matting_Anything_Haystack.ipynb` |
| `Sam2_Transparent_Background_Haystack.py` | `Sam2_Transparent_Background_Haystack.ipynb` |

*最終更新: 2026-05-25*
