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
| `.github/skills/report-for-leader-denshi/` | 電子さん向けリーダー報告書 作成スキル（SKILL.md / assets / references） | 処理の流れは ASCII フロー図1つを必須（writing_guide「5.」）。未確定機能は▲注記で明示 |
| `報告書/リーダー電子様：…/` | リーダー電子さん向け報告書（5/19 baseline → 6/4 進化版） | スキル `report-for-leader-denshi` のテンプレに準拠して書く |

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

#### SAMURAI（SAM2 互換 fork）

| 項目 | 値 |
|------|-----|
| 配置 | `samurai/`（git 管理外。直接変更禁止） |
| 設定ファイル | `samurai/sam2/sam2/configs/samurai/sam2.1_hiera_*.yaml`（`samurai_mode: true`） |
| API | SAM2 と同一（`build_sam2_video_predictor` 等）。独自抽象を増やさない（YAGNI） |
| 切替方法 | `SAM2_CONFIG_NAME` / `SAM2_CKPT_PATH`（config / 環境変数）と `samurai_mode` で切替える |
| metadata | tracker 切替の痕跡を masks metadata に残す（`tracker_config` / `tracker_checkpoint` / `samurai_mode`） |

- `samurai/` は `segment-anything/` と同様に直接変更しない。変更が必要ならラッパーで対応可能か検討し、不可能ならユーザーへ確認する。
- `SAM2VideoPropagator.tracker_metadata()` が `tracker_config` / `tracker_checkpoint` / `samurai_mode` を返し、`TrackingOverlayWriter` がその metadata を overlay へ伝搬する。

#### Tracking Overlay（追跡確認用 UI）

- SAM2 / SAMURAI の追跡が対象へ追従しているか目視確認できるよう、動画版は追跡 mask の輪郭+半透明塗りを元動画へ重ねた Tracking Overlay 出力を提供する。
- 描画は純粋関数 `pipelines/components/common.py` の `render_tracking_overlay_frame(frame, mask, color, fill_alpha, contour_thickness)` に分離する。
- 書き出しは専用 Component `TrackingOverlayWriter`（`pipelines/components/video_model_components.py`）へ委譲し、frame ごとに mp4 / PNG を逐次保存する（RAM に全 frame を保持しない、ERR030）。
- 進捗は `progress_callback` で stage `tracking_overlay` を出す（ERR029）。
- overlay metadata には tracker 種別（`tracker_config` / `tracker_checkpoint` / `samurai_mode`）を残す。
- Gradio 動画アプリは `overlay_enabled` Checkbox（`info=` 完備）と `Tracking Overlay (追跡確認用)` Video 出力を提供し、callback の `include_outputs_from` に `tracking_overlay` を含める（ERR018）。

#### SAM2 bbox UI 仕様

- SAM2 の bbox 座標をユーザーに数値手入力させない。
- Gradio UI では画像上のマウス操作で bbox を指定する。
- Haystack 版 SAM2 アプリでは、アップロード用 `Input Image` と prompt 編集用 `SAM2 Prompt Canvas` を分離する。`SAM2 Prompt Canvas` は `sources=[]` の編集面にし、画像のドラッグ＆ドロップ先にしない。クリックイベントは `prompt_canvas.select(...)` に紐づけ、推論には原本の `input_image` を渡す。
- `SAM2 Prompt Canvas` はアップロード先にしないため `sources=[]` を維持しつつ、point / bbox click を受けるため `interactive=True` にする（ERR026）。
- bbox は 2 点クリックで確定し、クリック順序に依存せず `[x_min, y_min, x_max, y_max]` へ正規化する。
- 画像端付近のクリックは端座標へ吸着し、被写体が画面外へ続くケースでも画面端まで選択できるようにする。
- Haystack 版 SAM2 アプリでは `gr.Image(type="numpy", interactive=True)` と `normalize_box_from_points()` / `clamp_prompt_point()` を使う。bbox / point 用途で `gr.ImageEditor` を使わない。
- 予測画像と prompt canvas は `Image Display Size` で `window`（既定・固定高さ）/ `original`（原寸）を切り替えられるようにする。

#### UI パラメーター解説 (info=) 規約

- Gradio UI の全パラメーター（`gr.Slider` / `gr.Radio` / `gr.Checkbox` / `gr.Textbox` 等）は、誤解のない解説を `info=` で完備する（プロジェクトルール）。
- `info=` には次の 3 点を必ず含める。
  1. **数値の単位**: px / frame 数 / `0.0〜1.0` の正規化アルファ / 個数 / 真偽値 / 単位なしの選択値 など。
  2. **各値の具体的な意味**: 低い値・高い値・ON・OFF・各選択肢が結果に何を与えるか。
  3. **推奨の目安値**。
- 選択肢型（`gr.Radio` 等）は各 choice の意味を列挙する。
- UI パラメーターを追加・変更した場合は `info=` の完備を同時に行う。
- 対象ファイル例: `gradio_app_sam2_transparent_BG.py`, `gradio_app_sam2_transparent_BG_haystack.py`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`。

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
- **mask dtype 契約（bool / soft 確率の両対応）**: `MatteExtractor`（`TransparentBGExtractor`）が受ける `mask` は (H,W) の **bool** または **float32 soft 確率 [0,1]** のいずれかを許容する。float の場合は前景判定（有無・bbox）を閾値 0.5 で行い、guard には soft 確率をそのまま使う（`soft_probability_guard`）。`max(probA,probB) >= 0.5 ⟺ binaryA OR binaryB` のため bool との前景領域互換は保たれる。`mask + image -> MatteResult` adapter を別実装に差し替える際は、この bool/float 両対応を満たすこと。

### 5-3. 動画版 Haystack I/O 契約

動画版は `pipelines/sam2_tb_video_pipeline.py` に 4 つの builder を持つ。

| builder | 用途 |
|---------|------|
| `build_video_reader_pipeline()` | 動画から第 1 フレームと metadata を取得する軽量 Pipeline |
| `build_sam2_video_propagation_pipeline()` | `VideoReader` + `SAM2VideoPropagator` の mask 伝搬確認 |
| `build_sam2_tb_video_pipeline()` | SAM2 video propagation → transparent-background → 動画/連番出力 |
| `build_tb_only_video_pipeline()` | SAM2/DINO なし。`VideoReader` → `TransparentBGVideoExtractor`（masks 未接続＝全画面 tb）→ 動画/連番出力。グリーンバック等の追跡不要ケース向け軽量経路 |

> **tb-only 経路**: `build_tb_only_video_pipeline()` は masks ソケットを接続しないため `TransparentBGVideoExtractor` に `mask=None` が渡り、各フレームを全画面のまま salient/human matting する（crop / guard / 所有権合成 / tracking overlay なし）。動画 UI では「背景除去のみ (tb only)」タブから利用し、コールバックは `run_tb_only_background_removal`（prompt 不要、6-tuple 返却）。


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
    "frame_masks": dict[int, np.ndarray],  # 値は (H,W) bool または float32 soft 確率 [0,1]
    "object_ids": list[int],  # 初版は [1] 固定 / 複合対象では box 分の 1..N
    "frame_indices": list[int],
    "source": str,
    "metadata": dict,
    # 動画版 5-6: per-object logit 保持。OwnershipResolver 用に各 frame の
    # (N,H,W) logit を同梱（任意キー）。OwnershipResolver 通過後は `ownership`
    # （per frame (N+1,H,W)、最終チャネルが背景）も付与される。
    "per_object_logits": dict[int, np.ndarray],  # frame_idx -> (N,H,W) logits
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

SAM2 prompt の端吸着・bbox 正規化・overlay 描画は `pipelines/components/ui_helpers.py` を共通利用する。対象関数は `clamp_prompt_point`, `normalize_box_from_points`, `draw_prompt_overlay`, `select_sam2_prompt`, `extend_box_to_edge`。静止画版と動画版 UI は同じ helper を import する。`prompt_state` は単一 `box` に加えて複合対象用の `boxes`（int 4 要素 list の list）を保持し、`empty_prompt_state()` / `copy_prompt_state()` / `draw_prompt_overlay()` が複数 bbox を扱う（`draw_prompt_overlay` は色循環 + 番号ラベルで `boxes` を描画してから単一 gold `box` を重ねる）。

### 5-5. 動画版 Text Prompt / GroundingDINO 導線 + 複合対象 union / フレーム選択 / 双方向伝播

動画版 Haystack UI でも静止画版と同じく、複合対象（例: `person playing drums`, `person riding bicycle`）を意味プロンプトで選ぶ導線を維持する。`gradio_app_sam2_transparent_BG_haystack_for_Movie.py` は GroundingDINO をボタン押下時まで遅延構築し、Text Prompt 検出を実行する。検出結果は top bbox を `prompt_state["box"]` に、全候補 bbox を `prompt_state["boxes"]` にコピーする。候補は `CheckboxGroup`（`populate_candidate_choices` がラベル生成、`apply_selected_boxes` が選択 bbox を `prompt_state["boxes"]` へ反映）でユーザーが複数選び、複合対象として union できる。

`SAM2VideoPropagator.run` は `boxes`（複数）/ `prompt_frame_idx`（起点フレーム）/ `bidirectional`（双方向）を受け取る。複数 bbox は obj_id 1..N として登録し、frame ごとに全 obj の mask を **soft max union（各 obj を `stable_sigmoid` で確率化し `np.maximum`）** して 1 枚へ統合するため、下流契約 `frame_masks: source_index → 1 枚` は不変（`TransparentBGVideoExtractor` / writer 改修不要）。ただし統合結果は二値化せず **float32 soft 確率 [0,1]** で `frame_masks` に格納する（消える線=継ぎ目の根治, ERR044）。`max(prob) >= 0.5 ⟺ binary OR` のため前景領域は従来 OR と一致し、差は継ぎ目が黒線でなく中間 alpha になる点のみ。`bidirectional=True` のとき forward / backward の 2 パスを走らせ各 frame で結果を `np.maximum` 統合する。`boxes` と補正 `points` を併用する場合、点群を別 obj にまとめず `assign_points_to_boxes` で各点を最近傍 box の object prompt に同梱する（複数インスタンスで点が落ちるのを防ぐ, ERR043）。これらの新パラメータは Component run kwargs の auto-socket 経由で `pipeline.run(data={...})` から渡せるため pipeline 結線は不変。`boxes=None` のときは従来の単一 box/point・`prompt_frame_idx=0`・forward only パスを完全に維持する（後方互換）。

### 5-6. 動画版 per-object logit 保持 + OwnershipResolver（画素 softmax 所有権）

継ぎ目の根治をさらに進め、`SAM2VideoPropagator` は各 obj の logit を二値化・union せず **per-object logit を保持**して下流へ渡す。`propagate_in_video` の forward / reverse 2 pass は **必ず object_id をキーに整列**してマージする（中間構造 `source_index → {obj_id → (H,W)}`、欠損 obj は `-1e6` 埋め、`np.maximum` で pass 統合）。位置ベースで stack すると追跡途切れ/再出現で pass 間の obj 数が変わった際に別 obj の logit が混入するため避ける。整列後に `target_object_ids` 順で `(N,H,W)` を構築し `FrameMaskSequence["per_object_logits"]` に同梱する。overlay / 後方互換用の union soft `frame_masks` は per-object logit から `stable_sigmoid` → 画素 max で派生し、既存契約（`metadata` / `object_ids` / soft union `frame_masks`）を温存する。

`OwnershipResolver`（`pipelines/components/ownership_resolver.py`）は propagator と `TransparentBGVideoExtractor` の間に挿入され、`masks` dict を受けて per-object logits に **背景 logit=0 を明示チャネルとして加えた N+1 チャネル**の温度 τ softmax で **画素ごと和=1 の所有権**を算出する。前景 soft = `clip(1 - 背景所有権, 0, 1)` を `frame_masks` に差し替え、`ownership`（per frame `(N+1,H,W)`、最終チャネルが背景）も同梱して下流へ渡す。単一 obj 画素は sigmoid 相当、重なり画素のみ softmax で所有権を分配する。`_softmax_across_objects` は `temperature <= 0` を `ValueError` で防御し、max 減算で数値安定化する。

温度 τ は `config/inference_models.toml` の各 background entry に `ownership_temperature`（>0）で定義し、ハードコードしない。movie app（`gradio_app_sam2_transparent_BG_haystack_for_Movie.py`）は `bg_entry.get("ownership_temperature", 1.0)` を読み `pipeline.run` の `"ownership_resolver": {"temperature": ...}` へ配線する。配線は `sam2_video_propagator.masks → ownership_resolver.masks → transparent_bg_video.masks`。overlay は生トラッキング可視化として `sam2_video_propagator.masks` を継続使用する。

#### per_object 連続アルファ合成（video_matte_mode）

背景透過の経路は config `video_matte_mode` で切り替える（各 background entry に定義、ハードコード禁止）。

> **既定変更（2026-06-18）**: グリーンバック実動画で union モードが細い/前ボケ対象（ドラムスティック・前ボケシンバル）を under-matte する事象を受け、全 background entry の既定を `video_matte_mode="per_object"` + `mask_feather=0` に変更。union は軽量モードとして config で明示的に選択する。

- `"union"`（軽量モード）: フレームあたり tb 1 回。`OwnershipResolver` が差し替えた union soft `frame_masks` の外接矩形で 1 度だけ切り抜く軽量経路。従来挙動と後方互換。
- `"per_object"`（**現行既定**）: フレームあたり tb N 回（対象数）。`TransparentBGVideoExtractor._run_per_object_frame` が各対象の logit を `stable_sigmoid` → soft mask として既存 `TransparentBGExtractor.run` を呼び（bbox 導出・crop・tb・full frame 配置・soft guard を再利用）、対象ごとの連続アルファ `alpha_o`（full frame）を得る。最終アルファは純粋関数 `composite_alpha_by_ownership(per_object_alphas, ownership)`（`pipelines/components/video_common.py`）で `alpha_final(p) = max_{o=0..N-1} alpha_o(p)`（**比較明 / lighten**）として合成する。**RGB は元フレームのまま、アルファのみ合成**する。

> **合成方式変更（2026-06-18）**: 旧仕様の所有権加重和 `Σ_o ownership_o × alpha_o` は、対象が重なる画素で手前対象のアルファが 0（黒）のとき背後の残したい対象まで減衰して黒く潰す欠点があった。比較明 max は対象ごとアルファの最大値を採るため、どれか 1 対象でも前景なら最終アルファに残り、対象同士の重なりで黒抜けが起きない。`ownership` 引数は合成には乗じず、前景チャネル数 N と per_object_alphas 数の一致検証にのみ使用する。

`composite_alpha_by_ownership` は ownership 形状 `(N+1,H,W)` と per_object_alphas 数 N の一致を検証し、入力・出力ともに `[0,1]` に clip、不一致時は `ValueError`。

`run()` のフレームループは `video_matte_mode == "per_object"` かつ当該 frame の `per_object_logits`（`(N,H,W)`）と `ownership` が揃い対象数 ≥1 のときのみ per_object 経路を使い、欠如時は union 経路へフォールバックする。実行モードは matte メタデータの `video_matte_mode` に記録する。per_object は tb 呼び出し回数（frames × objects）が増えるため重いが、対象ごとに crop して salient 前景として扱うため細い/前ボケ対象の欠落を抑える。エッジの半透明（tb 連続アルファ＋sigmoid 由来ソフト）を硬化したい場合は movie app の Alpha threshold スライダで二値化する。`mask_feather` は既定 0（guard 境界の Gaussian feather オフ）。

movie app は `bg_entry.get("video_matte_mode", "union")` を読み `transparent_bg_video` へ配線する。CLI `run_video_matting_headless.py` は `--matte-mode {union,per_object}`（未指定で config 既定）で上書きできる。

#### ヘッドレス実行 CLI

`run_video_matting_headless.py` は Gradio を起動せず end-to-end でパイプラインを実行する検証用エントリポイント。`--video`（fail-fast でパス存在検証）/ `--box` / `--point`（繰り返し可）/ `--tracker` / `--background` / `--temperature`（未指定で config の `ownership_temperature`）/ `--matte-mode`（未指定で config の `video_matte_mode`）/ `--output-mode` 等を受ける。`_parse_box("x1,y1,x2,y2")` / `_parse_point("x,y[,label]")` / `build_arg_parser`。GroundingDINO テキストプロンプトは扱わず box/point 直接指定のみ。

UI では入力動画のシーク位置を起点フレームに同期する。`gr.Video` の seek / pause / loadedmetadata は `Blocks(js=...)` のブラウザ hook で検出し、`prompt_frame_idx` Slider（サンプリング後シーケンス index 0〜max_frames-1）へ反映する。`extract_prompt_frame` は raw_index = slider × frame_step で抽出し index 整合を担保する。`frame_step` はフレーム取得系に配置し、`prompt_frame_idx` は「表示中フレームを再取得」ボタンと併用できる。双方向は `gr.Checkbox` で切り替え、フレーム選択 Slider は座標手入力ではないため ERR017 に抵触しない。画面の見出し順は `フレーム取得 → DINO → SAM → 背景透過`。

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

---

## 10. モデルレジストリとプルダウン切替

Gradio UI のプルダウンメニューで検知器・トラッカー・背景除去モデルを切り替えられる仕組み。モデルの追加・変更は **TOML を編集するだけ**で、Python コードは変更しない。

### 10-1. 関連ファイル

| ファイル | 役割 |
|---------|------|
| `config/inference_models.toml` | モデルレジストリ設定ファイル（正本）。モデル追加はここに entry を書く |
| `pipelines/components/model_registry.py` | TOML を読む純粋ローダ。モデル重みは一切読まない |
| `gradio_app_sam2_transparent_BG_haystack.py` | 静止画版 Gradio アプリ。背景モデル Dropdown 結線済み |
| `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` | 動画版 Gradio アプリ。トラッカー + 背景モデル Dropdown 結線済み（実装中） |
| `gradio_app_haystack.py` | MAM Haystack 版 Gradio アプリ。背景モデル Dropdown 結線済み（暫定：MAM Pipeline 未接続） |

### 10-2. 3 つの役割（role）

| role | 説明 | 対応 Dropdown |
|------|------|--------------|
| `detector` | テキストプロンプトから bbox を検出する視覚理解モデル | Text Prompt タブ内（将来結線予定） |
| `tracker` | 動画フレーム間でマスクを伝播するトラッキングモデル | 動画版 Tracker Model Dropdown（実装中） |
| `background` | アルファマット・背景除去モデル | 全アプリの Background Model Dropdown |

### 10-3. 現在登録されているモデル

#### detector（検知器）

| id | label | component | 備考 |
|----|-------|-----------|------|
| `groundingdino_swint_ogc` | GroundingDINO SwinT-OGC (default) | `GroundingDINODetector` | 唯一の detector エントリ |

#### tracker（トラッカー）

| id | label | component | 有効条件 |
|----|-------|-----------|---------|
| `sam2_hiera_l` | SAM2.1 Hiera-Large (standard) | `SAM2VideoPropagator` | `INFERENCE_TRACKER_VARIANT=sam2_facebook` または未設定 |
| `sam2_hiera_b_plus` | SAM2.1 Hiera-B+ (lighter) | `SAM2VideoPropagator` | `INFERENCE_TRACKER_VARIANT=sam2_facebook` または未設定 |
| `samurai_hiera_l` | SAMURAI Hiera-Large (motion-aware) | `SAM2VideoPropagator` | `INFERENCE_TRACKER_VARIANT=sam2_samurai` |
| `samurai_hiera_b_plus` | SAMURAI Hiera-B+ (motion-aware / lighter) | `SAM2VideoPropagator` | `INFERENCE_TRACKER_VARIANT=sam2_samurai` |

SAM2 と SAMURAI は `INFERENCE_TRACKER_VARIANT` 環境変数で切り替える。未設定時は全 tracker entry が可用とみなされる（後方互換）。

```powershell
# SAM2 標準版を使う場合（デフォルト）
$env:INFERENCE_TRACKER_VARIANT = "sam2_facebook"

# SAMURAI（モーションアウェア）を使う場合
$env:INFERENCE_TRACKER_VARIANT = "sam2_samurai"
```

#### background（背景除去）

| id | label | component | `tb_mode` |
|----|-------|-----------|----------|
| `tb_base` | transparent-background base（高精度） | `TransparentBGExtractor` | `base` |
| `tb_fast` | transparent-background fast（軽量） | `TransparentBGExtractor` | `fast` |
| `tb_base_nightly` | transparent-background base-nightly（最新実験版） | `TransparentBGExtractor` | `base-nightly` |

### 10-4. model_registry.py API

```python
from pipelines.components.model_registry import (
    build_dropdown_choices,
    entries_for,
    entry_by_id,
    is_available,
    load_model_registry,
    clear_registry_cache,
)

# Gradio Dropdown に渡す (label, id) リストを取得
choices = build_dropdown_choices("background")
# → [("transparent-background base（高精度）", "tb_base"), ...]

# role の全 entry を取得
entries = entries_for("tracker")

# id で entry を 1 件取得（Gradio callback の中で tb_mode などを参照）
entry = entry_by_id("background", "tb_base")
entry["tb_mode"]   # → "base"

# 特定 entry が現在の環境で利用可能か判定
is_available(entry)  # → True / False

# TOML を再編集した後にキャッシュをクリア（テスト時）
clear_registry_cache()
```

### 10-5. Gradio Dropdown の使い方パターン

```python
# UI 定義
background_model_dd = gr.Dropdown(
    choices=build_dropdown_choices("background"),
    value=build_dropdown_choices("background")[0][1],  # 先頭 entry の id
    label="Background Model",
    info="背景除去モデルを選択。base=高精度・推奨、fast=軽量・速度重視、base-nightly=実験的最新版。",
)

# callback の中で id → entry の詳細フィールドを参照
def _run_inference(..., background_model_id: str):
    entry = entry_by_id("background", background_model_id)
    tb_mode = entry["tb_mode"]   # → "base" / "fast" / "base-nightly"
    ...
```

### 10-6. モデルを新規追加する手順

1. `config/inference_models.toml` に entry を追記する（Python コード変更不要）。  
   - 必須フィールド: `id`（一意な識別子）、`label`（Gradio 表示名）、`component`（既知クラス名）。  
   - 既知でない `component` クラス名を書くと `load_model_registry()` が `ValueError` を raise する。
2. 新 Component クラスを実装する場合は `pipelines/components/model_registry.py` の `_KNOWN_COMPONENTS` に追加する。
3. `clear_registry_cache()` を呼ぶか、プロセスを再起動して TOML キャッシュをリフレッシュする。
4. unit test を `tests/unit/test_model_registry.py` に追記し、`pytest -m "not integration" -q` でパスを確認する。

### 10-7. tracker 切替と SAMURAI の注意点

- `SAM2VideoPropagator` は `config_name` に渡されたパスで動作する。SAM2 と SAMURAI は同一クラスを共有し、`samurai_mode` フラグ（`samurai/*.yaml` 内 `samurai_mode: true`）で切り替わる（YAGNI）。
- tracker entry の metadata（`tracker_config` / `tracker_checkpoint` / `samurai_mode`）は `FrameMaskSequence.metadata` と `TrackingOverlayWriter` を通じて出力に伝搬する。
- `samurai/` ディレクトリは直接変更しない（`copilot-instructions.md` の禁止事項）。

*最終更新: 2026-06-03（セクション 10 追加）*
