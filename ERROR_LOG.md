# エラーログ — エラー履歴と対処法

> **ルール**: 作業開始前に必ずこのファイルを読む。新しいエラーを解決したら必ず追記する。同じエラーを二度繰り返さない。

---

## 記載フォーマット

```
### [ERRXXX] エラータイトル

| 項目 | 内容 |
|------|------|
| **深刻度** | Critical / High / Medium / Low |
| **頻度** | 頻発 / 時々 / 一度のみ |
| **初回発生日** | YYYY-MM-DD |

**エラー内容**:
（エラーメッセージ・スタックトレースの要点）

**原因**:
（根本原因の説明）

**対処法**:
（再現・解決手順）

**備考**:
（関連ファイル・PR・コミット等）
```

---

## エラー一覧

### [ERR-VID-GUARD] 動画背景除去：人物マスクが半透明化する（guard が内部を削る）【Phase1 修正済】

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | union モードで頻発 |
| **初回発生日** | 2026-06-18 |

**エラー内容**:
人体が得意なはずの transparent-background の人物アルファが半透明になる。トラッキング領域全体の信頼度がアルファに反映されているように見える。

**原因**:
`TransparentBGExtractor.run`（pipelines/components/model_components.py）の `full_alpha = full_alpha * guard`。union モードで `OwnershipResolver` が frame_masks を「前景 soft = 1 − 背景所有権」という**領域全体の連続確率**に差し替え、その float mask が guard 生成時に `soft_probability_guard` を経由して**領域内部も 1.0 未満**になり、tb の人物アルファ内部に乗算されて半透明化。guard は本来「形状外の漏れ alpha を 0 にするゲート」であるべきで、内部の信頼度を掛けるのは設計誤り。

**対処法**:
guard 分岐を「mask が float か binary か」ではなく **`mask_guard_feather` の有無**で分岐。
- `mask_guard_feather > 0`：soft guard（オプトイン。float→`soft_probability_guard`／binary→`feather_binary_mask`）。
- `mask_guard_feather <= 0`（既定）：float/binary を問わず **`dilate_binary_mask`（内部 1.0・外部 0 の二値ゲート）**。float は 0.5 閾値で二値化。

これで guard は形状外ゲートに徹し、tb の連続アルファ内部を一切減衰させない。

**備考**:
- テスト追加: `tests/unit/test_transparent_bg_mask_guard.py`（`test_float_soft_mask_guard_keeps_interior_alpha_unscaled` / `test_float_soft_mask_guard_feather_opt_in_softens_edge`）。
- 検証: 非 integration 全体 180 passed / 1 skipped。サブエージェントレビュー APPROVE。
- 教訓: guard は「拡張(dilate)／形状外ゲート」のみ。内部を乗算で削るのは禁止。根本原因はコード計測で確定してから対処する。

### [ERR001] Gradio 5 で `block = block.queue()` が None を返す

| 項目 | 内容 |
|------|------|
| **深刻度** | Critical |
| **頻度** | 毎回（Gradio 5 以上） |
| **初回発生日** | 2025-07-23 |

**エラー内容**:
`AttributeError: 'NoneType' object has no attribute 'launch'` — `block.queue()` が `None` を返すため `with block:` が失敗。

**原因**:
Gradio 5 で `queue()` が `self` を返さなくなった（in-place 操作に変更）。

**対処法**:
```python
# NG (Gradio 4 パターン)
block = gr.Blocks()
block = block.queue()
with block:
    ...

# OK (Gradio 5 パターン)
with gr.Blocks() as block:
    ...
block.queue()
block.launch(...)
```

**備考**: gradio_app.py 修正済み（2025-07-23）

---

### [ERR002] Gradio 5 で `gr.Image(tool="sketch")` が AttributeError

| 項目 | 内容 |
|------|------|
| **深刻度** | Critical |
| **頻度** | 毎回（Gradio 5 以上） |
| **初回発生日** | 2025-07-23 |

**エラー内容**:
`AttributeError: tool parameter is not supported` — Gradio 5 で `gr.Image` の `tool` 引数が廃止。

**原因**:
Gradio 5 で描画機能が `gr.ImageEditor` に分離された。

**対処法**:
```python
# NG
gr.Image(type="numpy", tool="sketch")
# OK
gr.ImageEditor(type="numpy", value="assets/demo.jpg", label="Upload Image")
```
`ImageEditor` の戻り値は `{"background": ndarray, "layers": [ndarray (H,W,4)], "composite": ndarray}` の dict。

**備考**: gradio_app.py 修正済み（2025-07-23）

---

### [ERR003] Gradio 5 で `input_image["image"]` が KeyError

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 毎回（Gradio 5 以上） |
| **初回発生日** | 2025-07-23 |

**エラー内容**:
`KeyError: 'image'` — `gr.ImageEditor` の戻り値に `"image"` キーが存在しない。

**原因**:
Gradio 4 では `gr.Image` が `{"image": ndarray}` を返していたが、Gradio 5 の `gr.ImageEditor` は `{"background", "layers", "composite"}` を返す。

**対処法**:
```python
image_ori = input_image.get('background', input_image.get('composite', input_image.get('image')))
```
`scribble` は `layers[0]` の (H, W, 4) RGBA ndarray を使用。後続の `scribble.transpose(2,1,0)[0]` と互換性あり。

**備考**: gradio_app.py 修正済み（2025-07-23）

---

### [ERR004] GroundingDINO が CPU モードにフォールバック（`Failed to load custom C++ ops`）

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 毎回（環境変数未設定時） |
| **初回発生日** | 2025-07-24 |

**エラー内容**:
```
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
```
GroundingDINO の CUDA カーネル（`ms_deform_attn`）がロードできず CPU 強制になる。

**原因**:
インストールセルで `!export CUDA_HOME=...` を使っていたが、`!` コマンドは**別のサブシェル**で実行されるため、次の `!pip install -e GroundingDINO` には環境変数が引き継がれない。
結果として `CUDA_HOME=None` の状態でビルドされ CUDA 拡張がスキップされる。
`BUILD_WITH_CUDA` は GroundingDINO の setup.py で `AM_I_DOCKER=True` と AND 条件のため単独では機能しない。実際に効くのは `CUDA_HOME` を設定して `torch.cuda.is_available()` が True の場合のみ。

**対処法**:
```python
import os
# Python から環境変数を設定（!コマンドにも引き継がれる）
os.environ['CUDA_HOME'] = '/usr/local/cuda'

# --no-build-isolation: 既インストール済みの torch/numpy を使ってビルド（CUDA 拡張に必要）
# -q は外す: CUDA ビルドエラーが出力に隠れないようにする
# 再ビルドのため事前にアンインストール
!pip uninstall groundingdino -y -q
!pip install -e GroundingDINO --no-build-isolation
```

**備考**: Matting_Anything.ipynb セル5 修正済み（2025-07-24）

**2026-05-28 追記**: SAM2 / GroundingDINO Haystack 系と legacy `gradio_app.py` の本番・映像制作導線では CPU fallback は緊急回避専用。`MATTING_ANYTHING_ALLOW_CPU=1` を明示しない限り、SAM2 / GroundingDINO / MAM の重い推論は CUDA 不可時に fail fast する。

---

### [ERR005] `TypeError: to() received an invalid combination of arguments - got (dtype=torch.device, )`

| 項目 | 内容 |
|------|------|
| **深刻度** | Critical |
| **頻度** | 毎回（新しい transformers との組み合わせ） |
| **初回発生日** | 2026-05-14 |

**エラー内容**:
```
File ".../bertwarper.py", line 109, in forward
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
File ".../transformers/modeling_utils.py", line 974, in get_extended_attention_mask
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
TypeError: to() received an invalid combination of arguments - got (dtype=torch.device, )
```

**原因**:
新しい `transformers`（4.x 以降のある時点から）では `get_extended_attention_mask` の第3引数が `device` → `dtype` に変更された。`bertwarper.py` が旧シグネチャで `device` オブジェクトを渡していたため、内部で `dtype` として使われ `.to(dtype=<device>)` が例外。

**対処法**:
`GroundingDINO/groundingdino/models/GroundingDINO/bertwarper.py` の該当行から `device` 引数を削除する:
```python
# 修正前
extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
    attention_mask, input_shape, device
)
# 修正後
extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
    attention_mask, input_shape
)
```

**備考**: `bertwarper.py` 修正済み（2026-05-14）

---

### [ERR006] `NameError: name '_C' is not defined` in ms_deform_attn.py

| 項目 | 内容 |
|------|------|
| **深刻度** | Critical |
| **頻度** | 毎回（CUDA ops ビルド失敗時） |
| **初回発生日** | 2026-05-14 |

**エラー内容**:
```
File ".../ms_deform_attn.py", line 53, in forward
    output = _C.ms_deform_attn_forward(
             ^^
NameError: name '_C' is not defined
```

**原因**:
`_C`（GroundingDINO の CUDA カスタムカーネル）のインポートが失敗すると警告のみ出して `_C` は未定義のままになる。
しかし `MultiScaleDeformableAttention.forward()` は `torch.cuda.is_available() and value.is_cuda` が True なら `MultiScaleDeformableAttnFunction` 経由で `_C` を呼ぶため、CUDA テンソルが来ると NameError になる。

**原因コード（修正前）**:
```python
try:
    from groundingdino import _C
except:
    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")
# _C が未定義のまま、以下で呼ばれる
output = _C.ms_deform_attn_forward(...)
```

**対処法**:
1. import 部分に `CUDA_OPS_AVAILABLE` フラグを追加
2. `forward()` の分岐条件に `CUDA_OPS_AVAILABLE` を追加してCPU フォールバックを確実にする

```python
try:
    from groundingdino import _C
    CUDA_OPS_AVAILABLE = True
except:
    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")
    CUDA_OPS_AVAILABLE = False

# forward() 内:
if torch.cuda.is_available() and value.is_cuda and CUDA_OPS_AVAILABLE:
    output = MultiScaleDeformableAttnFunction.apply(...)
else:
    output = multi_scale_deformable_attn_pytorch(...)
```

**備考**: `ms_deform_attn.py` 修正済み（2026-05-14）。CUDA ops が使えない場合でも PyTorch 純正実装でフォールバック動作する。

---

### [ERR007] `UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly`

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium（現在は警告、PyTorch 2.9 以降は例外） |
| **頻度** | `use_checkpoint=True` 設定時に毎回 |
| **初回発生日** | 2026-05-14 |

**エラー内容**:
```
UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.
Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception.
use_reentrant=False is recommended for most use cases.
```

**原因**:
PyTorch 2.x 途中から `torch.utils.checkpoint.checkpoint()` に `use_reentrant` の明示が必要になった。
GroundingDINO の `transformer.py`（2箇所）と `backbone/swin_transformer.py`（1箇所）が引数なしで呼んでいた。

**対処法**:
`checkpoint.checkpoint(...)` 呼び出しすべてに `use_reentrant=False` を追加する:
```python
# 修正前
output = checkpoint.checkpoint(layer, *args)
# 修正後
output = checkpoint.checkpoint(layer, *args, use_reentrant=False)
```

**備考**: `transformer.py` 2箇所、`backbone/swin_transformer.py` 1箇所を修正済み（2026-05-14）。

---

### [ERR008] `RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0`

| 項目 | 内容 |
|------|------|
| **深刻度** | Critical |
| **頻度** | PNG アップロード時・Gradio 5 環境で頻発 |
| **初回発生日** | 2026-05-14 |

**エラー内容**:
```
RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
```
`gradio_app.py` 内の `image = (image - pixel_mean) / pixel_std` で発生。

**原因**:
`gr.ImageEditor` の `background` キーは RGBA (4ch) ndarray を返す場合がある（PNG アップロード時、または Gradio が内部で RGBA 変換する場合）。
これを `torch.as_tensor().permute(2,0,1)` すると `(4,H,W)` テンソルになるが、`pixel_mean` は `(3,1,1)` なのでブロードキャスト時に次元不一致エラーが発生する。

**対処法**:
`image_ori` 取得直後に RGBA → RGB 変換を追加:
```python
if isinstance(image_ori, np.ndarray) and image_ori.ndim == 3 and image_ori.shape[2] == 4:
    image_ori = image_ori[:, :, :3]
```
`scribble`（`layers[0]`、RGBA マスク）は変換不要。`transpose(2,1,0)[0]` でチャンネル0のみ取り出すため RGBA のままでも問題なし。

**備考**: `gradio_app.py` の `image_ori` 抽出直後（line 85 付近）に修正済み（2026-05-14）。

---

### [ERR009] `FutureWarning: torch.cuda.amp.autocast(args...) is deprecated`

| 項目 | 内容 |
|------|------|
| **深刻度** | Low（警告のみ、将来は例外になる可能性） |
| **頻度** | 毎回（PyTorch 2.x 以降） |
| **初回発生日** | 2026-05-14 |

**エラー内容**:
```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
Please use `torch.amp.autocast('cuda', args...)` instead.
```
`transformer.py` の `forward_ffn()` メソッド内で発生。

**原因**:
PyTorch 2.x 以降、`torch.cuda.amp.autocast` は deprecated。
`torch.amp.autocast(device_type, ...)` が新しい推奨 API。

**対処法**:
```python
# 修正前（deprecated）
with torch.cuda.amp.autocast(enabled=False):
    ...
# 修正後
with torch.amp.autocast('cuda', enabled=False):
    ...
```

**備考**: `GroundingDINO/groundingdino/models/GroundingDINO/transformer.py` line 864 付近を修正済み（2026-05-14）。`enabled=False` の場合は no-op なので動作上の差異なし。

---

### [ERR010] SAM2 pip install で CUDA ops ビルドエラーが `-q` フラグで隠れる

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | 時々（Colab 環境変動時） |
| **初回発生日** | 2026-05-22 |

**エラー内容**:
`!pip install -q git+https://github.com/facebookresearch/sam2.git` 実行時にビルドエラーが `-q` フラグで完全に抑制され、次のセルで `ModuleNotFoundError: No module named 'sam2'` が発生する。

**原因**:
Colab の CUDA バージョンやツールチェーンのアップデートにより SAM2 の C++ 拡張ビルドが失敗する場合がある。`-q` フラグがあるとエラーメッセージが見えず原因調査が困難になる。

**対処法**:
`-q` フラグを削除してインストールエラーを可視化する。
```python
# NG
!pip install -q git+https://github.com/facebookresearch/sam2.git

# OK
!pip install git+https://github.com/facebookresearch/sam2.git
```
`Sam2_Transparent_Background.ipynb` Cell 1 で修正済み（2026-05-22）。

**備考**: GroundingDINO も同様のリスクがある（既存 ERR004/ERR006 参照）。`-q` は Colab ノートブックで禁止パターン。

**2026-05-28 追記（Haystack 版 / エラーログ_07）**:
`Sam2_Transparent_Background_Haystack.py` の Gradio 起動後、`SAM2Segmenter.warm_up()` の `from sam2.build_sam import build_sam2` で `ModuleNotFoundError: No module named 'sam2'` が発生した。直接原因は SAM2 package が Colab runtime に入っていない、または install cell 失敗後に Gradio 起動セルへ進んだこと。静止画・動画 Haystack Notebook の診断セルに `import sam2`, `from sam2.build_sam import build_sam2`, `from sam2.sam2_image_predictor import SAM2ImagePredictor`, `from sam2.build_sam import build_sam2_video_predictor` の起動前 preflight を追加し、未導入なら Gradio を公開する前に停止する。

**2026-05-28 再発防止チェック**:
- Cell 1 の `git+https://github.com/facebookresearch/sam2.git` install を `-q` なしで実行し、エラーがないことを確認する。
- 診断セルで `sam2 package = ...` と `sam2 image imports = OK` / `sam2 video imports = OK` が出てから Gradio を起動する。
- install 後にランタイムを再起動した場合は、必ず install cell から再実行する。
- SAM2 import を直した後も、ERR025（GPU runtime）、ERR004 / ERR006（GroundingDINO CUDA ops）、ERR023 / ERR024 / ERR005（GroundingDINO 依存・transformers 互換）を続けて確認する。

---


### [ERR011] Gradio 5.x で /info エンドポイントが ASGI 例外（`api_info` 失敗）

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | Gradio 5.x + 特定 schema で再現 |
| **初回発生日** | 2026-05-22 |
| **解決日** | 2026-05-22 |

**エラー内容**:
```
TypeError: argument of type 'bool' is not iterable
  File ".../gradio_client/utils.py", line 887, in get_type
    if "const" in schema:   ← schema が bool のためクラッシュ
  File ".../gradio_client/utils.py", line 982, in _json_schema_to_python_type
    ... schema['additionalProperties'] ...  ← additionalProperties が bool
  File ".../gradio/blocks.py", line 2925, in get_api_info
    python_type = client_utils.json_schema_to_python_type(info)
  File ".../gradio/routes.py", line 582, in api_info
    api_info = utils.safe_deepcopy(app.get_blocks().get_api_info())
```

**原因**:
`gradio_client/utils.py` の `_json_schema_to_python_type` が JSON Schema の boolean 値
（`additionalProperties: true` / `false`）を受け取った場合を未処理のままにしているバグ。
`additionalProperties: true` は「追加プロパティ自由」を意味する正当な JSON Schema 記法だが、
Gradio の変換ロジックは dict のみを想定している。
`show_api=False` は UI ボタンを隠すだけで `/info` ルートは残るため回避不可。

**注意 — 失敗したアプローチ**:
`gradio.routes.App.api_info` メソッドをパッチしても効果なし。
FastAPI はルート登録時に関数オブジェクトを参照コピーするため、クラス上のメソッドを後から
差し替えても登録済みルートハンドラには反映されない。

**対処法（確定）**:
クラッシュ箇所の上流 `gradio_client.utils._json_schema_to_python_type` を直接パッチする。

```python
try:
    import gradio_client.utils as _gc_utils

    _orig_inner = _gc_utils._json_schema_to_python_type

    def _patched_inner(schema, defs=None):
        # JSON Schema では boolean も有効な schema 値（true=何でも許可, false=拒否）
        # Gradio の変換関数はこれを未処理のためクラッシュする → "Any" を返して回避
        if isinstance(schema, bool):
            return "Any"
        return _orig_inner(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_inner
except Exception:
    pass  # Gradio バージョン差異による patch 失敗は無視
```

加えて:
- `demo.launch(..., show_api=False)` を指定する（UIボタン非表示）
- `gr.Radio` の choices は文字列値で運用し、ハンドラ側で `int(idx)` にキャスト（保険）

**備考**: `gradio_app_sam2_transparent_BG.py` のサードパーティ import 直後に配置。
Gradio が修正版をリリースした場合もパッチは `try/except` で保護されているため害なし。
参照ログ: `ログ_01.md`

---

### [ERR012] uv に `pip index` サブコマンドがない

| 項目 | 内容 |
|------|------|
| **深刻度** | Low |
| **頻度** | 一度のみ |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
```
error: unrecognized subcommand 'index'
```

**原因**:
`uv pip` は通常の `pip index versions ...` と同じサブコマンドを提供していない。

**対処法**:
パッケージのバージョン確認は `.venv\Scripts\python.exe -m pip index versions <package>` を使う。pip がない環境では先に `ensurepip` で復旧する。

**備考**:
Haystack 2.x の最新安定版確認時に発生。`haystack-ai==2.29.0` を採用。

---

### [ERR013] `.venv` に pip がなく `No module named pip` が発生

| 項目 | 内容 |
|------|------|
| **深刻度** | Low |
| **頻度** | 一度のみ |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
```
No module named pip
```

**原因**:
uv で作成した `.venv` に pip がインストールされていなかった。

**対処法**:
```powershell
.venv\Scripts\python.exe -m ensurepip --upgrade
.venv\Scripts\python.exe -m pip --version
```

**備考**:
pip 24.0 と setuptools 65.5.0 が `.venv` に導入され、以降 `pip index versions haystack-ai` が実行可能になった。

---

### [ERR014] Haystack Component import 時に `No module named 'torch'` が発生

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | Haystack unit test 環境で再現 |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
```
ModuleNotFoundError: No module named 'torch'
```
`tests/unit/test_common_components.py` / `tests/unit/test_pipeline_wiring.py` の collection 時に発生。

**原因**:
`pipelines/components/__init__.py` がモデル Component も import し、`model_components.py` のトップレベル import が `torch` / `torchvision` / `diffusers` を即時要求していた。Haystack の Pipeline 構築や純粋 Component テストでは checkpoint や外部モデルを初期化しない設計だが、import 時依存により unit test が重い ML 依存へ引きずられていた。

**対処法**:
`torch` / `torchvision` / `diffusers` は Component の `warm_up()` または `run()` 内で遅延 import する。`device` は import 時に `torch.device` を作らず文字列（`"cuda"` / `"cpu"`）で保持する。

**備考**:
`pipelines/components/model_components.py` 修正済み。`pytest -m "not integration" -v` は 11 passed, 2 deselected。

---

### [ERR015] Gradio 再導入後に `No module named 'gradio._simple_templates'` が発生

| 項目 | 内容 |
|------|------|
| **深刻度** | Low |
| **頻度** | 一度のみ |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
```
ModuleNotFoundError: No module named 'gradio._simple_templates'
```
`gradio_app_haystack.py --help` と `gradio_app_sam2_transparent_BG_haystack.py --help` の import 時に発生。

**原因**:
`.venv` の Gradio パッケージが不完全な状態でインストールされ、内部モジュール `gradio._simple_templates` が欠落していた。

**対処法**:
Gradio を再インストールする。
```powershell
.venv\Scripts\python.exe -m pip install --force-reinstall gradio==5.9.1
```
通常の不足だけであれば以下でも復旧する。
```powershell
.venv\Scripts\python.exe -m pip install gradio==5.9.1
```

**備考**:
復旧後、Haystack 版 Gradio entrypoint の `--help` smoke test は成功。

---

### [ERR016] SAM2 Haystack 版で Gradio `/info` が `TypeError: argument of type 'bool' is not iterable`

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | Gradio 5.9.x + Haystack 版 SAM2 アプリで再現 |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
```
TypeError: argument of type 'bool' is not iterable
  File ".../gradio_client/utils.py", line 887, in get_type
    if "const" in schema:
```
`Sam2_Transparent_Background_Haystack.ipynb` から `gradio_app_sam2_transparent_BG_haystack.py` を起動した後、Gradio の `/info` 生成時に ASGI 例外が繰り返し発生。

**原因**:
通常版 `gradio_app_sam2_transparent_BG.py` には ERR011 対策の bool schema patch が入っていたが、Haystack 版 `gradio_app_sam2_transparent_BG_haystack.py` には未適用だった。Gradio 5.9.x の `gradio_client.utils._json_schema_to_python_type` は JSON Schema の boolean schema（`additionalProperties: true/false`）を dict として扱い、`"const" in schema` で TypeError になる。

**対処法**:
Haystack 版 SAM2 Gradio アプリの `gradio` import 前に `gradio_client.utils._json_schema_to_python_type` を patch し、boolean schema の場合は `"Any"` を返す。`demo.launch(..., show_api=False)` は API 表示を隠す補助設定として併用する。

```python
import warnings

import gradio_client.utils as _gradio_client_utils

_original_json_schema_to_python_type = getattr(
    _gradio_client_utils,
    "_matting_anything_original_json_schema_to_python_type",
    _gradio_client_utils._json_schema_to_python_type,
)
_gradio_client_utils._matting_anything_original_json_schema_to_python_type = _original_json_schema_to_python_type

def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)

_gradio_client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
```

**備考**:
`gradio_app_sam2_transparent_BG_haystack.py` 修正済み。回帰テストとして `tests/unit/test_jupytext_notebooks.py` に bool schema patch の実行確認を追加。

---

### [ERR017] SAM2 bbox 座標を手入力 UI にすると端まで選択できない

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | SAM2 bbox prompt UI 設計時に再発しやすい |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
SAM2 Haystack 版 UI で prompt 座標を `Point X` / `Point Y` の `gr.Number` 手入力にしていた。bbox prompt を追加する場合も同じ手入力方式では、ユーザーが画像上で対象を直接選べず、画面端まで自然に選択できない。

**原因**:
被写体が画面内に完全に収まる前提で、座標値をフォーム入力させる UI になっていた。実画像では被写体が画面端や画面外へ続くことが多く、bbox を端まで伸ばす操作が必要になる。

**対処法**:
SAM2 の point / bbox prompt は画像上のマウス操作から生成する。bbox は 2 クリックで確定し、クリック順序に依存せず `[x_min, y_min, x_max, y_max]` に正規化する。さらに端付近のクリックは 0 / `width - 1` / `height - 1` に吸着させる。

**備考**:
`gradio_app_sam2_transparent_BG_haystack.py` で `Point X` / `Point Y` の `gr.Number` を削除し、`ImageEditor.select` + `select_sam2_prompt()` + `normalize_box_from_points()` に置き換えた。回帰テストとして edge snap と手入力 UI 不在を `tests/unit/test_jupytext_notebooks.py` に追加。

---

### [ERR018] Haystack Pipeline 実行結果に `transparent_bg` がなく `KeyError`

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | Haystack 2.x で中間 Component 出力を Gradio 側で読む場合に再発 |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
```
KeyError: 'transparent_bg'
    File "gradio_app_sam2_transparent_BG_haystack.py", line 108, in run_transparent_bg
        rgba = result["transparent_bg"]["rgba"]
```

**原因**:
Haystack Pipeline は既定では leaf 出力中心に結果を返すため、後続 Component に接続されている `transparent_bg` の中間出力が `result` に含まれない場合がある。

**対処法**:
Gradio callback で中間 Component の出力を参照する場合は、`Pipeline.run(..., include_outputs_from={"transparent_bg", "sam2_guard", "output_saver"})` を指定する。

**備考**:
`gradio_app_sam2_transparent_BG_haystack.py` 修正済み。回帰テストとして `tests/unit/test_jupytext_notebooks.py` に `include_outputs_from` の存在確認を追加。

---

### [ERR019] SAM2 bbox を画像端まで届かせる UI 補助・positive/negative 明示化

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | SAM2 bbox / point prompt UI 設計時に再発しやすい |
| **初回発生日** | 2026-05-25 |

**エラー内容**:
ERR017 で導入したマウスクリック式 bbox は、Gradio のクリックイベントが画像の外側で発火しないため、被写体が画面端に接している場合に bbox を画面端まで正確に届かせるのが難しかった。また point prompt の正負を `gr.Checkbox("Positive Point")` で表現していたため、ユーザーから「positive / negative の選択 UI が見当たらない」と認識されやすかった。

**原因**:
1. `EDGE_SNAP_PIXELS = 8` ではエッジ吸着範囲が狭く、ユーザーが意識的に画面端付近をクリックする必要があった。
2. 画像外をドラッグして bbox を作る UX は Gradio の `Image.select` では実現できない。
3. `gr.Checkbox` は二値だが「positive 以外＝negative」が UI 上で暗黙的だった。
4. `gr.ImageEditor` は sketch ツールが前面に出るため、シンプルなクリック取得用途では `gr.Image(interactive=True)` のほうが意図が伝わりやすい。

**対処法**:
1. `EDGE_SNAP_PIXELS` を 16 に拡大し、より広い範囲のクリックを画像端へ吸着。
2. `extend_box_to_edge(input_image, prompt_state, side)` を追加し、確定済み bbox の left / right / top / bottom 辺をそれぞれ 0 / `width - 1` / `height - 1` まで延長する 4 ボタン (`Extend Left/Right/Top/Bottom`) を UI に配置。bbox 未確定時は `gr.Error` を送出。
3. point の正負を `gr.Radio(["positive", "negative"], value="positive", label="Point Label")` で明示化。`select_sam2_prompt` は文字列 / bool 両対応で後方互換維持。
4. SAM2 prompt 入力を `gr.ImageEditor` から `gr.Image(type="numpy", interactive=True)` に変更し、クリック UX を単純化。

**備考**:
`gradio_app_sam2_transparent_BG_haystack.py` を更新。回帰テストとして `tests/unit/test_jupytext_notebooks.py` に以下を追加:
- `test_sam2_haystack_extend_box_to_edge_modifies_each_side`（4 方向延長）
- `test_sam2_haystack_extend_box_to_edge_requires_existing_box`（bbox 未確定時の `gr.Error`）
- `test_sam2_haystack_app_uses_positive_negative_radio`
- `test_sam2_haystack_app_uses_image_for_prompt_input`
- `test_sam2_haystack_app_has_edge_extend_buttons`

---

### [ERR020] Skill 診断で推奨項目が「使わない場面」に見える

| 項目 | 内容 |
|------|------|
| **深刻度** | Low |
| **頻度** | Skill / instruction 文面更新時に再発しやすい |
| **初回発生日** | 2026-05-26 |

**エラー内容**:
Chat Customizations Evaluations analyzer が `.github/skills/haystack-pipeline/SKILL.md` に対し、推奨項目を含む表が「使わない場面」見出し配下にあること、`1 副作用境界` やサブエージェントレビュー手順が曖昧であること、device / model 共有方針が不足していることを診断した。

**原因**:
適用範囲の表が「禁止・非推奨」だけでなく「推奨」も含む比較表だったが、見出しが否定形だったため意図が反転して読めた。また Component 粒度、I/O 配置、レビュー完了条件が短い抽象語に寄っていた。

**対処法**:
見出しを「推奨する場面」「領域別の適用判断」に変更し、I/O は専用 Component の `run()` に置くこと、device は `__init__(device: str)` で受けて `warm_up()` で `.to(device)` すること、同一モデル共有は依存注入すること、レビュー観点と完了条件を明文化した。

**備考**:
診断取得ツールは修正後も古い診断文を返す場合があるため、実ファイル上で旧見出し・旧表現が消えていることを確認する。

---

### [ERR021] SAM2 prompt 指定がアップロード欄に埋もれて行方不明になる

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | SAM2 prompt UI を 1 つの画像入力に集約した場合に再発しやすい |
| **初回発生日** | 2026-05-26 |

**エラー内容**:
SAM2 Haystack 版 UI で、画像アップロード欄と point / bbox 指定欄が同じ `gr.Image` にまとまっていたため、ユーザーが「どこにポイントを打つのか」「bbox をどのウィンドウで指定するのか」を見失いやすかった。

**原因**:
アップロードというファイル入力の役割と、マスク編集という直接操作の役割を同じコンポーネントに持たせていた。SAM2 prompt のクリック対象が UI 上で独立しておらず、マスク編集モードの存在が視覚的に弱かった。

**対処法**:
アップロード用 `Input Image` と、prompt 編集専用 `SAM2 Prompt Canvas` を分離する。`input_image.change(sync_prompt_canvas, ...)` で入力画像をキャンバスへコピーし、クリックイベントは `prompt_canvas.select(...)` に紐づける。推論本体には原本の `input_image` を渡し、prompt canvas は点・bbox・mask overlay の表示に専念させる。`SAM2 Prompt Canvas` は `sources=[]` と独自 placeholder で、ドラッグ＆ドロップ先に見えないようにする。

**追加対処**:
予測画像が原寸で大きく表示されると操作面が流れるため、`Image Display Size` を追加する。既定は `window` の固定高さ表示、必要時のみ `original` で原寸表示に切り替える。

**備考**:
`gradio_app_sam2_transparent_BG_haystack.py` と `tests/unit/test_jupytext_notebooks.py` を更新。回帰テストで `input_image.select(` が存在せず、`prompt_canvas.select(` が存在することと、画像同期時に prompt state / `SAM2_STATE` がリセットされることを確認する。

---

### [ERR022] Haystack Pipeline 接続で `dict` と `dict[str, Any] | None` が不一致になる

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | Haystack 2.x Component の dict 契約を接続する時に再発しやすい |
| **初回発生日** | 2026-05-26 |

**エラー内容**:
```
PipelineConnectError: Cannot connect 'sam2_segmenter.mask_set' with 'mask_preview.mask_set':
their declared input and output types do not match.
'sam2_segmenter':
 - mask_set: dict
'mask_preview':
 - mask_set: dict[str, Any] | None
```

**原因**:
Haystack の Pipeline 接続検証は `dict` と `dict[str, Any]`、または `dict | None` を同一型として扱わない。標準 I/O 契約を Python の詳細型ヒントで書くと、Component 間の実データは同じ dict でも Pipeline builder が接続時に失敗する。

**対処法**:
Haystack の接続対象になる `MaskSet` / `SelectedMask` の入出力 socket は、`@component.output_types(mask_set=dict)` と `run(..., mask_set: dict, ...)` のように単純な `dict` 型に揃える。詳細仕様は docstring / tests / `REFERENCE.md` に記録し、Pipeline socket 型には持ち込まない。

**備考**:
`MaskCandidateSelector`, `MaskUnion`, `MaskPreviewComposer`, `SAM2Segmenter` の `mask_set` 接続を `dict` に統一して解消。回帰テストとして `tests/unit/test_pipeline_wiring.py` に `build_sam2_maskset_pipeline()` / `build_mask_union_pipeline()` / `build_sam2_union_tb_pipeline()` の builder smoke を追加。

**追加事例（2026-05-27）**:
SAM2 動画版 Pipeline 追加時にも `VideoReader.frames: list` と `SAM2VideoPropagator.frames: list[np.ndarray]`、および `VideoReader.metadata: dict` と `metadata: dict[str, Any] | None` の接続で同じ `PipelineConnectError` が再発した。動画版でも接続される socket は `frames: list`, `metadata: dict`, `masks: dict`, `matte: dict` に統一し、詳細契約は `REFERENCE.md` と unit test 側で固定する。

**追加回帰テスト**:
`tests/unit/test_video_pipeline_wiring.py` で `build_video_reader_pipeline()` / `build_sam2_video_propagation_pipeline()` / `build_sam2_tb_video_pipeline()` が接続エラーなく構築できることを確認する。

---

### [ERR023] SAM2 Haystack Notebook の Text Prompt 検出で `No module named 'supervision'`

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | Colab で GroundingDINO Text Prompt 検出を初回実行した時 |
| **初回発生日** | 2026-05-26 |

**エラー内容**:
```
ModuleNotFoundError: No module named 'supervision'
  File ".../GroundingDINO/groundingdino/util/inference.py", line 5, in <module>
    import supervision as sv
gradio.exceptions.Error: "Text Prompt 検出に失敗しました: No module named 'supervision'"
```

**原因**:
`Sam2_Transparent_Background_Haystack.py` の Colab install cell が SAM2 / transparent-background / Gradio 依存だけを入れており、Text Prompt 検出で使う GroundingDINO の runtime 依存 (`supervision`, `addict`, `yapf`, `timm`, `pycocotools`, `transformers`) をインストールしていなかった。`requirements.txt` には `supervision` があったが、Notebook は `requirements.txt` を使わず個別 `pip install` していたため Colab に反映されなかった。

**対処法**:
Notebook の install cell に以下を追加する。`bertwarper.py` は新しい `transformers` の `get_extended_attention_mask` signature に合わせてあるため、`transformers>=4.26.0` を明示する。
```python
!{sys.executable} -m pip install "transformers>=4.26.0" addict yapf timm supervision pycocotools
```
あわせて `checkpoints/groundingdino_swint_ogc.pth` を自動ダウンロードし、`requirements.txt` に不足していた `timm` を追加する。

**備考**:
`Sam2_Transparent_Background_Haystack.py` を修正し、Jupytext で `.ipynb` を再生成済み。回帰テストとして `tests/unit/test_jupytext_notebooks.py::test_sam2_haystack_notebook_installs_groundingdino_runtime_dependencies` を追加。

---

### [ERR024] GroundingDINO 初期化で `'BertModel' object has no attribute 'get_head_mask'`

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 新しい `transformers` で GroundingDINO Text Prompt を初回実行した時 |
| **初回発生日** | 2026-05-26 |

**エラー内容**:
```
AttributeError: 'BertModel' object has no attribute 'get_head_mask'
  File ".../GroundingDINO/groundingdino/models/GroundingDINO/bertwarper.py", line 29, in __init__
    self.get_head_mask = bert_model.get_head_mask
gradio.exceptions.Error: "Text Prompt 検出に失敗しました: 'BertModel' object has no attribute 'get_head_mask'"
```

**原因**:
GroundingDINO の `BertModelWarper` は旧 `transformers` の `BertModel.get_head_mask` を前提にしている。Colab に入る新しい `transformers` ではこの helper が削除されており、GroundingDINO model build 時に `AttributeError` になった。ERR005 で `get_extended_attention_mask` は新シグネチャに合わせていたが、別の BERT helper 互換性が残っていた。

**対処法**:
Haystack の GroundingDINO Component 初期化前に `patch_transformers_bert_for_groundingdino()` を呼び、`BertModel.get_head_mask` が存在しない場合だけ互換実装を追加する。`GroundingDINODetector` と `GroundingDINOMultiBoxDetector` の両方の `warm_up()` で GroundingDINO import 前に実行する。

**再発防止**:
- Text Prompt 系 Component を追加する時は、GroundingDINO import 前に BERT 互換パッチを呼ぶ。
- 既存 `gradio_app.py` 側の互換パッチも、`head_mask is None` では `[None] * num_hidden_layers` を返し、`is_attention_chunked` と `self.dtype` 変換を反映する。
- Colab で ERR023 の依存を入れ直した後は、起動済み Gradio プロセスではなく Notebook の app 起動セルを再実行する。
- `tests/unit/test_pipeline_wiring.py::test_groundingdino_transformers_bert_compat_patch_is_called_before_model_import` で互換パッチの存在と呼び出しを確認する。

**備考**:
既知の `UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!` と `FutureWarning: Importing from timm.models.layers is deprecated` は今回の停止原因ではない。

---

### [ERR025] GPU 必須推論が Colab CPU / CPU-only torch で fail fast

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | GPU ランタイム未選択、または CPU-only PyTorch が入った状態で SAM2 / GroundingDINO を実行した時 |
| **初回発生日** | 2026-05-28 |

**エラー内容**:
```
RuntimeError: GroundingDINOMultiBoxDetector requires a CUDA GPU for production inference.
CPU execution is reserved for emergency fallback only and is disabled by default.
selected_device=cpu cuda_available=False torch_available=True torch_cuda_version=None
```

**原因**:
映像制作向け運用では SAM2 / GroundingDINO / SAM2 video / MAM の重い推論は GPU 前提であり、CPU fallback は緊急回避専用として既定で禁止している。Colab 側で GPU ランタイムが選択されていない、または `torch.version.cuda=None` の CPU-only PyTorch が入っていると、Gradio 起動後の `Detect Text Boxes` / `Predict SAM2 Candidate Masks` 実行時に GPU 必須ガードで停止する。

**確認結果**:
2026-05-28 にユーザー確認により、該当ログの Colab ランタイムは CPU だったと判明。GPU first fail fast は意図通り動作していた。

**対処法**:
1. Colab の「ランタイム > ランタイムのタイプを変更」で T4 GPU 以上を選択する。
2. ランタイムを再起動し、install cell から実行し直す。
3. Notebook 診断セルで `nvidia-smi`, `torch.cuda.is_available() == True`, `torch.version.cuda != None` を確認してから Gradio を起動する。
4. CPU での超低速実行を意図する緊急回避時だけ `MATTING_ANYTHING_ALLOW_CPU=1` を設定する。

**再発防止**:
- Gradio 起動前の Notebook 診断セルで CUDA 不可なら即停止し、公開 Gradio URL を出さない。
- `torch_cuda_version=None` は CPU-only torch のサインとして扱い、正常運用に進めない。
- `MATTING_ANYTHING_ALLOW_CPU=1` は本番・映像制作導線では使わず、検証・緊急回避に限定する。
- GroundingDINO Text Prompt では ERR004 / ERR006 / ERR023 / ERR024 / ERR005 も同時に確認する。
- `tests/unit/test_jupytext_notebooks.py` で Notebook の GPU 起動前 preflight 文言を固定する。

---

### [ERR026] SAM2 positive point クリックで Gradio が `Connection errored out`

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 静止画 Haystack 版で point prompt をクリックした時 |
| **初回発生日** | 2026-05-28 |

**エラー内容**:
```
Error
Connection errored out.
```
ブラウザ側では汎用接続エラーだけが表示され、positive point 選択時に SAM2 prompt 操作が続行できない。

**原因**:
静止画 Haystack 版の `SAM2 Prompt Canvas` は `prompt_canvas.select(...)` でクリックイベントを受ける設計なのに、`gr.Image(... interactive=False)` になっていた。プロジェクト仕様では SAM2 prompt 入力は `gr.Image(type="numpy", interactive=True)` を使う必要があり、動画版は `interactive=True` だったため静止画版だけ仕様から外れていた。

**対処法**:
`gradio_app_sam2_transparent_BG_haystack.py` の `SAM2 Prompt Canvas` を `interactive=True` にする。アップロード先化は避けるため `sources=[]` は維持する。

**再発防止**:
- SAM2 prompt canvas は `sources=[]` と `interactive=True` をセットで使う。
- `input_image.select(...)` ではなく `prompt_canvas.select(...)` に click handler を結線する。
- positive / negative は `gr.Radio(["positive", "negative"])` で受け、helper 側で label `1` / `0` に変換する。
- `tests/unit/test_jupytext_notebooks.py` で prompt canvas block 自体が `interactive=True` かつ `interactive=False` を含まないことを確認する。
- `Connection errored out` はブラウザ側の汎用表示なので、再発時は Gradio 起動セル / サーバー stdout の traceback を一次情報として確認する。

---

### [ERR027] Colab Gradio share link 用 frpc 欠落で 127.0.0.1 しか表示されない

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | Colab 上で Gradio share 用 frpc バイナリの自動取得に失敗した時 |
| **初回発生日** | 2026-05-29 |

**エラー内容**:
```
* Running on local URL:  http://127.0.0.1:7861

Could not create share link. Missing file:
/usr/local/lib/python3.12/dist-packages/gradio/frpc_linux_amd64_v0.3
```
ブラウザで `http://127.0.0.1:7861` を開くと `ERR_CONNECTION_REFUSED` になる。

**原因**:
Colab の `127.0.0.1` は Colab VM 内部を指すため、手元ブラウザから直接アクセスできない。Colab では Gradio の public share URL を使う必要があるが、share tunnel 用の `frpc_linux_amd64_v0.3` が Gradio package 配下に存在せず、ネットワーク制限や一時的な取得失敗により public URL が生成されていなかった。

**対処法**:
Haystack 版 Colab Notebook の Gradio 起動セルでは、Colab 判定を `google.colab` の import spec で行い、Colab では必ず `--share` を渡す。frpc 取得や checksum 検証を Notebook 側で過剰に先取りせず、Gradio 5 の既定の share link 生成に任せる。Notebook 出力には、`Running on public URL: https://...gradio.live` を開き、local `127.0.0.1` URL は開かないことを明示する。

**再発防止**:
- Colab では `Running on local URL` ではなく `Running on public URL` の有無を成功判定にする。
- Notebook の Gradio 起動セルは `google.colab` の import spec で Colab 判定し、Colab では `!python app.py --share` を実行する。
- `frpc` の手動取得・checksum 検証を Notebook 側で先取りして Gradio 起動前に止めない。
- public URL が出ない状態で `127.0.0.1` を開くよう案内しない。
- `Connection errored out` や `ERR_CONNECTION_REFUSED` はブラウザ表示だけで判断せず、Colab stdout の share link / traceback ログを一次情報にする。

---

### [ERR028] VideoWriter.warm_up が Haystack の no-arg warm_up 契約に反する

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 動画版 Haystack Pipeline warm_up 時 |
| **初回発生日** | 2026-05-29 |

**エラー内容**:
```
動画処理に失敗しました: VideoWriter.warm_up() missing 1 required positional argument: 'frame_shape'
```

**原因**:
Haystack は Component の `warm_up()` を実行時入力なしで呼ぶ。一方で動画 writer は RGBA codec の可用性確認に frame shape が必要なため、`warm_up(self, frame_shape, preferred_rgba_codec=...)` として実装していた。これは Component lifecycle と runtime 入力依存処理を混在させた設計で、Pipeline 側から no-arg warm_up された時に `TypeError` になる。

同時に動画版 UI は静止画版にあった Text Prompt / GroundingDINO から bbox 候補を作る導線を持たず、複合対象（`person playing drums`, `person riding bicycle`）を第 1 フレームで意味的に指定する実験目的から外れていた。

**対処法**:
- `VideoWriter.warm_up()` は Haystack 契約通り no-arg / no-op に戻す。
- frame shape が必要な codec 選択は `_select_rgba_codec(frame_shape, preferred_rgba_codec)` に分離し、`run()` 内で RGBA frame が確定してから実行する。
- 動画版 UI に任意の `Text Prompt to Box (GroundingDINO)` accordion を追加し、第 1 フレーム検出の top bbox を SAM2 video prompt state にコピーする。
- Movie Notebook で GroundingDINO checkpoint を取得し、`GROUNDING_DINO_CKPT_PATH` を Gradio 実行プロセスへ渡す。

**再発防止**:
- Haystack Component の `warm_up()` は runtime 入力に依存させない。入力 shape / fps / codec など実行時にしか分からない値は `run()` で扱う。
- 静止画版 / 動画版の SAM2 Haystack UI を変更する時は、Text Prompt / GroundingDINO 導線が両方に残っているか確認する。
- 複合対象選択の要件は `REFERENCE.md` と `.github/copilot-instructions.md` に明記し、動画版でも `person playing drums` / `person riding bicycle` を UI placeholder またはテストで固定する。

---

### [ERR029] 動画版 Pipeline が 5% 表示のまま長時間進み、失敗 stage が分からない

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | 動画版 Haystack Pipeline の初回実行 / 長尺動画 |
| **初回発生日** | 2026-05-29 |

**エラー内容**:
```
動画背景除去を実行後、UI は Pipeline 5% 付近のまま約10分待機し、その後エラーで停止する。
```

`エラーログ/エラーログ_09.md` では final traceback が欠けていたが、ログには `frame loading (JPEG): 240/240 [00:43]`、`propagate in video: 240/240 [04:29]`、`Settings -> Mode=base, Device=cuda` が出ており、SAM2 video propagation までは進んでいた。

**原因**:
Gradio callback が `progress(0.05, desc="Pipeline を起動しています")` を表示した後、end-to-end Haystack Pipeline 完了まで次の進捗を出していなかった。動画読込、SAM2 一時 JPEG 化、SAM2 video propagation、transparent-background frame 処理、動画/連番書き出しが直列で走るため、長時間の初回実行が「5%で止まった」ように見えた。加えて、例外発生時にどの stage で落ちたかを UI エラーへ含めていなかった。

**対処法**:
- end-to-end Pipeline は維持し、`VideoReader` / `SAM2VideoPropagator` / `TransparentBGVideoExtractor` / `VideoWriter` / `FrameSequenceWriter` に任意の `progress_callback` 入力を追加する。
- Gradio 側で Component 内部進捗を全体進捗へマッピングし、動画読込、SAM2 伝搬、transparent-background、書き出しの stage と frame 数を表示する。
- 例外時は `stage=<最後に報告された処理>` と elapsed 秒を `gr.Error` に含め、final traceback が欠けても切り分け可能にする。
- 初回 UX の既定を短尺クイックプレビューへ変更し、長尺/全 frame は Advanced で明示的に増やす導線にする。

**再発防止**:
- 5分超が見込まれる Component は、Gradio callback の固定 progress だけでなく Component 内部 progress を返す。
- 動画読込を重複させる stage 分割は避け、1 回の `VideoReader` 出力を downstream に接続する end-to-end Pipeline を維持する。
- エラーログには final traceback と `stage=` 付き Gradio error を必ず残す。
- 初回確認用 default は短尺にし、UI に処理 frame 数・各パラメーターの意味・品質/速度トレードオフを明記する。

---

### [ERR030] 動画版 transparent-background が出力 frame を全保持し Colab RAM を使い切る

| 項目 | 内容 |
|------|------|
| **深刻度** | Critical |
| **頻度** | 高解像度動画 / `both` 出力 / 長尺処理で発生 |
| **初回発生日** | 2026-05-29 |

**エラー内容**:
```
使用可能な RAM をすべて使用した後でセッションがクラッシュ。
...
propagate in video: 100% 60/60
Settings -> Mode=base, Device=cuda, Torchscript=disabled
```

`エラーログ/エラーログ_10.md` では Python traceback が残る前に Colab runtime が kill されている。SAM2 の 60/60 伝搬完了後、transparent-background 初期化ログまで進んでいるため、SAM2 ではなく後段の frame matting / output retention が主要因。

**原因**:
動画版 `TransparentBGVideoExtractor` が `rgba_frames`, `alpha_frames`, `preview_frames` を全 frame 分 list に保持し、その後 `VideoWriter` / `FrameSequenceWriter` に渡していた。入力 RGB frame list と SAM2 mask list に加えて、RGBA(4ch)・alpha(1ch)・preview(3ch) を全保持するため、特に高解像度・長尺・`both` 出力で peak RAM が急増する。Colab では OS がプロセスを kill するため、Gradio 側に `gr.Error` や traceback が出ないことがある。

**対処法**:
- `TransparentBGVideoExtractor` を streaming 出力へ変更し、frame ごとに transparent-background 結果を動画/PNG へ即時保存する。
- `matte` dict は `rgba_frames` / `alpha_frames` / `preview_frames` を空 list にし、保存済み path と metadata だけを下流へ渡す compact contract にする。
- 既存 `VideoWriter` / `FrameSequenceWriter` は保存済み path/dir を持つ compact matte を pass-through できるようにする。
- Gradio callback は `include_outputs_from` で `video_reader` / `sam2_video_propagator` / `transparent_bg_video` の巨大中間出力を返さず、writer の compact 結果だけを読む。
- Text Prompt 使用後に GroundingDINO/BERT cache が残る副因を避けるため、動画実行直前に `release_text_detector()` で semantic detector を解放する。
- 初回既定を `max_frames=30` に下げ、まず短尺で prompt と品質を確認してから長尺へ増やす。

**再発防止**:
- 動画 pipeline で RGB / mask / RGBA / alpha / preview の全 frame list を同時保持しない。
- Haystack の中間出力を Gradio callback に返す場合は、返却 dict に numpy frame list が含まれないか確認する。
- `output_mode=both` は動画と連番の二重書き出しになるため、初回確認では `video` か `sequence` の片方を推奨する。
- Text Prompt 後に動画処理へ進む場合は、GroundingDINO/BERT cache を解放してから SAM2 / transparent-background を走らせる。
- 高解像度・長尺処理で runtime kill が起きた場合は、最終 traceback がないこと自体を OOM の兆候として扱い、最後に出た stage ログから peak RAM 箇所を切り分ける。

---

### [ERR031] 動画版 SAM2 Prompt Canvas が常に空白で何も映らない

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 動画版 UI で常に発生 |
| **初回発生日** | 2026-06-06 |

**エラー内容**:
```
ユーザー報告: 「sam2 prompt キャンバス何も映らない」
```
動画版 `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` の SAM2 Prompt Canvas にプレースホルダーもフレームも表示されず、クリックで bbox / point を打てない。

**原因**:
`prompt_canvas = gr.Image(..., type="numpy", interactive=True)` に `sources=[]` が抜けていた。`sources` 未指定の `gr.Image` は既定でアップロードソース（upload/clipboard 等）を持つアップロード UI として描画され、`value=create_prompt_canvas_placeholder()` で渡したプレースホルダー画像がアップロードドロップゾーンに上書きされて表示されない。静止画版 `gradio_app_sam2_transparent_BG_haystack.py` は `sources=[]` を持っており正しく表示されていたが、動画版へのコピー時に欠落した（ERR026 / ERR021 と同根のリグレッション）。

**対処法**:
- `prompt_canvas` に `sources=[]`（アップロード無効・クリック専用）を付与。併せて `show_download_button=False`, `show_fullscreen_button=False` を付け、静止画版と挙動を揃えた。
- アップロードは別の `Input Video`（`gr.Video(sources=["upload"])`）に分離済みなので、Prompt Canvas は表示+クリック専用で十分。

**再発防止**:
- SAM2 Prompt Canvas（静止画 / 動画とも）は `gr.Image(type="numpy", sources=[], interactive=True)` を固定パターンとする（copilot-instructions ERR021 / ERR026）。
- UI 変更後は必ず Gradio 実起動 + Playwright でプレースホルダー表示とクリック可否を目視確認する。
- 静止画版から動画版へ UI をコピーするときは `sources=[]` の欠落を最初にチェックする。

---

### [ERR032] 動画版のモデル変更プルダウンが Advanced アコーディオン内に埋もれて見えない

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | 動画版 UI で常に発生 |
| **初回発生日** | 2026-06-06 |

**エラー内容**:
```
ユーザー報告: 「モデル変更プルダウンメニューがない」「samuraiの機能チェックもしたかった」
```
tracker / background のモデル選択 Dropdown は実装されていたが、`Advanced: 動画処理設定`（`open=False`）アコーディオン内に置かれていたため、デフォルト折りたたみ状態では見えず「存在しない」と認識された。結果 SAMURAI への切替（`samurai_hiera_*`）も試せなかった。

**原因**:
`tracker_model` / `background_model` の `gr.Dropdown` 定義が Advanced アコーディオンの `gr.Row` 内にあり、`tb_jit` / `tb_threshold` / `crop_padding` などの詳細パラメータと同居していた。モデル選択は基本操作なのに詳細設定扱いになっていた。

**対処法**:
- `tracker_model` Dropdown を `## 3. SAM系` セクション直下（可視）へ移動。
- `background_model` Dropdown を `## 4. 背景透過系` セクション直下・実行ボタンの前（可視）へ移動。
- Advanced アコーディオンには `tb_jit` / `tb_threshold` / `crop_padding` のみ残した。
- 選択肢は `build_dropdown_choices("tracker"|"background")` で生成し、`INFERENCE_TRACKER_VARIANT` による可視フィルタ（SAM2 / SAMURAI 切替）を維持。`info=` に SAMURAI 利用条件（環境変数 + samurai パッケージ）を明記。

**再発防止**:
- モデル選択など「基本操作」の UI は折りたたみアコーディオンに入れず、対応セクション（SAM系 / 背景透過系）の可視領域に置く。
- アコーディオンは速度/品質の微調整パラメータ専用とする。
- UI 変更後は Playwright で折りたたみ初期状態のままドロップダウンが見えることを確認する。

---

### [ERR033] 動画版「表示中フレームを再取得」がシーク位置を無視し常に第1フレームを取得

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 動画版でシーク後に再取得する度に発生 |
| **初回発生日** | 2026-06-06 |

**エラー内容**:
```
ユーザー報告: 「表示中のフレームを再取得」と「シーク位置をSAM2に反映」「この二つが連動しない」
```
動画プレイヤーを任意位置へシークしてから「表示中フレームを再取得」ボタンを押しても、起点フレーム位置（`prompt_frame_idx`）が反映されず常にフレーム 0 が Prompt Canvas に出る。シーク連動の `prompt_frame_idx` と再取得ボタンが分離していた。

**原因**:
`load_first_frame_btn.click(...)` が `extract_first_frame, inputs=[input_video], outputs=[..., prompt_frame_idx]` に配線されていた。`extract_first_frame` は常にフレーム 0 を抽出し、さらに出力で `prompt_frame_idx` を 0 に上書きしていたため、シークで更新された起点フレーム位置が破棄されていた。シーク連動用の正しいハンドラ `extract_prompt_frame(input_video, prompt_frame_idx, frame_step)` は別ボタン（`show_frame_btn`）と `prompt_frame_idx.change` にのみ配線されていた。

**対処法**:
- `load_first_frame_btn.click` を `extract_prompt_frame, inputs=[input_video, prompt_frame_idx, frame_step], outputs=[prompt_canvas, prompt_state, prompt_status]` へ再配線。`prompt_frame_idx` は入力として読むだけにし、出力で上書きしない。
- これによりシーク → `prompt_frame_idx` 自動更新（`build_video_seek_sync_js`）→ 再取得ボタン or `prompt_frame_idx.change` のいずれでも同じ起点フレームが Canvas に出るよう統一。
- 初回アップロード時の `input_video.change(extract_first_frame_outputs, ...)` は第1フレーム自動表示の用途なので据え置き。

**再発防止**:
- 「シーク連動」と銘打つ UI 要素は、フレーム取得系の全ボタンで同一の `extract_prompt_frame(video, prompt_frame_idx, frame_step)` を共有する。
- フレーム取得ハンドラは `prompt_frame_idx` を入力として読み、出力で上書きしない（初回自動表示を除く）。
- UI 配線変更後は Playwright でシーク → 再取得が同じフレームを返すフローを確認する。

---

### [ERR034] 動画版 Pipeline が tracker 選択を無視し常に既定 SAM2 を構築

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 動画版で tracker / SAMURAI を切替えても常に発生 |
| **初回発生日** | 2026-06-06 |

**エラー内容**:
```
tracker_model ドロップダウンで samurai_hiera_l を選んでも SAM2 標準が動く（切替が効かない）。
```
ERR032 で Dropdown を可視化しても、選択した tracker モデルが実際の推論に反映されなかった。

**原因**:
`get_video_pipeline(tracker_model, background_model)` が `(tracker_model, background_model)` でキャッシュしていたが、内部で `build_sam2_tb_video_pipeline()` を引数なしで呼び、`SAM2VideoPropagator()` を既定 config_name / checkpoint_path で構築していた。registry（`config/inference_models.toml`）の `config_name` / `checkpoint_path` が伝搬されず、SAMURAI config（`configs/samurai/...`）への切替が無効だった。

**対処法**:
- `build_sam2_tb_video_pipeline(propagator: SAM2VideoPropagator | None = None)` に変更し、注入された propagator を `add_component("sam2_video_propagator", propagator or SAM2VideoPropagator())` で使用（疎結合・依存注入）。
- `get_video_pipeline` で `entry_by_id("tracker", tracker_model)` から `config_name` / `checkpoint_path` を解決し、`SAM2VideoPropagator(checkpoint_path=..., config_name=...)` を構築して `build_sam2_tb_video_pipeline(propagator=...)` に渡す。
- checkpoint の相対パスは `_resolve_project_path`（`PROJECT_ROOT` 環境変数 or ファイル基準）で絶対化。
- SAMURAI 選択時は `warm_up()` で samurai パッケージ / config が無ければ fail fast（許容挙動）。`tracker_metadata()` で `tracker_config` / `tracker_checkpoint` / `samurai_mode` を masks metadata に残す。

**再発防止**:
- UI のモデル選択は registry（TOML）→ Component 構築引数まで一気通貫で伝搬されているか確認する。Dropdown を出すだけでは推論に反映されない。
- Pipeline ビルダーは重い Component を依存注入で受け取れる形にし（YAGNI を守りつつ差し替え可能に）、`pipeline.get_component(name) is injected` をテストで固定する。
- tracker 切替の痕跡（config / checkpoint / samurai_mode）を必ず metadata に記録する（copilot-instructions の samurai 切替ルール）。

---

### [ERR035] 動画版 シーク連動 JS ブリッジが Gradio 5/Svelte で実行時に機能せず3コントロールが無反応

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 動画版でシーク連動を使う度に常に発生（実行時） |
| **初回発生日** | 2026-06-06 |

**エラー内容**:
```
ユーザー報告: 「プロンプト起点フレーム位置（シーク連動）機能せず / 表示中フレーム再取得 機能せず / シーク位置をsam2に反映 機能せず」
```
ERR033 ではソーステキスト上の配線を修正し「シーク連動が直った」と記録したが、実行時には依然として3つのコントロール（シーク連動スライダー自動更新、「表示中フレームを再取得」ボタン、「シーク位置を SAM2 に反映」ボタン）がすべて無反応だった。ERR033 の「fixed」記述は実行時検証を伴わない誤りだった。

**原因**:
`build_video_seek_sync_js`（`gr.Blocks(js=...)` で注入）が動画要素の `seeked` / `pause` イベントを拾い、`prompt_frame_idx` スライダーの DOM `input.value` を書き換えて native `input` / `change` イベントを dispatch していた。Gradio 5（Svelte）では、コンポーネント内部 state は Svelte が管理しており、DOM へ直接 `value=` を代入し native イベントを発火させてもバックエンドの `.change` は発火しない。結果、`prompt_frame_idx` に依存する3コントロールすべてがシーク位置に追従しなかった。さらに2ボタンは同一の `extract_prompt_frame` を呼ぶ完全な冗長 UI だった。

**対処法（Option A: スライダー1本へ集約）**:
- 不安定な JS ブリッジ（`VIDEO_SEEK_SYNC_JS` 定数 / `build_video_seek_sync_js()` 関数 / `gr.Blocks(js=...)` 引数）を削除。
- 冗長な2ボタン（`load_first_frame_btn` =「表示中フレームを再取得」/ `show_frame_btn` =「シーク位置を SAM2 に反映」）と未使用 hidden `video_fps`（`elem_id="movie-video-fps"`）、およびそれらへ供給する fps 返却値を削除。
- ネイティブに動作する `prompt_frame_idx.change(extract_prompt_frame, ...)` 1本へ集約。スライダーをドラッグすると Gradio ネイティブの `.change` が確実に発火し Canvas が更新される。スライダー label を「プロンプト起点フレーム位置（ドラッグで Canvas 更新）」へ変更し、操作の唯一性を明示。
- `extract_first_frame` / `extract_first_frame_outputs` を 4-tuple（fps を除外）へ、`input_video.change` の outputs から `video_fps` を除外。

**再発防止**:
- Gradio/Svelte コンポーネントへ DOM 直接書き換え + native イベント dispatch で値を流し込む JS ブリッジは実行時に機能しない前提とし、採用しない。値の連動は Gradio ネイティブのイベント（`.change` / `.select` 等）で構成する。
- ソーステキスト一致のテストは実行時挙動を保証しない。UI 配線変更は Playwright で実起動し、実際のユーザー操作（スライダードラッグ → Canvas 更新）が反応することを確認してから「fixed」と記録する。
- 同一ハンドラを呼ぶだけの冗長ボタンは増やさず、単一の操作元（single source of truth）へ集約する（ui-ux-pro-max: 冗長コントロール削減）。

### [ERR036] 動画版 候補bbox選択肢生成が gr.Dataframe の pandas DataFrame 真偽評価で失敗

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | テキストプロンプト→検出ボタン押下の度に常に発生（実行時） |
| **初回発生日** | 2026-06-06 |
| **関連ファイル** | `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` |

**エラー内容**:
```
ユーザー報告: 「複合対象に使う候補 bbox を選択（union 用）テキストプロンプトを入れ 検出ボタンを押すとエラー」
ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```
`detect_text_btn.click(...).then(populate_candidate_choices, inputs=[detected_boxes], ...)` 経由で `populate_candidate_choices` が呼ばれた際に送出。後段の `apply_selected_boxes` の「少なくとも 1 つの候補 bbox を選択してください」もこの連鎖失敗（候補が空のまま）による派生だった。

**原因**:
`populate_candidate_choices` が `rows = list(detected_rows or [])` で入力を扱っていた。Gradio 5 の `gr.Dataframe`（既定 `type="pandas"`）はハンドラへ値を **pandas DataFrame** で渡すため、`detected_rows or []` の真偽評価が `ValueError: truth value of a DataFrame is ambiguous` を送出した。

**対処法**:
- `_normalize_dataframe_rows(detected_rows)` ヘルパを追加し、入力型を明示判別: `None`→`[]`、pandas DataFrame（`hasattr "values"` かつ `hasattr "columns"`）→`.values.tolist()`、その他→`list(...)`。真偽評価を一切行わない。
- `populate_candidate_choices` を `rows = _normalize_dataframe_rows(detected_rows)` に変更。list 入力（後方互換）も従来通り処理。

**再発防止**:
- `gr.Dataframe` の値はハンドラに pandas DataFrame として渡る。`x or []` / `if rows:` 等の真偽評価は禁止。型を明示判別して `.values.tolist()` でリスト化する。
- 回帰テスト `tests/unit/test_movie_runtime_bugs.py`（DataFrame / 空 DataFrame / list の3ケース）を追加。

### [ERR037] 動画版 prompt_frame_idx 範囲外がモデル読込後（約18秒後）に発覚

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | フレーム数より大きい起点位置を指定する度に発生 |
| **初回発生日** | 2026-06-06 |
| **関連ファイル** | `gradio_app_sam2_transparent_BG_haystack_for_Movie.py` |

**エラー内容**:
```
prompt_frame_idx が範囲外です: 75（許容 0〜29）
```
スライダー `prompt_frame_idx` は 0〜1999 を許容するが、実際にサンプリングされるのは `max_frames`（例: 30）枚のみ。範囲外位置を起点にすると `SAM2VideoPropagator.run` がモデル読込後（十数秒後）にようやく `ValueError` を送出し、待ち時間が無駄になっていた。

**原因**:
`run_video_background_removal` に pipeline.run 前の事前検証がなく、`prompt_frame_idx >= processed_frames` の判定が propagator 内部の伝搬段階まで遅延していた。

**対処法**:
- `processed_frames = _estimate_processed_frames(...)` 算出直後（`build_video_progress_callback` / `release_text_detector` / pipeline.run より前）に fail-fast 検証を追加:
  `if int(prompt_frame_idx) >= processed_frames: raise gr.Error(... 起点位置 ... 処理フレーム数 ... {processed_frames - 1} 以下 ...)`。
- `except gr.Error: raise` により汎用エラーメッセージに包まれず即時通知される。

**再発防止**:
- スライダー上限と実処理レンジが乖離する UI では、重い処理（GPU/pipeline）に入る前に範囲を fail-fast 検証する。
- 回帰テスト `test_run_video_validates_prompt_frame_idx_before_pipeline` を追加。

### [ERR038] SAMURAI tracker 選択時 SAMURAI config が installed sam2 の Hydra 検索パスになく MissingConfigException

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | Colab 等 facebook 版 sam2 環境で SAMURAI tracker を選ぶ度に発生 |
| **初回発生日** | 2026-06-06 |
| **関連ファイル** | `pipelines/components/video_model_components.py` |

**エラー内容**:
```
ユーザー報告: 「sam3系のプルダウンメニュー 切り替えて 動画背景除去を実行ボタン押すとエラー スキーマがちがう？」
hydra.errors.MissingConfigException: Cannot find primary config 'configs/samurai/sam2.1_hiera_l.yaml'
```
`config/inference_models.toml` の SAMURAI tracker エントリ（config_name `configs/samurai/sam2.1_hiera_*.yaml`）を選び実行すると送出。Colab で facebook 版 sam2 が入っていると、その sam2 package には `configs/samurai/` が無く Hydra 検索パスで解決できない。

**原因**:
`SAM2VideoPropagator.warm_up` は `build_sam2_video_predictor(config_name, ...)` を直接呼ぶのみで、SAMURAI fork 同梱の configs（`samurai/sam2/sam2/configs/samurai/`）を Hydra 検索パスへ登録していなかった。

**対処法（samurai/ は変更しない: config/検索パスのみで対応）**:
- `_samurai_config_root(config_name)`: config 名が "samurai" を含み、ローカル `samurai/sam2/sam2/configs/samurai` が存在する場合のみ `samurai/sam2/sam2` package root を返す（非 samurai は None）。
- `_ensure_samurai_config_searchpath(config_name)`: samurai config のときのみ、Hydra `GlobalHydra` 検索パスへ `sam2_root.as_uri()` を重複排除して append。未初期化時は `import sam2` で初期化。解決不能時はエラーを握り潰さず `MissingConfigException` を伝搬。
- `warm_up()` の `build_sam2_video_predictor` 直前で `_ensure_samurai_config_searchpath(self.config_name)` を呼ぶ。

**再発防止**:
- fork 同梱 config（SAMURAI）は installed sam2 の Hydra 検索パスに自動では載らない。env / 検索パス登録で解決し、`samurai/` ディレクトリ自体は変更しない。
- URI は `Path.as_uri()` で RFC 準拠形式（Windows: `file:///J:/...`）を使い、自作 `f"file://{as_posix()}"` を避ける（重複排除比較の堅牢性）。
- 回帰テスト `_samurai_config_root`（samurai / 非 samurai）と warm_up のヘルパ呼出契約テストを追加。


### [ERR041] SAMURAI tracker が facebook 版 sam2 では動かない（検索パス追加だけでは不十分）— fork 導入が必須

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | Colab 等 facebook 版 sam2 環境で SAMURAI tracker を選ぶ度に再発（ERR038 対処後も） |
| **初回発生日** | 2026-06-19 |
| **関連ファイル** | `pipelines/components/video_model_components.py`, `Sam2_Transparent_Background_Haystack_for_Movie.py`, `config/inference_models.toml` |

**エラー内容**:
```
hydra.errors.MissingConfigException: Cannot find primary config 'configs/samurai/sam2.1_hiera_l.yaml'
Config search path:
	provider=hydra, path=pkg://hydra.conf
	provider=main, path=pkg://sam2
	provider=schema, path=structured://
```
ERR038 の `_ensure_samurai_config_searchpath`（検索パス append）を入れても、Colab で SAMURAI tracker 選択時に同じ MissingConfigException が再発（append した `samurai-local` provider が compose 時の検索パスに現れない）。

**原因（真因）**:
Colab notebook が **facebook 版 sam2**（`pip install git+https://github.com/facebookresearch/sam2.git`）を導入していた。SAMURAI は config だけの問題ではなく、
1. facebook 版 sam2 package に `configs/samurai/` が無い（→ MissingConfigException）。
2. 仮に config を解決できても、`configs/samurai/sam2.1_hiera_l.yaml` は `sam2.modeling.sam2_base.SAM2Base` に `samurai_mode` / `stable_frames_threshold` / `kf_score_weight` 等を渡すが、facebook 版 `SAM2Base` はこれらを受け付けず instantiate で `TypeError`。

つまり SAMURAI には **samurai 対応のモデルコードを持つ fork（同梱 `samurai/sam2`）を `sam2` として import させること**が必須。ERR038 の Hydra 検索パス append は config 配置の症状対処に過ぎず、モデルコード差異は解決できない（false hope）。SAMURAI は訓練不要モジュールで checkpoint は標準 `sam2.1_hiera_large.pt` を再利用するため追加 DL は不要。

**対処法（samurai/ は変更しない: 読み取り＋ installed package 差し替え）**:
- Colab notebook（`Sam2_Transparent_Background_Haystack_for_Movie.py` 正本）: Cell 1 から facebook sam2 install を削除し、Cell 2（Drive マウント後）で同梱 fork を editable 導入 `pip install -e "{PROJECT_ROOT}/samurai/sam2"`。fork は `configs/sam2.1/` と `configs/samurai/` の両方を含むため facebook / SAMURAI 両 tracker を 1 つの sam2 で賄う。Cell 2.5 の診断メッセージも fork 参照へ更新。`.ipynb` は jupytext で再生成。
- `_require_samurai_capable_sam2(config_name)` を追加し `warm_up()` の build 直前で呼ぶ。samurai config のとき installed sam2 の `Path(sam2.__file__).parent / configs / samurai` を検査し、無ければ **actionable な RuntimeError**（`pip install -e samurai/sam2` を促す / 標準 SAM2 への切替案内）を raise。cryptic な MissingConfigException / TypeError を回避。非 samurai config では `import sam2` せず no-op。

**再発防止**:
- SAMURAI tracker は「config 配置」ではなく「import される sam2 が SAMURAI fork であること」が前提。env / 検索パス append だけでは不十分（ERR038 の教訓を更新）。
- fork は標準 config も内包するため、Colab では sam2 を fork 一本に統一してよい（facebook tracker も動作）。追加 checkpoint は不要（標準 sam2.1 重みを再利用）。
- 新カーネルで「上から順に全セル実行」する前提。同一カーネルで先に facebook sam2 を import 済みの場合は editable install が反映されないため、ランタイム再起動が必要（診断メッセージに明記）。
- 回帰テスト: `_require_samurai_capable_sam2`（facebook で raise / fork で pass / 非 samurai で no-op、`sys.modules` の fake sam2 で GPU 非依存に検証）を追加。


### [ERR045] Colab で SAMURAI fork の editable install（Drive）が失敗し sam2 が入らない + `_C` 未ビルドで伝搬が落ちる

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | Colab で Drive 上の `pip install -e samurai/sam2` を使う度に再発（ERR041 の対処方法を実機適用したら失敗） |
| **初回発生日** | 2026-06-20 |
| **関連ファイル** | `Sam2_Transparent_Background_Haystack_for_Movie.py`（Cell 2 / Cell 2.5）, `samurai/sam2/setup.py`（参照のみ・変更不可） |

**エラー内容**:
```
ModuleNotFoundError: No module named 'sam2'
```
ERR041 の対処として Cell 2 で `!{sys.executable} -m pip install -e "{PROJECT_ROOT}/samurai/sam2"`（editable）を実行したが、`!pip` は失敗しても例外を出さないためそのまま進み、Cell 2.5 の `import sam2` で `ModuleNotFoundError` が表面化した。

**原因（2 つの独立した問題）**:
1. **Drive(FUSE) 上の editable install が失敗**: `pip install -e` は `.pth` / `*.egg-info` をソースディレクトリ（= Google Drive マウント）に書き込むが、Drive FUSE はこれらの書き込み・ロックに失敗しやすく、結果 sam2 が site-packages に登録されない。`!pip` は非ゼロ終了でもセルを止めないため、失敗が後段の cryptic なエラーに化ける。
2. **`_C`（CUDA 拡張）未ビルド**: `samurai/sam2/sam2/build_sam.py` が `build_sam2_video_predictor` で Hydra override `++model.fill_hole_area=8` を強制する。伝搬中 `sam2_video_predictor.py` の `if self.fill_hole_area > 0:` → `fill_holes_in_mask_scores` → `sam2/utils/misc.py` の `get_connected_components` が `from sam2 import _C; return _C.get_connected_componnets(...)` を **CPU fallback 無し**で呼ぶ。通常の `pip install`（build isolation 有効）では分離環境に torch が無く、setup.py の `get_extensions()` が `BUILD_ALLOW_ERRORS=1` で `ext_modules=[]` に縮退 → `_C` がビルドされず、伝搬時に `ModuleNotFoundError: sam2._C` で落ちる。

**対処法（samurai/ は変更しない: notebook の install 方法のみ修正）**:
- Cell 2 の install を **非 editable + `--no-build-isolation`** に変更:
  `!{sys.executable} -m pip install --no-build-isolation "{SAMURAI_SAM2_DIR_POSIX}"`。
  - 非 editable: pip がソースを temp に複製してビルドするため Drive の `.pth`/egg 書き込み失敗を回避。`configs/*.yaml`（`configs/sam2.1`・`configs/samurai`）は `MANIFEST.in` の `recursive-include sam2 *.yaml` + `include_package_data` で wheel に同梱される。
  - `--no-build-isolation`: 現環境の torch を見せて `_C`（connected_components）を Colab GPU の nvcc でビルド。これで伝搬時の `fill_hole_area=8` 経路が動く。
- install 直後に **fail-loud 検証**を追加: `importlib.invalidate_caches()` 後 `importlib.util.find_spec("sam2") is None` なら actionable な RuntimeError を raise（pip ログ確認 / GPU ランタイム・CUDA 整合を案内）。`!pip` の沈黙失敗が Cell 2.5 で cryptic 化するのを防ぐ。
- Cell 2.5 の診断メッセージも `pip install --no-build-isolation samurai/sam2`（非 editable）参照へ更新。`.ipynb` は jupytext 再生成。

**再発防止**:
- Google Drive 上のパッケージは **editable(`-e`) を避け非 editable で install** する（FUSE の .pth/egg 書き込み失敗回避）。
- notebook の `!pip install` は **fail-loud 化**（直後に import/find_spec で成功確認し、失敗なら raise）して沈黙失敗を後段に伝播させない。
- SAM2 動画伝搬は `fill_hole_area>0`（既定 8）で `sam2._C` が必須・CPU fallback 無し。`_C` をビルドするには **`--no-build-isolation`**（torch を見せる）と nvcc が要る。build isolation 既定の `pip install` だけでは `_C` が入らない（facebook 版でも同様）。
- 同一カーネルで先に sam2 を import 済みなら再 install が反映されないため、Colab はランタイム再起動 → 上から順に全セル実行を前提にする。


### [ERR046] SAMURAI fork の `sam2_base.py` が `loguru` を import するが fork の依存に未宣言で `ModuleNotFoundError`

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | SAMURAI fork を install 後、`build_sam2_video_predictor` 初回呼び出しで再発 |
| **初回発生日** | 2026-06-21 |
| **関連ファイル** | `Sam2_Transparent_Background_Haystack_for_Movie.py`（Cell 1）, `samurai/sam2/sam2/modeling/sam2_base.py`（参照のみ・変更不可）, `samurai/sam2/setup.py`（参照のみ・変更不可） |

**エラー内容**:
```
ModuleNotFoundError: No module named 'loguru'
```
`build_sam2_video_predictor` → `instantiate(cfg.model)` → `sam2.sam2_video_predictor` → `sam2.modeling.sam2_base` の冒頭 `from loguru import logger` で発生。

**原因**:
SAMURAI fork の `samurai/sam2/sam2/modeling/sam2_base.py` が `loguru` を import するが、fork の `samurai/sam2/setup.py` の `REQUIRED_PACKAGES`（torch / torchvision / numpy / tqdm / hydra-core / iopath / pillow のみ）に `loguru` が含まれない。このため fork を `pip install` しても `loguru` が入らず、伝搬器のビルド時に表面化する。facebook 版 sam2 には無い fork 固有の追加依存。

**対処法（samurai/ は変更しない: notebook の install のみ修正）**:
- Cell 1 に `!{sys.executable} -m pip install loguru` を追加して明示導入。理由コメント（ERR046）を併記。`.ipynb` は jupytext 再生成。
- 回帰テスト `tests/unit/test_jupytext_notebooks.py::test_sam2_movie_notebook_installs_loguru_for_samurai_fork` で `pip install loguru` の存在を検証。

**再発防止**:
- vendored fork は宣言されない実行時依存（`loguru` 等）を持つことがある。fork の import を grep し、`setup.py` の `install_requires` に無いものは notebook 側で明示 install する。


### [ERR047] RGBA(透過)動画が `cv2.VideoWriter` で全 frame skip され空動画になる（4ch 非対応 / VP90 webm 不可）

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 動画モード + RGBA 出力で毎回 |
| **初回発生日** | 2026-06-21 |
| **関連ファイル** | `pipelines/components/video_model_components.py`（`_OpenCVFrameVideoWriter` / `_select_rgba_codec` / `TransparentBGVideoExtractor.run` / `VideoWriter`）, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`（RGBA codec radio info） |

**エラー内容**:
```
OpenCV: FFMPEG: tag 0x30395056/'VP90' is not supported with codec id 167 and format 'webm / WebM'
global cap_ffmpeg_impl.hpp:2774 writeFrame write frame skipped - expected 3 channels but got 4
global cap_ffmpeg.cpp:218 write FFmpeg: Failed to write frame
UserWarning: Video does not have browser-compatible container or codec. Converting to mp4.
```
SAMURAI 伝搬・transparent-background までは成功するが、RGBA 動画の書き出し段で全 frame が skip され、透過 webm が空（中身なし）になる。

**原因**:
`cv2.VideoWriter` は 4ch(RGBA/BGRA) frame を書けない（`isColor` は 1ch/3ch のみ）。BGRA を渡すと FFmpeg が "expected 3 channels but got 4" で **毎 frame skip**。さらに OpenCV の VP9/webm 経路は `VP90` fourcc を webm コンテナで拒否する。旧 `_select_rgba_codec` は `cv2.VideoWriter.isOpened()` だけで可用性判定していたが、VP90 では **open は成功する（偽陽性）**ため、実書き込みの失敗を検知できなかった。OpenCV は本質的に alpha 動画を書けない（真の透過は ffmpeg の `libvpx-vp9`+`yuva420p` 等が必要）。

**対処法（samurai/ は変更しない）**:
- RGBA stream を `cv2.VideoWriter` から **imageio+ffmpeg** に置換。`_require_imageio()`（`imageio.v2` + `imageio_ffmpeg` を遅延 import、無ければ握り潰さず連番(PNG)出力を促す `RuntimeError`）、`_RgbaCodecSpec`、`_ImageioAlphaVideoWriter`（`append_data` で RGB order RGBA をそのまま書く）を追加。
- `_select_rgba_codec` は cv2 fourcc ではなく alpha 対応 imageio spec を返す: `webm_vp9` → codec `libvpx-vp9` / pixelformat `yuva420p` / `output_params=("-auto-alt-ref","0")` / `macro_block_size=2`（奇数解像度を偶数へ自動スケール、yuv420p 要件）。`mov_png` → codec `png` / pixelformat `rgba` / `macro_block_size=1`。
- alpha(1ch) / preview(3ch) は従来通り `_OpenCVFrameVideoWriter`（cv2 で問題なし）。4ch を cv2 に渡す経路は撤廃。
- 偽陽性の元だった `_test_codec` / `_codec_cache` を削除。UI の RGBA codec info から誤解を招く「自動で他方式に fallback」を削除し「imageio+ffmpeg で alpha 保持・書けない環境は連番(PNG)が確実」に更新。
- 依存: Cell 1 は既に `imageio[ffmpeg]` を install 済みのため notebook 変更不要。
- 回帰テスト `tests/unit/test_movie_runtime_bugs.py`（Bug D）: spec が alpha 対応 imageio パラメータを返す / imageio 欠如時に明確エラー / 動画モードの RGBA stream が 4ch を imageio へ append し cv2 へ渡さない。

**再発防止**:
- `cv2.VideoWriter` は RGB(3ch)/gray(1ch) 専用。alpha 動画は ffmpeg 直叩き（imageio+ffmpeg, `yuva420p`/`rgba`）でしか作れない。codec 可用性を `isOpened()` だけで判定しない（実書き込みの channel mismatch を見逃す）。
- `macro_block_size` は imageio-ffmpeg の正規パラメータ（奇数解像度を偶数へスケール）。`2` で yuva420p 要件を満たす。
- 真の alpha 動画の実エンコード検証は Colab（imageio+ffmpeg+GPU）でのみ可能。ローカルは mock + 全テスト + smoke の論理検証に留まる。


### [ERR048] Colab/gradio.live で長時間の動画処理中に SSE 接続が idle 切断され全出力が「Error」表示になる（処理は継続）

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | Colab 共有リンクで動画処理が数分かかる時（特に低速 GPU） |
| **初回発生日** | 2026-06-22 |
| **関連ファイル** | `pipelines/components/video_model_components.py`（`_ProgressKeepAlive` / `SAM2VideoPropagator.run` / `TransparentBGVideoExtractor.run` / tracking overlay ループ） |

**エラー内容**:
```
Error
Connection errored out.
```
UI の出力欄（RGBA / Alpha / Preview Video, Tracking Overlay, 連番 PNG サンプル）が**全て "Error" 表示**になる。一方 Colab stdout には例外が出ず、`propagate in video: 24% 18/74 [00:52<03:33, 3.81s/it]` のように **SAM2 伝搬処理は継続している**。

**原因**:
Colab / gradio.live の共有トンネルは event SSE 接続に無通信が一定時間続くと idle 切断する。ブラウザ側は pending 中の全出力を "Error" にするが、サーバ側 Python 関数は最後まで実行を続ける（だから stdout に例外が出ない）。進捗通知が **frame 数ベースの間引き**（伝搬ループ `propagated_count % 10`、tb/overlay ループ `% 5`）だったため、実測 3.81s/it の低速 GPU では伝搬ループで **最大約38秒の無通信ギャップ**が生じ、その間に SSE が切れていた。`Connection errored out` はブラウザ汎用表示で、一次情報は Colab stdout（伝搬が継続している＝接続のみの問題）。

**対処法（samurai/ は変更しない）**:
- 時間ベースの keep-alive throttle `_ProgressKeepAlive` を追加。`maybe(index, total, fraction, description, force=False)` が「最初/最後の frame（`index<=0 or index+1>=total`）」「`force`」「前回送信から `_PROGRESS_KEEPALIVE_SEC=2.0` 秒経過」のいずれかで `_notify_progress` を呼ぶ。`clock` を注入可能にしテスト決定論化。
- `SAM2VideoPropagator.run` の frame 準備ループ・伝搬ループ、`TransparentBGVideoExtractor.run`(streaming) の tb ループ、tracking overlay ループの進捗通知を、frame 数間引き（`% 10` / `% 5`）から `_ProgressKeepAlive.maybe` に置換。frame 速度によらず無通信ギャップを最大 2.0 秒に抑え SSE 接続を維持する。
- legacy `VideoWriter._write_*` ループ（cv2 frame 書き込み・高速）は対象外（`% 20` 維持。1 frame が ms 単位で無通信ギャップ無し）。
- 回帰テスト `tests/unit/test_progress_keepalive.py`（5 件）: 境界 frame 発火 / 旧 frame 数ベースなら落ちる「非境界 frame でも経過時間で発火」/ 間隔内抑制 / force / None no-op を FakeClock 注入で検証。

**再発防止**:
- 長時間ループの進捗通知は **frame 数ベースの間引きにしない**（低速 frame で無通信ギャップが数十秒に広がり SSE が切れる）。時間ベース keep-alive（最大数秒間隔）で接続を保つ。
- `Connection errored out` + 全出力 "Error" でも、Colab stdout で処理（`propagate in video` 等）が継続していれば接続のみの問題。サーバ stdout を一次情報にする（ERR026/ERR027 と同方針）。
- keep-alive 間隔は内部チューニング定数 `_PROGRESS_KEEPALIVE_SEC`（モデルパラメータではない）。Colab SSE idle timeout（概ね 30s 以上）に対し十分小さい 2.0s。
- 実機 Colab での接続維持はユーザー要確認（ローカル .venv は torch/sam2/GPU 無しのためロジック検証に留まる）。


### [ERR049] Colab T4 で SAMURAI 動画伝搬が最初の重い frame で GPU メモリ枯渇 stall し stdout が `propagate 1/N` で凍結する

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | Colab T4(16GB) 等で SAMURAI tracker を使い双方向伝搬する時 |
| **初回発生日** | 2026-06-22 |
| **関連ファイル** | `pipelines/components/video_model_components.py`（`SAM2VideoPropagator.__init__` / `run` の `init_state`）, `config/inference_models.toml`（SAMURAI tracker entry）, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`（`get_video_pipeline`） |

**エラー内容**:
```
propagate in video:   1%|          | 1/67 [..:..<..:.., ...s/it]
（ここで stdout が完全に凍結し、例外も進捗更新も出ない）
```
動画背景透過が SAM2 *video propagation* の最初の重い frame で固まる。ERR048（SSE idle 切断）と異なり、**stdout 自体が `1/67` から進まない**（＝バックエンドの hang）。**標準 SAM2 では出ず、SAMURAI checkpoint を使った時だけ**発生する。

**原因**:
SAMURAI は motion-aware memory（Kalman filter 状態）を持ち GPU 常駐メモリが標準 SAM2 より大きい。加えて T4(Turing 7.5) は非 Ampere で Flash Attention が無効化され attention のメモリ実装が重い。双方向伝搬（forward + reverse）は 2 pass 分の per-frame memory が積み上がるため、伝搬の最初の重い frame で VRAM を使い切り、CUDA アロケータがメモリ確保待ち/スラッシングで事実上 stall する（OOM 例外を投げ切らず固まる）。`samurai/` の `init_state` / `propagate_in_video` 自体は bounded ループで無限ループではない（コード確認済み）＝hang ではなく stall。

**対処法（samurai/ は変更しない）**:
- SAM2 標準の CPU offload（`init_state(offload_video_to_cpu=..., offload_state_to_cpu=...)`）を有効化して常駐 VRAM を抑える。SAMURAI fork の `init_state` は両 kwarg を受け取る（レビューで署名確認済み: `samurai/sam2/sam2/sam2_video_predictor.py`）。
- ハードコードせず **config 駆動**: `config/inference_models.toml` の SAMURAI tracker entry（`samurai_hiera_l` / `samurai_hiera_b_plus`）にのみ `offload_video_to_cpu = true` / `offload_state_to_cpu = true` を追加。標準 SAM2 entry（`sam2_hiera_l` / `sam2_hiera_b_plus`）は無変更で動作実績のある経路を保つ。
- `SAM2VideoPropagator.__init__` に `offload_video_to_cpu: bool = False` / `offload_state_to_cpu: bool = False`（既定 False=現状維持）を追加し、`run` の `init_state` 呼び出しに転送。`get_video_pipeline` が tracker entry の `.get("offload_video_to_cpu", False)` 等を読んで propagator に渡す（非 SAMURAI entry は既定 False で無影響）。
- 回帰テスト `tests/unit/test_video_pipeline_wiring.py`（4 件）: 既定で offload 無効 / offload 有効時に `init_state` へ kwargs が届く / SAMURAI registry entry のみ offload 有効・標準 SAM2 は無効 / `get_video_pipeline` が offload 設定を読む。fake predictor の `init_state` を `**kwargs` 受け入れに更新。

**再発防止**:
- 低 VRAM GPU での長尺・双方向動画伝搬は **CPU offload を前提**にする。stall（stdout が `1/N` で凍結・例外なし）は OOM 例外を投げ切れないメモリ枯渇のサイン。一次情報は Colab stdout（`propagate in video` が進むか止まるか）。
- offload はモデルパラメータではなく実行環境向けの config。GPU メモリに余裕がある環境（標準 SAM2 / 大容量 VRAM）では無効のままにして I/O オーバーヘッドを避ける。
- offload でも解消しない場合の follow-up 候補（本タスク未実施）: ① SAMURAI の KF 状態が cache されたモデルに残り pass/run 間でリセットされない（正確性懸念）, ② 双方向 + SAMURAI は reverse pass が forward の KF 状態を流用する点が意味的に疑問。
- 実機 Colab T4 での stall 解消はユーザー要確認（ローカル .venv は torch/sam2/GPU 無しのため配線検証に留まり、freeze 自体は再現不能）。




### [ERR050] SAMURAI 動画伝搬の VRAM 枯渇 follow-up（autocast fp16 / 双方向自動 OFF / 起点先頭 / 推奨設定明記）

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | Colab T4(16GB) 等で SAMURAI tracker を使う時（ERR049 と同条件） |
| **初回発生日** | 2026-06-22 |
| **関連ファイル** | `pipelines/components/video_model_components.py`（`SAM2VideoPropagator.__init__` / `_autocast_context` / `run`）, `config/inference_models.toml`（tracker entry）, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`（`get_video_pipeline` / `update_bidirectional_for_tracker` / 冒頭 Markdown / info 文言）, `Sam2_Transparent_Background_Haystack_for_Movie.py`（先頭セル） |

**エラー内容**:
ERR049（CPU offload）後も SAMURAI 動画伝搬の VRAM が逼迫しうる。SAMURAI 本家の推論スクリプト（`samurai/scripts/main_inference.py`）は ① `torch.autocast("cuda", dtype=float16)` で伝搬、② 起点フレーム 0、③ forward-only（reverse を使わない）を前提とするが、本リポジトリの propagator は autocast 未適用（fp32）で VRAM を余計に使い、UI は SAMURAI でも双方向伝搬 ON を許していた（逆走は KF が破綻し per-frame memory も 2 倍）。

**原因**:
- 伝搬を fp32 で実行 → SAMURAI 本家比で VRAM 消費が大きい。
- 双方向伝搬（reverse pass）は SAMURAI の Kalman filter の速度ベクトルが反転し追跡が崩れる上、2 pass 分の常駐メモリが積み上がり ERR049 の stall を誘発する anti-pattern。
- 末尾起点フレームは forward-only の SAMURAI で逆走を強い、同様に stall を誘発しうる。

**対処法（samurai/ は変更しない）**:
- **autocast fp16（config 駆動）**: `SAM2VideoPropagator.__init__` に `autocast_dtype: str | None = "float16"` と helper `_autocast_context` を追加。`run` の `with torch.inference_mode():` を `with torch.inference_mode(), self._autocast_context(torch):` に変更。autocast は `device == "cuda"` かつ dtype が `None/""/"none"` 以外のときだけ適用し、それ以外は `contextlib.nullcontext()` で既存挙動を保つ。`float16`/`bfloat16` を map。`get_video_pipeline` が tracker entry の `.get("autocast_dtype", "none")` を渡す。
- **標準 SAM2 経路は無変更**: 標準 SAM2 entry（`sam2_hiera_l` / `sam2_hiera_b_plus`）は `autocast_dtype = "none"`（fp32 維持）。SAMURAI entry（`samurai_hiera_l` / `samurai_hiera_b_plus`）のみ `autocast_dtype = "float16"`。実績ある標準経路の数値挙動を一切変えない（レビュー指摘反映）。
- **双方向自動 OFF（config 駆動）**: tracker entry に `supports_bidirectional`（標準 SAM2=true / SAMURAI=false）を追加。`update_bidirectional_for_tracker` を追加し `tracker_model.change(..., outputs=[bidirectional])` で配線。SAMURAI 選択時は `gr.update(value=False, interactive=False)` で双方向 checkbox を自動 OFF・無効化、標準 SAM2 は `interactive=True`。未知 id は `KeyError` を捕捉し安全側（`interactive=True`）に倒す（握り潰しではなく明示）。
- **起点フレーム先頭**: `prompt_frame_idx` は既定 `value=0`（先頭）のまま。info 文言を「SAMURAI は forward-only なので 0（先頭）推奨」に強化。
- **推奨設定の明記**: Gradio 冒頭（タイトル直後の最上部 Markdown）と notebook 正本 `.py` の先頭セルに、SAMURAI 推奨設定（双方向 OFF / 起点 0 / offload / autocast fp16 / 初回 30 frame）を表で明記。`.ipynb` は jupytext で再生成。
- 回帰テスト `tests/unit/test_video_pipeline_wiring.py`（10 件追加）: autocast 既定 float16 / CPU は nullcontext / `none` で cuda でも無効 / cuda+float16 で `torch.autocast("cuda", dtype=float16)` / registry の autocast・supports_bidirectional フラグ（標準 SAM2 は autocast none・双方向 true、SAMURAI は float16・双方向 false） / app の autocast 配線 / SAMURAI 双方向自動 OFF 挙動 / change 配線 / 推奨設定 doc の存在。

**再発防止**:
- SAMURAI は forward-only（KF motion model）。**逆方向・双方向伝搬は使わない**。UI は registry の `supports_bidirectional` を見て自動制御する（tracker 追加時もハードコード不要）。
- 低 VRAM GPU での SAM2/SAMURAI 推論は autocast fp16/bf16 が前提（SAM2 本家・SAMURAI 本家とも mixed precision）。本リポジトリでは安全のため標準 SAM2 経路は `none`（fp32 維持）、SAMURAI のみ fp16 を既定にし、いずれも config で切替可。
- 実機 Colab T4 での stall 解消・出力品質はユーザー要確認（ローカル .venv は torch/sam2/GPU 無しのため配線・契約検証に留まり、伝搬の実挙動・VRAM 実測は再現不能）。




### [ERR051] SAMURAI で複数オブジェクト指定時に伝搬が `Boolean value of Tensor ... ambiguous` で落ちる

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | SAMURAI tracker を選び、複数 box（複数オブジェクト）を指定して動画処理した時 |
| **初回発生日** | 2026-06-22 |
| **関連ファイル** | `pipelines/components/video_model_components.py`（`SAM2VideoPropagator.__init__` / `run`）, `config/inference_models.toml`（tracker entry）, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`（`get_video_pipeline` / 冒頭 Markdown）, `Sam2_Transparent_Background_Haystack_for_Movie.py`（先頭セル）, （原因箇所＝変更不可）`samurai/sam2/sam2/modeling/sam2_base.py::_forward_sam_heads` |

**エラー内容**:
SAMURAI tracker で複数 box を指定して動画背景除去を実行すると、伝搬の最初のフレームで失敗する。
```
File ".../sam2/modeling/sam2_base.py", line 451, in _forward_sam_heads
    if ious[0][best_iou_inds] > self.stable_ious_threshold:
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```
Haystack 経由で `Component 'sam2_video_propagator' (SAM2VideoPropagator)` が失敗し、`gr.Error` に伝搬する。

**原因**:
- SAMURAI fork の `_forward_sam_heads` は **単一オブジェクト（バッチ B=1）前提**で書かれている。`best_iou_inds = torch.argmax(ious, dim=-1)` は形状 `[B]`、`ious[0]` は `[num_masks]`。`ious[0][best_iou_inds]` は `[B]` 形状になり、B≥2（複数オブジェクト）だと多要素テンソルになって `if tensor > threshold` の boolean 評価が曖昧になる。
- SAMURAI は Kalman filter の motion model（`kf_mean` / `stable_frames` 等）を**モデルインスタンスで共有**するため、そもそも複数オブジェクト同時追跡を想定していない（単一対象追跡専用）。
- 本リポジトリの propagator は `boxes` を渡すと obj_id 1..N を登録する複合対象 union 配線を持つため、SAMURAI と複数 box を組み合わせると上記 B≥2 に到達する。

**対処法（samurai/ は変更しない）**:
- **config 駆動の事前ガード**: tracker entry に `single_object_only`（SAMURAI=true / 標準 SAM2=false）を追加。`SAM2VideoPropagator.__init__` に `single_object_only: bool = False` を追加し保持。`run` の冒頭バリデーション直後（`warm_up` より前＝fail-fast）で `requested_object_count = len(boxes) if boxes else 1` を計算し、`self.single_object_only and requested_object_count > 1` のとき actionable な `ValueError`（「単一オブジェクト専用です。box を 1 つに減らすか標準 SAM2 tracker に切り替えてください」）を raise。GPU 確保・モデル build 前に止まる。
- `get_video_pipeline` が tracker entry の `.get("single_object_only", False)` を propagator へ渡す（既定 False で後方互換）。
- **推奨設定の明記**: Gradio 冒頭の SAMURAI 推奨設定 Markdown 表と notebook 正本 `.py` 先頭セルの表に「対象オブジェクト数＝1 個のみ（複数は標準 SAM2 へ）」行を追加。`.ipynb` は jupytext 再生成。
- 回帰テスト `tests/unit/test_video_pipeline_wiring.py`（4 件追加）: 複数 box で `ValueError` かつ `warm_up` 未到達（`_video_predictor is None`）/ 単一 box は `single_object_only=True` でも正常 / registry の `single_object_only`（SAMURAI=True・標準=False）/ app の配線。

**再発防止**:
- SAMURAI は単一対象追跡専用。複数オブジェクトを切り抜く場合は標準 SAM2 tracker（`INFERENCE_TRACKER_VARIANT=sam2_facebook`）を使う。
- tracker ごとの能力差（forward-only / 単一オブジェクト / autocast）は `config/inference_models.toml` のフラグ（`supports_bidirectional` / `single_object_only` / `autocast_dtype`）で宣言し、propagator・UI はそれを見て自動制御する（tracker 追加時もハードコード不要）。
- vendored fork（samurai/）のバグは fork を変更せず、本リポジトリ側で「能力外の入力を事前に actionable に拒否する」方針で回避する。
- 実機 Colab での挙動はユーザー要確認（ローカル .venv は torch/sam2/GPU 無しのため、ガードの fail-fast 配線・契約検証に留まる）。


### [ERR052] 新規 Gradio アプリ起動時の `ImportError`（pipeline ビルダーの export 元取り違え）と config セクション取り違えで設定が黙殺される

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | 既存パイプライン/設定を参考に新規 Gradio アプリ・Component を追加した時 |
| **初回発生日** | 2026-06-23（ルートA案 動画αマット新規実装中） |
| **関連ファイル** | `gradio_app_sam2_ben2_route_a_for_Movie.py`, `pipelines/route_a_video_pipeline.py`, `pipelines/sam2_tb_video_pipeline.py`, `config/route_a.toml` |

**エラー内容**:
1. 新規 `gradio_app_sam2_ben2_route_a_for_Movie.py` の `--help` smoke で
   `ImportError: cannot import name 'build_video_reader_pipeline' from 'pipelines.route_a_video_pipeline'`。
2. サブエージェントレビュー指摘（M-1）: UI 既定値 `refine_foreground` を `config/route_a.toml` の `[composite]`
   セクションから取得していたが、実際の定義は `[alpha]` セクション。`.get(..., False)` が常に False に
   フォールバックし、`[alpha].refine_foreground = true` を設定しても UI に反映されない（dead config）。

**原因**:
1. `build_video_reader_pipeline` は `pipelines/sam2_tb_video_pipeline.py` で定義されており、新規
   `route_a_video_pipeline.py` には存在しない。「参考元が同名関数を別モジュールで持つ」ことを確認せず、
   新規モジュールから import しようとした。
2. config の TOML セクション構造（`[alpha]` / `[blur_guide]` / `[composite]`）を UI 側ヘルパ（`_composite_defaults()`）
   と取り違えた。キー名だけ見て取得元セクションを誤った。

**対処法**:
1. import 元を実定義モジュールに修正: `from pipelines.sam2_tb_video_pipeline import build_video_reader_pipeline`。
   既存ビルダーは re-export せず、定義モジュールから直接 import する。
2. `_alpha_defaults()`（`[alpha]` を返す）を追加し、`refine_foreground` 既定はそこから取得。BEN2 のみタブの
   既定も config 化（ハードコード `value=False` を撤去）。

**再発防止**:
- 既存資産を参考に新規ファイルを作る時は、参照する関数/定数の **実定義モジュール** を grep で確認してから import する（同名再 export を仮定しない）。
- config から既定値を取得する時は、キー名だけでなく **取得元セクション** が TOML 定義と一致するか確認する。設定が `.get(default)` で黙って握りつぶされていないか、レビューで「設定値が実際に画面/挙動へ反映されるか」を確認する。
- 新規 Gradio アプリは作成直後に `get_errors` + `--help` smoke を必ず通し、import/構築エラーを早期に検出する。





| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | SAM2 mask が対象を囲みきれない素材で発生 |
| **初回発生日** | 2026-06-04（調査）/ 2026-06-12（修正） |
| **関連ファイル** | `pipelines/components/model_components.py`（`TransparentBGExtractor.run`） |

**エラー内容**:
ユーザー報告: 「画面下のマスクが横一直線でばっさり切れる。BBOX そのものをマスク範囲に使っている疑い」。グレースケール alpha で被写体の下半分が水平直線で切れる。

**原因**:
`TransparentBGExtractor.run` が SAM2 mask の **形状を使わず外接矩形**（`mask_to_bbox` + `crop_padding`）で画像をクロップして transparent-background を適用し、`full_alpha[y_min:y_max, x_min:x_max] = alpha_crop` で矩形範囲だけに貼り戻していた。矩形内・mask 形状外の領域に alpha が残るため、矩形下端＝横一直線で切れる。`SAM2GuardFilter`（mask 外 alpha 削り）は実装済みだが、動画パイプライン `sam2_tb_video_pipeline.py` には未接続だった（静止画でも mask 未接続パイプラインでは no-op）。

**対処法**:
- `TransparentBGExtractor.run` に `apply_mask_guard: bool = True` / `mask_guard_dilate: int = 21` を追加。`full_alpha` 算出後、mask があり `mask.any()` のとき `dilate_binary_mask(mask, kernel_size=mask_guard_dilate)` の guard を乗算し、mask 形状外の alpha を 0 にする（transparent-background のソフト境界は dilate 分だけ保持）。
- extractor 内で適用するため、frame ごとに同 run を呼ぶ動画版 `TransparentBGVideoExtractor` にも自動波及。preview/rgba も guard 後の `full_alpha` から生成。
- `build_sam2_union_tb_pipeline` は後段 `SAM2GuardFilter` を同一 mask で接続するが、二値 guard の乗算は冪等（`guard×guard=guard`）で二重適用しても結果不変（回帰テストで担保）。

**再発防止**:
- 横切れの真因は「mask 形状ではなく mask の外接矩形が最終 alpha 範囲を決める」こと。透過抽出は必ず mask 形状を最終 alpha に反映する。
- mask が対象を囲みきれない根本（複合対象を 1 box しか使わない配線）は別タスク（複合対象 union UI 復旧）で対応。guard は「直線切れ」を「mask 形状に沿った切れ」に変えるが、未検出領域そのものは復元しない。
- 回帰テスト `tests/unit/test_transparent_bg_mask_guard.py`（mask 形状反映 / mask 未指定の後方互換 / SAM2GuardFilter 二重適用の冪等 / guard 無効化で従来挙動）を追加。



### [ERR053] RouteA Movie UI で prompt の個別削除ができず調整反復が困難、かつ prompt 適用の可視化不足で「点が伝わっていない」と誤認しやすい

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | prompt 微調整（bbox/point を多数打って試行錯誤）時に発生 |
| **初回発生日** | 2026-06-23 |
| **関連ファイル** | `gradio_app_sam2_ben2_route_a_for_Movie.py`, `pipelines/components/ui_helpers.py`, `Sam2_BEN2_RouteA_for_Movie.py` |

**エラー内容**:
- 既存 UI は `Prompt をクリア`（全消し）しかなく、選択した bbox や point（positive/negative）だけを削除できなかった。
- そのため prompt 調整中に意図しない点/箱が残りやすく、実行結果のちらつき原因を切り分けにくい。
- 実際には points/labels は SAM2 へ渡っていたが、UI 上で割当・件数が見えず「点が伝わっていない」と判断されやすかった。

**原因**:
- prompt state（`points`/`labels`/`box`/`boxes`）の編集 API が「追加」「全クリア」に偏っており、個別削除が無かった。
- 実行 status に prompt 反映の診断情報（pos/neg 件数、box 件数、複数 box 時の point 割当）が無く、入力が下流へ届いたか即時確認できなかった。

**対処法**:
- `ui_helpers.py` に個別削除 API を追加:
    - `build_prompt_selection_choices(prompt_state)`
    - `remove_selected_points(prompt_state, selected_labels)`
    - `remove_selected_boxes(prompt_state, selected_labels)`
- RouteA Gradio に「Prompt 編集（個別削除）」Accordion を追加し、選択 point/bbox を個別削除可能にした。
- prompt 更新イベント（click/detect/apply/clear/frame切替）ごとに削除候補 UI を同期。
- run status に prompt デバッグ情報（point pos/neg 件数、manual/union box 件数、複数 box 時の point assignment）と flicker ヒント（標準 SAM2 の双方向 ON、難シーンは per_object）を表示。

**再発防止**:
- Prompt UI には「追加」「全消し」だけでなく「個別削除」を最初から用意する。
- 追跡結果が不安定な報告を受けたときは、まず入力反映可視化（件数/割当）を status へ出し、入力欠落とモデル限界（伝播方向・合成モード依存）を切り分ける。
- notebook 手順にも削除機能とちらつき対策（標準 SAM2 + 双方向 ON / per_object）を明記して誤操作を減らす。


### [ERR054] RouteA で「overlay が 検出→ID追跡(MOT)→SAM2.1 を通らない」「ポジ点でちらつきが変わらない＝伝わっていない」と誤認する（実は設計どおり）

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | RouteA でちらつきを point で抑えようとしたとき |
| **初回発生日** | 2026-06-23 |
| **関連ファイル** | `gradio_app_sam2_ben2_route_a_for_Movie.py`, `pipelines/route_a_video_pipeline.py`, `pipelines/components/ben2_components.py`, `pipelines/components/video_model_components.py`, `計画書/2026-06-22_動画αマット_ルートA案_ブラー誘導_仕様書.md` |

**エラー内容**:
- ユーザー報告: 「tracking overlay が 検出(RF-DETR)→ID追跡(ByteTrack/BoT-SORT)→SAM2.1 の経路を通っていないのでは」「ちらつき箇所にポジティブ点を打っても変わらない＝明らかに prompt が伝わっていない」。

**原因（コード追跡で確認）**:
1. **MOT 層は未実装**: 現実装の RouteA pipeline は `VideoReader → SAM2VideoPropagator → OwnershipResolver → BEN2RouteAVideoExtractor → VideoWriter/FrameSequenceWriter/TrackingOverlayWriter`。テキスト検出は GroundingDINO で、RF-DETR も ByteTrack/BoT-SORT(MOT) も存在しない。仕様書 line10 の「RF-DETR→ByteTrack/BoT-SORT→SAM2.1」は**設計意図（要件定義 §10.1）であり未実装**。spec と impl の乖離であってバグではない。
2. **point は SAM2 へ確実に渡っている**: `video_model_components.py` の `SAM2VideoPropagator.run` で、boxes 経路は各 point を最近傍 box へ同梱（L547-554）、単一経路は points/labels を `add_new_points_or_box` に追加（L557-562）。`TrackingOverlayWriter` は point を反映した SAM2 union soft mask（`frame_masks`）を描画する。よって overlay マスクは point を反映する。
3. **ちらつきが point で変わらない真因**: RouteA では SAM2 マスクは「背景ブラーのゲート G」生成にしか使われない。`gate_alpha=OFF`（既定）では**最終 α を BEN2 が単独生成**する（BEN2 はマスク入力ポートを持たない＝仕様 A-2）。よって point → union マスク微修正 → dilation_px(=24) 膨張で吸収 → BEN2 入力ほぼ不変 → 最終 α/ちらつき不変。これは設計どおりで配線欠落ではない。ちらつきの主因は BEN2 saliency の不安定性。

**対処法**:
- `run_route_a_background_removal` の実行 status に診断を追加（`points and not gate_alpha` のとき）:
    - 「point/SAM2 マスクは現在『背景ブラーの範囲』にのみ使われ、最終 α は BEN2 が単独生成。point で直接ちらつきを抑えるには『ゲートでαを制限（gate_alpha）』を ON に」
    - 「Tracking Overlay は point を反映した SAM2 マスクを描く。point 追加で overlay が変われば SAM2 へは伝達済み＝ちらつきは BEN2 側要因」
- これにより「入力欠落」と「RouteA アーキ由来（α は BEN2 単独）」を切り分けられる。

**再発防止**:
- RouteA は「マスク注入」ではなく「入力画像加工で間接誘導」する設計（仕様 A-2）。SAM2 マスク/point は α を直接拘束しない（gate_alpha=ON または per_object で初めて α へ波及）。
- 仕様書の検出→MOT→SAM2.1 は将来アーキの設計意図。現実装は GroundingDINO→SAM2 で MOT 層は無いことを混同しない。
- 「prompt が伝わらない」報告時は、まず overlay（SAM2 マスク）で入力反映を確認し、α 段（BEN2）の挙動と切り分ける。


### [ERR040] UI ファイルが未コミットの作業ツリー変更で過去版へ巻き戻り、復元中の git stash で全作業を退避してしまう

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 一度のみ（再発防止のため記録） |
| **初回発生日** | 2026-06-15 |
| **関連ファイル** | `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`, `gradio_app_sam2_transparent_BG_haystack.py`, `tests/unit/test_jupytext_notebooks.py` |

**エラー内容**:
動画 UI のシーク機能・複数 bbox 反映・mask union が「約1ヶ月前の版へ巻き戻った」とユーザー報告。調査の結果、動画版・静止画版 UI と jupytext notebook が**未コミットの作業ツリー変更**で全機能版から旧版へロールバックされていた（Component 層 `model_components.py` / `video_model_components.py` / `model_registry.py` は無傷）。さらに復元作業中に `git stash`（引数なし）を実行したところ、復元途中の全作業がまとめて退避され、`git stash pop` が EOL 正規化で2回失敗（`Your local changes would be overwritten by merge`、ただし `git diff HEAD` はクリーン）した。

**原因**:
1. 巻き戻り: UI 層の全機能実装（HEAD `2702d6b` より新しい未コミット差分）が一度も commit されておらず、作業ツリーで旧版へ上書きされていた。stash / branch / reflog のいずれにも残っておらず**復元不能**。Component 層は別ファイルのため影響を受けなかった。
2. stash 事故: `git stash` は引数なしだと**追跡中の全変更を退避**する。復元途中の作業ツリー全体が対象になった。`git stash pop` はマージを伴うため、`* text=auto` 等の EOL 正規化で内容が同一でも「上書きされる」と判定され失敗した。

**対処法**:
- 巻き戻り復元: `git checkout HEAD -- <file>` で HEAD の全機能版を復元。HEAD より新しい未コミット実装は失われているため、**RED テスト（`test_movie_app_ui_wiring` / `test_movie_runtime_bugs` / `test_video_pipeline_wiring` / `test_jupytext_notebooks`）を正本として再実装**して GREEN 化した（静止画版は HEAD が全機能版だったため checkout のみで復旧）。
- stash 事故の安全な回収: `git stash pop`（マージ）が EOL で失敗する場合は `git checkout 'stash@{0}' -- .` を使う。これは**マージせず stash のファイル内容を作業ツリーへ展開**するため EOL 競合を回避できる。回収後に内容とテストを検証し、冗長になった stash を `git stash drop 'stash@{0}'` で削除。
- PowerShell では `stash@{0}` の波括弧が誤解釈されるため**シングルクォート必須**。日本語ファイル名は `git -c core.quotepath=false` で文字化けを防ぐ。

**再発防止**:
- UI/配線の重要実装は**こまめに commit** する。未コミットの作業ツリーだけに依存しない（巻き戻りで復元不能になる）。
- 復元・整理作業中に `git stash`（引数なし）を安易に実行しない。退避したい範囲を明示するか、先に commit してから操作する。
- 巻き戻りの正本は**テスト**。RED テストが残っていれば、ソース実装が失われても再実装の指針になる（今回はこれで全機能を復元できた）。
- UI/配線の「fixed/完了」は ERR035 に従い Playwright 実起動で実行時検証してから記録する（今回 `prompt-frame-idx` シーク・複数 bbox CheckboxGroup・処理順表示・`movie-frame-step` のレンダリングを Playwright で確認済み）。

---

### [ERR041] 動画 SAM2 追跡で box と point prompt を併用すると point（positive/negative）が無視される

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | box 群と point 群を同時指定した時に常発 |
| **初回発生日** | 2026-06-15 |
| **関連ファイル** | `pipelines/components/video_model_components.py`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`, `tests/unit/test_video_pipeline_wiring.py` |

**エラー内容**:
文字プロンプト → bbox 候補 → bbox union までは追跡できるが、その後 Point mode で positive/negative の補正点を追加しても **point が追跡に反映されない**。UI 上の Prompt Status には点が登録され（`Point selected: ..., label=positive/negative`）、`state["points"]/["labels"]` にも保持されるのに、伝搬結果の mask へ寄与しない。

**原因**:
`SAM2VideoPropagator.run`（`pipelines/components/video_model_components.py`）の登録分岐が `if boxes:` の時に **box のみを `add_new_points_or_box(box=...)` で登録し、points/labels を一切渡していなかった**。`apply_selected_boxes`（テキスト/候補フロー）は `state["boxes"]` を設定する際に `state["points"]` をクリアしないため box と point が共存するが、propagator 側が box 分岐に入ると point が黙って捨てられていた。UI 層（`select_sam2_prompt`）は正常で、欠陥は propagator のみ。

**対処法（方針A: box 群と point 群をそれぞれ追跡対象 obj として登録し全て OR 統合）**:
- `target_object_ids` 構築時、`boxes` と `points` が両方あれば point 群用に `point_group_obj_id = len(boxes) + 1` を割り当てて `target_object_ids` に追加する。
- `if boxes:` 登録ブロックで各 box を obj 1..N として登録した後、`point_group_obj_id` があれば `add_new_points_or_box(obj_id=point_group_obj_id, points=..., labels=...)` で point 群（positive=前景／negative=除外）を追加 obj として登録する。
- union ロジックは既に `target_object_ids` を走査して OR 統合するため、追加した point 群 obj も自動で union される。
- `else`（point のみ／単一 box／object_id）分岐は**未変更**で後方互換を維持。

**再発防止**:
- Component 境界の I/O 契約（points/labels/box/boxes を全て受理し漏れなく登録）を崩さない。一方の prompt 種別だけを処理する分岐は片方を黙殺しやすい。
- TDD: boxes+points 併用時に point 群が obj N+1 として登録され union されることを検証する RED テスト（`test_sam2_video_propagator_registers_point_group_with_boxes`）を先に追加してから修正。torch 未導入環境では `monkeypatch` で `torch.inference_mode` のみの最小 stub を注入し、union が排他領域の OR で全面 True になることまで検証する。
- UI 実行時検証（ERR035）: Point mode で positive/negative ラジオ表示、両 label の点登録（Prompt Status 反映）を Playwright で確認済み。




### [ERR042] transparent-background の gradient alpha と SAM2/SAMURAI 二値 mask の合成で黒/白の2値エッジが出る

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | mask guard を適用する全切り抜きで境界に発生 |
| **初回発生日** | 2026-06-15 |
| **関連ファイル** | `pipelines/components/common.py`, `pipelines/components/model_components.py`, `pipelines/components/video_model_components.py`, `config/inference_models.toml`, `gradio_app_sam2_transparent_BG_haystack.py`, `gradio_app_sam2_transparent_BG_haystack_for_Movie.py`, `tests/unit/test_transparent_bg_mask_guard.py` |

**エラー内容**:
切り抜き結果のマスクエッジに2種類が混在する。(1) transparent-background が生成する自然なグラデーション境界と、(2) 黒/白の硬い2値境界。後者は出力が transparent-background と SAM2/SAMURAI mask の**合成**であることを示し、見た目の品質を損なう。

**原因**:
`TransparentBGExtractor.run` の最終 alpha = tb の連続 gradient alpha × **二値 guard**（`dilate_binary_mask` は bool→0/1 を返す）。ERR039（横一直線切れ）対策で導入した guard が二値のため、mask 境界で tb の gradient を硬く切断し2値エッジを生む。guard を単純除去すると ERR039 が再発するため除去不可。

**技術的制約**:
`transparent_background.Remover.process` は画像のみを受け取り、マスクをヒント入力として受け付けない。よって「feather したマスクを tb に入力」を画像前処理（マスク外を中立化）で行うと、tb の salient object 検出がマスク切断線を物体輪郭と誤認し劣化する。

**対処法（union マスクを feather して tb 出力 alpha に乗算 = 二値 guard の feather 版）**:
- 新規 `feather_binary_mask(mask, dilate_size=21, feather_radius=8)`（`pipelines/components/common.py`）を追加。`feather_radius<1` で従来二値（後方互換）。`>=1` で `effective_dilate = max(1, min(dilate_size, feather_radius))` で軽く dilate した base 境界を中心に符号付き距離変換（`cv2.distanceTransform`）で ±feather_radius を 0↔1 に滑らかに遷移させた float32 soft guard を返す。
- **要点**: 遷移帯が mask 境界（= tb 前景 alpha 境界）に重なる必要がある。`dilate_size` を大きく取り過ぎると遷移帯が前景の外側に出て中間 alpha が生まれず2値のままになるため、`effective_dilate` を `feather_radius` 以下に抑える。
- `TransparentBGExtractor.run` に `mask_guard_feather:int=0`、`SAM2GuardFilter.run` に `feather:int=0`、`TransparentBGVideoExtractor.run` に `mask_guard_feather:int=0` を追加し、`>0` で feather guard に分岐。
- 強度は `config/inference_models.toml` の `[[background]]` の `mask_feather`（既定8）で制御し、UI から `bg_entry.get("mask_feather",0)` 経由で渡す（ハードコード回避）。
- 静止画パイプラインは extractor と後段 SAM2GuardFilter が同一 mask で二重 guard になる。feather>0 のとき soft × soft / soft × 二値が2値エッジを再発させるため、**静止画 UI で feather>0 時は `sam2_guard` を enabled=False** にして extractor 段に soft guard を一元化する。動画版は extractor が最終段のため二重適用なし。

**再発防止**:
- mask を guard として乗算する箇所は、二値だと gradient を硬く切る。境界をぼかす必要がある場合は `feather_binary_mask` を使い、`feather_radius` は画像対角線の半分を超えない（推奨4〜16）値にする（過大だと距離変換が飽和し遷移帯が頭打ち）。
- guard の二重適用に注意。soft guard を2回乗算すると遷移帯が再び急峻化する。最終 alpha を出す段に guard を一元化する。
- TDD: feather guard が境界に中間 alpha を生み（2値でない）、feather=0 で従来二値を維持することを検証する RED テストを先に追加。極端な feather_radius・極小 mask・空 mask の範囲保持も検証。
- UI 実行時検証（ERR035）: 配線変更後に静止画版を起動し UI 描画が壊れないことを Playwright で確認。feather の視覚的品質は checkpoints+GPU の実モデル実行が必要なため、単体テスト＋UI 描画検証に留め、実素材での見た目はユーザー GPU 実行で要確認。

### [ERR043] 動画 SAM2 で box+point 併用時、point 群を1つの追加 obj にまとめると複数インスタンスで point が落ちる

| 項目 | 内容 |
|------|------|
| **深刻度** | Medium |
| **頻度** | box が2つ以上 + 補正 point を使う追跡で発生 |
| **初回発生日** | 2026-06-16 |
| **関連ファイル** | `pipelines/components/common.py`, `pipelines/components/video_model_components.py`, `tests/unit/test_common_components.py`, `tests/unit/test_video_pipeline_wiring.py` |

**エラー内容**:
ERR041 の方針A（全 point を末尾の追加 obj としてまとめて登録）では、複数 box（複数インスタンス）に対する補正 point が反映されないことがある。SAM2 は1つの obj_id に1インスタンスの mask しか割り当てられないため、複数インスタンスにまたがる point 群を1 obj にまとめると、最も強い1インスタンス分しか残らず他の point が union から落ちる。

**原因**:
`SAM2VideoPropagator.run` が `point_group_obj_id = len(boxes)+1` で全 point を1つの追加 obj に登録していた。positive 点で別 box を補強したくても、その obj が表現できるインスタンスは1つだけなので補強先が定まらず、negative 点も「どの box の内部をくり抜くか」が曖昧になる。

**対処法（修正1: 最近傍 box 割当 = 方針1）**:
- 新規 `assign_points_to_boxes(points, boxes) -> dict[obj_id, list[point_index]]`（`pipelines/components/common.py`）を追加。各 point を矩形距離（点が box 内なら0、外なら最寄り辺までの L2 二乗）が最小の box（obj_id 1..N）に割り当てる。box が無ければ空辞書、point が無くても全 obj_id を空リストで含む。
- `SAM2VideoPropagator.run` の `if boxes:` 分岐で `point_group_obj_id` を廃止。各 box を `add_new_points_or_box(box=single_box, points=割当点, labels=割当ラベル)` で登録し、割り当てられた point を**その box の object prompt に同梱**する。positive 点は最寄り box を補強、negative 点は box 内部をくり抜く。追加 obj は作らない。
- `else`（point のみ / 単一 box）分岐は未変更で後方互換維持。

**再発防止**:
- SAM2 video の複数インスタンス追跡では「1 obj = 1 インスタンス」を厳守。複数インスタンスにまたがる point を1 obj にまとめない。点は所属インスタンス（最寄り box）の prompt に同梱する。
- TDD: 最近傍割当（box1 が point(1,1,label=1)、box2 が point(4,2,label=0) を受け取り、追加 obj を作らず object_ids が box 分の 1,2 のみ）を検証する RED テストを先に追加。

### [ERR044] 動画 union の早期二値化 + binary OR + 二値 guard が継ぎ目（消える線）を出力に焼き込む

| 項目 | 内容 |
|------|------|
| **深刻度** | High |
| **頻度** | 複数 obj を union する全動画切り抜きで境界（継ぎ目）に発生 |
| **初回発生日** | 2026-06-16 |
| **関連ファイル** | `pipelines/components/common.py`, `pipelines/components/video_common.py`, `pipelines/components/video_model_components.py`, `pipelines/components/model_components.py`, `tests/unit/test_common_components.py`, `tests/unit/test_video_pipeline_wiring.py`, `tests/unit/test_transparent_bg_mask_guard.py` |

**エラー内容**:
複数 obj（複数 box）を union した動画切り抜きで、物体輪郭に沿った細い黒線（消える線）が出る。ERR042 の末端 feather だけでは消えきらない、union 境界由来の継ぎ目。

**原因**:
`SAM2VideoPropagator.run` が各 obj を早期に二値化（`logits>0.0`）し binary OR で union していた。隣接 obj の mask 境界がわずかにずれていると、OR の結果に細い谷（どちらの obj にも属さない継ぎ目）が残り、それが二値 guard を介して tb alpha に黒線として焼き込まれる。二値化を最終段まで遅延しないことが根本原因。

**対処法（修正2: soft 合成＋末端 feather = 根治）**:
- 新規 `stable_sigmoid(x)`（overflow 回避の数値安定 sigmoid, float32[0,1]）と `soft_probability_guard(prob, dilate_size=21, feather_radius=8)`（grayscale `cv2.morphologyEx(MORPH_CLOSE)` で継ぎ目谷を橋渡し → `cv2.GaussianBlur` で末端 feather、二値化なし、[0,1] float32）を `pipelines/components/common.py` に追加。
- `SAM2VideoPropagator.run`: 各 obj を二値化せず `stable_sigmoid(logits)` で確率化し `np.maximum` で union（forward/reverse の重複 frame も max 統合）。継ぎ目の谷は二値の「穴」ではなく確率の連続値になる。
- 契約を float32[0,1] のまま疎通。`build_frame_mask_sequence`（`video_common.py`）は float 入力を `clip(0,1).astype(float32)` 保持、bool 入力は従来 bool（後方互換）。`render_tracking_overlay_frame` は float mask を `>=0.5` 閾値、bool は従来通り。
- `TransparentBGExtractor.run`（`model_components.py`）を float(soft確率)/bool 両対応。float は `mask_soft=clip(0,1)`・`mask_binary=soft>=0.5`（has_mask/bbox 判定用）で、guard は `soft_probability_guard`（closing で継ぎ目谷を埋め、feather で末端をぼかす）。bool は従来パス。
- **後方互換**: `max(probA,probB)>=0.5 ⟺ binaryA OR binaryB` のため、閾値0.5判定の前景領域は従来 OR と一致。差は継ぎ目が黒線でなく中間 alpha になる点のみ。

**再発防止**:
- mask の二値化は**最終段まで遅延**する。中間表現（union、契約、guard）は soft 確率[0,1]（float32）で持ち、継ぎ目を二値の穴にしない。
- 複数 obj の union は binary OR ではなく確率の `np.maximum`。logit→確率は `stable_sigmoid` で overflow を避ける。
- 継ぎ目谷は closing（`MORPH_CLOSE`）で橋渡しし、末端は gaussian feather する。`soft_probability_guard` を使い、二値 guard を soft 確率 mask に乗算しない。
- TDD: soft union が float[0,1] を保ち閾値0.5で全面被覆すること、soft guard が中間値を保持し継ぎ目谷を橋渡しすること、extractor が float mask で中間 alpha を出し bbox は閾値0.5で決まることを RED テストで先に検証。
- 視覚品質（継ぎ目線の消滅・末端の自然さ）は checkpoints+GPU の実モデル実行が必要。単体テスト＋UI 描画検証に留め、実素材はユーザー GPU 実行で要確認。

