# Quickstart: DEVA方式 BEN2/TB ハイブリッド動画背景除去

このクイックスタートは `gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py` 用です。

## 何をするアプリか

Text Prompt で指定した人物を GroundingDINO + SAM2.1 の DEVA 方式で追跡し、人物 mask 内だけ `transparent-background`、それ以外は BEN2 で alpha を作ります。最後に 2 つの alpha を合成して RGBA / Alpha / Preview 動画を出力します。

## 起動

```powershell
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py
```

既定:

- URL: `http://127.0.0.1:7865`
- Gradio queue: 有効
- 出力先: `outputs\`

ポートを変える場合:

```powershell
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py --server-port 7866
```

## 最小手順

1. `Input Video` に動画をアップロードする。
2. `Text Prompt` に `person` または `person playing drums` などを入力する。
3. 最初は `最大処理フレーム数 = 30`、`フレーム間引きステップ = 1` のままにする。
4. `Alpha 合成方式` はまず `lighten / 比較明（推奨）` を選ぶ。
5. `Hybrid 実行` を押す。
6. `Tracking Overlay` で SAM2 mask が人物に合っているか確認する。
7. `RGBA Video` / `Alpha Video` / `Preview Video` を確認する。

## 主要パラメータ

| UI | 役割 | 初期の目安 |
| --- | --- | --- |
| `Text Prompt` | 追跡し、TB を使う人物領域の指定 | `person` |
| `再検出周期（frames）` | GroundingDINO→SAM2 検出島を走らせる周期 | 8〜15 |
| `未検出保持回数` | 連続未検出 track を削除するまでの回数 | 2〜4 |
| `consensus IoU しきい値` | 伝播 mask と再検出 mask の同一判定 | 0.4〜0.6 |
| `Alpha 合成方式` | BEN2 alpha と TB alpha の重ね方 | `lighten / 比較明` |
| `transparent-background mode` | TB の推論モード | `base` |
| `TB crop padding(px)` | 人物 mask bbox の外側余白 | 30〜60 |
| `人物領域 feather(px)` | BEN2/TB 切替境界のぼかし幅 | 8〜16 |
| `人物領域 dilate(px)` | TB 担当領域を外側へ広げる量 | 0〜12 |
| `TB mask guard dilate(px)` | TB alpha の mask 外漏れ許容量 | 21 |
| `TB mask guard feather(px)` | TB guard の境界ぼかし | 0、必要時 4〜12 |
| `検出起点フレーム（サンプリング後 index）` | 被写体が最大に映るフレームから検出・追跡を開始する | 0（先頭） |

## 検出起点フレーム（被写体が最大に映るフレームで検出）

被写体が動画の途中で一番大きく映る場合、先頭フレームから検出すると小さく写った被写体で追跡が始まり精度が落ちます。`検出起点フレーム（サンプリング後 index）` を、被写体が最大に映るフレーム（サンプリング後のインデックス）に設定すると、そのフレームで検出・追跡を開始し、双方向伝播で前方フレームも遡って追跡します。

手順:

1. アコーディオン `検出起点フレーム（被写体が最大に映るフレームで検出）` を開く。
2. `起点フレームを表示` を押してプレビュー画像で被写体が最大に映るフレームを確認する。
3. `検出起点フレーム（サンプリング後 index）` をそのフレームに合わせる。
4. `Hybrid 実行` を押す。

注意:

- インデックスは `フレーム間引きステップ` 適用後（サンプリング後）の位置です。
- 既定値 `0` は従来どおり先頭フレームから検出します（後方互換）。
- `0` より大きい値のときは、起点フレームより前のフレームも backward pass で追跡します。
- ステータス文字列に `start_frame=` が表示されます。

## Alpha 合成方式の選び方

| 方式 | 使う場面 |
| --- | --- |
| `lighten / 比較明（推奨）` | まず試す。黒抜けを避けやすい。 |
| `person_over_ben2` | 人物・髪の TB alpha を手前として優先したい。 |
| `ben2_over_person` | BEN2 の主体把握を優先したい。 |
| `screen` | 両 alpha を柔らかく加算的に残したい。 |

## よくある調整

### 髪が切れる

- `人物領域 dilate(px)` を少し上げる。
- `人物領域 feather(px)` を 8〜16 にする。
- `TB crop padding(px)` を増やす。

### 背景が人物側に残る

- `TB mask guard dilate(px)` を下げる。
- `TB mask guard feather(px)` を 0 に近づける。
- `Tracking Overlay` で SAM2 mask が広すぎないか確認する。

### SAM2 mask が人物から外れる

- `Text Prompt` を具体化する。
- `再検出周期（frames）` を短くする。
- `Box threshold` / `Text threshold` を調整する。
- 複数人物が混在する場合は `top-k` を下げて過検出を抑える。

## 検証コマンド

軽い起動確認:

```powershell
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py --help
```

関連 unit test:

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_hybrid_alpha_components.py tests\unit\test_route_a_deva_hybrid_video_pipeline_wiring.py tests\unit\test_routea_deva_hybrid_movie_sync_wiring.py -q
```

非 integration 全体:

```powershell
.venv\Scripts\python.exe -m pytest -m "not integration" -q
```

## ルートA案のブラー誘導について

既存の RouteA アプリでは、SAM2 由来の mask を膨張してゲートを作り、そのゲート外をブラーしてから BEN2 に渡す処理が入っています。

実装箇所:

- `pipelines\components\ben2_components.py`
  - `_process_union_frame()`
  - `dilate_mask_to_gate(mask, dilation_px)`
  - `blur_background_outside_gate(image_rgb, gate, blur_kernel, blur_sigma, feather_px)`
- `pipelines\components\route_a_common.py`
  - `blur_background_outside_gate()`

今回の hybrid アプリは RouteA の置き換えではなく、人物領域に TB、領域外に BEN2 を割り当てる別方式です。RouteA と比較できるよう、既存 RouteA / DEVA RouteA は残しています。

検出起点フレーム機能の実行時検証（Playwright, ERR035）:

```powershell
.venv\Scripts\python.exe outputs\verify_hybrid_deva_start_frame.py
```

## 注意

- `segment-anything\` と `samurai\` は直接変更しない。
- UI fixed として記録するには、短尺動画で Playwright などの実行時検証が必要。
- `outputs\` は出力用で、git 管理対象ファイルを追加しない。
