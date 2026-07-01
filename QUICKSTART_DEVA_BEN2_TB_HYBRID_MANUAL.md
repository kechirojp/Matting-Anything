# Quickstart: DEVA方式 BEN2/TB ハイブリッド動画背景除去（手動プロンプト版）

このクイックスタートは `gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py` 用です。

`gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py`（Text Prompt のみ）に、**SAM2.1 の手動 box / point（positive・negative）プロンプト**を追加した版です。

## 何をするアプリか

先頭フレーム（または指定した検出起点フレーム）で、対象を **box で囲む**か **point（positive/negative）でヒントを与える**と、DEVA 方式（周期再検出 × SAM2 伝播 × consensus 統合）で対象を追跡します。追跡した人物 mask 内だけ `transparent-background`、それ以外は BEN2 で alpha を作り、最後に 2 つの alpha を合成して RGBA / Alpha / Preview 動画を出力します。

## 2 つの動作モード

| モード | Text Prompt | Canvas の box/point | 挙動 |
| --- | --- | --- | --- |
| **A. 手動 seed のみ** | 空 | 必須 | 先頭フレームの box/point から SAM2 伝播（再検出なし） |
| **B. ハイブリッド（推奨）** | 入力 | 任意 | 先頭を box/point で精緻 seed → 以降 text で周期再検出し「はがれ復帰」 |

- **point の positive/negative** で前景・背景を補正できます（negative で誤検出領域を除外）。
- 手動 point は**先頭（起点）フレームの seed にのみ**反映されます（対象が動くため後続へ再投影しません）。
- **point は box とあわせて**指定してください（box で囲み、point で補正）。point 単独では追跡できません。

## 起動

```powershell
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py
```

既定:

- URL: `http://127.0.0.1:7866`
- Gradio queue: 有効
- 出力先: `outputs\`

ポートを変える場合:

```powershell
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py --server-port 7867
```

## 最小手順（モードB: ハイブリッド・推奨）

1. `Input Video` に動画をアップロードする（第 1 フレームが Canvas に自動表示される）。
2. `Input Mode` を `box` にし、`SAM2 Prompt Canvas` で対象の対角 2 点をクリックして囲む。**複数の対象を追いたい場合は、続けて別の対象を対角 2 点で囲むと box を何個でも追加できる**（各 box は番号・色分けで表示され、別オブジェクトとして追跡される）。
3. 必要に応じて `Input Mode` を `point` にし、`Point Label` を `positive`（前景）/`negative`（背景）で切り替えてクリック補正する。
4. `Text Prompt` に `person` または `person playing drums` などを入力する（周期再検出に使用）。
5. 最初は `最大処理フレーム数 = 30`、`フレーム間引きステップ = 1` のままにする。
6. `Alpha 合成方式` はまず `lighten / 比較明（推奨）` を選ぶ。
7. `DEVA + 手動 Hybrid 実行` を押す。
8. `Tracking Overlay` で SAM2 mask が対象に合っているか確認する。
9. `RGBA Video` / `Alpha Video` / `Preview Video` を確認する。

## 最小手順（モードA: 手動 seed のみ）

1. `Input Video` に動画をアップロードする。
2. `SAM2 Prompt Canvas` で対象を box で囲む（必要なら point で補正）。
3. `Text Prompt` は**空のまま**にする。
4. `DEVA + 手動 Hybrid 実行` を押す。

> モードA では再検出を行わないため、対象が大きく変形・遮蔽される動画では追跡がはがれることがあります。安定させたい場合はモードB を使ってください。

## 手動プロンプトの補助機能

| UI | 役割 |
| --- | --- |
| `第1フレームを再取得` | Canvas を先頭フレームに戻す |
| `Prompt をクリア` | 描いた box/point をすべて消す |
| `Extend Left/Right/Top/Bottom` | 直近に描いた box を画面端まで広げる |
| 複数 box | box を続けて囲むと何個でも追加でき、各 box を別オブジェクトとして同時に追跡（番号・色分け表示） |
| `Prompt 編集（個別削除）` | 誤って追加した point/bbox だけを選んで削除 |
| `Optional: Text Prompt から box を自動作成` | GroundingDINO で候補 bbox を作り Canvas にコピー（複数選択で union） |

## 主要パラメータ

| UI | 役割 | 初期の目安 |
| --- | --- | --- |
| `Input Mode` | box で囲む / point でヒント | `box` |
| `Point Label` | point の前景(positive)/背景(negative) | `positive` |
| `Text Prompt` | モードB の周期再検出に使う（空でモードA） | `person`（任意） |
| `再検出周期（frames）` | GroundingDINO→SAM2 検出島を走らせる周期（モードB のみ） | 8〜15 |
| `未検出保持回数` | 連続未検出 track を削除するまでの回数 | 2〜4 |
| `consensus IoU しきい値` | 伝播 mask と再検出 mask の同一判定 | 0.4〜0.6 |
| `Alpha 合成方式` | BEN2 alpha と TB alpha の重ね方 | `lighten / 比較明` |
| `transparent-background mode` | TB の推論モード | `base` |
| `TB crop padding(px)` | 人物 mask bbox の外側余白 | 30〜60 |
| `人物領域 feather(px)` | BEN2/TB 切替境界のぼかし幅 | 8〜16 |
| `人物領域 dilate(px)` | TB 担当領域を外側へ広げる量 | 0〜12 |
| `TB mask guard dilate(px)` | TB alpha の mask 外漏れ許容量 | 21 |
| `TB mask guard feather(px)` | TB guard の境界ぼかし | 0、必要時 4〜12 |
| `検出起点フレーム（サンプリング後 index）` | 被写体が最大に映るフレームで seed / 検出を開始 | 0（先頭） |

## 検出起点フレーム（被写体が最大に映るフレームで seed）

被写体が動画の途中で一番大きく映る場合、先頭フレームで box/point を描くと小さく映った状態で seed され精度が落ちます。`検出起点フレーム（サンプリング後 index）` を、被写体が最大に映るフレームに設定すると、**Canvas がそのフレームに張り替わる**ので、その位置で box/point を描けます。起点フレームより前のフレームは双方向逆伝播で追跡されます。

手順:

1. アコーディオン `検出起点フレーム（被写体が最大に映るフレームで seed）` を開く。
2. `検出起点フレーム（サンプリング後 index）` を動かす、または `起点フレームを Canvas へ表示` を押す。
3. 張り替わった Canvas で box/point を描く。
4. `DEVA + 手動 Hybrid 実行` を押す。

注意:

- インデックスは `フレーム間引きステップ` 適用後（サンプリング後）の位置です。
- 起点フレームを変えると **既存の box/point は初期化されます**（座標が無効になるため）。
- 既定値 `0` は従来どおり先頭フレームから seed します（後方互換）。
- ステータス文字列に `start_frame=` が表示されます。

## Alpha 合成方式の選び方

| 方式 | 使う場面 |
| --- | --- |
| `lighten / 比較明（推奨）` | まず試す。黒抜けを避けやすい。 |
| `person_over_ben2` | 人物・髪の TB alpha を手前として優先したい。 |
| `ben2_over_person` | BEN2 の主体把握を優先したい。 |
| `screen` | 両 alpha を柔らかく加算的に残したい。 |

## よくある調整

### 対象がうまく囲めない / 一部が切れる

- `Extend Left/Right/Top/Bottom` で box を画面端まで広げる。
- `point` の `positive` で拾いたい部分をクリックする。

### 背景の一部を対象と誤認する

- `point` の `negative` で誤認領域をクリックして除外する。

### 髪が切れる

- `人物領域 dilate(px)` を少し上げる。
- `人物領域 feather(px)` を 8〜16 にする。
- `TB crop padding(px)` を増やす。

### 背景が人物側に残る

- `TB mask guard dilate(px)` を下げる。
- `TB mask guard feather(px)` を 0 に近づける。
- `Tracking Overlay` で SAM2 mask が広すぎないか確認する。

### SAM2 mask が対象から外れる（動画途中で追跡がはがれる）

- モードB にして `Text Prompt` を具体化する。
- `再検出周期（frames）` を短くする。
- `Box threshold` / `Text threshold` を調整する。

## 検証コマンド

軽い起動確認:

```powershell
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py --help
```

関連 unit test:

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_hybrid_alpha_components.py tests\unit\test_route_a_deva_hybrid_video_pipeline_wiring.py -q
```

非 integration 全体:

```powershell
.venv\Scripts\python.exe -m pytest -m "not integration" -q
```

## 注意

- `segment-anything\` と `samurai\` は直接変更しない。
- 手動プロンプトの UI を「fixed」として記録するには、短尺動画で Playwright などの実行時検証が必要（ERR035）。
- `outputs\` は出力用で、git 管理対象ファイルを追加しない。
- このアプリはベース版（`gradio_app_sam2_ben2_tb_hybrid_deva_for_Movie.py`）を置き換えるものではなく、手動 seed が必要な場面向けの追加アプリです。ベース版・参考アプリ（`gradio_app_sam2_ben2_route_a_deva_manual_for_Movie.py`）はそのまま残しています。
