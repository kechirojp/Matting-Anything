# MAM を始める

このドキュメントでは MAM（Matting Anything Model）の再現手順を説明します。

## 目次

- [データ準備](#データ準備)
- [MAM の学習](#mam-の学習)
- [評価](#評価)
- [Google Colab での推論（Matting_Anything.ipynb）](#google-colab-での推論matting_anythingipynb)

---

## データ準備

### 学習データ

学習には以下のデータセットを使用しています。

**フォアグラウンド（前景）:**

| データセット | 内容 |
|------------|------|
| AIM | 高品質なインスタンスマット |
| Distinctions-646 | 多様な前景画像 |
| AM2K | 動物マッティング |
| Human-2K | 人物マッティング |
| RefMatte | 参照ベースのマット |

**バックグラウンド（背景）:**

| データセット | 内容 |
|------------|------|
| COCO | 実世界の多様な背景 |
| BG20K | 高品質な合成背景 |

### 評価データ

| ベンチマーク | 種別 | 説明 |
|------------|------|------|
| PPM-100 | セマンティックマッティング | ポートレート |
| AM2K | セマンティックマッティング | 動物 |
| PM-10K | セマンティックマッティング | 多様なクラス |
| RWP636 | インスタンスマッティング | 実世界ポートレート |
| HIM2K | インスタンスマッティング | 人物（自然光・合成） |
| RefMatte-RW100 | 参照マッティング | テキスト/ボックス/点プロンプト |

> 学習・評価に使用するデータセットのパスは各 `config/*.toml` ファイルで指定します。

---

## MAM の学習

### 前提条件

- [INSTALL_ja.md](INSTALL_ja.md) の手順で環境を構築してください。
- 上記のデータセットを準備し、`config/*.toml` にパスを記載してください。
- 8 GPU（分散学習）を推奨します。

### 学習コマンド

**SAM ViT-B バックボーンで学習（8 GPU）:**

```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py --config config/MAM-ViTB-8gpu.toml
```

**SAM ViT-L バックボーンで学習（8 GPU）:**

```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py --config config/MAM-ViTL-8gpu.toml
```

**SAM ViT-H バックボーンで学習（8 GPU）:**

```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py --config config/MAM-ViTH-8gpu.toml
```

---

## 評価

### 前提条件

- [INSTALL_ja.md](INSTALL_ja.md) の手順で環境を構築してください。
- 評価対象のベンチマークデータセットを準備し、`config/*.toml` にパスを記載してください。

### SAM ViT-B チェックポイントによる各ベンチマーク評価

#### PPM-100

```bash
python inference_benchmark.py --config config/MAM-ViTB-8gpu.toml \
    --checkpoint checkpoints/mam_vitb.pth \
    --benchmark ppm100 --output outputs/ppm100 --postprocess

python evaluation/evaluation_ppm100.py --pred-dir outputs/ppm100
```

#### AM2K

```bash
python inference_benchmark.py --config config/MAM-ViTB-8gpu.toml \
    --checkpoint checkpoints/mam_vitb.pth \
    --benchmark am2k --output outputs/am2k --postprocess

python evaluation/evaluation_am2k.py --pred-dir outputs/am2k
```

#### PM-10K

```bash
python inference_benchmark.py --config config/MAM-ViTB-8gpu.toml \
    --checkpoint checkpoints/mam_vitb.pth \
    --benchmark pm10k --output outputs/pm10k --postprocess

python evaluation/evaluation_pm10k.py --pred-dir outputs/pm10k
```

#### RWP636

```bash
python inference_benchmark.py --config config/MAM-ViTB-8gpu.toml \
    --checkpoint checkpoints/mam_vitb.pth \
    --benchmark rwp636 --output outputs/rwp636 --postprocess

python evaluation/IMQ_quick_rwp.py path/to/outputs/rwp636 path/to/RealWorldPortrait-636/alpha
```

#### HIM2K（自然光 / 合成）

```bash
# 自然光
python inference_benchmark.py --config config/MAM-ViTB-8gpu.toml \
    --checkpoint checkpoints/mam_vitb.pth \
    --benchmark him2k --output outputs/him2k/ --maskguide

python evaluation/IMQ.py path/to/outputs/him2k path/to/HIM2K/alphas/natural/

# 合成
python inference_benchmark.py --config config/MAM-ViTB-8gpu.toml \
    --checkpoint checkpoints/mam_vitb.pth \
    --benchmark him2k_comp --output outputs/him2k_comp --maskguide

python evaluation/IMQ.py path/to/outputs/him2k_comp path/to/HIM2K/alphas/comp/
```

#### RefMatte-RW100

```bash
python inference_benchmark.py --config config/MAM-ViTB-8gpu.toml \
    --checkpoint checkpoints/mam_vitb.pth \
    --benchmark rw100 --output outputs/rw100 \
    --maskguide --prompt text/box/point

python evaluation/evaluation_refmatte.py --pred-dir outputs/rw100
```

> 評価結果は `outputs/` 以下に保存されます（git 管理外）。

---

## Google Colab での推論（Matting_Anything.ipynb）

ローカル環境を構築せずに MAM を試したい場合は、付属のノートブックを使用してください。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/softmurata/colab_notebooks/blob/main/computervision/Matting_Anything.ipynb)

### ノートブックの概要

`Matting_Anything.ipynb` は Google Colab 上で MAM の Gradio WebUI をエンドツーエンドで実行するためのノートブックです。  
セルを上から順番に実行するだけで推論環境が構築されます。

### 実行手順

**ステップ 1: GPU ランタイムの確認**

> Colab メニュー > **ランタイム** > **ランタイムのタイプを変更** > ハードウェア アクセラレータ > **GPU (T4)**

**ステップ 2: ノートブックを開く**

上の「Open in Colab」バッジをクリックするか、Colab で `Matting_Anything.ipynb` を開いてください。

**ステップ 3: セルを順番に実行**

| # | セル内容 | 詳細 |
|---|---------|------|
| 1 | **GPU / CUDA 環境チェック** | GPU が正しく認識されているか確認。警告が出た場合はランタイムを GPU に変更してください。 |
| 2 | **Google Drive のマウント** | チェックポイントを永続化するために Google Drive をマウントします。 |
| 3 | **依存パッケージのインストール** | リポジトリのクローン、requirements.txt、segment-anything、GroundingDINO（CUDA ビルド）、diffusers をインストールします。数分かかります。 |
| 4 | **GroundingDINO チェックポイントのダウンロード** | `groundingdino_swint_ogc.pth` を Google Drive の `AI_picasso/Matting-Anything/checkpoints/` に保存します。 |
| 5 | **SAM チェックポイントディレクトリの作成** | ディレクトリを作成します（`sam_vit_b_01ec64.pth` は別途配置が必要）。 |
| 6 | **Gradio WebUI の起動** | `--share` オプション付きで起動し、外部からアクセス可能な URL が発行されます。 |

### Google Drive へのチェックポイント配置

セル 3 のインストール完了後、MAM のチェックポイントを Google Drive に手動でアップロードしてください。

```
マイドライブ/
└── AI_picasso/
    └── Matting-Anything/
        └── checkpoints/
            ├── groundingdino_swint_ogc.pth   ← セル 4 で自動ダウンロード
            ├── mam_vitb.pth                   ← 手動でアップロード（必須）
            ├── mam_vitl.pth                   ← 任意
            └── mam_vith.pth                   ← 任意
```

MAM チェックポイントのダウンロード元: [Google Drive](https://drive.google.com/drive/folders/1Bor2jRE0U-U6PIYaCm6SZY7qu_c1GYfq?usp=sharing)

### WebUI の操作方法

セル 6 を実行すると Gradio の公開 URL が表示されます。  
ブラウザで URL にアクセスして以下のプロンプト種別を選択してください。

| プロンプト | 操作 |
|----------|------|
| `scribble_point` | ターゲット上の1点をクリック |
| `scribble_box` | 左上・右下の2点でバウンディングボックスを指定 |
| `text` | テキストでターゲットを指定 |

### 注意事項

- Colab の無料枠では GPU セッションに利用時間の制限があります。
- セッションが切断されると `/content/` 以下のデータは消えますが、Google Drive 上のチェックポイントは保持されます。
- GroundingDINO の CUDA ビルドに失敗した場合でも CPU モードで推論は継続できます（低速）。
- 詳細なインストール手順は [INSTALL_ja.md](INSTALL_ja.md) を参照してください。

---

## Haystack パイプラインでの推論

既存の Gradio アプリはそのまま残し、Haystack 2.x の Component / Pipeline で推論 DAG を組んだ entrypoint を追加しています。

### 依存関係

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` には `haystack-ai==2.29.0` を固定しています。

### MAM + GroundingDINO / Scribble

```powershell
.venv\Scripts\python.exe gradio_app_haystack.py
```

既定ポートは `7861` です。

### SAM2 + transparent-background

```powershell
.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack.py
```

既定ポートは `7862` です。

### 単体テスト

```powershell
.venv\Scripts\python.exe -m pytest -m "not integration" -v
```

GPU、checkpoint、外部モデルが必要な検証は `@pytest.mark.integration` として分離しています。
