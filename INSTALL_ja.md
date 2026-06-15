# インストール手順

## 目次

- [ローカル環境でのセットアップ](#ローカル環境でのセットアップ)
- [事前学習済み重みのダウンロード](#事前学習済み重みのダウンロード)
- [Gradio デモの起動](#gradio-デモの起動)
- [Google Colab でのセットアップ（推奨）](#google-colab-でのセットアップ推奨)

---

## ローカル環境でのセットアップ

動作確認済み環境: Python 3.9 / PyTorch 1.13.1（CUDA 11.7）/ torchvision 0.14.1 / diffusers 0.17.0

> バージョン不一致が発生した場合は `requirements.txt` のバージョン指定を調整してください。

### 1. conda 環境の作成

```bash
conda create --name mam python=3.9 -y
conda activate mam
```

### 2. パッケージと依存関係のインストール

```bash
# ⚠ samurai サブモジュールを含むため --recurse-submodules を必ず付けること
git clone --recurse-submodules https://github.com/SHI-Labs/Matting-Anything
cd Matting-Anything

# すでに clone 済みでサブモジュールが空の場合は以下を実行
# git submodule update --init --recursive

# 主要依存パッケージ
pip install -r requirements.txt

# Segment Anything Model (SAM)
python -m pip install -e segment-anything

# SAM2 / SAMURAI（動画マッティング・テキストプロンプト機能に必要）
# samurai サブモジュール内の sam2 パッケージをインストール
python -m pip install -e samurai/sam2

# GroundingDINO（CUDA 対応ビルド）
# ⚠ CUDA_HOME を必ず自分の環境に合わせて設定すること
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda/   # 例: /usr/local/cuda-11.7
python -m pip install -e GroundingDINO --no-build-isolation

# diffusers（背景生成機能に必要）
pip install --upgrade diffusers[torch]
```

> **GroundingDINO の CUDA ビルドについて**  
> `--no-build-isolation` オプションを付けることで、既にインストール済みの torch/numpy を使って CUDA 拡張をビルドします。  
> `CUDA_HOME` が未設定の場合や nvcc が見つからない場合は CPU モードにフォールバックします（推論は可能ですが低速）。

詳しくは [segment-anything 公式](https://github.com/facebookresearch/segment-anything#installation) および [GroundingDINO 公式](https://github.com/IDEA-Research/GroundingDINO#install) を参照してください。

---

## 事前学習済み重みのダウンロード

### チェックポイントのディレクトリ構成

```
Matting-Anything/
├── checkpoints/
│   ├── groundingdino_swint_ogc.pth   # GroundingDINO（テキストプロンプト用）
│   ├── mam_vitb.pth                  # MAM（SAM ViT-B ベース）← デフォルト
│   ├── mam_vitl.pth                  # MAM（SAM ViT-L ベース）
│   └── mam_vith.pth                  # MAM（SAM ViT-H ベース）
└── segment-anything/
    └── checkpoints/
        └── sam_vit_b_01ec64.pth      # SAM ViT-B（MAM ViT-B 使用時に必要）
```

### GroundingDINO モデルのダウンロード

```bash
mkdir -p checkpoints
cd checkpoints

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### MAM モデルのダウンロード

以下の Google Drive からダウンロードし、`checkpoints/` ディレクトリに配置してください。

- [MAM チェックポイント（Google Drive）](https://drive.google.com/drive/folders/1Bor2jRE0U-U6PIYaCm6SZY7qu_c1GYfq?usp=sharing)

| ファイル名 | SAM バックボーン | 用途 |
|-----------|----------------|------|
| `mam_vitb.pth` | ViT-B | 軽量・高速（デフォルト） |
| `mam_vitl.pth` | ViT-L | 中程度の精度とスピード |
| `mam_vith.pth` | ViT-H | 最高精度（VRAM 多く必要） |

### SAM モデルのダウンロード

```bash
mkdir -p segment-anything/checkpoints
cd segment-anything/checkpoints

# ViT-B（MAM ViT-B と合わせて使用）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Gradio デモの起動

環境構築とチェックポイントの配置が完了したら、以下のコマンドでローカルデモを起動できます。

```bash
python gradio_app.py
```

対応するプロンプト種別：

| プロンプト名 | 操作内容 |
|------------|---------|
| `scribble_point` | ターゲットインスタンス上の1点をクリック |
| `scribble_box` | 左上・右下の2点をクリックしてバウンディングボックスを指定 |
| `text` | テキストでターゲットを指定（「Text Prompt」欄に入力） |

対応する背景合成方法：

| 背景タイプ | 説明 |
|-----------|------|
| `real_world_sample` | `assets/backgrounds/` からランダムに実写背景を選択 |
| `generated_by_text` | Stable Diffusion でテキストから背景を生成（「Background Prompt」欄に入力） |

オンラインでも [HuggingFace デモ](https://huggingface.co/spaces/shi-labs/Matting-Anything) で試せます。

---

## Google Colab でのセットアップ（推奨）

### 概要

本リポジトリには Google Colab 向けノートブック `Matting_Anything.ipynb` が付属しています。  
ローカル環境構築なしで、ブラウザ上で MAM を動かすことができます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/softmurata/colab_notebooks/blob/main/computervision/Matting_Anything.ipynb)

### 事前準備

1. [Google Colab](https://colab.research.google.com/) にアクセス
2. **ランタイムのタイプを変更** → ハードウェア アクセラレータ → **GPU (T4)** を選択

### セル構成と実行手順

| セル | 内容 | 備考 |
|------|------|------|
| **セル 1** | GPU / CUDA 環境チェック | GPU が検出されない場合は警告が表示される |
| **セル 2** | Google Drive のマウント | チェックポイントの永続化に使用 |
| **セル 3** | 依存パッケージのインストール | GroundingDINO の CUDA ビルドを含む（数分かかる） |
| **セル 4** | GroundingDINO チェックポイントのダウンロード | Google Drive の `AI_picasso/Matting-Anything/checkpoints/` に保存 |
| **セル 5** | SAM チェックポイントディレクトリの作成 | `sam_vit_b_01ec64.pth` は別途配置が必要 |
| **セル 6** | Gradio WebUI の起動（`--share`） | 公開 URL が発行される（有効期限あり） |

### チェックポイントの配置（Google Drive）

Google Drive 上の以下のパスにチェックポイントを配置してください。

```
マイドライブ/
└── AI_picasso/
    └── Matting-Anything/
        └── checkpoints/
            ├── groundingdino_swint_ogc.pth   ← セル 4 で自動ダウンロード
            ├── mam_vitb.pth                   ← 手動でアップロード
            ├── mam_vitl.pth                   ← 手動でアップロード（任意）
            └── mam_vith.pth                   ← 手動でアップロード（任意）
```

`segment-anything/checkpoints/sam_vit_b_01ec64.pth` は Google Drive 上のリポジトリに既に配置されているか、または手動でアップロードしてください。

### トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `⚠ GPU が検出されませんでした` | CPU ランタイムで実行中 | ランタイム設定で GPU を選択してセッションを再起動 |
| GroundingDINO CUDA ops のロード失敗 | `CUDA_HOME` が未設定またはビルド失敗 | セル 3 の出力を確認（CPU フォールバックで実行は継続可能） |
| `ModuleNotFoundError: groundingdino` | インストール未完了 | セル 3 を再実行 |
| WebUI 起動後に `mam_vitb.pth` not found | チェックポイントが見つからない | Google Drive のパスとファイル名を確認 |
