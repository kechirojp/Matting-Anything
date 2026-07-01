# クイックスタート — ローカル RTX 4090 / uv 環境

ローカル RTX 4090 で **gradio.live トンネルを使わず直結**（`127.0.0.1`）で動かすための手順です。
トンネルを使わないため、長時間処理での「Connection errored out.（SSE 切断）」が原理的に起きません（ERR058 / ERR059 参照）。

- パッケージ管理: **uv**（このプロジェクトは `.venv + pip` ではなく uv 運用）
- Python: **3.11**（`.python-version` で固定）
- torch: **2.6.0+cu124**（CUDA 12.4 ビルド）
- 同梱モデル: SAM2 / SAMURAI、GroundingDINO、BEN2、transparent-background
- 主なアプリ: **`gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py`**
  （DEVA方式 SAM2 + BEN2/TB ハイブリッド動画αマット・手動プロンプト版。
  詳細は `QUICKSTART_DEVA_BEN2_TB_HYBRID_MANUAL.md` 参照）

---

## 0. 前提

- Windows + PowerShell
- NVIDIA RTX 4090（ドライバ CUDA 12.x 対応）
- [uv](https://docs.astral.sh/uv/) がインストール済み（未導入なら `winget install astral-sh.uv` 等）
- リポジトリ直下（`pyproject.toml` がある場所）で作業する

---

## 1. 環境構築（初回のみ）

### 1-1. コア依存を導入

```powershell
uv sync
```

`torch 2.6.0+cu124` / `torchvision 0.21.0+cu124` / `transparent-background` / `ben2` などが `.venv` に入ります。

### 1-2. SAM-2 を editable で導入（別ステップ）

SAM-2 は `setup.py` が import 時に torch を参照するため `--no-build-isolation` が必須です。
CUDA 拡張（nvcc ビルド）は optional なので `SAM2_BUILD_CUDA=0` でスキップします。

```powershell
$env:SAM2_BUILD_CUDA = "0"
uv pip install --python .venv\Scripts\python.exe --no-build-isolation -e samurai/sam2
```

> `--python .venv\Scripts\python.exe` を必ず付けること。付けないとグローバルの uv 管理 Python を掴み
> 「externally managed」で失敗します。

---

## 2. 動作確認

### 2-1. CUDA と主要 import

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期待出力:

```text
2.6.0+cu124 True NVIDIA GeForce RTX 4090
```

### 2-2. アプリ起動スモーク

```powershell
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py --help
```

### 2-3. テスト（非 integration）

```powershell
.venv\Scripts\python.exe -m pytest -m "not integration" -q
```

---

## 3. アプリ起動（本番・ローカル直結）

`--share` を **付けない**でください（付けるとトンネル経由になり SSE 切断が再発します）。
既定で `server_name=127.0.0.1` です。

```powershell
# DEVA方式 SAM2 + BEN2/TB ハイブリッド動画αマット（手動プロンプト版）— 既定 port 7866
.venv\Scripts\python.exe gradio_app_sam2_ben2_tb_hybrid_deva_manual_for_Movie.py
```

ブラウザで表示された `http://127.0.0.1:<port>` を開きます。詳しい使い方（Canvas での box/point 指定、
DEVA周期再検出、Alpha合成方式など）は `QUICKSTART_DEVA_BEN2_TB_HYBRID_MANUAL.md` を参照してください。

### 主な起動オプション

| オプション | 説明 |
|------------|------|
| `--share` | gradio.live 公開リンク（**ローカル直結では使わない**） |
| `--debug` | Gradio debug モード |
| `--server-name` | バインドするホスト（既定 `127.0.0.1`） |
| `--server-port` | ポート番号 |

---

## 4. 補足・トラブルシュート

- **GroundingDINO は本アプリで必須**です（DEVA周期再検出・Text Prompt・「Text Prompt から box を自動作成」機能に使用）。
  CUDA custom ops（`ms_deform_attn`）は未導入でも pure PyTorch フォールバックで動作しますが、
  検出が遅くなります。高速化したい場合は `BUILD_WITH_CUDA=True` で `GroundingDINO` を
  ビルドしてください（`INSTALL.md` 参照）。
- **`flet` の `UserWarning`** は transparent-background の GUI モード由来で無害です。
- 依存を変えたら再度 `uv sync`。SAM-2 を入れ直す場合は手順 1-2 を再実行（`--no-build-isolation` 必須）。
- **BEN2 チェックポイントはローカル優先**です。`config/route_a.toml` の `ben2_checkpoint_path`（既定 `checkpoints/BEN2`）に重みがあればそれを使用し、無ければ初回のみ自動ダウンロードして同じ場所へ保存（永続化）します。
- **transparent-background の重み**（`checkpoints/transparent_BG/`）も初回推論時に自動ダウンロードされます。
- 詳細な背景・既知の落とし穴は `ERROR_LOG.md` の **ERR058 / ERR059**、設定の正本は `pyproject.toml` を参照。

---

## 5. なぜトンネルを使わないのか（要点）

数分かかる処理を 1 本の同期リクエスト（長時間 SSE）として無料 gradio.live トンネル越しに保持すると、
トンネルの総接続時間上限で切断され全出力が「Error」になります（ERR048〜058）。
**ローカル 4090 で `127.0.0.1` 直結**にすればトンネル自体が無くなり、この切断クラスは原理的に発生しません。
