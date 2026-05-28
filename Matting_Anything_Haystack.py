# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Matting-Anything Haystack Pipeline (Google Colab + Gradio 5)
#
# このノートブックは **Jupytext の `.py` を正本** として管理します。`.ipynb` は次のコマンドで生成します。
#
# ```powershell
# .venv\Scripts\python.exe -m jupytext --to ipynb Matting_Anything_Haystack.py
# ```
#
# 実装本体は [gradio_app_haystack.py](./gradio_app_haystack.py) と [pipelines/](./pipelines/) に集約しています。
#
# **手順:**
# 1. GPU / CUDA 環境を確認
# 2. Google Drive / `PROJECT_ROOT` を解決
# 3. 依存関係をインストール
# 4. checkpoint 配置を確認
# 5. Haystack 版 Gradio アプリを起動

# %%
# ======================================================
# Cell 1: GPU / CUDA 環境チェック
# ======================================================
import os
import subprocess
import sys

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode != 0:
    print("=" * 60)
    print("警告: GPU が検出されませんでした")
    print("Colab メニュー > ランタイム > ランタイムのタイプを変更 > GPU を選択してください")
    print("=" * 60)
else:
    name_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"GPU 検出: {name_result.stdout.strip()}")

cuda_path = "/usr/local/cuda"
if os.path.exists(cuda_path):
    os.environ["CUDA_HOME"] = cuda_path
    print(f"CUDA_HOME={cuda_path}")
else:
    print(f"{cuda_path} が見つかりません")

try:
    import torch

    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
except Exception as exc:
    print(f"torch 確認は依存関係インストール後に再確認してください: {exc}")

# %%
# ======================================================
# Cell 2: Google Drive マウント + PROJECT_ROOT 解決
# ======================================================
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules
COLAB_DRIVE_PROJECT = Path("/content/drive/MyDrive/AI_picasso/Matting-Anything")

if IS_COLAB:
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        print("Google Drive mounted at /content/drive")
    else:
        print("Google Drive already mounted")


def detect_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    if IS_COLAB and COLAB_DRIVE_PROJECT.exists():
        return COLAB_DRIVE_PROJECT
    return Path.cwd().resolve()


PROJECT_ROOT = detect_project_root()
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
if IS_COLAB and PROJECT_ROOT == COLAB_DRIVE_PROJECT:
    os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT = {PROJECT_ROOT}")

# %%
# ======================================================
# Cell 3: 依存関係インストール
# ======================================================
# -q は使わず、CUDA / build エラーを見える状態にします。
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")

%cd {PROJECT_ROOT}
!{sys.executable} -m pip install -r requirements.txt
!{sys.executable} -m pip install -e segment-anything
!{sys.executable} -m pip uninstall groundingdino -y
!{sys.executable} -m pip install -e GroundingDINO --no-build-isolation
!{sys.executable} -m pip install --upgrade diffusers[torch]

print("Install done")

# %%
# ======================================================
# Cell 4: checkpoint 配置確認
# ======================================================
CKPT_ROOT = PROJECT_ROOT / "checkpoints"
GROUNDING_DINO_CKPT = CKPT_ROOT / "groundingdino_swint_ogc.pth"
MAM_CKPT = CKPT_ROOT / "mam_vitb.pth"
SAM_CKPT_DIR = PROJECT_ROOT / "segment-anything" / "checkpoints"
SAM_CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

if not GROUNDING_DINO_CKPT.exists():
    !wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P "{CKPT_ROOT}"

print(f"GroundingDINO: {GROUNDING_DINO_CKPT} ({GROUNDING_DINO_CKPT.exists()})")
print(f"MAM ViT-B     : {MAM_CKPT} ({MAM_CKPT.exists()})")
print(f"SAM ckpt dir  : {SAM_CKPT_DIR}")
print("MAM と SAM v1 checkpoint は未配置なら手動配置してください。")

# %%
# ======================================================
# Cell 5: Haystack 版 Gradio アプリ起動
# ======================================================
APP_PATH = PROJECT_ROOT / "gradio_app_haystack.py"
assert APP_PATH.exists(), f"アプリが見つかりません: {APP_PATH}"

IS_COLAB = "google.colab" in sys.modules
SHARE_FLAG = "--share" if IS_COLAB else ""

!{sys.executable} "{APP_PATH}" {SHARE_FLAG}