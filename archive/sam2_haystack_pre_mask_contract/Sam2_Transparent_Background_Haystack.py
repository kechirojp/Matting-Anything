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
# # SAM2 + transparent-background Haystack Pipeline (Google Colab + Gradio 5)
#
# このノートブックは **Jupytext の `.py` を正本** として管理します。`.ipynb` は次のコマンドで生成します。
#
# ```powershell
# .venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py
# ```
#
# 実装本体は [gradio_app_sam2_transparent_BG_haystack.py](./gradio_app_sam2_transparent_BG_haystack.py) と [pipelines/](./pipelines/) に集約しています。
#
# **手順:**
# 1. 依存関係をインストール
# 2. Google Drive / `PROJECT_ROOT` / checkpoint 配置を解決
# 3. Haystack 版 Gradio アプリを起動

# %%
# ============================================================
# Cell 1: 依存関係インストール
# ============================================================
# -q は使わず、インストールエラーを見える状態にします。
import os
import sys

!{sys.executable} -m pip install gradio==5.9.1
!{sys.executable} -m pip install haystack-ai==2.29.0
!{sys.executable} -m pip install transparent-background
!{sys.executable} -m pip install pymatting
!{sys.executable} -m pip install git+https://github.com/facebookresearch/sam2.git
!{sys.executable} -m pip install opencv-python-headless pillow numpy

print("Install done")

# %%
# ============================================================
# Cell 2: Google Drive マウント + プロジェクトパス設定 + チェックポイント配置
# ============================================================
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

CKPT_ROOT = PROJECT_ROOT / "checkpoints"
SAM2_CKPT_DIR = CKPT_ROOT / "SAM2"
TB_CKPT_DIR = CKPT_ROOT / "transparent_BG"
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
for directory in (SAM2_CKPT_DIR, TB_CKPT_DIR, INPUT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SAM2_CKPT_PATH = SAM2_CKPT_DIR / "sam2.1_hiera_large.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"


def fetch_if_missing(path: Path, url: str) -> None:
    if path.exists():
        print(f"Found: {path.relative_to(PROJECT_ROOT)}")
        return
    print(f"Downloading: {path.name}")
    rc = os.system(f'wget -O "{path}" "{url}"')
    if rc != 0 or not path.exists():
        raise RuntimeError(f"Download failed: {url}")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


fetch_if_missing(
    SAM2_CKPT_PATH,
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
)

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"SAM2_CKPT_PATH = {SAM2_CKPT_PATH}")
print(f"TB_CKPT_DIR = {TB_CKPT_DIR}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")

# %%
# ============================================================
# Cell 3: Haystack 版 Gradio アプリ起動
# ============================================================
APP_PATH = PROJECT_ROOT / "gradio_app_sam2_transparent_BG_haystack.py"
assert APP_PATH.exists(), f"アプリが見つかりません: {APP_PATH}"

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["SAM2_CKPT_PATH"] = str(SAM2_CKPT_PATH)
os.environ["SAM2_CONFIG_NAME"] = SAM2_CONFIG_NAME

IS_COLAB = "google.colab" in sys.modules
SHARE_FLAG = "--share" if IS_COLAB else ""

!{sys.executable} "{APP_PATH}" {SHARE_FLAG}