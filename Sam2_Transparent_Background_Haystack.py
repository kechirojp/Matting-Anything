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
# # SAM2 + transparent-background Haystack Mask Union Pipeline
#
# このノートブックは **Jupytext の `.py` を正本** として管理します。`.ipynb` は次のコマンドで生成します。
#
# ```powershell
# .venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack.py
# ```
#
# 実装本体は [gradio_app_sam2_transparent_BG_haystack.py](./gradio_app_sam2_transparent_BG_haystack.py) と
# [pipelines/](./pipelines/) に集約しています。
#
# **今回の目的:**
# - Text Prompt から bbox 候補を作る
# - SAM2 multimask 候補を表示する
# - 複数 mask を union して「人 + ドラム」「人 + 自転車」のような複合対象を扱う
# - union mask を transparent-background MatteExtractor に渡す

# %%
# ============================================================
# Cell 1: 依存関係インストール
# ============================================================
# -q は使わず、CUDA / build エラーを見える状態にします。
import os
import sys

!{sys.executable} -m pip install gradio==5.9.1
!{sys.executable} -m pip install haystack-ai==2.29.0
!{sys.executable} -m pip install transparent-background
!{sys.executable} -m pip install pymatting
!{sys.executable} -m pip install git+https://github.com/facebookresearch/sam2.git
!{sys.executable} -m pip install opencv-python-headless pillow numpy
!{sys.executable} -m pip install "transformers>=4.26.0" addict yapf timm supervision pycocotools

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
GROUNDING_DINO_CKPT_PATH = CKPT_ROOT / "groundingdino_swint_ogc.pth"


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
fetch_if_missing(
    GROUNDING_DINO_CKPT_PATH,
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
)

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"SAM2_CKPT_PATH = {SAM2_CKPT_PATH}")
print(f"GROUNDING_DINO_CKPT_PATH = {GROUNDING_DINO_CKPT_PATH}")
print(f"TB_CKPT_DIR = {TB_CKPT_DIR}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")

# %%
# ============================================================
# Cell 2.5: Colab / CUDA / checkpoint 診断
# ============================================================
print("=== Runtime diagnostics ===")
if IS_COLAB:
    print("nvidia-smi:")
    os.system("nvidia-smi")
else:
    print("Not running on Colab; skipping nvidia-smi.")

try:
    import torch

    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
except ModuleNotFoundError as exc:
    print(f"torch import failed: {exc}")


try:
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"sam2 package = {sam2.__file__}")
    print("sam2 image imports = OK")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "SAM2 package import failed before launching Gradio. "
        "Cell 1 の install をエラーが見える状態（-q なし）で再実行し、"
        f"`{sys.executable} -m pip install git+https://github.com/facebookresearch/sam2.git` が成功したことを確認してください. "
        "その後、この診断セルと Gradio 起動セルを再実行してください。"
    ) from exc


def print_checkpoint_diagnostics(label: str, path: Path) -> None:
    exists = path.exists()
    size_mb = path.stat().st_size / (1024 * 1024) if exists and path.is_file() else None
    is_drive_path = "/drive/" in str(path).replace("\\", "/").lower()
    print(f"{label}: path={path}, exists={exists}, size_mb={size_mb}, drive_path={is_drive_path}")


print_checkpoint_diagnostics("SAM2", SAM2_CKPT_PATH)
print_checkpoint_diagnostics("GroundingDINO", GROUNDING_DINO_CKPT_PATH)
print(
    "GPU policy: SAM2 / GroundingDINO heavy inference requires CUDA by default. "
    "Set MATTING_ANYTHING_ALLOW_CPU=1 only for emergency CPU fallback."
)
CPU_FALLBACK_ALLOWED = os.environ.get("MATTING_ANYTHING_ALLOW_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
try:
    CUDA_AVAILABLE = bool(torch.cuda.is_available())
    TORCH_CUDA_VERSION = torch.version.cuda
except NameError:
    CUDA_AVAILABLE = False
    TORCH_CUDA_VERSION = None
if not CUDA_AVAILABLE and not CPU_FALLBACK_ALLOWED:
    raise RuntimeError(
        "CUDA GPU runtime is required before launching Gradio. "
        "Colab の ランタイム > ランタイムのタイプを変更 で T4 GPU 以上を選び、"
        "ランタイム再起動後に install cell から再実行してください。 "
        f"torch.version.cuda={TORCH_CUDA_VERSION}. "
        "CPU 緊急回避を意図する場合のみ MATTING_ANYTHING_ALLOW_CPU=1 を設定してください。"
    )

# %%
# ============================================================
# Cell 3: Haystack 版 Gradio アプリ起動
# ============================================================
APP_PATH = PROJECT_ROOT / "gradio_app_sam2_transparent_BG_haystack.py"
assert APP_PATH.exists(), f"アプリが見つかりません: {APP_PATH}"

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["SAM2_CKPT_PATH"] = str(SAM2_CKPT_PATH)
os.environ["SAM2_CONFIG_NAME"] = SAM2_CONFIG_NAME
os.environ["GROUNDING_DINO_CKPT_PATH"] = str(GROUNDING_DINO_CKPT_PATH)

IS_COLAB = "google.colab" in sys.modules
SHARE_FLAG = "--share" if IS_COLAB else ""

!{sys.executable} "{APP_PATH}" {SHARE_FLAG}
