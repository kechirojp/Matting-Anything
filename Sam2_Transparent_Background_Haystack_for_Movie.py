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
# # SAM2 + transparent-background Haystack Movie Pipeline
#
# このノートブックは **Jupytext の `.py` を正本** として管理します。`.ipynb` は次のコマンドで生成します。
#
# ```powershell
# .venv\Scripts\python.exe -m jupytext --to ipynb Sam2_Transparent_Background_Haystack_for_Movie.py
# ```
#
# 実装本体は [gradio_app_sam2_transparent_BG_haystack_for_Movie.py](./gradio_app_sam2_transparent_BG_haystack_for_Movie.py) と
# [pipelines/](./pipelines/) に集約しています。
#
# ## UI の使い方
#
# 1. **Cell 1〜2.5** を順に実行し、CUDA / SAM2 / GroundingDINO checkpoint 診断が OK であることを確認します。
# 2. **Cell 3** を実行し、Colab では `Running on public URL: https://...gradio.live` を開きます。
# 3. Gradio の **Input Video** に動画をアップロードします。右側の **SAM2 Prompt Canvas** に第 1 フレームが表示されます。
# 4. 複合対象を意味で選びたい場合は **Optional: Text Prompt to Box (GroundingDINO)** を開き、
#    `person playing drums` や `person riding bicycle` のように入力して bbox 候補を作ります。
# 5. **SAM2 Prompt Canvas** 上で bbox を確認・補正します。手動 bbox は対象を囲む四角形の **対角 2 点**
#    （例: 左上→右下、または右下→左上）をクリックする操作です。
# 6. まず既定値の **最大 30 frames / frame_step=1** でクイックプレビューし、問題なければ Advanced で
#    最大処理フレーム数を増やして最終出力します。transparent-background の出力は処理中に書き出され、
#    RGBA/alpha/preview frame 全体を RAM に保持しない設計です。
#
# **フローの意味**:
# 静止画版・動画版とも本質的な順序は
# `Text Prompt（画像意味解釈） → SAM2（マスク/トラッキング） → transparent-background（背景除去）` です。
# 動画版の SAM2 Prompt Canvas は SAM2 への入力先で、Text Prompt はそこへ bbox を自動で書き込む補助機能です。
#
# **今回の目的:**
# - 動画をアップロードする
# - 必要に応じて Text Prompt / GroundingDINO で複合対象の bbox 候補を作る
# - 第 1 フレーム上で SAM2 point / box prompt を指定する
# - SAM2 video predictor で mask を動画全体へ伝搬する
# - transparent-background を frame ごとに適用する
# - 出力を動画形式 / PNG 連番形式 / 両方から選ぶ

# %%
# ============================================================
# Cell 1: 依存関係インストール
# ============================================================
# -q は使わず、CUDA / build / ffmpeg エラーを見える状態にします。
import os
import sys

!{sys.executable} -m pip install gradio==5.9.1
!{sys.executable} -m pip install haystack-ai==2.29.0
!{sys.executable} -m pip install transparent-background
!{sys.executable} -m pip install pymatting
!{sys.executable} -m pip install git+https://github.com/facebookresearch/sam2.git
!{sys.executable} -m pip install opencv-python-headless pillow numpy imageio[ffmpeg]
!{sys.executable} -m pip install "transformers>=4.26.0" addict yapf timm supervision pycocotools

print("Install done")

# %%
# ============================================================
# Cell 2: Google Drive マウント + プロジェクトパス設定 + チェックポイント配置
# ============================================================
from pathlib import Path

def is_colab_runtime() -> bool:
    """Google Colab runtime かを、import 済み状態に依存せず判定する。"""
    import importlib.util

    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        return False


IS_COLAB = is_colab_runtime()
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
    from sam2.build_sam import build_sam2_video_predictor

    print(f"sam2 package = {sam2.__file__}")
    print("sam2 video imports = OK")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "SAM2 package import failed before launching Gradio. "
        "Cell 1 の install をエラーが見える状態（-q なし）で再実行し、"
        f"`{sys.executable} -m pip install git+https://github.com/facebookresearch/sam2.git` が成功したことを確認してください. "
        "その後、この診断セルと Gradio 起動セルを再実行してください。"
    ) from exc

exists = SAM2_CKPT_PATH.exists()
size_mb = SAM2_CKPT_PATH.stat().st_size / (1024 * 1024) if exists and SAM2_CKPT_PATH.is_file() else None
is_drive_path = "/drive/" in str(SAM2_CKPT_PATH).replace("\\", "/").lower()
print(f"SAM2 checkpoint: path={SAM2_CKPT_PATH}, exists={exists}, size_mb={size_mb}, drive_path={is_drive_path}")
grounding_exists = GROUNDING_DINO_CKPT_PATH.exists()
grounding_size_mb = (
    GROUNDING_DINO_CKPT_PATH.stat().st_size / (1024 * 1024)
    if grounding_exists and GROUNDING_DINO_CKPT_PATH.is_file()
    else None
)
print(
    "GroundingDINO checkpoint: "
    f"path={GROUNDING_DINO_CKPT_PATH}, exists={grounding_exists}, size_mb={grounding_size_mb}"
)
print(
    "GPU policy: SAM2 video tracking and GroundingDINO text detection require CUDA by default. "
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
# Cell 3: 動画版 Haystack Gradio アプリ起動
# ============================================================
APP_PATH = PROJECT_ROOT / "gradio_app_sam2_transparent_BG_haystack_for_Movie.py"
assert APP_PATH.exists(), f"アプリが見つかりません: {APP_PATH}"

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["SAM2_CKPT_PATH"] = str(SAM2_CKPT_PATH)
os.environ["SAM2_CONFIG_NAME"] = SAM2_CONFIG_NAME
os.environ["GROUNDING_DINO_CKPT_PATH"] = str(GROUNDING_DINO_CKPT_PATH)

SHARE_FLAG = "--share" if IS_COLAB else ""
if IS_COLAB:
    print("Colab detected: use the 'Running on public URL' gradio.live link, not the local 127.0.0.1 URL.")

!{sys.executable} "{APP_PATH}" {SHARE_FLAG}
