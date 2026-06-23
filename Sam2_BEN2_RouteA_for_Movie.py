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
# # SAM2 + BEN2 Route A（ブラー誘導 → 再α化）Haystack Movie Pipeline
#
# このノートブックは **Jupytext の `.py` を正本** として管理します。`.ipynb` は次のコマンドで生成します。
#
# ```powershell
# .venv\Scripts\python.exe -m jupytext --to ipynb Sam2_BEN2_RouteA_for_Movie.py
# ```
#
# ## ルートA案（ブラー誘導 → BEN2 再α化）とは
#
# SAM2 が出す下地マスク **M** を少し膨張させてゲート **G** を作り、**G の外側だけを強くブラー**した
# 誘導フレーム **I'** を **BEN2**（前景マッティングモデル）に渡して、α を高品質に作り直す方式です
# （仕様書: `計画書/2026-06-22_動画αマット_ルートA案_ブラー誘導_仕様書.md`）。背景がボケて「ここが前景」と
# 誘導されるため、髪や手足の細部のα品質が上がりやすくなります。背景透過モデルを transparent-background から
# **BEN2** に差し替えたのが本ノートブックの主眼です。
#
# ## ⚠️ SAMURAI トラッカー推奨設定（必読）
#
# SAMURAI（motion-aware）は Kalman filter による **forward-only（前方向のみ）** 設計です。Colab T4 等の
# 小 VRAM 環境ではメモリ枯渇で伝搬が `propagate 1/N` のまま凍結する stall が起きやすいため、以下を守ってください（ERR049 / ERR050）。
#
# | 項目 | 推奨 | 理由 |
# |---|---|---|
# | **対象オブジェクト数** | **1 個のみ**（複数は標準 SAM2 へ） | SAMURAI は Kalman filter による単一対象追跡専用。複数 box を渡すと fork が `Boolean value of Tensor ... ambiguous` で伝搬失敗（ERR051） |
# | **双方向伝播** | **OFF**（SAMURAI 選択時は UI が自動 OFF・無効化） | 逆方向は Kalman の速度ベクトルが反転し追跡が崩れ、per-frame memory も 2 倍 |
# | **プロンプト起点フレーム** | **0（先頭）** | forward-only のため先頭起点が安定。末尾起点は逆走 stall を誘発 |
# | **CPU offload** | 有効（config で自動 ON） | 常駐 VRAM を抑え stall を回避（`offload_video_to_cpu` / `offload_state_to_cpu`） |
# | **autocast** | fp16（config で自動 ON） | 伝搬を mixed precision で回し VRAM を削減・高速化（SAMURAI 本家と同じ） |
# | **最大処理フレーム数** | まず 30 でプレビュー | 初回から大量フレームにすると stall リスクが上がる |
#
# これらの値はすべて `config/inference_models.toml` の tracker entry 駆動です。ルートAの膨張量・ブラー強度などの
# チューニング値は `config/route_a.toml` 駆動で、UI の Advanced から微調整できます。
#
# 実装本体は [gradio_app_sam2_ben2_route_a_for_Movie.py](./gradio_app_sam2_ben2_route_a_for_Movie.py) と
# [pipelines/](./pipelines/) に集約しています。
#
# ## UI の使い方
#
# 1. **Cell 1〜2.5** を順に実行し、CUDA / SAM2 / BEN2 / GroundingDINO checkpoint 診断が OK であることを確認します。
# 2. **Cell 3** を実行し、Colab では `Running on public URL: https://...gradio.live` を開きます。
# 3. Gradio の **Input Video** に動画をアップロードします。右側の **SAM2 Prompt Canvas** に第 1 フレームが表示されます。
# 4. 複合対象を意味で選びたい場合は **Optional: Text Prompt to Box (GroundingDINO)** を開き、
#    `person playing drums` や `person riding bicycle` のように入力して bbox 候補を作ります。
# 5. **SAM2 Prompt Canvas** 上で bbox を確認・補正します。手動 bbox は対象を囲む四角形の **対角 2 点** をクリックします。
# 6. 誤って入れた prompt は **Prompt 編集（個別削除）** で、選択した bbox / point（positive・negative）だけ削除できます。
# 7. ちらつきが残る場合は、標準 SAM2 で **双方向伝播 ON**、難しい素材では **matte_mode=per_object** を試してください。
# 7.5. SAM2 マスクが最後まで追跡できているなら、**SAM2マスクでα底上げ（合成）** を `screen`（自然に底上げ）か
#      `lighten / 比較明`（確実に塗る）にすると、安定した SAM2 マスクを α の床にして BEN2 のちらつきを直接補えます。
#      前景ブラー・背景同系色・高速な被写体（例: ドラムスティック）で BEN2 の α が揺れる動画に有効です（既定は `none`=無効）。
# 8. まず既定値（最大 30 frames・union）で **ルートA実行** し、結果を確認してから Advanced で
#    膨張量・ブラー強度を微調整します。BEN2 の出力は処理中に書き出され、frame 全体を RAM に保持しない設計です。
#
# **フローの意味**:
# `Text Prompt（画像意味解釈） → SAM2（マスク/トラッキング） → ルートA（下地マスク膨張→背景ブラー→BEN2 再α化）→ 出力` です。
# SAM2 Prompt Canvas は SAM2 への入力先で、Text Prompt はそこへ bbox を自動で書き込む補助機能です。

# %%
# ============================================================
# Cell 1: 依存関係インストール
# ============================================================
# -q は使わず、CUDA / build / ffmpeg エラーを見える状態にします。
import os
import sys

!{sys.executable} -m pip install gradio==5.9.1
!{sys.executable} -m pip install haystack-ai==2.29.0
# BEN2（PramaLLC, MIT ライセンス・base は商用可）を背景透過モデルとして導入する。
# 重みは Cell 2.5 / 初回推論時に Hugging Face `PramaLLC/BEN2` から自動取得される（追加 wget 不要）。
!{sys.executable} -m pip install git+https://github.com/PramaLLC/BEN2.git
!{sys.executable} -m pip install pymatting
# sam2 は SAMURAI fork（同梱 samurai/sam2）を Cell 2 で Drive マウント後に
# 非 editable + --no-build-isolation で導入する（ERR045）。
# fork は標準 config（configs/sam2.1/）と SAMURAI config（configs/samurai/）の両方を含み、
# facebook / SAMURAI 両 tracker を 1 つの sam2 package で賄う（ERR038/ERR041）。
# loguru は SAMURAI fork の sam2/modeling/sam2_base.py が import するが fork の
# setup.py の install_requires に含まれないため、ここで明示的に導入する（ERR046）。
!{sys.executable} -m pip install loguru
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
BEN2_CKPT_DIR = CKPT_ROOT / "BEN2"
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
for directory in (SAM2_CKPT_DIR, BEN2_CKPT_DIR, INPUT_DIR, OUTPUT_DIR):
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

# BEN2 の重みは Hugging Face `PramaLLC/BEN2` から `BEN_Base.from_pretrained(...)` 経由で
# 自動ダウンロード・キャッシュされる（config/route_a.toml の [alpha].ben2_checkpoint_path が
# 空のとき from_pretrained を使う）。ローカル .pth を使う場合は config に絶対パスを設定する。
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"SAM2_CKPT_PATH = {SAM2_CKPT_PATH}")
print(f"GROUNDING_DINO_CKPT_PATH = {GROUNDING_DINO_CKPT_PATH}")
print(f"BEN2_CKPT_DIR = {BEN2_CKPT_DIR} (HF キャッシュ利用時は空のままで可)")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")

# SAM2 は SAMURAI fork（同梱 samurai/sam2）を installed package として導入する。
# facebook 版 sam2 だと configs/samurai/ も samurai_mode 対応モデルコードも無く、
# SAMURAI tracker 選択時に MissingConfigException になる（ERR038/ERR041）。
# fork は標準 config も含むため facebook tracker もそのまま動く。SAMURAI は訓練不要で
# 同じ sam2.1_hiera_large.pt を再利用する（追加チェックポイント不要）。
SAMURAI_SAM2_DIR = PROJECT_ROOT / "samurai" / "sam2"
assert SAMURAI_SAM2_DIR.exists(), f"SAMURAI sam2 fork が見つかりません: {SAMURAI_SAM2_DIR}"
SAMURAI_SAM2_DIR_POSIX = SAMURAI_SAM2_DIR.as_posix()  # shell 安全のため posix 区切りで渡す
# 重要(ERR045):
#  1) Google Drive(FUSE) 上で `pip install -e`（editable）は .pth/egg-info の書き込みに
#     失敗しやすく、sam2 が入らず `ModuleNotFoundError: No module named 'sam2'` になる。
#     → 非 editable で install（pip が temp に複製してビルドするため Drive 問題を回避）。
#  2) 動画伝搬は build_sam.py が fill_hole_area=8 を強制するため sam2._C
#     （connected_components CUDA 拡張）が実行時に必須（CPU fallback 無し）。通常の
#     `pip install`（build isolation 有効）では torch が見えず _C がビルドされない。
#     → --no-build-isolation で現環境の torch を見せ、Colab GPU の nvcc で _C をビルドする。
#  configs(*.yaml: configs/sam2.1, configs/samurai) は MANIFEST により wheel に同梱される。
!{sys.executable} -m pip install --no-build-isolation "{SAMURAI_SAM2_DIR_POSIX}"

# install が成功したかを Cell 2 の時点で fail-loud に検証する（`!pip` は失敗でも例外を
# 出さないため、ここで明示的に確認しないと Cell 2.5 で分かりにくいエラーになる / ERR045）。
import importlib

importlib.invalidate_caches()
if importlib.util.find_spec("sam2") is None:
    raise RuntimeError(
        "SAMURAI fork (sam2) の install に失敗しました。上の pip ログのエラーを確認してください。"
        " CUDA 拡張(_C)のビルドに失敗した場合は、GPU ランタイム（nvcc あり）で実行しているか、"
        " torch と CUDA toolkit の整合を確認してから Cell 2 を再実行してください。"
    )

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
        "Cell 2 の SAMURAI fork 導入（エラーが見える状態）を再実行し、"
        f"`{sys.executable} -m pip install --no-build-isolation samurai/sam2` が成功したことを確認してください. "
        "（Drive 上の editable(-e) は失敗しやすいので非 editable を使う / ERR042）。"
        "その後、この診断セルと Gradio 起動セルを再実行してください。"
    ) from exc

# BEN2 package が import 可能かを確認する（重み取得は初回推論時に遅延実行されるため、
# ここでは import 可否のみ fail-loud に検証する）。
try:
    from ben2 import BEN_Base  # noqa: F401

    print("ben2 import = OK")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "BEN2 package import failed before launching Gradio. "
        "Cell 1 の `pip install git+https://github.com/PramaLLC/BEN2.git` が成功したことを確認し、"
        "ランタイムを再起動してから install cell から再実行してください。"
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
    "GPU policy: SAM2 video tracking, BEN2 matting and GroundingDINO text detection require CUDA by default. "
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
# Cell 3: ルートA動画版 Haystack Gradio アプリ起動
# ============================================================
APP_PATH = PROJECT_ROOT / "gradio_app_sam2_ben2_route_a_for_Movie.py"
assert APP_PATH.exists(), f"アプリが見つかりません: {APP_PATH}"

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["SAM2_CKPT_PATH"] = str(SAM2_CKPT_PATH)
os.environ["SAM2_CONFIG_NAME"] = SAM2_CONFIG_NAME
os.environ["GROUNDING_DINO_CKPT_PATH"] = str(GROUNDING_DINO_CKPT_PATH)

SHARE_FLAG = "--share" if IS_COLAB else ""
if IS_COLAB:
    print("Colab detected: use the 'Running on public URL' gradio.live link, not the local 127.0.0.1 URL.")

!{sys.executable} "{APP_PATH}" {SHARE_FLAG}
