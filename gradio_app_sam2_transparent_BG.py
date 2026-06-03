"""
gradio_app_sam2_transparent_BG.py
SAM2 + transparent-background を組み合わせた背景除去 Gradio 5 デモ。

ローカル環境・Google Colab いずれでも動作する。
Colab で使用する場合は Sam2_Transparent_Background.ipynb の Cell 6 を参照。

依存パッケージ:
    pip install gradio==5.9.1 transparent-background pymatting
    pip install git+https://github.com/facebookresearch/sam2.git
    pip install opencv-python-headless pillow numpy

チェックポイント配置（デフォルト: プロジェクト内 ./checkpoints/）:
    checkpoints/SAM2/sam2.1_hiera_large.pt
    checkpoints/transparent_BG/ckpt_base.pth         （任意・無ければ初回 DL）
    checkpoints/transparent_BG/ckpt_fast.pth         （任意）
    checkpoints/transparent_BG/ckpt_base_nightly.pth （任意）
    ※ SAM2_CKPT_PATH / SAM2_CONFIG_NAME は環境変数で上書き可能
"""

# 標準ライブラリ
import datetime
import os
import sys
from pathlib import Path

# サードパーティ
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transparent_background import Remover

# ----------------------------------------------------------------
# Gradio 5.x /info エンドポイント クラッシュ回避パッチ（ERR011）
# 根本原因: gradio_client/utils.py の _json_schema_to_python_type が
# JSON Schema の boolean 値（additionalProperties: true/false）を受け取ると
#   TypeError: argument of type 'bool' is not iterable
# でクラッシュする。FastAPI はルート登録時に関数を参照コピーするため、
# App.api_info メソッドのモンキーパッチは効かない。
# → クラッシュ箇所の上流にある _json_schema_to_python_type を直接パッチする。
# ----------------------------------------------------------------
try:
    import gradio_client.utils as _gc_utils

    _orig_inner = _gc_utils._json_schema_to_python_type

    def _patched_inner(schema, defs=None):
        # JSON Schema では boolean も有効な schema 値（true=何でも許可, false=拒否）
        # Gradio の変換関数はこれを未処理のためクラッシュする → "Any" を返して回避
        if isinstance(schema, bool):
            return "Any"
        return _orig_inner(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_inner
except Exception:
    pass  # Gradio バージョン差異による patch 失敗は無視

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============================================================
# 設定（すべてプロジェクト内パス）
# ============================================================
# Colab 判定 + Drive マウント
# プロジェクト本体が Google Drive 上にある運用のため、Colab で実行された場合は
# 自動的に Drive をマウントしてプロジェクトルートを Drive 上のパスに合わせる。
IS_COLAB = "google.colab" in sys.modules
COLAB_DRIVE_PROJECT = Path("/content/drive/MyDrive/AI_picasso/Matting-Anything")

if IS_COLAB and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    print("[INFO] Google Drive mounted at /content/drive")


def _detect_project_root() -> Path:
    """プロジェクトルートを決定する。

    優先順位:
        1. 環境変数 PROJECT_ROOT
        2. Colab かつ Drive 上の所定パスが存在する場合 → そのパス
        3. それ以外: このファイルの親ディレクトリ
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    if IS_COLAB and COLAB_DRIVE_PROJECT.exists():
        return COLAB_DRIVE_PROJECT
    return Path(__file__).parent.resolve()


PROJECT_ROOT = _detect_project_root()
print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
CKPT_ROOT = PROJECT_ROOT / "checkpoints"
SAM2_CKPT_DIR = CKPT_ROOT / "SAM2"
TB_CKPT_DIR = CKPT_ROOT / "transparent_BG"
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
for _d in (SAM2_CKPT_DIR, TB_CKPT_DIR, INPUT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# SAM2 チェックポイント（環境変数で上書き可能・ハードコード禁止）
SAM2_CKPT_PATH = os.environ.get(
    "SAM2_CKPT_PATH",
    str(SAM2_CKPT_DIR / "sam2.1_hiera_large.pt"),
)
SAM2_CONFIG_NAME = os.environ.get(
    "SAM2_CONFIG_NAME",
    "configs/sam2.1/sam2.1_hiera_l.yaml",
)

# transparent-background チェックポイント（モード別・ローカル優先）
TB_CKPT_BY_MODE: dict = {
    "base":         TB_CKPT_DIR / "ckpt_base.pth",
    "fast":         TB_CKPT_DIR / "ckpt_fast.pth",
    "base-nightly": TB_CKPT_DIR / "ckpt_base_nightly.pth",
}

# GPU/CPU 自動フォールバック
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# ============================================================
# SAM2 初期化
# ============================================================
sam2_model = build_sam2(SAM2_CONFIG_NAME, SAM2_CKPT_PATH, device=str(DEVICE))
sam2_model.eval()
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("[INFO] SAM2 loaded")

# ============================================================
# transparent-background Remover キャッシュ
# ============================================================
_remover_cache: dict = {}


def get_remover(mode: str = "base", jit: bool = False, ckpt_path=None) -> Remover:
    """モードごとに Remover インスタンスをキャッシュして返す。

    Args:
        mode:      'base' / 'fast' / 'base-nightly'
        jit:       True で TorchScript 高速化（初回コンパイルに時間がかかる）
        ckpt_path: カスタムチェックポイントパス（None なら TB_CKPT_BY_MODE を参照）

    Returns:
        Remover インスタンス
    """
    # ckpt_path 未指定時はプロジェクト内のローカルチェックポイントを優先
    if ckpt_path is None:
        local = TB_CKPT_BY_MODE.get(mode)
        if local is not None and Path(local).exists():
            ckpt_path = local

    key = (mode, jit, str(ckpt_path))
    if key not in _remover_cache:
        kwargs: dict = dict(mode=mode, jit=jit, device=str(DEVICE))
        if ckpt_path and Path(str(ckpt_path)).exists():
            kwargs["ckpt"] = str(ckpt_path)
        _remover_cache[key] = Remover(**kwargs)
        print(f"[INFO] Loaded Remover(mode={mode}, jit={jit})")
    return _remover_cache[key]


def tb_process(
    pil_image: Image.Image,
    mode: str = "base",
    jit: bool = False,
    threshold: float | None = None,
    output_type: str = "rgba",
    ckpt_path=None,
) -> Image.Image:
    """transparent-background でアルファ抽出を行う。

    Args:
        pil_image:   入力 PIL.Image
        mode:        'base' / 'fast' / 'base-nightly'
        jit:         True で TorchScript 高速化
        threshold:   None=ソフトα / 0.0〜1.0 で二値化
        output_type: 'rgba' / 'map' / 'green' / 'white' / 'blur' / 'overlay'
        ckpt_path:   カスタムチェックポイントパス

    Returns:
        PIL.Image（output_type に応じた形式）
    """
    remover = get_remover(mode=mode, jit=jit, ckpt_path=ckpt_path)
    return remover.process(pil_image, type=output_type, threshold=threshold)


# ============================================================
# SAM2 推論ヘルパー
# ============================================================
def sam2_predict(
    image_np: np.ndarray,
    points: list | None,
    labels: list | None,
    box: list | None = None,
    multimask: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """SAM2 でマスクを予測する。

    Args:
        image_np: HxWx3 uint8 RGB 配列
        points:   [(x, y), ...] または None
        labels:   [1/0, ...]  1=positive, 0=negative
        box:      [x1, y1, x2, y2] または None
        multimask: True で複数候補マスクを返す

    Returns:
        masks (N, H, W) bool, scores (N,)
    """
    with torch.inference_mode(), torch.autocast(str(DEVICE), dtype=torch.bfloat16):
        sam2_predictor.set_image(image_np)
        kwargs: dict = {"multimask_output": multimask}
        if points is not None and len(points) > 0:
            kwargs["point_coords"] = np.array(points, dtype=np.float32)
            kwargs["point_labels"] = np.array(labels, dtype=np.int32)
        if box is not None:
            kwargs["box"] = np.array(box, dtype=np.float32)
        masks, scores, _ = sam2_predictor.predict(**kwargs)
    return masks, scores


# ============================================================
# パイプライン関数
# ============================================================
def bbox_from_mask(
    mask: np.ndarray, padding: int = 20, img_shape=None
) -> tuple | None:
    """bool マスクから padding 込みの bounding box を返す。

    Args:
        mask:      HxW bool 配列
        padding:   bbox 周囲の余白 px
        img_shape: 画像の shape（境界クリップ用）

    Returns:
        (x1, y1, x2, y2) または None（マスクが空の場合）
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if img_shape is not None:
        H, W = img_shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(W, x2 + padding)
        y2 = min(H, y2 + padding)
    return x1, y1, x2, y2


def dilate_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """マスクを膨張させる（細毛のはみ出し許容）。"""
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)


def run_pipeline(
    image_pil: Image.Image,
    sam2_mask: np.ndarray | None,
    tb_mode: str,
    tb_jit: bool,
    tb_threshold: float,
    tb_output_type: str,
    crop_padding: int,
    use_sam2_as_guard: bool,
    sam2_guard_dilate: int,
    apply_decontam: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SAM2マスクで前景を切り出し → tb でα抽出 → 元位置に戻す → SAM2ガード。

    Args:
        image_pil:         入力 PIL.Image
        sam2_mask:         HxW bool マスク（None の場合は全体を処理）
        tb_mode:           transparent-background モード
        tb_jit:            JIT 高速化フラグ
        tb_threshold:      0=ソフトα / >0 で二値化
        tb_output_type:    プレビュー背景タイプ
        crop_padding:      SAM2 bbox の余白 px
        use_sam2_as_guard: True で SAM2 マスク外のαをゼロにする
        sam2_guard_dilate: ガードマスクの dilate kernel サイズ
        apply_decontam:    True で pymatting による Color decontamination

    Returns:
        (rgba, alpha_u8, preview) それぞれ HxWx4, HxW, HxWx3 or HxWx4 の ndarray
    """
    img_np = np.array(image_pil.convert("RGB"))
    H, W = img_np.shape[:2]

    # 1) SAM2 マスクから bbox を取得（マスクがない場合は全体）
    if sam2_mask is not None and sam2_mask.any():
        bbox = bbox_from_mask(sam2_mask, padding=crop_padding, img_shape=img_np.shape)
        x1, y1, x2, y2 = bbox
        crop_np = img_np[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, W, H
        crop_np = img_np

    crop_pil = Image.fromarray(crop_np)

    # 2) transparent-background でα抽出（出力は RGBA）
    rgba_crop = tb_process(
        crop_pil,
        mode=tb_mode,
        jit=tb_jit,
        threshold=tb_threshold if tb_threshold > 0 else None,
        output_type="rgba",
    )
    rgba_crop_np = np.array(rgba_crop)              # (h, w, 4)
    alpha_crop = rgba_crop_np[..., 3].astype(np.float32) / 255.0
    rgb_crop = rgba_crop_np[..., :3]

    # 3) 元サイズキャンバスに貼り戻し
    full_alpha = np.zeros((H, W), dtype=np.float32)
    full_alpha[y1:y2, x1:x2] = alpha_crop
    full_rgb = img_np.copy()
    full_rgb[y1:y2, x1:x2] = rgb_crop

    # 4) SAM2 マスクでガード（tb の誤検出を除去）
    if use_sam2_as_guard and sam2_mask is not None:
        guard = dilate_mask(sam2_mask, kernel_size=sam2_guard_dilate)
        full_alpha = full_alpha * guard.astype(np.float32)

    # 5) Color decontamination（緑被り除去）
    if apply_decontam:
        try:
            from pymatting import estimate_foreground_ml
            fg = estimate_foreground_ml(
                img_np.astype(np.float64) / 255.0,
                full_alpha.astype(np.float64),
            )
            full_rgb = np.clip(fg * 255.0, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WARN] Decontam skipped: {e}")

    # 6) 出力（RGBA & プレビュー用）
    alpha_u8 = (full_alpha * 255).astype(np.uint8)
    rgba = np.dstack([full_rgb, alpha_u8])

    # プレビュー用背景を合成
    if tb_output_type == "green":
        bg = np.full_like(full_rgb, [0, 255, 0])
    elif tb_output_type == "white":
        bg = np.full_like(full_rgb, [255, 255, 255])
    elif tb_output_type == "blur":
        bg = cv2.GaussianBlur(img_np, (51, 51), 0)
    else:
        bg = None

    if bg is not None:
        a3 = full_alpha[..., None]
        preview = (full_rgb * a3 + bg * (1 - a3)).astype(np.uint8)
    else:
        preview = rgba  # rgba そのまま

    return rgba, alpha_u8, preview


# ============================================================
# UI 初期状態
# ============================================================
INITIAL_STATE: dict = {
    "image": None,          # 元画像 (np.ndarray HxWx3)
    "points": [],           # [(x, y), ...]
    "labels": [],           # [1/0, ...]  1=positive, 0=negative
    "box": None,            # [x1, y1, x2, y2] or None
    "box_buffer": [],       # box 指定用の一時クリック
    "masks": None,          # SAM2 候補マスク (N, H, W) bool
    "scores": None,         # SAM2 候補スコア (N,)
    "selected_mask": None,
    "input_mode": "point",  # "point" / "box"
}


def draw_overlay(
    img_np: np.ndarray,
    points: list,
    labels: list,
    box,
    mask=None,
) -> np.ndarray:
    """クリック点・box・マスクを画像に重ねて可視化する。"""
    vis = img_np.copy()
    # マスクを半透明青で重畳
    if mask is not None:
        color = np.array([30, 144, 255], dtype=np.uint8)
        m = mask.astype(bool)
        vis[m] = (vis[m] * 0.5 + color * 0.5).astype(np.uint8)
    # Box を黄色で描画
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 215, 0), 3)
    # クリック点を描画（緑=positive / 赤=negative）
    for (x, y), lbl in zip(points, labels):
        color_dot = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(vis, (int(x), int(y)), 8, color_dot, -1)
        cv2.circle(vis, (int(x), int(y)), 9, (255, 255, 255), 2)
    return vis


# ============================================================
# イベントハンドラ
# ============================================================

def on_upload(img, state: dict) -> tuple:
    """画像アップロード時に state を初期化する。"""
    if img is None:
        return None, state, "画像をアップロードしてください"
    state = {**INITIAL_STATE}
    # gr.Image(type="pil") は PIL.Image を返すが numpy の場合も許容
    state["image"] = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else img
    return state["image"], state, "✅ 画像読み込み完了。点 or box で対象を指定してください"


def on_click(
    evt: gr.SelectData, state: dict, current_label: str, current_mode: str
) -> tuple:
    """画像クリック時に point/box を蓄積し可視化を更新する。"""
    if state["image"] is None:
        return None, state, "先に画像をアップロードしてください"
    x, y = evt.index
    state["input_mode"] = current_mode

    if current_mode == "point":
        state["points"].append((x, y))
        state["labels"].append(1 if current_label == "positive" else 0)
        msg = f"📍 Point 追加: ({x},{y}) {current_label}  / 計 {len(state['points'])} 点"
    else:  # box モード: 2 点クリックで box 確定
        state["box_buffer"].append((x, y))
        if len(state["box_buffer"]) == 2:
            (x1, y1_), (x2, y2_) = state["box_buffer"]
            state["box"] = [min(x1, x2), min(y1_, y2_), max(x1, x2), max(y1_, y2_)]
            state["box_buffer"] = []
            msg = f"📦 Box 確定: {state['box']}"
        else:
            msg = "📦 Box 左上クリック完了 → 右下をクリックしてください"

    vis = draw_overlay(
        state["image"], state["points"], state["labels"],
        state["box"], state.get("selected_mask"),
    )
    return vis, state, msg


def on_clear(state: dict) -> tuple:
    """点・box・マスクをすべてクリアする。"""
    if state["image"] is None:
        return None, state, ""
    state["points"] = []
    state["labels"] = []
    state["box"] = None
    state["box_buffer"] = []
    state["masks"] = None
    state["selected_mask"] = None
    return state["image"], state, "🧹 クリアしました"


def on_predict(state: dict, multimask: bool) -> tuple:
    """SAM2 で候補マスクを推論する。"""
    if state["image"] is None:
        return None, None, None, "0", state, "画像がありません"
    if len(state["points"]) == 0 and state["box"] is None:
        return None, None, None, "0", state, "Point か Box を指定してください"

    masks, scores = sam2_predict(
        state["image"],
        state["points"] if state["points"] else None,
        state["labels"] if state["labels"] else None,
        state["box"],
        multimask=multimask,
    )
    state["masks"] = masks
    state["scores"] = scores
    # スコア最良を初期選択
    best = int(np.argmax(scores))
    state["selected_mask"] = masks[best]

    previews = []
    for i in range(len(masks)):
        previews.append(draw_overlay(state["image"], [], [], None, masks[i]))
    # 3 枚に満たない場合は None で埋める
    while len(previews) < 3:
        previews.append(None)

    info = " | ".join([f"Mask{i}: {s:.3f}" for i, s in enumerate(scores)])
    info += f"  → ベスト = Mask{best}"
    # Gradio 5.x の api_info schema 変換で int 値の Radio が落ちるため str で返す
    return previews[0], previews[1], previews[2], str(best), state, info


def on_select_mask(idx, state: dict) -> tuple:
    """候補マスクのうち採用するものを切り替える。"""
    if state["masks"] is None:
        return None, state, ""
    state["selected_mask"] = state["masks"][int(idx)]
    vis = draw_overlay(
        state["image"], state["points"], state["labels"],
        state["box"], state["selected_mask"],
    )
    return vis, state, f"✅ Mask{idx} を採用しました"


def on_run_tb(
    state: dict,
    tb_mode: str,
    tb_jit: bool,
    tb_thresh: float,
    tb_output: str,
    crop_pad: float,
    use_guard: bool,
    guard_dilate: float,
    decontam: bool,
    save_to_disk: bool,
) -> tuple:
    """transparent-background パイプラインを実行して結果を返す。SAM2 マスクなしでも動作する。"""
    if state["image"] is None:
        raise gr.Error("画像がありません。先に画像をアップロードしてください。")
    # selected_mask が None の場合は画像全体を tb で直接処理（SAM2 なしモード）
    pil_img = Image.fromarray(state["image"])
    rgba, alpha, preview = run_pipeline(
        pil_img, state["selected_mask"],
        tb_mode=tb_mode,
        tb_jit=tb_jit,
        tb_threshold=tb_thresh,
        tb_output_type=tb_output,
        crop_padding=int(crop_pad),
        use_sam2_as_guard=use_guard,
        sam2_guard_dilate=int(guard_dilate),
        apply_decontam=decontam,
    )

    # プロジェクト内 OUTPUT_DIR 配下にタイムスタンプ付きで保存
    no_sam2 = state["selected_mask"] is None
    msg = "✅ 抽出完了（SAM2 なし・全体処理）" if no_sam2 else "✅ 抽出完了"
    if save_to_disk:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = OUTPUT_DIR / ts
        save_root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(save_root / "rgba.png")
        Image.fromarray(alpha).save(save_root / "alpha.png")
        Image.fromarray(preview).save(save_root / "preview.png")
        rel = save_root.relative_to(PROJECT_ROOT)
        msg = f"✅ 抽出完了 / 保存: {rel}"

    return rgba, alpha, preview, msg


# ============================================================
# Gradio 5 UI 構築
# ============================================================
with gr.Blocks(title="SAM2 + transparent-background", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 SAM2 + transparent-background 背景除去パイプライン")
    gr.Markdown(
        "**手順**: 画像アップ → （任意）SAM2 で対象指定 → マスク選択 → tb パラメータ調整 → 抽出  \n"
        "💡 **SAM2 はスキップ可能**です。画像をアップロードして「背景除去を実行」ボタンを押すだけで"
        " transparent-background が単体で動作します。"
    )

    state = gr.State(value=dict(INITIAL_STATE))

    with gr.Row():
        # ============ 左カラム: SAM2 ============
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ SAM2 で対象を指定")
            input_image = gr.Image(label="画像アップロード", type="pil", height=400)
            interactive_image = gr.Image(
                label="クリックして指定（緑=前景 / 赤=背景 / 黄=box）",
                interactive=True,
                height=400,
            )

            with gr.Row():
                input_mode = gr.Radio(
                    ["point", "box"],
                    value="point",
                    label="入力モード",
                    info="point: クリック1点ごとに前景/背景ヒントを与える（複数点可）。box: 2クリックで対象を囲む矩形を1個指定する。単位なしの選択値。",
                )
                current_label = gr.Radio(
                    ["positive", "negative"],
                    value="positive",
                    label="Point 種別 (positive=対象 / negative=除外)",
                    info="point モード時のみ有効。positive: クリック位置を前景（残す）とする。negative: クリック位置を背景（除外）とする。box モードでは無視される。",
                )

            with gr.Row():
                btn_clear = gr.Button("🧹 全クリア", variant="secondary")
                multimask = gr.Checkbox(
                    value=True,
                    label="複数候補マスク出力",
                    info="ON: SAM2 が候補マスクを最大3枚（Mask 0/1/2）返し、採用マスクを選べる。OFF: スコア最上位の1枚のみ返す。真偽値。",
                )
                btn_predict = gr.Button("🚀 SAM2 推論", variant="primary")

            sam_info = gr.Markdown()

            gr.Markdown("### 2️⃣ 候補マスクから採用を選ぶ")
            with gr.Row():
                mask_view_0 = gr.Image(label="Mask 0", height=180)
                mask_view_1 = gr.Image(label="Mask 1", height=180)
                mask_view_2 = gr.Image(label="Mask 2", height=180)
            # Gradio 5.x の /info 生成バグ回避のため choices は文字列値で運用
            mask_idx = gr.Radio(
                ["0", "1", "2"],
                value="0",
                label="採用マスク",
                info="transparent-background のガードに使う SAM2 マスク番号。0/1/2 は上の Mask 0/1/2 に対応する候補インデックス（単位なし）。複数候補が無い場合は 0 のまま。",
            )

        # ============ 右カラム: transparent-background ============
        with gr.Column(scale=1):
            gr.Markdown("### 3️⃣ transparent-background パラメータ")

            with gr.Group():
                tb_mode = gr.Radio(
                    ["base", "fast", "base-nightly"],
                    value="base",
                    label="モード",
                    info="base: 高精度（推奨）/ fast: 軽量高速 / base-nightly: 最新実験版（不安定）",
                )
                tb_jit = gr.Checkbox(
                    value=False,
                    label="JIT（高速化、初回コンパイル時間あり）",
                    info="TorchScript JIT でモデルをコンパイル。2 回目以降の推論が約 1.5〜2× 高速化。初回のみコンパイル待機が発生する。",
                )
                tb_thresh = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Threshold (0=ソフトα / >0 で二値化)",
                    info="0.0: 半透明ピクセルを保持するソフトアルファ（髪・煙・ガラス向き）。0.5 付近: 境界をくっきり二値化（輪郭がはっきりしている場合）。値が大きいほど前景領域が小さくなる。",
                )
                tb_output = gr.Radio(
                    ["rgba", "green", "white", "blur"],
                    value="rgba",
                    label="プレビュー背景",
                    info="rgba: 透過 PNG そのまま / green: 緑背景（クロマキー確認用）/ white: 白背景（印刷・合成確認用）/ blur: 元画像ぼかし背景（ポートレート風）",
                )

            gr.Markdown("### 4️⃣ パイプライン制御")
            with gr.Group():
                crop_pad = gr.Slider(
                    0, 64, value=5, step=1,
                    label="SAM2 bbox の padding (px)",
                    info="SAM2 が検出した bounding box の四辺に加える余白（px、整数）。主目的は髪・衣服端の検出漏れ防止。2K 解像度基準で目安 5px。大きすぎると mask が背景を巻き込み壊れるため、細部が切れる時のみ 10〜30 に増やす。SAM2 なしモードでは無視される。",
                )
                use_guard = gr.Checkbox(
                    value=True,
                    label="SAM2 マスクで tb 結果をガード（誤検出除去）",
                    info="ON: SAM2 マスク外のアルファをゼロに強制し、tb が背景を誤って前景と判定した領域を除去する。OFF: tb の予測をそのまま使用。SAM2 マスクなしモードでは自動的に無効になる。",
                )
                guard_dilate = gr.Slider(
                    1, 81, value=21, step=2,
                    label="ガードマスクの dilate kernel",
                    info="Dilate kernel（膨張カーネル）サイズ（ピクセル）。SAM2 マスクをこのサイズで膨らませて、境界からはみ出た髪の毛・産毛などを tb が拾えるよう余白を確保する。値が大きいほどゆるいガード（細部を拾いやすい）、小さいほどマスクに厳密に従う。",
                )
                decontam = gr.Checkbox(
                    value=True,
                    label="Color decontamination（緑被り除去）",
                    info="ON: pymatting の estimate_foreground_ml で背景色が前景ピクセルに混入する「色汚染」を除去する。グリーンバック撮影などで輪郭が緑がかる場合に有効。処理時間が若干増加する。",
                )
                save_to_disk = gr.Checkbox(
                    value=True,
                    label="結果を outputs/ に保存する",
                    info="ON: RGBA・アルファ・プレビュー画像を outputs/<日時>/ に PNG 保存する。OFF: 画面表示のみで保存しない。真偽値。",
                )

            with gr.Accordion("📖 用語・アルゴリズム解説（クリックで展開）", open=False):
                gr.Markdown("""
**transparent-background（tb）とは**  
ディープラーニングモデル（IS-Net ベース）が各ピクセルに「前景らしさ（アルファ値 0〜255）」を予測する背景除去ツール。  
単色背景がなくても機能し、複雑な背景でも人物・物体を切り抜ける。

**SAM2 マスクの役割**  
Segment Anything Model 2 がクリック・ボックスを手がかりにセグメンテーションマスクを生成する。  
このマスクを tb の「ガード（範囲制限）」として使うことで誤検出を大幅に減らせる。  
**SAM2 なしでも tb 単体で動作するが、tb が背景を前景と誤認識することがある。**

**Alpha matte（アルファマット）**  
各ピクセルの前景透明度を表す 0〜255 のグレースケール画像。  
255 = 完全前景 / 0 = 完全背景 / 中間値 = 半透明（髪・ガラス・煙など）。

**Dilate kernel（膨張カーネル）**  
形態学的処理の一種。SAM2 マスク境界を指定サイズの正方形で膨らませ、  
マスク外縁の細毛などが tb の結果に含まれるよう余白を確保する。

**Soft alpha vs 二値化 threshold**  
- Soft alpha（threshold=0）: 半透明ピクセルを保持。髪の毛やフワフワした輪郭に最適。  
- 二値化（threshold>0）: α ≥ threshold → 不透明 / それ未満 → 透明。輪郭をくっきりさせたい場合に使用。

**Color decontamination（色汚染除去）**  
アルファ合成の計算式 `observed = alpha × fg + (1−alpha) × bg` を逆算して  
純粋な前景色を推定し直す処理。境界付近の緑がかり・色にじみを除去する（pymatting 使用）。
""")

            btn_run = gr.Button("✨ 背景除去を実行（SAM2 なしでも動作）", variant="primary", size="lg")
            run_info = gr.Markdown()

            gr.Markdown("### 5️⃣ 結果")
            with gr.Tabs():
                with gr.Tab("RGBA"):
                    out_rgba = gr.Image(label="透過 PNG", type="numpy", height=400)
                with gr.Tab("Alpha"):
                    out_alpha = gr.Image(label="アルファマスク", type="numpy", height=400)
                with gr.Tab("Preview"):
                    out_preview = gr.Image(
                        label="背景合成プレビュー", type="numpy", height=400
                    )

    # ----- イベント結線 -----
    input_image.upload(
        on_upload, [input_image, state],
        [interactive_image, state, sam_info],
    )
    interactive_image.select(
        on_click, [state, current_label, input_mode],
        [interactive_image, state, sam_info],
    )
    btn_clear.click(
        on_clear, [state],
        [interactive_image, state, sam_info],
    )
    btn_predict.click(
        on_predict, [state, multimask],
        [mask_view_0, mask_view_1, mask_view_2, mask_idx, state, sam_info],
    )
    mask_idx.change(
        on_select_mask, [mask_idx, state],
        [interactive_image, state, sam_info],
    )
    btn_run.click(
        on_run_tb,
        [state, tb_mode, tb_jit, tb_thresh, tb_output, crop_pad,
         use_guard, guard_dilate, decontam, save_to_disk],
        [out_rgba, out_alpha, out_preview, run_info],
    )

# Gradio 5 のキュー設定は launch 前に呼び出す（ERR001 対策）
demo.queue()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM2 + transparent-background Gradio デモ",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Gradio share URL を有効化（Colab で必須）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Gradio debug モードを有効化",
    )
    parser.add_argument(
        "--server-name", default="127.0.0.1",
        help="待ち受けホスト（既定: 127.0.0.1。LAN 公開や Colab share では Gradio が自動で上書き）",
    )
    parser.add_argument(
        "--server-port", type=int, default=None,
        help="待ち受けポート（デフォルト: Gradio 自動選択）",
    )
    args = parser.parse_args()

    # show_api=False: Gradio 5.x の /info エンドポイント schema 変換クラッシュ回避（ERR011）
    demo.launch(
        share=args.share,
        debug=args.debug,
        server_name=args.server_name,
        server_port=args.server_port,
        show_api=False,
    )
