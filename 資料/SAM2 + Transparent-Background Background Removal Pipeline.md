

SAM2 + transparent-background パイプライン (Google Colab + Gradio 5)

了解です。親会社が後で見ても「これは引き継げる」と思える構成にします。
- ✅ チェックポイントは Google Drive 集約（差し替えだけで OK）
- ✅ SAM2 のインタラクティブ機能（point / box / multi-mask）を全部 UI に出す
- ✅ transparent-background の全パラメータを UI で操作
- ✅ 中間結果（SAM2マスク、tbアルファ、合成後）全部見える
---

📓 Colab Notebook（セル分割）


Cell 1：環境セットアップ

```
# ============================================================
# Cell 1: 依存関係インストール
# ============================================================
!pip install -q gradio==5.9.1
!pip install -q transparent-background
!pip install -q pymatting  # color decontamination 後処理用

# SAM2 (Meta公式の sam2 リポジトリ)
!pip install -q git+https://github.com/facebookresearch/sam2.git

# 念のため
!pip install -q opencv-python-headless pillow numpy
print("✅ Install done")

```

---

Cell 2：Google Drive マウント＆チェックポイント配置

```
# ============================================================
# Cell 2: Drive マウント + チェックポイントパス設定
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

# ====== ここを親会社のフォルダ構成に合わせて変更 ======
DRIVE_ROOT = Path('/content/drive/MyDrive/bg_removal_ckpts')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# SAM2 checkpoint（差し替えたい場合はここを変えるだけ）
SAM2_CKPT_PATH   = DRIVE_ROOT / 'sam2.1_hiera_large.pt'
SAM2_CONFIG_NAME = 'configs/sam2.1/sam2.1_hiera_l.yaml'  # パッケージ内パス

# transparent-background checkpoint（モード別）
TB_CKPT_BASE        = DRIVE_ROOT / 'ckpt_base.pth'
TB_CKPT_FAST        = DRIVE_ROOT / 'ckpt_fast.pth'
TB_CKPT_BASE_NIGHT  = DRIVE_ROOT / 'ckpt_base_nightly.pth'
# =====================================================

# ----- 初回のみダウンロード（Drive に無ければ取得） -----
def fetch_if_missing(path: Path, url: str):
    if path.exists():
        print(f"✅ Found: {path.name}")
        return
    print(f"⬇️  Downloading: {path.name}")
    os.system(f'wget -q -O "{path}" "{url}"')
    print(f"✅ Saved to Drive: {path}")

# SAM2.1 large
fetch_if_missing(
    SAM2_CKPT_PATH,
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)

# transparent-background (公式リリースから自動DLされるので、Drive保存は手動でOK)
# 初回はライブラリのデフォルトキャッシュからコピーする運用にすると楽
print("\n📂 Drive checkpoints:")
for p in DRIVE_ROOT.iterdir():
    print(f"  - {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

```

---

Cell 3：SAM2 ラッパー

```
# ============================================================
# Cell 3: SAM2 Predictor ラッパー
# ============================================================
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ----- SAM2 ロード -----
sam2_model = build_sam2(
    SAM2_CONFIG_NAME,
    str(SAM2_CKPT_PATH),
    device=DEVICE,
)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("✅ SAM2 loaded")


def sam2_predict(image_np, points, labels, box=None, multimask=True):
    """
    image_np: HxWx3 uint8 RGB
    points:   [(x,y), ...] or None
    labels:   [1/0, ...] (1=positive, 0=negative)
    box:      [x1,y1,x2,y2] or None
    returns:  masks (N,H,W) bool, scores (N,)
    """
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        sam2_predictor.set_image(image_np)
        kwargs = {"multimask_output": multimask}
        if points is not None and len(points) > 0:
            kwargs["point_coords"] = np.array(points, dtype=np.float32)
            kwargs["point_labels"] = np.array(labels, dtype=np.int32)
        if box is not None:
            kwargs["box"] = np.array(box, dtype=np.float32)
        masks, scores, _ = sam2_predictor.predict(**kwargs)
    return masks, scores

```

---

Cell 4：transparent-background ラッパー（パラメータ全部出し）

```
# ============================================================
# Cell 4: transparent-background Remover ラッパー
# ============================================================
from transparent_background import Remover

_remover_cache = {}  # mode -> Remover インスタンス（キャッシュ）

def get_remover(mode='base', jit=False, ckpt_path=None):
    """モードごとに Remover をキャッシュ"""
    key = (mode, jit, str(ckpt_path))
    if key not in _remover_cache:
        kwargs = dict(mode=mode, jit=jit, device=DEVICE)
        if ckpt_path and Path(ckpt_path).exists():
            kwargs['ckpt'] = str(ckpt_path)
        _remover_cache[key] = Remover(**kwargs)
        print(f"✅ Loaded Remover(mode={mode}, jit={jit})")
    return _remover_cache[key]


def tb_process(pil_image, mode='base', jit=False, threshold=None,
               output_type='rgba', ckpt_path=None):
    """
    pil_image:   PIL.Image
    mode:        'base' / 'fast' / 'base-nightly'
    threshold:   None or 0.0~1.0（None=ソフトα、値指定で二値化）
    output_type: 'rgba' / 'map' / 'green' / 'white' / 'blur' / 'overlay'
    """
    remover = get_remover(mode=mode, jit=jit, ckpt_path=ckpt_path)
    out = remover.process(pil_image, type=output_type, threshold=threshold)
    return out

```

---

Cell 5：合成パイプライン（SAM2 → crop → tb → paste back）

```
# ============================================================
# Cell 5: 合成パイプライン
# ============================================================
import cv2

def bbox_from_mask(mask, padding=20, img_shape=None):
    """bool マスクから padding 込み bbox を返す"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    H, W = img_shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W, x2 + padding)
    y2 = min(H, y2 + padding)
    return int(x1), int(y1), int(x2), int(y2)


def dilate_mask(mask, kernel_size=15):
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)


def run_pipeline(image_pil, sam2_mask, tb_mode, tb_jit, tb_threshold,
                 tb_output_type, crop_padding, use_sam2_as_guard,
                 sam2_guard_dilate, apply_decontam):
    """
    SAM2マスクで前景を切り出し → tb でα抽出 → 元位置に戻す → SAM2ガード
    """
    img_np = np.array(image_pil.convert('RGB'))
    H, W = img_np.shape[:2]

    # 1) SAM2マスクからbbox（ない場合は全体）
    if sam2_mask is not None and sam2_mask.any():
        bbox = bbox_from_mask(sam2_mask, padding=crop_padding, img_shape=img_np.shape)
        x1, y1, x2, y2 = bbox
        crop_np = img_np[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, W, H
        crop_np = img_np

    crop_pil = Image.fromarray(crop_np)

    # 2) transparent-background でα抽出（出力はRGBA）
    rgba_crop = tb_process(
        crop_pil, mode=tb_mode, jit=tb_jit,
        threshold=tb_threshold if tb_threshold > 0 else None,
        output_type='rgba',
    )
    rgba_crop_np = np.array(rgba_crop)  # (h,w,4)
    alpha_crop = rgba_crop_np[..., 3].astype(np.float32) / 255.0
    rgb_crop = rgba_crop_np[..., :3]

    # 3) 元サイズキャンバスに貼り戻し
    full_alpha = np.zeros((H, W), dtype=np.float32)
    full_alpha[y1:y2, x1:x2] = alpha_crop
    full_rgb = img_np.copy()
    full_rgb[y1:y2, x1:x2] = rgb_crop

    # 4) SAM2マスクでガード（tbの誤検出を除去）
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
            print(f"Decontam skipped: {e}")

    # 6) 出力（RGBA & プレビュー用）
    alpha_u8 = (full_alpha * 255).astype(np.uint8)
    rgba = np.dstack([full_rgb, alpha_u8])

    # 出力タイプ別レンダリング
    if tb_output_type == 'green':
        bg = np.full_like(full_rgb, [0, 255, 0])
    elif tb_output_type == 'white':
        bg = np.full_like(full_rgb, [255, 255, 255])
    elif tb_output_type == 'blur':
        bg = cv2.GaussianBlur(img_np, (51, 51), 0)
    else:
        bg = None

    if bg is not None:
        a3 = full_alpha[..., None]
        preview = (full_rgb * a3 + bg * (1 - a3)).astype(np.uint8)
    else:
        preview = rgba  # rgba そのまま

    return rgba, alpha_u8, preview

```

---

Cell 6：Gradio 5 UI

```
# ============================================================
# Cell 6: Gradio 5 UI
# ============================================================
import gradio as gr

# UI 状態
INITIAL_STATE = {
    "image": None,        # 元画像 (np)
    "points": [],         # [(x,y), ...]
    "labels": [],         # [1/0, ...]
    "box": None,          # [x1,y1,x2,y2] or None
    "box_buffer": [],     # box指定用の一時クリック
    "masks": None,        # SAM2 候補マスク (N,H,W)
    "scores": None,
    "selected_mask": None,
    "input_mode": "point" # point / box
}


def draw_overlay(img_np, points, labels, box, mask=None):
    """クリック点・box・マスクを画像に重ねる"""
    vis = img_np.copy()
    if mask is not None:
        color = np.array([30, 144, 255], dtype=np.uint8)
        m = mask.astype(bool)
        vis[m] = (vis[m] * 0.5 + color * 0.5).astype(np.uint8)
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 215, 0), 3)
    for (x, y), lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(vis, (int(x), int(y)), 8, color, -1)
        cv2.circle(vis, (int(x), int(y)), 9, (255, 255, 255), 2)
    return vis


# ---------- イベントハンドラ ----------
def on_upload(img, state):
    if img is None:
        return None, state, "画像をアップロードしてください"
    state = {**INITIAL_STATE}
    state["image"] = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else img
    return state["image"], state, "✅ 画像読み込み完了。点 or boxで対象を指定"


def on_click(evt: gr.SelectData, state, current_label, current_mode):
    if state["image"] is None:
        return None, state, "先に画像をアップロードしてください"
    x, y = evt.index
    state["input_mode"] = current_mode

    if current_mode == "point":
        state["points"].append((x, y))
        state["labels"].append(1 if current_label == "positive" else 0)
        msg = f"📍 Point追加: ({x},{y}) {current_label}  / 計{len(state['points'])}点"
    else:  # box
        state["box_buffer"].append((x, y))
        if len(state["box_buffer"]) == 2:
            (x1, y1), (x2, y2) = state["box_buffer"]
            state["box"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            state["box_buffer"] = []
            msg = f"📦 Box確定: {state['box']}"
        else:
            msg = "📦 Box左上クリック完了 → 右下をクリック"

    vis = draw_overlay(state["image"], state["points"], state["labels"],
                       state["box"], state.get("selected_mask"))
    return vis, state, msg


def on_clear(state):
    if state["image"] is None:
        return None, state, ""
    state["points"] = []
    state["labels"] = []
    state["box"] = None
    state["box_buffer"] = []
    state["masks"] = None
    state["selected_mask"] = None
    return state["image"], state, "🧹 クリアしました"


def on_predict(state, multimask):
    if state["image"] is None:
        return None, None, None, None, state, "画像なし"
    if len(state["points"]) == 0 and state["box"] is None:
        return None, None, None, None, state, "Point か Box を指定してください"

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
    # 3枚に満たない場合の埋め
    while len(previews) < 3:
        previews.append(None)

    info = " | ".join([f"Mask{i}: {s:.3f}" for i, s in enumerate(scores)])
    info += f"  → ベスト=Mask{best}"
    return previews[0], previews[1], previews[2], best, state, info


def on_select_mask(idx, state):
    if state["masks"] is None:
        return None, state, ""
    state["selected_mask"] = state["masks"][int(idx)]
    vis = draw_overlay(state["image"], state["points"], state["labels"],
                       state["box"], state["selected_mask"])
    return vis, state, f"✅ Mask{idx} を採用"


def on_run_tb(state, tb_mode, tb_jit, tb_thresh, tb_output, crop_pad,
              use_guard, guard_dilate, decontam):
    if state["image"] is None or state["selected_mask"] is None:
        return None, None, None, "SAM2マスクが未確定です"
    pil_img = Image.fromarray(state["image"])
    rgba, alpha, preview = run_pipeline(
        pil_img, state["selected_mask"],
        tb_mode=tb_mode, tb_jit=tb_jit,
        tb_threshold=tb_thresh, tb_output_type=tb_output,
        crop_padding=int(crop_pad),
        use_sam2_as_guard=use_guard,
        sam2_guard_dilate=int(guard_dilate),
        apply_decontam=decontam,
    )
    return rgba, alpha, preview, "✅ 抽出完了"


# ---------- UI 構築 ----------
with gr.Blocks(title="SAM2 + transparent-background", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 SAM2 + transparent-background パイプライン")
    gr.Markdown("**手順**: 画像アップ → SAM2で対象指定 → マスク選択 → tbパラメータ調整 → 抽出")

    state = gr.State(value=dict(INITIAL_STATE))

    with gr.Row():
        # ============ 左カラム: SAM2 ============
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ SAM2 で対象を指定")
            input_image = gr.Image(label="画像アップロード", type="pil", height=400)
            interactive_image = gr.Image(label="クリックして指定（緑=前景/赤=背景/黄=box）",
                                         interactive=True, height=400)

            with gr.Row():
                input_mode = gr.Radio(["point", "box"], value="point", label="入力モード")
                current_label = gr.Radio(["positive", "negative"], value="positive",
                                         label="Point種別 (positive=対象/negative=除外)")

            with gr.Row():
                btn_clear = gr.Button("🧹 全クリア", variant="secondary")
                multimask = gr.Checkbox(value=True, label="複数候補マスク出力")
                btn_predict = gr.Button("🚀 SAM2 推論", variant="primary")

            sam_info = gr.Markdown()

            gr.Markdown("### 2️⃣ 候補マスクから採用を選ぶ")
            with gr.Row():
                mask_view_0 = gr.Image(label="Mask 0", height=180)
                mask_view_1 = gr.Image(label="Mask 1", height=180)
                mask_view_2 = gr.Image(label="Mask 2", height=180)
            mask_idx = gr.Radio([0, 1, 2], value=0, label="採用マスク")

        # ============ 右カラム: transparent-background ============
        with gr.Column(scale=1):
            gr.Markdown("### 3️⃣ transparent-background パラメータ")

            with gr.Group():
                tb_mode = gr.Radio(
                    ["base", "fast", "base-nightly"],
                    value="base", label="モード"
                )
                tb_jit = gr.Checkbox(value=False, label="JIT（高速化、初回コンパイル時間あり）")
                tb_thresh = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                      label="Threshold (0=ソフトα / >0で二値化)")
                tb_output = gr.Radio(
                    ["rgba", "green", "white", "blur"],
                    value="rgba", label="プレビュー背景"
                )

            gr.Markdown("### 4️⃣ パイプライン制御")
            with gr.Group():
                crop_pad = gr.Slider(0, 200, value=40, step=5,
                                     label="SAM2 bbox の padding (px)")
                use_guard = gr.Checkbox(value=True,
                                        label="SAM2マスクでtb結果をガード（誤検出除去）")
                guard_dilate = gr.Slider(1, 81, value=21, step=2,
                                         label="ガードマスクの dilate kernel")
                decontam = gr.Checkbox(value=True,
                                       label="Color decontamination（緑被り除去）")

            btn_run = gr.Button("✨ 背景除去を実行", variant="primary", size="lg")
            run_info = gr.Markdown()

            gr.Markdown("### 5️⃣ 結果")
            with gr.Tabs():
                with gr.Tab("RGBA"):
                    out_rgba = gr.Image(label="透過PNG", type="numpy", height=400)
                with gr.Tab("Alpha"):
                    out_alpha = gr.Image(label="アルファマスク", type="numpy", height=400)
                with gr.Tab("Preview"):
                    out_preview = gr.Image(label="背景合成プレビュー", type="numpy", height=400)

    # ----- イベント結線 -----
    input_image.upload(
        on_upload, [input_image, state],
        [interactive_image, state, sam_info]
    )
    interactive_image.select(
        on_click, [state, current_label, input_mode],
        [interactive_image, state, sam_info]
    )
    btn_clear.click(
        on_clear, [state],
        [interactive_image, state, sam_info]
    )
    btn_predict.click(
        on_predict, [state, multimask],
        [mask_view_0, mask_view_1, mask_view_2, mask_idx, state, sam_info]
    )
    mask_idx.change(
        on_select_mask, [mask_idx, state],
        [interactive_image, state, sam_info]
    )
    btn_run.click(
        on_run_tb,
        [state, tb_mode, tb_jit, tb_thresh, tb_output, crop_pad,
         use_guard, guard_dilate, decontam],
        [out_rgba, out_alpha, out_preview, run_info]
    )

demo.launch(share=True, debug=False)

```

---

🔑 設計ポイント解説


SAM2 機能の活かし方

UI要素
SAM2機能
Point モード（positive/negative）
クリックプロンプト
Box モード（2点クリック）
Boxプロンプト
複数候補マスク
multimask_output=True で3候補
マスク選択ラジオ
候補から人手で採用決定
Point + Box 併用可
プロンプトの mix

transparent-background 調整パラメータ

パラメータ
UI
用途
mode
base/fast/base-nightly
モデル切替
jit
チェックボックス
推論高速化
threshold
0〜1スライダー
ソフトα vs 二値化
output_type
プレビュー切替
確認用背景色

パイプライン制御（独自）

- crop padding：bbox周りの余白（細毛が切れないように）
- SAM2 guard：tbの誤検出を SAM2 マスクで殴り消す
- dilate kernel：ガードマスクの膨張量（細毛のはみ出し許容）
- decontam：pymattingで緑被り除去

親会社へのバトンパス設計

- DRIVE_ROOT 配下のckptを差し替えるだけでモデル更新可
- SAM2_CONFIG_NAME で SAM2 のサイズ変更可（large→base_plus 等）
- Cell 4 の Remover(ckpt=...) でカスタムckptを読める
---

🧪 まず試してほしいこと

1. 今回の逆光細毛画像で実行
2. SAM2 で人物を point 1〜2発でセグメント
3. tb は最初 mode=base, threshold=0, decontam=ON で実行
4. 細毛が切れる → crop_pad を 80〜120 に上げる
5. 顔以外が混じる → guard_dilate を 31〜51 に上げる
6. 緑被りが残る → decontam ON のまま threshold を 0.05 程度に
---

❓ 確認

このまま貼って動かせる構成ですが、以下どうしますか？
- A: このまま実行 → 動作確認後に細部調整
- B: 親会社の Drive 構成（フォルダ名）に先に合わせる → パス書き換え
- C: tb の追加パラメータ（reverse、custom resize等）も UI に出す
- D: 複数オブジェクトの逐次セグメント（人＋自転車を別alphaで）対応も入れる
個人的には A → 結果見て C/D 判断 が早いと思います。
