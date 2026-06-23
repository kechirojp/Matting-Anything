"""ルートA案（ブラー誘導 → BEN2 再α化）の純関数ユーティリティ。

副作用を持たない関数のみを置く。重いモデルは読み込まない（import 時に checkpoint を
触らない原則を維持）。各関数は GPU 非依存で単体テスト可能であり、ルートAの「合成軸」
（ブラー誘導）の内部機構を機能分割して提供する。

処理フロー（per frame）:
    入力フレーム I + SAM2.1 マスク M
      → ① ``dilate_mask_to_gate`` で M を膨張させゲート G を作る（毛先の逃げ代）
      → ② ``blur_background_outside_gate`` で G 外を強ブラーして I' を作る
      → ③ I' を BEN2 へ（``ben2_components`` 側）
      → ④ ``alpha_to_rgba`` で α を RGBA へ合成
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


# ルートA既定値。config/route_a.toml が無い/キー欠落時のフォールバック。
_DEFAULT_ROUTE_A_CONFIG: dict[str, dict[str, Any]] = {
    "alpha": {
        "ben2_repo_id": "PramaLLC/BEN2",
        "ben2_checkpoint_path": "",
        "refine_foreground": False,
    },
    "blur_guide": {
        "dilation_px": 24,
        "blur_kernel": 41,
        "blur_sigma": 0.0,
        "feather_px": 12,
    },
    "composite": {
        "matte_mode": "union",
        "gate_alpha": False,
        "mask_floor_mode": "none",
    },
}


def _default_route_a_config_path() -> Path:
    """既定の config/route_a.toml パスを返す。"""
    return Path(__file__).resolve().parents[2] / "config" / "route_a.toml"


def load_route_a_config(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """ルートA設定 TOML を読み込み、既定値とマージした dict を返す純関数。

    モデルは一切読まない。ファイルが存在しない場合は既定値をそのまま返す。

    Args:
        path: route_a.toml のパス。None の場合は config/route_a.toml を使う。

    Returns:
        ``{"alpha": {...}, "blur_guide": {...}, "composite": {...}}`` 形式の設定 dict。
        各セクションは既定値で補完される。

    Raises:
        ValueError: TOML として解釈できない場合（tomllib の例外を握り潰さず伝搬）。
    """
    resolved = Path(path) if path is not None else _default_route_a_config_path()
    merged: dict[str, dict[str, Any]] = {
        section: dict(values) for section, values in _DEFAULT_ROUTE_A_CONFIG.items()
    }
    if not resolved.exists():
        return merged
    with resolved.open("rb") as f:
        raw = tomllib.load(f)
    for section, defaults in _DEFAULT_ROUTE_A_CONFIG.items():
        section_raw = raw.get(section, {})
        if not isinstance(section_raw, dict):
            raise ValueError(f"route_a.toml の [{section}] は table である必要があります: {section_raw!r}")
        merged_section = dict(defaults)
        merged_section.update(section_raw)
        merged[section] = merged_section
    return merged


def _coerce_odd_kernel(kernel: int) -> int:
    """ガウシアン kernel サイズを 1 以上の奇数へ補正する。

    Args:
        kernel: 要求 kernel サイズ。

    Returns:
        1 以上の奇数。0 以下は 1 に丸める。
    """
    value = int(kernel)
    if value < 1:
        return 1
    if value % 2 == 0:
        return value + 1
    return value


def _binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """soft 確率 / bool / uint8 mask を bool 前景マスクへ正規化する。

    Args:
        mask: (H,W) の mask。float（[0,1] 確率）/ bool / uint8（0-255）を受け付ける。
        threshold: float mask の前景閾値。

    Returns:
        (H,W) bool の前景マスク。
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask は (H,W) 形式である必要があります: shape={array.shape}")
    if array.dtype == bool:
        return array
    array_float = array.astype(np.float32)
    if array_float.max() > 1.0:
        array_float = array_float / 255.0
    return array_float >= float(threshold)


def dilate_mask_to_gate(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """① SAM2 マスク M を膨張させ、ブラー誘導用のゲート G（二値）を作る。

    毛先の逃げ代を確保するため M を ``dilation_px`` だけ膨張させる。膨張が大きすぎると
    背景を巻き込み、小さすぎると毛先が落ちるため、値はチューニング対象（仕様書 A-4）。

    Args:
        mask: (H,W) の SAM2 マスク。soft 確率 / bool / uint8 を受け付ける。
        dilation_px: 膨張量（px）。0 以下なら二値化のみで膨張しない。

    Returns:
        (H,W) uint8 のゲート G。前景=1, 背景=0。
    """
    binary = _binarize_mask(mask).astype(np.uint8)
    if int(dilation_px) <= 0:
        return binary
    kernel_size = int(dilation_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(binary, kernel, iterations=1).astype(np.uint8)


def blur_background_outside_gate(
    image_rgb: np.ndarray,
    gate: np.ndarray,
    blur_kernel: int = 41,
    blur_sigma: float = 0.0,
    feather_px: int = 12,
) -> np.ndarray:
    """② ゲート G の外側を強ブラーし、被写体をシャープに保った誘導フレーム I' を作る。

    BEN2 は mask 入力を持たないため、G 外を背景としてぼかすことで「どこに注目すべきか」を
    間接誘導する。G 境界は ``feather_px`` で羽根を付け、シャープ域とブラー域の遷移を滑らかにする。

    Args:
        image_rgb: (H,W,3) uint8 の RGB フレーム I。
        gate: (H,W) のゲート G（0/1 もしくは 0-255）。``dilate_mask_to_gate`` の出力。
        blur_kernel: 背景ブラーのガウシアン kernel サイズ（奇数へ自動補正）。
        blur_sigma: ガウシアン sigma。0 で kernel から自動計算。
        feather_px: G 境界の羽根幅（px）。0 で羽根なし。

    Returns:
        (H,W,3) uint8 の誘導フレーム I'。
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb は (H,W,3) 形式である必要があります: shape={image.shape}")
    image_f = image.astype(np.float32)
    kernel = _coerce_odd_kernel(blur_kernel)
    blurred = cv2.GaussianBlur(image_f, (kernel, kernel), float(blur_sigma))

    gate_array = np.asarray(gate)
    gate_f = gate_array.astype(np.float32)
    if gate_f.max() > 1.0:
        gate_f = gate_f / 255.0
    gate_f = np.clip(gate_f, 0.0, 1.0)
    if int(feather_px) > 0:
        feather_kernel = _coerce_odd_kernel(int(feather_px) * 2 + 1)
        gate_f = cv2.GaussianBlur(gate_f, (feather_kernel, feather_kernel), 0)
        gate_f = np.clip(gate_f, 0.0, 1.0)

    weight = gate_f[..., None]
    guided = weight * image_f + (1.0 - weight) * blurred
    return np.clip(guided, 0, 255).astype(np.uint8)


def ben2_rgba_to_alpha(rgba: Any) -> np.ndarray:
    """BEN2 の RGBA 出力から α チャネル（H,W）を取り出す。

    BEN2 ``inference()`` は背景除去後の前景を RGBA で返し、α が matte に相当する。

    Args:
        rgba: PIL.Image（RGBA）もしくは (H,W,4) の np 配列。

    Returns:
        (H,W) uint8 の α。

    Raises:
        ValueError: RGBA でない（α チャネルが無い）場合。
    """
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"BEN2 出力は RGBA (H,W,4) である必要があります: shape={array.shape}")
    return array[..., 3].astype(np.uint8, copy=False)


def alpha_to_rgba(image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """④ 元フレームの RGB と α を結合して RGBA を作る。

    Args:
        image_rgb: (H,W,3) uint8 の RGB フレーム。
        alpha: (H,W) の α。uint8（0-255）または float（[0,1]）。

    Returns:
        (H,W,4) uint8 の RGBA。
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb は (H,W,3) 形式である必要があります: shape={image.shape}")
    alpha_array = np.asarray(alpha)
    if alpha_array.ndim != 2:
        raise ValueError(f"alpha は (H,W) 形式である必要があります: shape={alpha_array.shape}")
    alpha_f = alpha_array.astype(np.float32)
    if alpha_f.max() <= 1.0:
        alpha_f = alpha_f * 255.0
    alpha_u8 = np.clip(alpha_f, 0, 255).astype(np.uint8)
    if alpha_u8.shape != image.shape[:2]:
        alpha_u8 = cv2.resize(alpha_u8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.dstack([image.astype(np.uint8, copy=False), alpha_u8])


def apply_gate_to_alpha(alpha: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """最終 α に膨張ゲート G を乗算し、G 外の遠景誤検出を抑える（任意）。

    純ルートAは誘導のみで G 外を刈らないが、遠景に BEN2 が誤反応する場合の安全弁として
    ``composite.gate_alpha = true`` で有効化できる。

    Args:
        alpha: (H,W) の α。uint8（0-255）または float（[0,1]）。
        gate: (H,W) のゲート G（0/1 もしくは 0-255）。

    Returns:
        (H,W) uint8 のゲート適用後 α。
    """
    alpha_array = np.asarray(alpha)
    alpha_f = alpha_array.astype(np.float32)
    if alpha_f.max() <= 1.0:
        alpha_f = alpha_f * 255.0
    gate_f = np.asarray(gate).astype(np.float32)
    if gate_f.max() > 1.0:
        gate_f = gate_f / 255.0
    gate_f = np.clip(gate_f, 0.0, 1.0)
    if gate_f.shape != alpha_f.shape:
        gate_f = cv2.resize(gate_f, (alpha_f.shape[1], alpha_f.shape[0]), interpolation=cv2.INTER_LINEAR)
    gated = alpha_f * gate_f
    return np.clip(gated, 0, 255).astype(np.uint8)


def combine_alpha_with_mask(alpha: np.ndarray, mask: np.ndarray, mode: str = "none") -> np.ndarray:
    """SAM2 マスク M を α の「床（floor）」として加算的に合成し、BEN2 の α 抜け落ち（ちらつき）を補う。

    ``gate_alpha`` が α を G 内へ絞る（乗算＝減算的）のに対し、本関数は SAM2 が安定追跡している
    領域を α へ底上げ（加算的）する。BEN2 がフレーム単位で被写体を取りこぼしても、SAM2 マスクが
    床を張るため時間方向のちらつきが減る。前景ブラーや背景同系色で BEN2 saliency が不安定な動画向け。

    Args:
        alpha: (H,W) の BEN2 α。uint8（0-255）または float（[0,1]）。
        mask: (H,W) の SAM2 soft マスク M（膨張前）。float（[0,1] 確率）/ bool / uint8（0-255）。
        mode: 合成方式。``"none"``（無効・既定）/ ``"screen"``（1-(1-a)(1-m)）/
            ``"lighten"``（= ``"max"``、画素ごと max。比較明）。

    Returns:
        (H,W) uint8 の合成後 α。``mode="none"`` または ``mask is None`` のとき入力 α をそのまま返す。

    Raises:
        ValueError: mode が未知の場合（黙殺せず通知する）。
    """
    normalized = str(mode).strip().lower()
    if normalized in {"none", "", "off"} or mask is None:
        alpha_f = np.asarray(alpha).astype(np.float32)
        if alpha_f.max() <= 1.0:
            alpha_f = alpha_f * 255.0
        return np.clip(alpha_f, 0, 255).astype(np.uint8)
    alpha_f = np.asarray(alpha).astype(np.float32)
    if alpha_f.max() <= 1.0:
        alpha_f = alpha_f * 255.0
    alpha_norm = np.clip(alpha_f / 255.0, 0.0, 1.0)
    mask_f = np.asarray(mask).astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0
    mask_norm = np.clip(mask_f, 0.0, 1.0)
    if mask_norm.shape != alpha_norm.shape:
        mask_norm = cv2.resize(mask_norm, (alpha_norm.shape[1], alpha_norm.shape[0]), interpolation=cv2.INTER_LINEAR)
    if normalized == "screen":
        combined = 1.0 - (1.0 - alpha_norm) * (1.0 - mask_norm)
    elif normalized in {"lighten", "max"}:
        combined = np.maximum(alpha_norm, mask_norm)
    else:
        raise ValueError(f"mask_floor_mode は 'none' / 'screen' / 'lighten' のいずれかです: {mode!r}")
    return np.clip(combined * 255.0, 0, 255).astype(np.uint8)


def resolve_ben2_device(device: str | None = None) -> str:
    """BEN2 推論に使うデバイス文字列を解決する。

    Args:
        device: 明示デバイス（"cuda" / "cpu" など）。None なら環境から推定する。

    Returns:
        "cuda" もしくは "cpu"。
    """
    if device:
        return device
    env_device = os.environ.get("ROUTE_A_DEVICE")
    if env_device:
        return env_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"
