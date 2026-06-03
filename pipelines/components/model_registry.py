"""推論モデルレジストリローダ。

副作用なしの純粋なローダ。TOML を読むだけでモデルは読まない。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

# Gradio callback が使う Component の既知ホワイトリスト（クラス名文字列）
_KNOWN_COMPONENTS: frozenset[str] = frozenset(
    {
        "GroundingDINODetector",
        "GroundingDINOMultiBoxDetector",
        "SAM2Segmenter",
        "SAM2VideoPropagator",
        "TransparentBGExtractor",
        "MAMAlphaPredictor",
    }
)

# 役割別の必須フィールド
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "detector": ["id", "label", "component"],
    "tracker": ["id", "label", "component"],
    "background": ["id", "label", "component"],
}

# ModelEntry は役割別フィールドを含む dict のエイリアス
ModelEntry = dict[str, Any]

# モジュールレベルのキャッシュ（path -> registry）
_REGISTRY_CACHE: dict[Path, dict[str, list[ModelEntry]]] = {}
_REGISTRY_CACHE_LOCK = Lock()


def _get_default_registry_path() -> Path:
    """デフォルトの inference_models.toml パスを返す。"""
    return Path(__file__).resolve().parents[2] / "config" / "inference_models.toml"


def load_model_registry(path: Path | str | None = None) -> dict[str, list[ModelEntry]]:
    """TOML からモデルレジストリを読み込み、役割別 entry list の dict を返す。

    モデルは一切読まない（import 時に checkpoint を触らない原則を維持）。

    Args:
        path: inference_models.toml のパス。None の場合は config/inference_models.toml を使う。

    Returns:
        dict: 役割（"detector" / "tracker" / "background"）→ ModelEntry list。

    Raises:
        FileNotFoundError: TOML ファイルが存在しない場合。
        ValueError: 必須フィールド欠落または未知 component クラス名の場合。
    """
    resolved = Path(path) if path is not None else _get_default_registry_path()

    with _REGISTRY_CACHE_LOCK:
        if resolved in _REGISTRY_CACHE:
            return _REGISTRY_CACHE[resolved]

        if not resolved.exists():
            raise FileNotFoundError(f"モデルレジストリ TOML が見つかりません: {resolved}")

        with resolved.open("rb") as f:
            raw = tomllib.load(f)

        registry: dict[str, list[ModelEntry]] = {}

        for role in ("detector", "tracker", "background"):
            entries_raw = raw.get(role, [])
            required = _REQUIRED_FIELDS.get(role, ["id", "label", "component"])
            entries: list[ModelEntry] = []

            for entry in entries_raw:
                # 必須フィールドチェック
                for field in required:
                    if field not in entry:
                        raise ValueError(
                            f"[{role}] entry に必須フィールド '{field}' が欠落しています: {entry}"
                        )
                # ホワイトリストチェック（未知 component は明示 raise）
                component = entry["component"]
                if component not in _KNOWN_COMPONENTS:
                    raise ValueError(
                        f"[{role}] entry の component '{component}' は未知です。"
                        f" 既知の Component: {sorted(_KNOWN_COMPONENTS)}"
                    )
                entries.append(dict(entry))

            if entries:
                registry[role] = entries

        _REGISTRY_CACHE[resolved] = registry
        return registry


def clear_registry_cache(path: Path | str | None = None) -> None:
    """モデルレジストリのキャッシュをクリアする。

    テスト・設定ファイルの動的再読み込み時に使用する。
    ``path`` が None の場合は全キャッシュをクリアする。

    Args:
        path: クリアするキャッシュのパス。None の場合は全件クリア。
    """
    with _REGISTRY_CACHE_LOCK:
        if path is None:
            _REGISTRY_CACHE.clear()
        else:
            _REGISTRY_CACHE.pop(Path(path), None)


def entries_for(role: str, registry_path: Path | str | None = None) -> list[ModelEntry]:
    """指定した役割の全 entry を返す。

    Args:
        role: "detector" / "tracker" / "background" のいずれか。
        registry_path: TOML パス（None でデフォルト）。

    Returns:
        ModelEntry のリスト。役割が存在しない場合は空リスト。
    """
    registry = load_model_registry(registry_path)
    return list(registry.get(role, []))


def entry_by_id(role: str, entry_id: str, registry_path: Path | str | None = None) -> ModelEntry:
    """指定した役割・id の entry を返す。

    Args:
        role: "detector" / "tracker" / "background" のいずれか。
        entry_id: entry の id フィールド値。
        registry_path: TOML パス（None でデフォルト）。

    Returns:
        一致する ModelEntry。

    Raises:
        KeyError: 役割または id が見つからない場合。
    """
    registry = load_model_registry(registry_path)
    if role not in registry:
        raise KeyError(f"役割 '{role}' がレジストリに存在しません。利用可能: {list(registry.keys())}")
    for entry in registry[role]:
        if entry["id"] == entry_id:
            return entry
    raise KeyError(f"役割 '{role}' に id='{entry_id}' の entry が存在しません。")


def is_available(entry: ModelEntry) -> bool:
    """entry が現在の環境で利用可能かどうかを返す。

    `requires` フィールドが未指定の場合は常に True。
    `requires` が "sam2_*" の場合は環境変数 ``INFERENCE_TRACKER_VARIANT`` を参照する。
    環境変数が未設定の場合は True（後方互換性）。

    Args:
        entry: ModelEntry dict。

    Returns:
        bool: 利用可能なら True。
    """
    requires = entry.get("requires")
    if requires is None:
        return True

    if requires.startswith("sam2"):
        variant = os.environ.get("INFERENCE_TRACKER_VARIANT")
        if variant is None:
            # 環境変数未設定 → デグレ防止のため可用とみなす
            return True
        return variant == requires

    # 認識できない requires 値は将来拡張用。警告を出して可用とみなす。
    warnings.warn(
        f"model_registry: 未知の requires 値 '{requires}' を検出しました。"
        " 常時利用可能として扱います。",
        UserWarning,
        stacklevel=2,
    )
    return True


def build_dropdown_choices(
    role: str, registry_path: Path | str | None = None
) -> list[tuple[str, str]]:
    """役割に属する可用 entry の (label, id) タプルリストを返す純関数。

    Gradio の ``gr.Dropdown(choices=...)`` に直接渡せる形式。
    モデルは読まない。

    Args:
        role: "detector" / "tracker" / "background" のいずれか。
        registry_path: TOML パス（None でデフォルト）。

    Returns:
        [(label, id), ...] の順序付きリスト。可用 entry のみ。
    """
    all_entries = entries_for(role, registry_path)
    return [(e["label"], e["id"]) for e in all_entries if is_available(e)]
