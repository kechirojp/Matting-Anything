"""model_registry のユニットテスト（RED: model_registry.py 未実装時は失敗する）。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# model_registry は pipelines/components/model_registry.py に実装される予定
from pipelines.components.model_registry import (
    ModelEntry,
    build_dropdown_choices,
    entries_for,
    entry_by_id,
    is_available,
    load_model_registry,
)

# テスト用 TOML の最小定義
_TOML_CONTENT = """\
[[detector]]
id = "groundingdino_swint_ogc"
label = "GroundingDINO SwinT-OGC (default)"
component = "GroundingDINODetector"
config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "checkpoints/groundingdino_swint_ogc.pth"

[[tracker]]
id = "sam2_hiera_l"
label = "SAM2.1 Hiera-Large (default)"
component = "SAM2VideoPropagator"
config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint_path = "checkpoints/SAM2/sam2.1_hiera_large.pt"
requires = "sam2_facebook"

[[tracker]]
id = "samurai_hiera_l"
label = "SAMURAI Hiera-Large (motion-aware)"
component = "SAM2VideoPropagator"
config_name = "configs/samurai/sam2.1_hiera_l.yaml"
checkpoint_path = "checkpoints/SAM2/sam2.1_hiera_large.pt"
requires = "sam2_samurai"

[[background]]
id = "tb_base"
label = "transparent-background base"
component = "TransparentBGExtractor"
tb_mode = "base"

[[background]]
id = "tb_fast"
label = "transparent-background fast"
component = "TransparentBGExtractor"
tb_mode = "fast"
"""

_TOML_INVALID_COMPONENT = """\
[[detector]]
id = "evil"
label = "Evil Component"
component = "EvilUnknownComponent"
"""

_TOML_MISSING_REQUIRED_KEY = """\
[[detector]]
label = "Missing ID"
component = "GroundingDINODetector"
"""


@pytest.fixture
def toml_path(tmp_path: Path) -> Path:
    p = tmp_path / "inference_models.toml"
    p.write_text(_TOML_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def invalid_component_toml_path(tmp_path: Path) -> Path:
    p = tmp_path / "invalid_component.toml"
    p.write_text(_TOML_INVALID_COMPONENT, encoding="utf-8")
    return p


@pytest.fixture
def missing_key_toml_path(tmp_path: Path) -> Path:
    p = tmp_path / "missing_key.toml"
    p.write_text(_TOML_MISSING_REQUIRED_KEY, encoding="utf-8")
    return p


# --- load_model_registry ---


def test_load_model_registry_returns_role_keyed_dict(toml_path: Path) -> None:
    registry = load_model_registry(toml_path)

    assert set(registry.keys()) == {"detector", "tracker", "background"}


def test_load_model_registry_detector_count(toml_path: Path) -> None:
    registry = load_model_registry(toml_path)

    assert len(registry["detector"]) == 1


def test_load_model_registry_tracker_count(toml_path: Path) -> None:
    registry = load_model_registry(toml_path)

    assert len(registry["tracker"]) == 2


def test_load_model_registry_background_count(toml_path: Path) -> None:
    registry = load_model_registry(toml_path)

    assert len(registry["background"]) == 2


def test_load_model_registry_entry_fields(toml_path: Path) -> None:
    registry = load_model_registry(toml_path)
    entry = registry["detector"][0]

    assert entry["id"] == "groundingdino_swint_ogc"
    assert entry["label"] == "GroundingDINO SwinT-OGC (default)"
    assert entry["component"] == "GroundingDINODetector"


def test_load_model_registry_invalid_component_raises(invalid_component_toml_path: Path) -> None:
    with pytest.raises(ValueError, match="EvilUnknownComponent"):
        load_model_registry(invalid_component_toml_path)


def test_load_model_registry_missing_id_raises(missing_key_toml_path: Path) -> None:
    with pytest.raises(ValueError, match="'id' が欠落"):
        load_model_registry(missing_key_toml_path)


def test_load_model_registry_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_model_registry(Path("/nonexistent/path/to/file.toml"))


# --- entries_for / entry_by_id ---


def test_entries_for_returns_list(toml_path: Path) -> None:
    load_model_registry(toml_path)
    result = entries_for("detector", registry_path=toml_path)

    assert isinstance(result, list)
    assert len(result) == 1


def test_entry_by_id_returns_matching_entry(toml_path: Path) -> None:
    entry = entry_by_id("background", "tb_base", registry_path=toml_path)

    assert entry["id"] == "tb_base"
    assert entry["tb_mode"] == "base"


def test_entry_by_id_unknown_id_raises(toml_path: Path) -> None:
    with pytest.raises(KeyError):
        entry_by_id("background", "nonexistent", registry_path=toml_path)


def test_entry_by_id_unknown_role_raises(toml_path: Path) -> None:
    with pytest.raises(KeyError):
        entry_by_id("nonexistent_role", "tb_base", registry_path=toml_path)


# --- is_available ---


def test_is_available_no_requires_returns_true(toml_path: Path) -> None:
    entry = entry_by_id("background", "tb_base", registry_path=toml_path)

    assert is_available(entry) is True


def test_is_available_sam2_facebook_with_env_var(toml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_TRACKER_VARIANT", "sam2_facebook")
    entry = entry_by_id("tracker", "sam2_hiera_l", registry_path=toml_path)

    assert is_available(entry) is True


def test_is_available_sam2_facebook_unavailable_for_samurai_env(toml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_TRACKER_VARIANT", "sam2_samurai")
    entry = entry_by_id("tracker", "sam2_hiera_l", registry_path=toml_path)

    assert is_available(entry) is False


def test_is_available_samurai_with_matching_env_var(toml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_TRACKER_VARIANT", "sam2_samurai")
    entry = entry_by_id("tracker", "samurai_hiera_l", registry_path=toml_path)

    assert is_available(entry) is True


def test_is_available_sam2_requires_no_env_var_defaults_to_true(toml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """INFERENCE_TRACKER_VARIANT 未設定なら requires=sam2_* の entry を可用とする（デグレ防止）。"""
    monkeypatch.delenv("INFERENCE_TRACKER_VARIANT", raising=False)
    entry = entry_by_id("tracker", "sam2_hiera_l", registry_path=toml_path)

    assert is_available(entry) is True


# --- build_dropdown_choices ---


def test_build_dropdown_choices_returns_label_id_tuples(toml_path: Path) -> None:
    choices = build_dropdown_choices("background", registry_path=toml_path)

    assert isinstance(choices, list)
    for label, val in choices:
        assert isinstance(label, str)
        assert isinstance(val, str)


def test_build_dropdown_choices_background_has_two_entries(toml_path: Path) -> None:
    choices = build_dropdown_choices("background", registry_path=toml_path)

    assert len(choices) == 2


def test_build_dropdown_choices_excludes_unavailable_tracker(toml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_TRACKER_VARIANT", "sam2_facebook")
    choices = build_dropdown_choices("tracker", registry_path=toml_path)

    ids = [val for _, val in choices]
    assert "sam2_hiera_l" in ids
    assert "samurai_hiera_l" not in ids


def test_build_dropdown_choices_first_entry_is_default(toml_path: Path) -> None:
    """background の最初の entry が choices[0] になる。"""
    choices = build_dropdown_choices("background", registry_path=toml_path)

    assert choices[0][1] == "tb_base"
