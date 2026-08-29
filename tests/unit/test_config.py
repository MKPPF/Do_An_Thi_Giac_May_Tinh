from __future__ import annotations

from pathlib import Path

import pytest

from crackspot.config import ConfigError, deep_merge, load_config

BASE = """
experiment:
  name: base
  seed: 42
data:
  image_size: [224, 224]
  require_source_group: true
  split: {train: 0.7, val: 0.15, test: 0.15}
model:
  architecture: MobileNetV2
  output_units: 1
training:
  monitor: val_loss
evaluation:
  threshold: 0.5
"""


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"x": 3}}
    result = deep_merge(base, override)
    assert result == {"a": {"x": 3, "y": 2}}
    assert base["a"]["x"] == 1


def test_load_config_with_relative_extends(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(BASE, encoding="utf-8")
    (tmp_path / "e1.yaml").write_text(
        "extends: base.yaml\nexperiment:\n  name: e1_baseline\n", encoding="utf-8"
    )
    loaded = load_config(tmp_path / "e1.yaml")
    assert loaded.name == "e1_baseline"
    assert loaded.seed == 42
    assert len(loaded.sha256) == 64


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("  image_size: [224, 224]", "  image_size: [128, 128]"),
        ("  require_source_group: true", "  require_source_group: false"),
    ],
)
def test_invalid_data_invariants(tmp_path: Path, original: str, replacement: str) -> None:
    text = BASE.replace(original, replacement)
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
