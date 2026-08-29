"""Validated YAML configuration loading with deterministic deep merging."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from crackspot.constants import DEFAULT_THRESHOLD, IMAGE_SIZE, SEED
from crackspot.utils.hashing import sha256_json


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either argument."""

    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Không tìm thấy file cấu hình: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Cấu hình phải là mapping YAML: {path}")
    return payload


def _resolve_base(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base_ref = config.pop("extends", None)
    defaults = config.pop("defaults", None)
    if base_ref and defaults:
        raise ConfigError("Chỉ dùng một trong extends hoặc defaults")
    if defaults:
        if not isinstance(defaults, list) or len(defaults) != 1:
            raise ConfigError("defaults hiện chỉ hỗ trợ một base YAML")
        base_ref = defaults[0]
    if not base_ref:
        return config
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    base = _read_yaml(base_path)
    base = _resolve_base(base, base_path)
    return deep_merge(base, config)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate invariants that protect comparability and label semantics."""

    experiment = config.get("experiment", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})

    name = experiment.get("name") or experiment.get("slug") or experiment.get("id")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("Thiếu experiment.name")

    seed = int(experiment.get("seed", config.get("seed", SEED)))
    if seed < 0:
        raise ConfigError("experiment.seed phải không âm")

    image_size = tuple(
        data.get("image_size", config.get("pipeline", {}).get("image_size", IMAGE_SIZE))
    )
    if image_size != IMAGE_SIZE:
        raise ConfigError(f"CrackSpot yêu cầu image_size={IMAGE_SIZE}, nhận {image_size}")

    if model.get("architecture", "MobileNetV2") != "MobileNetV2":
        raise ConfigError("Mô hình chính bắt buộc là MobileNetV2")
    if int(model.get("output_units", 1)) != 1:
        raise ConfigError("Đầu ra bắt buộc là một neuron sigmoid P(Crack)")

    monitor = training.get("monitor", training.get("checkpoint_monitor", "val_loss"))
    if monitor != "val_loss":
        raise ConfigError("Checkpoint/model selection bắt buộc monitor val_loss")

    evaluation = config.get("evaluation", {})
    threshold = float(
        evaluation.get(
            "threshold",
            evaluation.get("test_threshold", training.get("default_threshold", DEFAULT_THRESHOLD)),
        )
    )
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("evaluation.threshold phải nằm trong [0, 1]")

    split = data.get("split", data.get("split_ratios", {}))
    ratios = [
        float(split.get(key, split.get(alias, default)))
        for key, alias, default in (
            ("train", "train", 0.7),
            ("val", "validation", 0.15),
            ("test", "test", 0.15),
        )
    ]
    if abs(sum(ratios) - 1.0) > 1e-9 or any(value <= 0 for value in ratios):
        raise ConfigError("Tỷ lệ train/val/test phải dương và có tổng bằng 1")
    require_source_group = data.get(
        "require_source_group", data.get("fail_without_verified_group", True)
    )
    if require_source_group is not True:
        raise ConfigError("Full experiment bắt buộc require_source_group=true")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable loaded configuration plus its content hash."""

    path: Path
    values: dict[str, Any]
    sha256: str

    @property
    def name(self) -> str:
        experiment = self.values["experiment"]
        return str(experiment.get("name") or experiment.get("slug") or experiment["id"])

    @property
    def seed(self) -> int:
        return int(self.values.get("experiment", {}).get("seed", self.values.get("seed", SEED)))

    def to_json(self) -> str:
        return json.dumps(self.values, ensure_ascii=False, sort_keys=True, indent=2)


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    values = _resolve_base(_read_yaml(config_path), config_path)
    if overrides:
        values = deep_merge(values, overrides)
    validate_config(values)
    return ExperimentConfig(path=config_path, values=values, sha256=sha256_json(values))
