"""Safe artifact serialization and atomic JSON output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any, *, overwrite: bool = True) -> Path:
    """Atomically write UTF-8 JSON so interrupted runs do not leave partial evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
