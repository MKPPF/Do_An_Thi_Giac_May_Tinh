"""Stable SHA-256 helpers used to lock data, configuration, and checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO


def sha256_stream(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return sha256_stream(handle)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
