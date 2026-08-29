from __future__ import annotations

from pathlib import Path

from crackspot.utils.hashing import sha256_file, sha256_json


def test_json_hash_is_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
