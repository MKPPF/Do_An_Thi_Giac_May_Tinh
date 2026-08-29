from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from crackspot.data.integrity import (
    DatasetIntegrityError,
    verify_dataset_integrity,
    verify_official_dataset_preconditions,
)
from crackspot.utils.hashing import sha256_file


def _manifest(root: Path) -> pd.DataFrame:
    first = root / "D" / "CD" / "001-1.jpg"
    second = root / "P" / "UP" / "002-1.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    return pd.DataFrame(
        {
            "relative_path": ["P/UP/002-1.jpg", "D/CD/001-1.jpg"],
            "sha256": [sha256_file(second), sha256_file(first)],
        }
    )


def test_dataset_integrity_hashes_every_file_deterministically(tmp_path: Path) -> None:
    frame = _manifest(tmp_path)

    first = verify_dataset_integrity(frame, tmp_path, workers=1)
    second = verify_dataset_integrity(frame.sample(frac=1), tmp_path, workers=2)

    assert first.status == "DATASET_BYTES_VERIFIED"
    assert first.checked_rows == 2
    assert first.checked_bytes == len(b"first-image") + len(b"second-image")
    assert first.content_fingerprint_sha256 == second.content_fingerprint_sha256
    assert first.verification_scope == "relative_path_sha256_and_encoded_file_bytes_only"
    assert first.label_values_accessed is False


def test_dataset_integrity_rejects_image_changed_after_audit(tmp_path: Path) -> None:
    frame = _manifest(tmp_path)
    (tmp_path / "D" / "CD" / "001-1.jpg").write_bytes(b"tampered")

    with pytest.raises(DatasetIntegrityError, match="does not match manifest"):
        verify_dataset_integrity(frame, tmp_path, workers=1)


@pytest.mark.parametrize("relative_path", ["../escape.jpg", "C:/escape.jpg", ""])
def test_dataset_integrity_rejects_unsafe_paths(tmp_path: Path, relative_path: str) -> None:
    frame = pd.DataFrame({"relative_path": [relative_path], "sha256": ["a" * 64]})

    with pytest.raises(DatasetIntegrityError, match="relative_path"):
        verify_dataset_integrity(frame, tmp_path, workers=1)


def test_dataset_integrity_rejects_malformed_hash_before_reading(tmp_path: Path) -> None:
    frame = pd.DataFrame({"relative_path": ["missing.jpg"], "sha256": ["not-a-hash"]})

    with pytest.raises(DatasetIntegrityError, match="malformed SHA-256"):
        verify_dataset_integrity(frame, tmp_path, workers=1)


def test_official_preconditions_bind_exact_bundle_manifest_and_byte_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crackspot.data import split
    from crackspot.data.split import manifest_sha256

    frame = _manifest(tmp_path / "images")
    bundle_dir = tmp_path / "split_v1"
    bundle_dir.mkdir()
    manifest_path = bundle_dir / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    inventory = bundle_dir / "manifest_hashes.json"
    completion = bundle_dir / "split_complete.json"
    inventory.write_bytes(b"inventory")
    completion.write_bytes(b"completion")
    fake_bundle = SimpleNamespace(
        directory=bundle_dir.resolve(),
        manifest_path=manifest_path.resolve(),
        inventory_path=inventory,
        completion_path=completion,
        manifest_sha256=manifest_sha256(frame),
    )
    monkeypatch.setattr(split, "verify_locked_split_bundle", lambda path: fake_bundle)

    result = verify_official_dataset_preconditions(
        frame,
        manifest_path,
        tmp_path / "images",
        workers=1,
    )

    assert result.status == "OFFICIAL_DATASET_PRECONDITIONS_VERIFIED"
    assert result.manifest_sha256 == manifest_sha256(frame)
    assert result.dataset_integrity.checked_rows == 2
    assert result.dataset_integrity.label_values_accessed is False
    assert result.split_inventory_sha256 == sha256_file(inventory)
    assert result.split_completion_sha256 == sha256_file(completion)


def test_official_preconditions_reject_manifest_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crackspot.data import split
    from crackspot.data.split import manifest_sha256

    frame = _manifest(tmp_path / "images")
    manifest_path = tmp_path / "copied.csv"
    manifest_path.write_text("copy", encoding="utf-8")
    other = tmp_path / "bundle" / "manifest.csv"
    other.parent.mkdir()
    other.write_text("locked", encoding="utf-8")
    fake_bundle = SimpleNamespace(
        manifest_path=other,
        manifest_sha256=manifest_sha256(frame),
    )
    monkeypatch.setattr(split, "verify_locked_split_bundle", lambda path: fake_bundle)

    with pytest.raises(DatasetIntegrityError, match="bundle's own manifest"):
        verify_official_dataset_preconditions(frame, manifest_path, tmp_path / "images")
