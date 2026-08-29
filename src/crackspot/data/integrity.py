"""Verify that dataset bytes still match an audited manifest."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd

from crackspot.utils.hashing import sha256_file


class DatasetIntegrityError(RuntimeError):
    """Raised when a locked manifest no longer describes the dataset bytes."""


@dataclass(frozen=True, slots=True)
class DatasetIntegrityReport:
    """Deterministic summary of one complete byte-integrity pass."""

    schema_version: int
    status: str
    checked_rows: int
    checked_bytes: int
    content_fingerprint_sha256: str
    dataset_root: str
    verification_scope: str
    label_values_accessed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OfficialDatasetPreconditions:
    """Verified split lineage plus one complete encoded-byte integrity pass."""

    schema_version: int
    status: str
    manifest_path: str
    manifest_sha256: str
    locked_split_bundle: str
    split_inventory_sha256: str
    split_completion_sha256: str
    dataset_integrity: DatasetIntegrityReport

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_relative_path(value: object) -> str:
    if value is None or value is pd.NA:
        raise DatasetIntegrityError("relative_path cannot be blank")
    text = str(value).strip().replace("\\", "/")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or not posix.parts
        or posix.as_posix() == "."
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
    ):
        raise DatasetIntegrityError(f"unsafe relative_path: {value!r}")
    return posix.as_posix()


def _resolve_image(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DatasetIntegrityError(f"relative_path escapes dataset root: {relative_path}") from exc
    if not candidate.is_file():
        raise DatasetIntegrityError(f"dataset image is missing: {relative_path}")
    return candidate


def verify_dataset_integrity(
    frame: pd.DataFrame,
    dataset_root: str | Path,
    *,
    workers: int | None = None,
) -> DatasetIntegrityReport:
    """Hash every manifest image and reject any missing, changed, or unsafe file.

    Rows are verified in canonical path order.  The returned fingerprint covers
    each relative path, audited SHA-256, and observed byte size; it is evidence
    of the pass, while the manifest's own canonical hash remains the experiment
    identity.  This function deliberately accesses only ``relative_path`` and
    ``sha256`` from the manifest.  It reads encoded bytes from every split,
    including test images, but never reads or filters label/split values.
    """

    required = {"relative_path", "sha256"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetIntegrityError(f"manifest is missing required columns: {missing}")
    if frame.empty:
        raise DatasetIntegrityError("cannot verify an empty manifest")

    paths = frame["relative_path"].map(_canonical_relative_path)
    if paths.duplicated().any():
        examples = paths.loc[paths.duplicated(keep=False)].tolist()[:5]
        raise DatasetIntegrityError(f"manifest contains duplicate paths: {examples}")
    hashes = frame["sha256"].astype("string").fillna("").str.strip().str.lower()
    malformed = ~hashes.str.fullmatch(r"[0-9a-f]{64}")
    if malformed.any():
        examples = paths.loc[malformed].tolist()[:5]
        raise DatasetIntegrityError(f"manifest contains malformed SHA-256: {examples}")

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetIntegrityError(f"dataset root does not exist: {root}")
    records = sorted(zip(paths.tolist(), hashes.tolist(), strict=True))
    resolved: list[tuple[str, str, Path]] = [
        (relative_path, expected_hash, _resolve_image(root, relative_path))
        for relative_path, expected_hash in records
    ]

    def inspect(record: tuple[str, str, Path]) -> tuple[str, str, int]:
        relative_path, expected_hash, image_path = record
        observed_hash = sha256_file(image_path)
        if observed_hash != expected_hash:
            raise DatasetIntegrityError(
                "dataset image SHA-256 does not match manifest: "
                f"{relative_path}; expected {expected_hash}, observed {observed_hash}"
            )
        return relative_path, observed_hash, image_path.stat().st_size

    worker_count = workers
    if worker_count is None:
        worker_count = min(32, max(1, (os.cpu_count() or 1) + 4))
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise ValueError("workers must be a positive integer")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        verified = list(executor.map(inspect, resolved))

    fingerprint = hashlib.sha256()
    checked_bytes = 0
    for relative_path, observed_hash, byte_count in verified:
        checked_bytes += int(byte_count)
        fingerprint.update(relative_path.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(observed_hash.encode("ascii"))
        fingerprint.update(b"\0")
        fingerprint.update(str(int(byte_count)).encode("ascii"))
        fingerprint.update(b"\n")
    return DatasetIntegrityReport(
        schema_version=1,
        status="DATASET_BYTES_VERIFIED",
        checked_rows=len(verified),
        checked_bytes=checked_bytes,
        content_fingerprint_sha256=fingerprint.hexdigest(),
        dataset_root=str(root),
        verification_scope="relative_path_sha256_and_encoded_file_bytes_only",
        label_values_accessed=False,
    )


def verify_official_dataset_preconditions(
    frame: pd.DataFrame,
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    workers: int | None = None,
) -> OfficialDatasetPreconditions:
    """Verify the exact locked bundle and all image bytes for an official run.

    The selected manifest must be the bundle's own ``manifest.csv`` rather than
    a copy.  Bundle verification checks immutable curation/split lineage.  The
    subsequent byte pass reads every encoded image but deliberately does not
    inspect ``label`` or ``split`` values.
    """

    # Runtime imports avoid a data-package import cycle: split itself uses the
    # lightweight image/path manifest primitives.
    from .split import manifest_sha256, verify_locked_split_bundle

    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise DatasetIntegrityError(f"official manifest does not exist: {manifest}")
    try:
        bundle = verify_locked_split_bundle(manifest.parent)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise DatasetIntegrityError(f"locked split bundle verification failed: {exc}") from exc
    if bundle.manifest_path.resolve() != manifest:
        raise DatasetIntegrityError(
            "official manifest must be the verified bundle's own manifest.csv"
        )
    canonical_hash = manifest_sha256(frame)
    if canonical_hash != bundle.manifest_sha256:
        raise DatasetIntegrityError(
            "loaded manifest does not match the verified locked split bundle"
        )
    integrity = verify_dataset_integrity(frame, dataset_root, workers=workers)
    return OfficialDatasetPreconditions(
        schema_version=1,
        status="OFFICIAL_DATASET_PRECONDITIONS_VERIFIED",
        manifest_path=str(manifest),
        manifest_sha256=canonical_hash,
        locked_split_bundle=str(bundle.directory),
        split_inventory_sha256=sha256_file(bundle.inventory_path),
        split_completion_sha256=sha256_file(bundle.completion_path),
        dataset_integrity=integrity,
    )


__all__ = [
    "DatasetIntegrityError",
    "DatasetIntegrityReport",
    "OfficialDatasetPreconditions",
    "verify_dataset_integrity",
    "verify_official_dataset_preconditions",
]
