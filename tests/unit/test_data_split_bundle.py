from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crackspot.data import (
    SPLIT_BUNDLE_FILENAMES,
    SPLIT_COMPLETION_FILENAME,
    SPLIT_INVENTORY_FILENAME,
    SplitValidationError,
    create_curation_bundle,
    create_locked_split_bundle,
    load_manifest_table,
    manifest_sha256,
    verify_locked_split_bundle,
)
from crackspot.utils.hashing import sha256_file


def _hash(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _row(
    relative_path: str,
    label: int,
    *,
    source_group: str,
    exact_hash: str | None = None,
) -> dict[str, object]:
    surface = relative_path.split("/", maxsplit=1)[0]
    return {
        "relative_path": relative_path,
        "label": label,
        "class_name": "Crack" if label else "Non-crack",
        "surface": surface,
        "source_group": source_group,
        "source_group_verified": True,
        "group_resolution_method": "verified_group_map",
        "sha256": exact_hash or _hash(relative_path),
        "perceptual_hash": "0123456789abcdef",
        "width": 256,
        "height": 256,
        "source_width": 256,
        "source_height": 256,
        "source_mode": "RGB",
        "image_format": "JPEG",
        "file_size_bytes": 1234,
        "exif_orientation": "",
        "audit_status": "ok",
        "audit_error": "",
        "split": "",
    }


def _official_synthetic_parent(groups_per_stratum: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_number = 1
    for surface in ("D", "P", "W"):
        for label in (0, 1):
            folder = f"{'C' if label else 'U'}{surface}"
            for index in range(groups_per_stratum):
                rows.append(
                    _row(
                        f"{surface}/{folder}/{surface}{label}-{index}.jpg",
                        label,
                        source_group=f"{group_number:04d}",
                    )
                )
                group_number += 1

    conflict_hash = _hash("contradictory-exact-bytes")
    rows.extend(
        [
            _row(
                "D/CD/conflicting-crack.jpg",
                1,
                source_group="9001",
                exact_hash=conflict_hash,
            ),
            _row(
                "D/UD/conflicting-non-crack.jpg",
                0,
                source_group="9002",
                exact_hash=conflict_hash,
            ),
        ]
    )
    return pd.DataFrame(rows)


def _create_bundle(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent_path = tmp_path / "audit_manifest.csv"
    _official_synthetic_parent().to_csv(parent_path, index=False, lineterminator="\n")
    curation = create_curation_bundle(parent_path, tmp_path / "curation")
    split = create_locked_split_bundle(
        curation.cleaned_manifest_path,
        tmp_path / "split_v1",
        restarts=8,
    )
    return split.directory, curation.conflict_report_path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_completion_inventory_hash(bundle_dir: Path) -> None:
    inventory_path = bundle_dir / SPLIT_INVENTORY_FILENAME
    completion_path = bundle_dir / SPLIT_COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    inventory_hash = sha256_file(inventory_path)
    completion["inventory_sha256"] = inventory_hash
    completion["artifact_sha256"][SPLIT_INVENTORY_FILENAME] = inventory_hash
    _write_json(completion_path, completion)


def _refresh_all_internal_hashes(bundle_dir: Path) -> None:
    """Refresh untrusted metadata so semantic lineage checks are exercised."""

    parent_path = bundle_dir / "parent_manifest.csv"
    cleaned_path = bundle_dir / "pre_split_manifest.csv"
    conflict_rows_path = bundle_dir / "conflict_rows.csv"
    report_path = bundle_dir / "conflict_report.json"

    parent = load_manifest_table(parent_path)
    cleaned = load_manifest_table(cleaned_path)
    conflicts = pd.read_csv(conflict_rows_path, dtype=str, keep_default_na=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["parent_manifest"].update(
        {
            "file_sha256": sha256_file(parent_path),
            "canonical_sha256": manifest_sha256(parent),
            "rows": len(parent),
        }
    )
    report["artifacts"]["cleaned_manifest"].update(
        {
            "file_sha256": sha256_file(cleaned_path),
            "canonical_sha256": manifest_sha256(cleaned),
            "rows": len(cleaned),
        }
    )
    report["artifacts"]["conflict_rows_csv"].update(
        {
            "file_sha256": sha256_file(conflict_rows_path),
            "rows": len(conflicts),
        }
    )
    _write_json(report_path, report)

    inventory_path = bundle_dir / SPLIT_INVENTORY_FILENAME
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    lineage = inventory["curation_lineage"]
    lineage["conflict_report_file_sha256"] = sha256_file(report_path)
    lineage["parent_manifest"].update(
        {
            "file_sha256": sha256_file(parent_path),
            "canonical_sha256": manifest_sha256(parent),
            "rows": len(parent),
        }
    )
    lineage["cleaned_manifest"].update(
        {
            "file_sha256": sha256_file(cleaned_path),
            "canonical_sha256": manifest_sha256(cleaned),
            "rows": len(cleaned),
        }
    )
    lineage["conflict_rows"].update(
        {
            "file_sha256": sha256_file(conflict_rows_path),
            "rows": len(conflicts),
        }
    )
    files = inventory["files"]
    files["parent_manifest.csv"].update(
        {
            "file_sha256": sha256_file(parent_path),
            "canonical_sha256": manifest_sha256(parent),
            "rows": len(parent),
        }
    )
    files["pre_split_manifest.csv"].update(
        {
            "file_sha256": sha256_file(cleaned_path),
            "canonical_sha256": manifest_sha256(cleaned),
            "rows": len(cleaned),
        }
    )
    files["conflict_rows.csv"].update(
        {"file_sha256": sha256_file(conflict_rows_path), "rows": len(conflicts)}
    )
    files["conflict_report.json"]["file_sha256"] = sha256_file(report_path)
    _write_json(inventory_path, inventory)

    completion_path = bundle_dir / SPLIT_COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for name in completion["artifact_sha256"]:
        completion["artifact_sha256"][name] = sha256_file(bundle_dir / name)
    completion["inventory_sha256"] = sha256_file(inventory_path)
    _write_json(completion_path, completion)


def test_locked_bundle_is_atomic_portable_and_verifies(tmp_path: Path) -> None:
    bundle_dir, conflict_report = _create_bundle(tmp_path)

    verified = verify_locked_split_bundle(
        bundle_dir,
        conflict_report_path=conflict_report,
    )
    manifest = load_manifest_table(verified.manifest_path)

    assert {path.name for path in bundle_dir.iterdir()} == set(SPLIT_BUNDLE_FILENAMES)
    assert verified.completion["status"] == "LOCKED_SPLIT_COMPLETE"
    assert verified.audit["valid"] is True
    assert verified.audit["protocol"]["official_balance_enforced"] is True
    assert "0001" in set(manifest["source_group"])
    assert set(verified.completion["artifact_sha256"]) == set(SPLIT_BUNDLE_FILENAMES) - {
        SPLIT_COMPLETION_FILENAME
    }


def test_completion_marker_is_deliberately_written_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crackspot.data.split as split_module

    parent_path = tmp_path / "audit_manifest.csv"
    _official_synthetic_parent().to_csv(parent_path, index=False, lineterminator="\n")
    curation = create_curation_bundle(parent_path, tmp_path / "curation")
    written: list[str] = []
    original = split_module._write_json_file

    def recording_write(value: Any, path: Path) -> None:
        written.append(path.name)
        original(value, path)

    monkeypatch.setattr(split_module, "_write_json_file", recording_write)
    create_locked_split_bundle(curation.cleaned_manifest_path, tmp_path / "split", restarts=8)

    assert written[-1] == SPLIT_COMPLETION_FILENAME


def test_failed_staging_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crackspot.data.split as split_module

    parent_path = tmp_path / "audit_manifest.csv"
    _official_synthetic_parent().to_csv(parent_path, index=False, lineterminator="\n")
    curation = create_curation_bundle(parent_path, tmp_path / "curation")
    destination = tmp_path / "split"

    def fail_verification(*args: Any, **kwargs: Any) -> None:
        raise SplitValidationError("injected staging failure")

    monkeypatch.setattr(split_module, "verify_locked_split_bundle", fail_verification)
    with pytest.raises(SplitValidationError, match="injected staging failure"):
        create_locked_split_bundle(curation.cleaned_manifest_path, destination, restarts=8)

    assert not destination.exists()
    assert not list(tmp_path.glob(".split-staging-*"))


@pytest.mark.parametrize(
    "filename",
    [
        "manifest.csv",
        "train.csv",
        "split_audit.json",
        "manifest_hashes.json",
        "conflict_report.json",
        "parent_manifest.csv",
        "pre_split_manifest.csv",
        "conflict_rows.csv",
    ],
)
def test_verifier_rejects_any_altered_artifact(tmp_path: Path, filename: str) -> None:
    bundle_dir, _ = _create_bundle(tmp_path)
    with (bundle_dir / filename).open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(SplitValidationError):
        verify_locked_split_bundle(bundle_dir)


def test_verifier_rejects_missing_completion_and_unexpected_files(tmp_path: Path) -> None:
    missing_dir, _ = _create_bundle(tmp_path / "missing")
    (missing_dir / SPLIT_COMPLETION_FILENAME).unlink()
    with pytest.raises(SplitValidationError, match="file set mismatch"):
        verify_locked_split_bundle(missing_dir)

    extra_dir, _ = _create_bundle(tmp_path / "extra")
    (extra_dir / "partial.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(SplitValidationError, match="file set mismatch"):
        verify_locked_split_bundle(extra_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", 41), ("ratios", {"train": 0.6, "validation": 0.2, "test": 0.2})],
)
def test_verifier_rejects_wrong_protocol_in_completion(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle_dir, _ = _create_bundle(tmp_path)
    completion_path = bundle_dir / SPLIT_COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion[field] = value
    _write_json(completion_path, completion)

    with pytest.raises(SplitValidationError):
        verify_locked_split_bundle(bundle_dir)


def test_verifier_rejects_wrong_protocol_in_rehashed_inventory(tmp_path: Path) -> None:
    bundle_dir, _ = _create_bundle(tmp_path)
    inventory_path = bundle_dir / SPLIT_INVENTORY_FILENAME
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["seed"] = 41
    _write_json(inventory_path, inventory)
    _refresh_completion_inventory_hash(bundle_dir)

    with pytest.raises(SplitValidationError, match="official seed"):
        verify_locked_split_bundle(bundle_dir)


def test_explicit_conflict_report_must_match_snapshot(tmp_path: Path) -> None:
    bundle_dir, conflict_report = _create_bundle(tmp_path)
    different = tmp_path / "different_conflict_report.json"
    different.write_bytes(conflict_report.read_bytes() + b"\n")

    with pytest.raises(SplitValidationError, match="differs from bundled"):
        verify_locked_split_bundle(bundle_dir, conflict_report_path=different)


@pytest.mark.parametrize("artifact", ["parent", "cleaned", "conflicts"])
def test_semantic_lineage_tampering_fails_even_after_internal_rehash(
    tmp_path: Path, artifact: str
) -> None:
    bundle_dir, _ = _create_bundle(tmp_path)
    if artifact == "parent":
        path = bundle_dir / "parent_manifest.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[0, "width"] = "999"
    elif artifact == "cleaned":
        path = bundle_dir / "pre_split_manifest.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[0, "width"] = "999"
    else:
        path = bundle_dir / "conflict_rows.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[0, "label"] = "1" if frame.loc[0, "label"] == "0" else "0"
    frame.to_csv(path, index=False, lineterminator="\n")
    _refresh_all_internal_hashes(bundle_dir)

    with pytest.raises(SplitValidationError, match="curation"):
        verify_locked_split_bundle(bundle_dir)


def test_existing_output_directory_is_unchanged(tmp_path: Path) -> None:
    parent_path = tmp_path / "audit_manifest.csv"
    _official_synthetic_parent().to_csv(parent_path, index=False, lineterminator="\n")
    curation = create_curation_bundle(parent_path, tmp_path / "curation")
    destination = tmp_path / "split_v1"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_locked_split_bundle(curation.cleaned_manifest_path, destination, restarts=8)

    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
