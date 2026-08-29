from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from crackspot.data import (
    EXCLUSION_REASON,
    ManifestCurationError,
    create_curation_bundle,
    create_group_splits,
    curate_exact_label_conflicts,
    manifest_sha256,
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
    surface = relative_path.split("/")[0]
    return {
        "relative_path": relative_path,
        "label": label,
        "class_name": "Crack" if label else "Non-crack",
        "surface": surface,
        "source_group": source_group,
        "source_group_verified": True,
        "group_resolution_method": "verified_regex",
        "sha256": exact_hash or _hash(relative_path),
        "perceptual_hash": "0123456789abcdef",
        "width": "256",
        "height": "256",
        "source_width": "256",
        "source_height": "256",
        "source_mode": "RGB",
        "image_format": "JPEG",
        "file_size_bytes": "1234",
        "exif_orientation": "",
        "audit_status": "ok",
        "audit_error": "",
        "split": "",
    }


def _manifest_with_conflicts() -> pd.DataFrame:
    conflict_a = _hash("conflict-a")
    conflict_b = _hash("conflict-b")
    retained_duplicate = _hash("same-label-duplicate")
    return pd.DataFrame(
        [
            _row("D/CD/7039-112.jpg", 1, source_group="7039", exact_hash=conflict_a),
            _row("D/UD/7039-112_2.jpg", 0, source_group="7039", exact_hash=conflict_a),
            _row("W/UW/7074-105_2.jpg", 0, source_group="7074", exact_hash=conflict_b),
            _row("W/CW/7074-105.jpg", 1, source_group="7074", exact_hash=conflict_b),
            _row("P/UP/001-1.jpg", 0, source_group="001", exact_hash=retained_duplicate),
            _row("P/UP/002-1.jpg", 0, source_group="002", exact_hash=retained_duplicate),
            _row("P/CP/003-1.jpg", 1, source_group="003"),
        ]
    )


def test_curation_excludes_complete_contradictory_groups_only() -> None:
    parent = _manifest_with_conflicts()
    original = parent.copy(deep=True)

    result = curate_exact_label_conflicts(parent)

    pdt.assert_frame_equal(parent, original)
    assert result.cleaned_manifest["relative_path"].tolist() == [
        "P/UP/001-1.jpg",
        "P/UP/002-1.jpg",
        "P/CP/003-1.jpg",
    ]
    assert result.cleaned_manifest["sha256"].duplicated(keep=False).sum() == 2
    assert result.excluded_rows["relative_path"].tolist() == [
        "D/CD/7039-112.jpg",
        "D/UD/7039-112_2.jpg",
        "W/CW/7074-105.jpg",
        "W/UW/7074-105_2.jpg",
    ]
    assert result.excluded_rows["exclusion_reason"].eq(EXCLUSION_REASON).all()
    assert result.counts["contradictory_exact_hash_groups"] == 2
    assert result.counts["excluded_rows"] == 4
    assert result.counts["retained_same_label_exact_duplicate_groups"] == 1
    assert result.counts["retained_same_label_exact_duplicate_rows"] == 2
    assert result.counts["parent_rows"] == (
        result.counts["cleaned_rows"] + result.counts["excluded_rows"]
    )


def test_curation_is_deterministic_for_shuffled_input() -> None:
    frame = _manifest_with_conflicts()

    first = curate_exact_label_conflicts(frame)
    second = curate_exact_label_conflicts(frame.sample(frac=1, random_state=17))

    assert manifest_sha256(first.cleaned_manifest) == manifest_sha256(second.cleaned_manifest)
    pdt.assert_frame_equal(first.excluded_rows, second.excluded_rows)
    assert first.conflict_groups == second.conflict_groups


def test_curation_bundle_locks_parent_hashes_and_refuses_overwrite(tmp_path: Path) -> None:
    parent = _manifest_with_conflicts()
    parent_path = tmp_path / "audit_manifest.csv"
    # Exercise preservation of a leading-zero source_group through disk I/O.
    parent.to_csv(parent_path, index=False, lineterminator="\n")
    output = tmp_path / "pre_split_curation_v1"

    bundle = create_curation_bundle(parent_path, output)
    report = json.loads(bundle.conflict_report_path.read_text(encoding="utf-8"))
    cleaned = pd.read_csv(bundle.cleaned_manifest_path, dtype=str, keep_default_na=False)
    conflict_rows = pd.read_csv(bundle.conflict_rows_path, dtype=str, keep_default_na=False)

    assert (
        cleaned.loc[cleaned["relative_path"].eq("P/UP/001-1.jpg"), "source_group"].item() == "001"
    )
    assert report["parent_manifest"]["file_sha256"] == sha256_file(parent_path)
    assert report["parent_manifest"]["canonical_sha256"] == manifest_sha256(
        pd.read_csv(parent_path, dtype=str, keep_default_na=False)
    )
    assert report["artifacts"]["cleaned_manifest"]["file_sha256"] == sha256_file(
        bundle.cleaned_manifest_path
    )
    assert report["artifacts"]["cleaned_manifest"]["canonical_sha256"] == manifest_sha256(cleaned)
    assert report["artifacts"]["conflict_rows_csv"]["file_sha256"] == sha256_file(
        bundle.conflict_rows_path
    )
    assert report["counts"]["excluded_rows"] == len(conflict_rows) == 4
    assert report["policy"]["test_set_used_for_decision"] is False
    assert report["policy"]["labels_rewritten"] is False
    assert report["validation"]["same_label_exact_duplicates_retained"] is True

    before = {
        path.name: path.read_bytes()
        for path in (
            bundle.cleaned_manifest_path,
            bundle.conflict_rows_path,
            bundle.conflict_report_path,
        )
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_curation_bundle(parent_path, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.assign(source_group_verified=False), "verified"),
        (lambda frame: frame.assign(source_group=""), "source_group is blank"),
        (lambda frame: frame.assign(audit_status="invalid"), "invalid or unaudited"),
        (lambda frame: frame.assign(split="train"), "pre-split only"),
    ],
)
def test_curation_rejects_unverified_or_non_pre_split_input(mutate, message: str) -> None:
    with pytest.raises(ManifestCurationError, match=message):
        curate_exact_label_conflicts(mutate(_manifest_with_conflicts()))


def test_curation_rejects_manifest_label_that_disagrees_with_folder() -> None:
    frame = _manifest_with_conflicts()
    frame.loc[0, "label"] = 0

    with pytest.raises(ManifestCurationError, match="disagrees with SDNET class folder"):
        curate_exact_label_conflicts(frame)


def test_cleaned_manifest_flows_to_split_without_dropping_same_label_duplicates() -> None:
    rows: list[dict[str, object]] = []
    for surface in ("D", "P", "W"):
        for label in (0, 1):
            class_folder = ("C" if label else "U") + surface
            for index in range(12):
                rows.append(
                    _row(
                        f"{surface}/{class_folder}/{surface}{label}{index}-1.jpg",
                        label,
                        source_group=f"{surface}-{label}-{index}",
                    )
                )
    conflict_hash = _hash("conflicting")
    rows[0]["sha256"] = conflict_hash
    rows[12]["sha256"] = conflict_hash
    duplicate_hash = _hash("retained")
    rows[24]["sha256"] = duplicate_hash
    rows[48]["sha256"] = duplicate_hash
    parent = pd.DataFrame(rows)

    curated = curate_exact_label_conflicts(parent).cleaned_manifest
    split = create_group_splits(curated, seed=42, restarts=8)

    assert len(split) == len(parent) - 2
    retained = split.loc[split["sha256"].eq(duplicate_hash)]
    assert len(retained) == 2
    assert retained["split"].nunique() == 1


def test_curation_bundle_refuses_existing_empty_directory(tmp_path: Path) -> None:
    parent_path = tmp_path / "audit_manifest.csv"
    _manifest_with_conflicts().to_csv(parent_path, index=False)
    output = tmp_path / "already-there"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_curation_bundle(parent_path, output)
    assert list(output.iterdir()) == []
