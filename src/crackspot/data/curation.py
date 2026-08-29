"""Immutable pre-split curation for contradictory exact duplicates.

SDNET2018 contains a small number of byte-identical files whose class-folder
labels disagree.  Such a hash cannot be assigned a trustworthy binary label.
This module applies one narrow, auditable policy before any split exists:
exclude every row in every contradictory SHA-256 group, while preserving all
other rows (including same-label exact duplicates) unchanged.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

import pandas as pd

from crackspot.utils.hashing import sha256_file

from .manifest import load_manifest_table, parse_sdnet_path
from .split import manifest_sha256

CURATION_SCHEMA_VERSION: Final[int] = 1
EXCLUSION_REASON: Final[str] = "exact_sha256_group_has_conflicting_labels"
CLEANED_MANIFEST_FILENAME: Final[str] = "pre_split_manifest.csv"
CONFLICT_ROWS_FILENAME: Final[str] = "conflict_rows.csv"
CONFLICT_REPORT_FILENAME: Final[str] = "conflict_report.json"
CURATION_OUTPUT_FILES: Final[tuple[str, ...]] = (
    CLEANED_MANIFEST_FILENAME,
    CONFLICT_ROWS_FILENAME,
    CONFLICT_REPORT_FILENAME,
)
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "relative_path",
        "label",
        "surface",
        "source_group",
        "source_group_verified",
        "sha256",
        "audit_status",
        "split",
    }
)
CONFLICT_ROW_COLUMNS: Final[tuple[str, ...]] = (
    "exact_sha256",
    "relative_path",
    "label",
    "surface",
    "source_group",
    "source_group_verified",
    "hash_group_row_count",
    "conflicting_labels",
    "exclusion_reason",
)


class ManifestCurationError(ValueError):
    """Raised when an input manifest is unsafe for pre-split curation."""


@dataclass(frozen=True, slots=True)
class ExactConflictCuration:
    """In-memory result of the exact-label-conflict policy."""

    cleaned_manifest: pd.DataFrame
    excluded_rows: pd.DataFrame
    counts: dict[str, Any]
    conflict_groups: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CurationBundle:
    """Paths and headline counts for one immutable curation bundle."""

    output_dir: Path
    cleaned_manifest_path: Path
    conflict_rows_path: Path
    conflict_report_path: Path
    parent_rows: int
    cleaned_rows: int
    excluded_rows: int
    conflicting_hash_groups: int


def _verified_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "verified"})
    )


def _label_counts(frame: pd.DataFrame, labels: pd.Series) -> dict[str, int]:
    if frame.empty:
        return {}
    return {
        str(int(label)): int(count)
        for label, count in labels.loc[frame.index].value_counts(sort=False).sort_index().items()
    }


def _surface_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(surface): int(count)
        for surface, count in frame["surface"].astype(str).value_counts().sort_index().items()
    }


def _same_values(first: pd.Series, second: pd.Series) -> bool:
    return first.astype("string").fillna("").equals(second.astype("string").fillna(""))


def _assert_exact_policy(
    parent: pd.DataFrame,
    cleaned: pd.DataFrame,
    excluded: pd.DataFrame,
    *,
    hashes: pd.Series,
    conflict_hashes: tuple[str, ...],
) -> None:
    """Prove that the output is exactly parent minus complete conflict groups."""

    expected_excluded_paths = set(
        parent.loc[hashes.isin(set(conflict_hashes)), "relative_path"].astype(str)
    )
    observed_excluded_paths = set(excluded["relative_path"].astype(str))
    if observed_excluded_paths != expected_excluded_paths:
        raise RuntimeError("internal error: exclusion rows do not match conflict groups")
    if set(cleaned["relative_path"].astype(str)).intersection(observed_excluded_paths):
        raise RuntimeError("internal error: an excluded path remains in the cleaned manifest")

    parent_by_path = parent.set_index(parent["relative_path"].astype(str), drop=False)
    cleaned_by_path = cleaned.set_index(cleaned["relative_path"].astype(str), drop=False)
    expected_cleaned_paths = set(parent_by_path.index) - expected_excluded_paths
    if set(cleaned_by_path.index) != expected_cleaned_paths:
        raise RuntimeError("internal error: curation added or dropped a non-conflict row")
    expected_rows = parent_by_path.loc[cleaned_by_path.index, cleaned.columns]
    for column in cleaned.columns:
        if not _same_values(cleaned_by_path[column], expected_rows[column]):
            raise RuntimeError(f"internal error: curation changed parent column {column!r}")


def _validate_manifest(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ManifestCurationError(f"manifest is missing required columns: {missing}")
    if frame.empty:
        raise ManifestCurationError("cannot curate an empty manifest")

    paths = frame["relative_path"].astype("string").fillna("").str.strip()
    if (paths == "").any():
        raise ManifestCurationError("relative_path cannot be blank")
    if paths.duplicated().any():
        examples = paths.loc[paths.duplicated(keep=False)].tolist()[:5]
        raise ManifestCurationError(f"relative_path values are not unique: {examples}")

    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ManifestCurationError("labels must be exactly 0 (Non-crack) or 1 (Crack)")
    labels = labels.astype(int)

    surfaces = frame["surface"].astype("string").fillna("").str.strip().str.upper()
    if (surfaces == "").any():
        raise ManifestCurationError("surface cannot be blank")
    for index, relative_path in paths.items():
        portable = PurePosixPath(str(relative_path).replace("\\", "/"))
        if portable.is_absolute() or ".." in portable.parts:
            raise ManifestCurationError(f"relative_path is unsafe: {relative_path!r}")
        try:
            expected_surface, expected_label = parse_sdnet_path(portable.as_posix())
        except ValueError as exc:
            raise ManifestCurationError(str(exc)) from exc
        if surfaces.loc[index] != expected_surface or labels.loc[index] != expected_label:
            raise ManifestCurationError(
                f"manifest label/surface disagrees with SDNET class folder for {relative_path}"
            )

    source_groups = frame["source_group"].astype("string").fillna("").str.strip()
    if (source_groups == "").any():
        examples = paths.loc[source_groups == ""].tolist()[:5]
        raise ManifestCurationError(f"source_group is blank for: {examples}")
    verified = _verified_series(frame["source_group_verified"])
    if not verified.all():
        examples = paths.loc[~verified].tolist()[:5]
        raise ManifestCurationError(
            "source_group must be verified for every row before curation; "
            f"unverified examples: {examples}"
        )

    hashes = frame["sha256"].astype("string").fillna("").str.strip().str.lower()
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        examples = paths.loc[~hashes.str.fullmatch(r"[0-9a-f]{64}")].tolist()[:5]
        raise ManifestCurationError(f"SHA-256 is blank or malformed for: {examples}")

    audit_status = frame["audit_status"].astype("string").fillna("").str.strip().str.lower()
    if not audit_status.eq("ok").all():
        examples = paths.loc[~audit_status.eq("ok")].tolist()[:5]
        raise ManifestCurationError(
            f"manifest contains invalid or unaudited rows; curation cannot hide them: {examples}"
        )
    assigned = frame["split"].astype("string").fillna("").str.strip()
    if assigned.ne("").any():
        examples = paths.loc[assigned.ne("")].tolist()[:5]
        raise ManifestCurationError(
            f"curation is pre-split only and refuses an already assigned manifest: {examples}"
        )
    return labels, hashes, verified


def curate_exact_label_conflicts(frame: pd.DataFrame) -> ExactConflictCuration:
    """Exclude complete contradictory hash groups and no other rows.

    The input frame is never mutated.  Row order and every original column are
    preserved in the cleaned manifest so it can be passed directly to
    :func:`create_group_splits`.  The conflict table is sorted by hash and path
    to make the evidence deterministic.
    """

    labels, hashes, verified = _validate_manifest(frame)
    work = frame.assign(_normalised_sha256=hashes, _normalised_label=labels)
    hash_sizes = work.groupby("_normalised_sha256", sort=True).size()
    duplicate_hashes = hash_sizes.loc[lambda values: values > 1]
    label_cardinality = work.groupby("_normalised_sha256", sort=True)["_normalised_label"].nunique()
    conflict_hashes = tuple(label_cardinality.loc[lambda values: values > 1].index.tolist())
    conflict_hash_set = set(conflict_hashes)
    exclusion_mask = hashes.isin(conflict_hash_set)

    cleaned = frame.loc[~exclusion_mask].copy()
    excluded_records: list[dict[str, Any]] = []
    conflict_groups: list[dict[str, Any]] = []
    for exact_hash in conflict_hashes:
        group = work.loc[work["_normalised_sha256"].eq(exact_hash)].copy()
        group = group.sort_values("relative_path", kind="stable")
        group_labels = sorted(int(value) for value in group["_normalised_label"].unique())
        encoded_labels = "|".join(str(value) for value in group_labels)
        json_rows: list[dict[str, Any]] = []
        for index, row in group.iterrows():
            record = {
                "exact_sha256": exact_hash,
                "relative_path": str(row["relative_path"]),
                "label": int(row["_normalised_label"]),
                "surface": str(row["surface"]),
                "source_group": str(row["source_group"]),
                "source_group_verified": bool(verified.loc[index]),
                "hash_group_row_count": len(group),
                "conflicting_labels": encoded_labels,
                "exclusion_reason": EXCLUSION_REASON,
            }
            excluded_records.append(record)
            json_rows.append(
                {
                    "relative_path": record["relative_path"],
                    "label": record["label"],
                    "surface": record["surface"],
                    "source_group": record["source_group"],
                    "exclusion_reason": EXCLUSION_REASON,
                }
            )
        conflict_groups.append(
            {
                "sha256": exact_hash,
                "row_count": len(group),
                "labels": group_labels,
                "exclusion_reason": EXCLUSION_REASON,
                "rows": json_rows,
            }
        )

    excluded = pd.DataFrame(excluded_records, columns=CONFLICT_ROW_COLUMNS)
    if not excluded.empty:
        excluded = excluded.sort_values(
            ["exact_sha256", "relative_path"], kind="stable"
        ).reset_index(drop=True)
    _assert_exact_policy(
        frame,
        cleaned,
        excluded,
        hashes=hashes,
        conflict_hashes=conflict_hashes,
    )

    cleaned_hashes = hashes.loc[cleaned.index]
    cleaned_labels = labels.loc[cleaned.index]
    remaining_conflicts = (
        pd.DataFrame({"sha256": cleaned_hashes, "label": cleaned_labels})
        .groupby("sha256", sort=True)["label"]
        .nunique()
        .loc[lambda values: values > 1]
    )
    if not remaining_conflicts.empty:  # pragma: no cover - protected by construction.
        raise RuntimeError("internal error: contradictory hashes remain after curation")
    if len(cleaned) + len(excluded) != len(frame):  # pragma: no cover - invariant.
        raise RuntimeError("internal error: curation did not account for every parent row")

    retained_duplicate_hashes = sorted(set(duplicate_hashes.index) - conflict_hash_set)
    retained_duplicate_rows = int(duplicate_hashes.loc[retained_duplicate_hashes].sum())
    excluded_frame = frame.loc[exclusion_mask]
    counts: dict[str, Any] = {
        "parent_rows": len(frame),
        "cleaned_rows": len(cleaned),
        "excluded_rows": len(excluded),
        "contradictory_exact_hash_groups": len(conflict_hashes),
        "exact_duplicate_groups_before_curation": len(duplicate_hashes),
        "exact_duplicate_rows_before_curation": int(duplicate_hashes.sum()),
        "retained_same_label_exact_duplicate_groups": len(retained_duplicate_hashes),
        "retained_same_label_exact_duplicate_rows": retained_duplicate_rows,
        "source_groups_before_curation": int(
            frame["source_group"].astype("string").str.strip().nunique()
        ),
        "source_groups_after_curation": int(
            cleaned["source_group"].astype("string").str.strip().nunique()
        ),
        "labels_before_curation": _label_counts(frame, labels),
        "labels_after_curation": _label_counts(cleaned, labels),
        "excluded_labels": _label_counts(excluded_frame, labels),
        "excluded_surfaces": _surface_counts(excluded_frame),
    }
    return ExactConflictCuration(
        cleaned_manifest=cleaned,
        excluded_rows=excluded,
        counts=counts,
        conflict_groups=tuple(conflict_groups),
    )


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_curation_bundle(
    parent_manifest_path: str | Path,
    output_dir: str | Path,
) -> CurationBundle:
    """Create an atomic, create-once curation evidence directory.

    Textual identifiers are loaded losslessly so leading zeroes in verified
    ``source_group`` values survive the round trip.  The output directory itself
    is the immutability boundary: any existing file, directory or symbolic link
    at that path causes a hard failure before artifacts are made.
    """

    parent = Path(parent_manifest_path)
    destination = Path(output_dir)
    if not parent.is_file():
        raise FileNotFoundError(f"parent manifest does not exist: {parent}")
    if parent.suffix.lower() != ".csv":
        raise ManifestCurationError("parent manifest must be a CSV file")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable curation directory: {destination}")

    try:
        frame = load_manifest_table(parent)
    except (OSError, pd.errors.ParserError) as exc:
        raise ManifestCurationError(f"cannot read parent manifest: {exc}") from exc
    result = curate_exact_label_conflicts(frame)
    parent_file_hash = sha256_file(parent)
    parent_canonical_hash = manifest_sha256(frame)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-", dir=destination.parent
    ) as temporary_name:
        staging = Path(temporary_name) / "bundle"
        staging.mkdir()
        cleaned_path = staging / CLEANED_MANIFEST_FILENAME
        conflicts_path = staging / CONFLICT_ROWS_FILENAME
        report_path = staging / CONFLICT_REPORT_FILENAME
        result.cleaned_manifest.to_csv(cleaned_path, index=False, lineterminator="\n")
        result.excluded_rows.to_csv(conflicts_path, index=False, lineterminator="\n")

        report: dict[str, Any] = {
            "schema_version": CURATION_SCHEMA_VERSION,
            "artifact_type": "pre_split_exact_label_conflict_curation",
            "created_utc": datetime.now(UTC).isoformat(),
            "immutable": True,
            "parent_manifest": {
                "path": str(parent.resolve()),
                "file_sha256": parent_file_hash,
                "canonical_sha256": parent_canonical_hash,
                "rows": len(frame),
            },
            "policy": {
                "exclusion_reason": EXCLUSION_REASON,
                "action": (
                    "Before splitting, exclude every row in every exact SHA-256 group "
                    "whose audited labels disagree."
                ),
                "same_label_exact_duplicates": (
                    "Retain unchanged; create_group_splits links their source groups into "
                    "one allocation unit."
                ),
                "test_set_used_for_decision": False,
                "labels_rewritten": False,
            },
            "counts": result.counts,
            "conflict_groups": list(result.conflict_groups),
            "artifacts": {
                "cleaned_manifest": {
                    "filename": CLEANED_MANIFEST_FILENAME,
                    "file_sha256": sha256_file(cleaned_path),
                    "canonical_sha256": manifest_sha256(result.cleaned_manifest),
                    "rows": len(result.cleaned_manifest),
                },
                "conflict_rows_csv": {
                    "filename": CONFLICT_ROWS_FILENAME,
                    "file_sha256": sha256_file(conflicts_path),
                    "rows": len(result.excluded_rows),
                },
            },
            "validation": {
                "all_parent_rows_accounted_for": True,
                "all_source_groups_nonblank_and_verified": True,
                "only_contradictory_hash_groups_excluded": True,
                "same_label_exact_duplicates_retained": True,
                "contradictory_hash_groups_after_curation": 0,
                "split_assignments_present_during_curation": False,
            },
            "immutable_protocol": (
                "Archive this directory with the split evidence. Do not delete and recreate "
                "it after viewing validation or test results."
            ),
        }
        _write_json(report, report_path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite immutable curation directory: {destination}"
            )
        try:
            staging.rename(destination)
        except OSError:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite immutable curation directory: {destination}"
                ) from None
            raise

    final_report = destination / CONFLICT_REPORT_FILENAME
    if not all((destination / name).is_file() for name in CURATION_OUTPUT_FILES):
        # This is a create-only destination; removing an incomplete bundle is
        # safe here because it was created by this call and never returned.
        shutil.rmtree(destination, ignore_errors=True)
        raise OSError("curation bundle finalization was incomplete")
    counts = result.counts
    return CurationBundle(
        output_dir=destination.resolve(),
        cleaned_manifest_path=(destination / CLEANED_MANIFEST_FILENAME).resolve(),
        conflict_rows_path=(destination / CONFLICT_ROWS_FILENAME).resolve(),
        conflict_report_path=final_report.resolve(),
        parent_rows=int(counts["parent_rows"]),
        cleaned_rows=int(counts["cleaned_rows"]),
        excluded_rows=int(counts["excluded_rows"]),
        conflicting_hash_groups=int(counts["contradictory_exact_hash_groups"]),
    )
