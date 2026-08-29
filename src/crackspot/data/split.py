"""Leakage-resistant, deterministic group-aware dataset splitting."""

from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from crackspot.utils.hashing import sha256_file

from .manifest import load_manifest_table

DEFAULT_SPLIT_RATIOS: Final[dict[str, float]] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
OFFICIAL_SPLIT_SEED: Final[int] = 42
EXPECTED_SURFACES: Final[tuple[str, ...]] = ("D", "P", "W")
EXPECTED_LABELS: Final[tuple[int, ...]] = (0, 1)
EXPECTED_STRATA: Final[tuple[tuple[str, int], ...]] = tuple(
    (surface, label) for surface in EXPECTED_SURFACES for label in EXPECTED_LABELS
)

# Group-aware allocation cannot normally hit fractional targets exactly.  These
# are absolute percentage-point tolerances, not relative-error tolerances.  The
# official SDNET2018 split must keep total images and source groups within five
# percentage points of 70/15/15, and every surface x label stratum within ten.
MAX_IMAGE_FRACTION_DEVIATION: Final[float] = 0.05
MAX_SOURCE_GROUP_FRACTION_DEVIATION: Final[float] = 0.05
MAX_STRATUM_FRACTION_DEVIATION: Final[float] = 0.10

SPLIT_BUNDLE_SCHEMA_VERSION: Final[int] = 2
SPLIT_INVENTORY_FILENAME: Final[str] = "manifest_hashes.json"
SPLIT_COMPLETION_FILENAME: Final[str] = "split_complete.json"
SPLIT_MANIFEST_FILENAME: Final[str] = "manifest.csv"
SPLIT_SUBSET_FILENAMES: Final[dict[str, str]] = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
SPLIT_AUDIT_FILENAME: Final[str] = "split_audit.json"
PARENT_MANIFEST_SNAPSHOT_FILENAME: Final[str] = "parent_manifest.csv"
CURATED_MANIFEST_SNAPSHOT_FILENAME: Final[str] = "pre_split_manifest.csv"
CONFLICT_ROWS_SNAPSHOT_FILENAME: Final[str] = "conflict_rows.csv"
CONFLICT_REPORT_SNAPSHOT_FILENAME: Final[str] = "conflict_report.json"
SPLIT_BUNDLE_ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    SPLIT_MANIFEST_FILENAME,
    *SPLIT_SUBSET_FILENAMES.values(),
    SPLIT_AUDIT_FILENAME,
    PARENT_MANIFEST_SNAPSHOT_FILENAME,
    CURATED_MANIFEST_SNAPSHOT_FILENAME,
    CONFLICT_ROWS_SNAPSHOT_FILENAME,
    CONFLICT_REPORT_SNAPSHOT_FILENAME,
)
SPLIT_BUNDLE_FILENAMES: Final[tuple[str, ...]] = (
    *SPLIT_BUNDLE_ARTIFACT_FILENAMES,
    SPLIT_INVENTORY_FILENAME,
    SPLIT_COMPLETION_FILENAME,
)
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "relative_path",
        "label",
        "surface",
        "source_group",
        "source_group_verified",
        "sha256",
    }
)


class SplitValidationError(ValueError):
    """Raised when a report-valid, leakage-safe split cannot be produced."""


@dataclass(frozen=True, slots=True)
class LockedSplitBundle:
    """Verified paths and provenance for one immutable official split bundle."""

    directory: Path
    manifest_path: Path
    audit_path: Path
    inventory_path: Path
    completion_path: Path
    manifest_sha256: str
    audit: dict[str, Any]
    inventory: dict[str, Any]
    completion: dict[str, Any]


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if root_first < root_second:
            self.parent[root_second] = root_first
        else:
            self.parent[root_first] = root_second


def _normalise_ratios(
    ratios: Mapping[str, float] | Sequence[float],
) -> dict[str, float]:
    if isinstance(ratios, Mapping):
        provided = {str(name): float(value) for name, value in ratios.items()}
        if set(provided) != set(DEFAULT_SPLIT_RATIOS):
            raise SplitValidationError("split names must be train, validation and test")
        result = {name: provided[name] for name in DEFAULT_SPLIT_RATIOS}
    else:
        values = [float(value) for value in ratios]
        if len(values) != 3:
            raise SplitValidationError("ratio sequence must contain train/validation/test")
        result = dict(zip(DEFAULT_SPLIT_RATIOS, values, strict=True))
    if tuple(result) != tuple(DEFAULT_SPLIT_RATIOS):
        raise SplitValidationError("split names and order must be train, validation, test")
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise SplitValidationError("all split ratios must be finite and positive")
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise SplitValidationError("split ratios must sum to 1.0")
    mismatches = {
        name: value
        for name, value in result.items()
        if not math.isclose(
            value,
            DEFAULT_SPLIT_RATIOS[name],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if mismatches:
        raise SplitValidationError(
            "official split ratios are fixed at train=0.70, validation=0.15, "
            f"test=0.15; received {mismatches}"
        )
    return result


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


def _normalise_labels(series: pd.Series) -> pd.Series:
    is_boolean = series.map(lambda value: isinstance(value, bool | np.bool_))
    if is_boolean.any():
        raise SplitValidationError("labels must be integer values 0 or 1, not booleans")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    invalid = (
        ~np.isfinite(numeric_array)
        | (numeric_array != np.floor(numeric_array))
        | ~np.isin(numeric_array, EXPECTED_LABELS)
    )
    if invalid.any():
        examples = series.loc[series.index[invalid]].astype(str).tolist()[:5]
        raise SplitValidationError(
            "labels must be exactly integer 0 (Non-crack) or 1 (Crack); "
            f"invalid examples: {examples}"
        )
    labels = numeric.astype(int)
    observed = set(labels.unique())
    if observed != set(EXPECTED_LABELS):
        raise SplitValidationError(
            f"binary manifest must contain labels 0 and 1; observed {sorted(observed)}"
        )
    return labels


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise SplitValidationError(
            f"manifest is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise SplitValidationError("cannot split an empty manifest")
    work = frame.copy()
    if "audit_status" in work.columns:
        invalid = work["audit_status"].fillna("").astype(str).ne("ok")
        if invalid.any():
            examples = work.loc[invalid, "relative_path"].astype(str).tolist()[:5]
            raise SplitValidationError(
                f"manifest contains {int(invalid.sum())} unaudited/invalid files: {examples}"
            )
    work["relative_path"] = work["relative_path"].fillna("").astype(str).str.strip()
    work["source_group"] = work["source_group"].fillna("").astype(str).str.strip()
    work["sha256"] = work["sha256"].fillna("").astype(str).str.strip().str.lower()
    if (work["relative_path"] == "").any():
        raise SplitValidationError("relative_path cannot be blank")
    if work["relative_path"].duplicated().any():
        duplicates = work.loc[
            work["relative_path"].duplicated(keep=False), "relative_path"
        ].tolist()[:5]
        raise SplitValidationError(f"relative_path values are not unique: {duplicates}")
    if (work["source_group"] == "").any():
        raise SplitValidationError(
            "source_group is unresolved; create and review group_map.csv before a "
            "report-valid split"
        )
    if not _verified_series(work["source_group_verified"]).all():
        raise SplitValidationError(
            "source_group rule/map is not verified for every image; random or inferred "
            "patch-level splitting is NOT_VALID_FOR_REPORT"
        )
    if (work["sha256"] == "").any():
        raise SplitValidationError("SHA-256 is required for every audited image")
    malformed_hash = ~work["sha256"].str.fullmatch(r"[0-9a-f]{64}")
    if malformed_hash.any():
        raise SplitValidationError("manifest contains malformed SHA-256 values")
    work["label"] = _normalise_labels(work["label"])
    work["surface"] = work["surface"].fillna("").astype(str).str.strip().str.upper()
    surfaces = set(work["surface"].unique())
    unexpected_surfaces = sorted(surfaces.difference(EXPECTED_SURFACES))
    missing_surfaces = sorted(set(EXPECTED_SURFACES).difference(surfaces))
    if unexpected_surfaces or missing_surfaces:
        raise SplitValidationError(
            "surface must contain exactly D, P and W; "
            f"unexpected={unexpected_surfaces}, missing={missing_surfaces}"
        )

    conflicting_hashes = (
        work.groupby("sha256", sort=False)["label"].nunique().loc[lambda values: values > 1]
    )
    if not conflicting_hashes.empty:
        raise SplitValidationError(
            "exact duplicate bytes have conflicting labels; hashes: "
            f"{conflicting_hashes.index.tolist()[:5]}"
        )
    return work


def _build_units(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    """Collapse groups connected by exact duplicates into indivisible units."""

    groups = sorted(frame["source_group"].unique().tolist())
    sets = _DisjointSet(groups)
    for _, duplicate_rows in frame.groupby("sha256", sort=False):
        hash_groups = sorted(duplicate_rows["source_group"].unique().tolist())
        if len(hash_groups) > 1:
            first = hash_groups[0]
            for other in hash_groups[1:]:
                sets.union(first, other)

    unit_by_group = {group: sets.find(group) for group in groups}
    strata = sorted({(str(row.surface), int(row.label)) for row in frame.itertuples(index=False)})
    stratum_index = {stratum: index for index, stratum in enumerate(strata)}
    frame_with_units = frame.assign(_allocation_unit=frame["source_group"].map(unit_by_group))
    units: list[dict[str, Any]] = []
    for unit_id, rows in frame_with_units.groupby("_allocation_unit", sort=True):
        counts = np.zeros(len(strata), dtype=np.int64)
        for row in rows.itertuples(index=False):
            counts[stratum_index[(str(row.surface), int(row.label))]] += 1
        units.append(
            {
                "id": str(unit_id),
                "source_groups": tuple(sorted(rows["source_group"].unique())),
                "counts": counts,
                "size": len(rows),
            }
        )
    return units, strata


def _allocation_score(
    counts: np.ndarray,
    group_counts: np.ndarray,
    target_counts: np.ndarray,
    target_groups: np.ndarray,
) -> float:
    count_scale = np.maximum(target_counts, 1.0)
    count_error = np.square((counts - target_counts) / count_scale).mean()
    total_counts = counts.sum(axis=1)
    total_targets = target_counts.sum(axis=1)
    total_error = np.square((total_counts - total_targets) / np.maximum(total_targets, 1.0)).mean()
    group_error = np.square((group_counts - target_groups) / np.maximum(target_groups, 1.0)).mean()
    # Per-label/surface balance dominates; total and group ratios regularise ties.
    return float(count_error + 0.35 * total_error + 0.05 * group_error)


def _greedy_allocate(
    units: list[dict[str, Any]],
    ratios: np.ndarray,
    *,
    seed: int,
    restarts: int,
) -> list[int]:
    split_count = len(ratios)
    total_by_stratum = np.sum([unit["counts"] for unit in units], axis=0)
    target_counts = ratios[:, None] * total_by_stratum[None, :]
    target_groups = ratios * len(units)
    best_assignment: list[int] | None = None
    best_score = math.inf

    for restart in range(restarts):
        rng = random.Random(seed + 1_000_003 * restart)
        ordered = list(range(len(units)))
        rng.shuffle(ordered)
        # Place large/rare groups first; the shuffle makes exact-size ties explore
        # alternate deterministic restarts without making the final result random.
        ordered.sort(
            key=lambda index: (
                units[index]["size"],
                float(np.sum(units[index]["counts"] / np.maximum(total_by_stratum, 1))),
            ),
            reverse=True,
        )
        counts = np.zeros_like(target_counts, dtype=np.float64)
        group_counts = np.zeros(split_count, dtype=np.float64)
        assignment = [-1] * len(units)

        for unit_index in ordered:
            choices = list(range(split_count))
            rng.shuffle(choices)
            selected = min(
                choices,
                key=lambda split_index: _allocation_score(
                    counts
                    + np.eye(split_count, dtype=np.float64)[split_index, :, None]
                    * units[unit_index]["counts"][None, :],
                    group_counts + np.eye(split_count, dtype=np.float64)[split_index],
                    target_counts,
                    target_groups,
                ),
            )
            assignment[unit_index] = selected
            counts[selected] += units[unit_index]["counts"]
            group_counts[selected] += 1

        if np.any(group_counts == 0):
            continue

        # Deterministic hill-climb: move one unit when it improves global balance.
        improved = True
        while improved:
            improved = False
            current_score = _allocation_score(counts, group_counts, target_counts, target_groups)
            for unit_index in ordered:
                old_split = assignment[unit_index]
                if group_counts[old_split] <= 1:
                    continue
                for new_split in range(split_count):
                    if new_split == old_split:
                        continue
                    candidate_counts = counts.copy()
                    candidate_groups = group_counts.copy()
                    candidate_counts[old_split] -= units[unit_index]["counts"]
                    candidate_counts[new_split] += units[unit_index]["counts"]
                    candidate_groups[old_split] -= 1
                    candidate_groups[new_split] += 1
                    candidate_score = _allocation_score(
                        candidate_counts,
                        candidate_groups,
                        target_counts,
                        target_groups,
                    )
                    if candidate_score < current_score - 1e-12:
                        assignment[unit_index] = new_split
                        counts = candidate_counts
                        group_counts = candidate_groups
                        current_score = candidate_score
                        improved = True
                        break
                if improved:
                    break

        score = _allocation_score(counts, group_counts, target_counts, target_groups)
        signature = tuple(assignment)
        incumbent_signature = tuple(best_assignment or [])
        if score < best_score - 1e-12 or (
            math.isclose(score, best_score, abs_tol=1e-12)
            and (best_assignment is None or signature < incumbent_signature)
        ):
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise SplitValidationError("unable to allocate at least one group to every split")
    return best_assignment


def create_group_splits(
    frame: pd.DataFrame,
    ratios: Mapping[str, float] | Sequence[float] = DEFAULT_SPLIT_RATIOS,
    *,
    seed: int = 42,
    restarts: int = 64,
) -> pd.DataFrame:
    """Assign train/validation/test while isolating source groups and hashes.

    Exact duplicates that cross source groups connect those groups into one
    indivisible allocation unit.  The allocator balances every ``surface x
    label`` stratum, total image counts and group counts.  It is deterministic
    for a fixed input, seed and ratio set.
    """

    if restarts <= 0:
        raise ValueError("restarts must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed != OFFICIAL_SPLIT_SEED:
        raise SplitValidationError(
            f"official split seed is fixed at {OFFICIAL_SPLIT_SEED}; received {seed!r}"
        )
    split_ratios = _normalise_ratios(ratios)
    work = _validate_frame(frame)
    units, strata = _build_units(work)
    if len(units) < len(split_ratios):
        raise SplitValidationError(
            f"need at least {len(split_ratios)} independent allocation units; "
            f"observed {len(units)} after exact-duplicate linkage"
        )
    for stratum in strata:
        containing = sum(int(unit["counts"][strata.index(stratum)] > 0) for unit in units)
        if containing < len(split_ratios):
            raise SplitValidationError(
                f"stratum surface={stratum[0]}, label={stratum[1]} appears in only "
                f"{containing} independent units; cannot represent it in all splits"
            )

    names = list(split_ratios)
    assignment = _greedy_allocate(
        units,
        np.asarray(list(split_ratios.values()), dtype=np.float64),
        seed=seed,
        restarts=restarts,
    )
    split_by_group: dict[str, str] = {}
    for unit, split_index in zip(units, assignment, strict=True):
        for group in unit["source_groups"]:
            split_by_group[group] = names[split_index]

    result = work.copy()
    result["split"] = work["source_group"].map(split_by_group)
    if result["split"].isna().any():  # pragma: no cover - internal invariant.
        raise RuntimeError("internal error: not every source group was assigned")
    report = audit_split(result)
    if not report["valid"]:
        raise SplitValidationError(
            "split audit failed after allocation: " + "; ".join(report["errors"])
        )
    result.attrs.update(frame.attrs)
    result.attrs["split_seed"] = int(seed)
    result.attrs["split_ratios"] = split_ratios
    result.attrs["split_audit"] = report
    return result


def _pairwise_overlaps(
    frame: pd.DataFrame, column: str, split_names: Sequence[str]
) -> dict[str, list[str]]:
    values = {
        name: set(
            frame.loc[frame["split"] == name, column]
            .dropna()
            .astype(str)
            .loc[lambda series: series != ""]
        )
        for name in split_names
    }
    overlaps: dict[str, list[str]] = {}
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            overlaps[f"{first}__{second}"] = sorted(values[first].intersection(values[second]))
    return overlaps


def audit_split(frame: pd.DataFrame, *, enforce_official_balance: bool = False) -> dict[str, Any]:
    """Audit leakage/strata and report official balance deviations.

    Tiny synthetic smoke manifests may be too small to meet 70/15/15 while
    representing every stratum.  They still receive the complete deviation
    report, but only locked official bundles set ``enforce_official_balance``.
    """

    errors: list[str] = []
    balance_violations: list[str] = []
    required = REQUIRED_COLUMNS.union({"split"})
    missing = sorted(required.difference(frame.columns))
    protocol = {
        "seed": OFFICIAL_SPLIT_SEED,
        "target_ratios": dict(DEFAULT_SPLIT_RATIOS),
        "tolerances": {
            "units": "absolute_fraction_percentage_points",
            "image_fraction": MAX_IMAGE_FRACTION_DEVIATION,
            "source_group_fraction": MAX_SOURCE_GROUP_FRACTION_DEVIATION,
            "surface_label_fraction": MAX_STRATUM_FRACTION_DEVIATION,
        },
        "required_surfaces": list(EXPECTED_SURFACES),
        "required_labels": list(EXPECTED_LABELS),
        "required_surface_label_strata_per_split": [
            f"{surface}|{label}" for surface, label in EXPECTED_STRATA
        ],
        "official_balance_enforced": bool(enforce_official_balance),
    }
    if missing:
        return {
            "schema_version": SPLIT_BUNDLE_SCHEMA_VERSION,
            "valid": False,
            "errors": [f"missing columns: {missing}"],
            "overlaps": {},
            "counts": {},
            "rows": len(frame),
            "protocol": protocol,
        }

    try:
        work = _validate_frame(frame)
    except SplitValidationError as exc:
        errors.append(str(exc))
        work = frame.copy()
        for column in ("relative_path", "source_group", "sha256", "surface"):
            work[column] = work[column].fillna("").astype(str).str.strip()
        work["sha256"] = work["sha256"].str.lower()
        work["surface"] = work["surface"].str.upper()
        work["label"] = pd.to_numeric(work["label"], errors="coerce")

    split_names = list(DEFAULT_SPLIT_RATIOS)
    work["split"] = work["split"].fillna("").astype(str).str.strip().str.lower()
    observed = set(work["split"].loc[lambda values: values.ne("")])
    unexpected = sorted(observed.difference(split_names))
    absent = sorted(set(split_names).difference(observed))
    if unexpected:
        errors.append(f"unexpected split names: {unexpected}")
    if absent:
        errors.append(f"empty/missing splits: {absent}")
    if (work["split"] == "").any():
        errors.append("some rows have no split assignment")

    overlap_columns = {
        "path": "relative_path",
        "source_group": "source_group",
        "sha256": "sha256",
    }
    overlaps = {
        output_name: _pairwise_overlaps(work, column, split_names)
        for output_name, column in overlap_columns.items()
    }
    for category, pairs in overlaps.items():
        leaking = {pair: values for pair, values in pairs.items() if values}
        if leaking:
            errors.append(f"{category} overlaps across splits: {leaking}")

    counts: dict[str, Any] = {}
    total = len(work)
    total_source_groups = int(work["source_group"].nunique())
    stratum_totals = {
        (surface, label): int((work["surface"].eq(surface) & work["label"].eq(label)).sum())
        for surface, label in EXPECTED_STRATA
    }
    for name in split_names:
        subset = work.loc[work["split"] == name]
        by_surface_label = subset.groupby(["surface", "label"], dropna=False).size().sort_index()
        image_fraction = float(len(subset) / total) if total else 0.0
        source_groups = int(subset["source_group"].nunique())
        source_group_fraction = (
            float(source_groups / total_source_groups) if total_source_groups else 0.0
        )
        image_deviation = image_fraction - DEFAULT_SPLIT_RATIOS[name]
        source_group_deviation = source_group_fraction - DEFAULT_SPLIT_RATIOS[name]
        if abs(image_deviation) > MAX_IMAGE_FRACTION_DEVIATION + 1e-12:
            balance_violations.append(
                f"{name} image fraction deviation {image_deviation:+.6f} exceeds "
                f"tolerance {MAX_IMAGE_FRACTION_DEVIATION:.6f}"
            )
        if abs(source_group_deviation) > MAX_SOURCE_GROUP_FRACTION_DEVIATION + 1e-12:
            balance_violations.append(
                f"{name} source-group fraction deviation {source_group_deviation:+.6f} "
                f"exceeds tolerance {MAX_SOURCE_GROUP_FRACTION_DEVIATION:.6f}"
            )

        stratum_fractions: dict[str, float] = {}
        stratum_deviations: dict[str, float] = {}
        surface_label_counts: dict[str, int] = {}
        for surface, label in EXPECTED_STRATA:
            key = f"{surface}|{label}"
            count = int(by_surface_label.get((surface, label), 0))
            total_in_stratum = stratum_totals[(surface, label)]
            fraction = float(count / total_in_stratum) if total_in_stratum else 0.0
            deviation = fraction - DEFAULT_SPLIT_RATIOS[name]
            surface_label_counts[key] = count
            stratum_fractions[key] = fraction
            stratum_deviations[key] = deviation
            if count == 0:
                errors.append(
                    f"{name} is missing required stratum surface={surface}, label={label}"
                )
            if total_in_stratum and abs(deviation) > MAX_STRATUM_FRACTION_DEVIATION + 1e-12:
                balance_violations.append(
                    f"{name} stratum {key} fraction deviation {deviation:+.6f} exceeds "
                    f"tolerance {MAX_STRATUM_FRACTION_DEVIATION:.6f}"
                )
        counts[name] = {
            "images": len(subset),
            "fraction": image_fraction,
            "target_fraction": DEFAULT_SPLIT_RATIOS[name],
            "fraction_deviation": image_deviation,
            "source_groups": source_groups,
            "source_group_fraction": source_group_fraction,
            "source_group_fraction_deviation": source_group_deviation,
            "labels": {
                str(key): int(value)
                for key, value in subset.groupby("label", dropna=False).size().items()
            },
            "surfaces": {
                str(key): int(value)
                for key, value in subset.groupby("surface", dropna=False).size().items()
            },
            "surface_label": surface_label_counts,
            "surface_label_fraction": stratum_fractions,
            "surface_label_fraction_deviation": stratum_deviations,
        }
    if enforce_official_balance:
        errors.extend(balance_violations)
    return {
        "schema_version": SPLIT_BUNDLE_SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "rows": int(total),
        "source_groups": total_source_groups,
        "surface_label_totals": {
            f"{surface}|{label}": count for (surface, label), count in stratum_totals.items()
        },
        "counts": counts,
        "overlaps": overlaps,
        "protocol": protocol,
        "balance": {
            "within_official_tolerances": not balance_violations,
            "violations": balance_violations,
        },
    }


def manifest_sha256(frame: pd.DataFrame) -> str:
    """Hash canonical CSV content independent of DataFrame row order."""

    if "relative_path" not in frame.columns:
        raise ValueError("manifest requires relative_path for canonical hashing")
    ordered_columns = sorted(str(column) for column in frame.columns)
    canonical = frame.loc[:, ordered_columns].copy()
    canonical = canonical.sort_values("relative_path", kind="stable").reset_index(drop=True)
    content = canonical.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.17g",
    ).encode("utf-8")
    return sha256(content).hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise SplitValidationError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitValidationError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitValidationError(f"{description} must be a JSON object: {path}")
    return value


def _required_sha256(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SplitValidationError(f"{field} must be a 64-character SHA-256 hex digest")
    return normalized


def _required_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SplitValidationError(f"{field} must be a non-negative integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise SplitValidationError(f"{field} must be a non-negative integer") from exc
    if converted < 0 or converted != value:
        raise SplitValidationError(f"{field} must be a non-negative integer")
    return converted


def _require_official_protocol(payload: Mapping[str, Any], description: str) -> None:
    if payload.get("seed") != OFFICIAL_SPLIT_SEED:
        raise SplitValidationError(
            f"{description}.seed must equal official seed {OFFICIAL_SPLIT_SEED}"
        )
    ratios = payload.get("ratios")
    if not isinstance(ratios, Mapping):
        raise SplitValidationError(f"{description}.ratios must be an object")
    _normalise_ratios(ratios)


def _resolve_report_path(base: Path, value: Any, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise SplitValidationError(f"{field} is missing")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _validate_curation_lineage(
    *,
    conflict_report_path: Path,
    cleaned_manifest_path: Path,
    parent_manifest_path: Path | None = None,
    conflict_rows_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the exact parent -> excluded conflicts -> cleaned hash chain."""

    report_path = conflict_report_path.resolve()
    report = _read_json_object(report_path, "curation conflict report")
    if report.get("schema_version") != 1:
        raise SplitValidationError("curation conflict report has unsupported schema_version")
    if report.get("artifact_type") != "pre_split_exact_label_conflict_curation":
        raise SplitValidationError("curation conflict report has an unexpected artifact_type")
    if report.get("immutable") is not True:
        raise SplitValidationError("curation conflict report must declare immutable=true")
    policy = report.get("policy")
    validation = report.get("validation")
    if not isinstance(policy, Mapping) or not isinstance(validation, Mapping):
        raise SplitValidationError("curation conflict report lacks policy/validation objects")
    if policy.get("test_set_used_for_decision") is not False:
        raise SplitValidationError("curation must declare test_set_used_for_decision=false")
    if policy.get("labels_rewritten") is not False:
        raise SplitValidationError("curation must declare labels_rewritten=false")
    required_validation = {
        "all_parent_rows_accounted_for": True,
        "all_source_groups_nonblank_and_verified": True,
        "only_contradictory_hash_groups_excluded": True,
        "same_label_exact_duplicates_retained": True,
        "contradictory_hash_groups_after_curation": 0,
        "split_assignments_present_during_curation": False,
    }
    for field, expected in required_validation.items():
        if validation.get(field) != expected:
            raise SplitValidationError(f"curation validation.{field} must equal {expected!r}")

    artifacts = report.get("artifacts")
    parent_record = report.get("parent_manifest")
    counts = report.get("counts")
    if not isinstance(artifacts, Mapping) or not isinstance(parent_record, Mapping):
        raise SplitValidationError("curation report lacks artifact/parent provenance")
    if not isinstance(counts, Mapping):
        raise SplitValidationError("curation report lacks counts")
    cleaned_record = artifacts.get("cleaned_manifest")
    conflicts_record = artifacts.get("conflict_rows_csv")
    if not isinstance(cleaned_record, Mapping) or not isinstance(conflicts_record, Mapping):
        raise SplitValidationError("curation report lacks cleaned/conflict artifact records")

    cleaned_path = cleaned_manifest_path.resolve()
    if parent_manifest_path is None:
        parent_path = _resolve_report_path(
            report_path.parent,
            parent_record.get("path"),
            "curation parent_manifest.path",
        )
    else:
        parent_path = parent_manifest_path.resolve()
    if conflict_rows_path is None:
        conflict_path = _resolve_report_path(
            report_path.parent,
            conflicts_record.get("filename"),
            "curation artifacts.conflict_rows_csv.filename",
        )
    else:
        conflict_path = conflict_rows_path.resolve()

    recorded_cleaned_path = _resolve_report_path(
        report_path.parent,
        cleaned_record.get("filename"),
        "curation artifacts.cleaned_manifest.filename",
    )
    # An embedded split snapshot intentionally lives beside a copied report and
    # therefore resolves to the copied pre_split_manifest.csv.  At creation this
    # also proves that the CLI input is exactly the artifact named by the report.
    if recorded_cleaned_path != cleaned_path:
        raise SplitValidationError(
            "input cleaned manifest is not the artifact referenced by conflict_report.json"
        )

    for path, description in (
        (parent_path, "curation parent manifest"),
        (cleaned_path, "curation cleaned manifest"),
        (conflict_path, "curation conflict rows"),
    ):
        if not path.is_file():
            raise SplitValidationError(f"missing {description}: {path}")

    parent = load_manifest_table(parent_path)
    cleaned = load_manifest_table(cleaned_path)
    try:
        conflict_rows = pd.read_csv(conflict_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise SplitValidationError(f"cannot read curation conflict rows: {exc}") from exc

    parent_rows = _required_nonnegative_int(parent_record.get("rows"), "parent_manifest.rows")
    cleaned_rows = _required_nonnegative_int(cleaned_record.get("rows"), "cleaned_manifest.rows")
    excluded_rows = _required_nonnegative_int(
        conflicts_record.get("rows"), "conflict_rows_csv.rows"
    )
    if (
        len(parent) != parent_rows
        or len(cleaned) != cleaned_rows
        or len(conflict_rows) != excluded_rows
    ):
        raise SplitValidationError("curation artifact row counts do not match conflict report")
    if parent_rows != cleaned_rows + excluded_rows:
        raise SplitValidationError("curation parent rows are not cleaned rows plus excluded rows")
    for field, observed in (
        ("parent_rows", parent_rows),
        ("cleaned_rows", cleaned_rows),
        ("excluded_rows", excluded_rows),
    ):
        if _required_nonnegative_int(counts.get(field), f"counts.{field}") != observed:
            raise SplitValidationError(f"curation counts.{field} does not match artifact rows")

    expected_parent_file_hash = _required_sha256(
        parent_record.get("file_sha256"), "parent_manifest.file_sha256"
    )
    expected_parent_canonical_hash = _required_sha256(
        parent_record.get("canonical_sha256"), "parent_manifest.canonical_sha256"
    )
    expected_cleaned_file_hash = _required_sha256(
        cleaned_record.get("file_sha256"), "cleaned_manifest.file_sha256"
    )
    expected_cleaned_canonical_hash = _required_sha256(
        cleaned_record.get("canonical_sha256"), "cleaned_manifest.canonical_sha256"
    )
    expected_conflict_file_hash = _required_sha256(
        conflicts_record.get("file_sha256"), "conflict_rows_csv.file_sha256"
    )
    checks = (
        (sha256_file(parent_path), expected_parent_file_hash, "parent manifest file"),
        (manifest_sha256(parent), expected_parent_canonical_hash, "parent manifest canonical"),
        (sha256_file(cleaned_path), expected_cleaned_file_hash, "cleaned manifest file"),
        (
            manifest_sha256(cleaned),
            expected_cleaned_canonical_hash,
            "cleaned manifest canonical",
        ),
        (sha256_file(conflict_path), expected_conflict_file_hash, "conflict rows file"),
    )
    for observed, expected, description in checks:
        if observed != expected:
            raise SplitValidationError(f"curation {description} hash does not match report")

    required_conflict_columns = {"exact_sha256", "relative_path", "label"}
    missing_conflict_columns = required_conflict_columns.difference(conflict_rows.columns)
    if missing_conflict_columns:
        raise SplitValidationError(
            f"conflict_rows.csv lacks columns: {sorted(missing_conflict_columns)}"
        )
    parent_paths = parent["relative_path"].astype(str)
    cleaned_paths = cleaned["relative_path"].astype(str)
    excluded_paths = conflict_rows["relative_path"].astype(str)
    if (
        parent_paths.duplicated().any()
        or cleaned_paths.duplicated().any()
        or excluded_paths.duplicated().any()
    ):
        raise SplitValidationError("curation lineage contains duplicate relative_path values")
    parent_path_set = set(parent_paths)
    cleaned_path_set = set(cleaned_paths)
    excluded_path_set = set(excluded_paths)
    if cleaned_path_set.intersection(excluded_path_set):
        raise SplitValidationError("curation cleaned and excluded path sets overlap")
    if parent_path_set != cleaned_path_set.union(excluded_path_set):
        raise SplitValidationError(
            "curation path sets do not account for the parent manifest exactly"
        )

    parent_by_path = parent.assign(_path=parent_paths).set_index("_path", drop=True)
    cleaned_by_path = cleaned.assign(_path=cleaned_paths).set_index("_path", drop=True)
    for column in cleaned.columns:
        expected_values = parent_by_path.loc[cleaned_by_path.index, column]
        if (
            not cleaned_by_path[column]
            .astype("string")
            .fillna("")
            .equals(expected_values.astype("string").fillna(""))
        ):
            raise SplitValidationError(f"curation changed retained parent column {column!r}")

    normalized_parent_hashes = parent_by_path["sha256"].astype(str).str.strip().str.lower()
    normalized_parent_labels = _normalise_labels(parent_by_path["label"])
    conflicting_hashes = set(
        pd.DataFrame(
            {"sha256": normalized_parent_hashes, "label": normalized_parent_labels},
            index=parent_by_path.index,
        )
        .groupby("sha256")["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    recorded_excluded_hashes = set(
        conflict_rows["exact_sha256"].astype(str).str.strip().str.lower()
    )
    expected_excluded_hashes = set(normalized_parent_hashes.loc[list(excluded_path_set)])
    if (
        recorded_excluded_hashes != conflicting_hashes
        or expected_excluded_hashes != conflicting_hashes
    ):
        raise SplitValidationError(
            "curation exclusions are not exactly all contradictory SHA-256 groups"
        )
    conflict_labels = pd.to_numeric(
        conflict_rows.set_index("relative_path")["label"], errors="coerce"
    )
    expected_labels = normalized_parent_labels.loc[conflict_labels.index]
    if not conflict_labels.equals(expected_labels.astype(conflict_labels.dtype)):
        raise SplitValidationError("curation conflict labels do not match parent manifest")

    _validate_frame(cleaned)
    if (
        "split" in cleaned.columns
        and cleaned["split"].fillna("").astype(str).str.strip().ne("").any()
    ):
        raise SplitValidationError("curation cleaned manifest must not contain split assignments")
    return {
        "conflict_report_file_sha256": sha256_file(report_path),
        "parent_manifest": {
            "file_sha256": expected_parent_file_hash,
            "canonical_sha256": expected_parent_canonical_hash,
            "rows": parent_rows,
        },
        "cleaned_manifest": {
            "file_sha256": expected_cleaned_file_hash,
            "canonical_sha256": expected_cleaned_canonical_hash,
            "rows": cleaned_rows,
        },
        "conflict_rows": {
            "file_sha256": expected_conflict_file_hash,
            "rows": excluded_rows,
            "conflicting_sha256_groups": len(conflicting_hashes),
        },
        "resolved_paths": {
            "parent_manifest": parent_path,
            "cleaned_manifest": cleaned_path,
            "conflict_rows": conflict_path,
            "conflict_report": report_path,
        },
    }


def _write_json_file(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _inventory_entry(
    path: Path, *, rows: int | None = None, canonical_sha256: str | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {"file_sha256": sha256_file(path)}
    if rows is not None:
        value["rows"] = int(rows)
    if canonical_sha256 is not None:
        value["canonical_sha256"] = canonical_sha256
    return value


def create_locked_split_bundle(
    input_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    conflict_report_path: str | Path | None = None,
    seed: int = OFFICIAL_SPLIT_SEED,
    ratios: Mapping[str, float] | Sequence[float] = DEFAULT_SPLIT_RATIOS,
    restarts: int = 64,
) -> LockedSplitBundle:
    """Atomically create one immutable, portable official split bundle."""

    input_manifest = Path(input_manifest_path).resolve()
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable split directory: {destination}")
    report_source = (
        Path(conflict_report_path).resolve()
        if conflict_report_path is not None
        else input_manifest.with_name(CONFLICT_REPORT_SNAPSHOT_FILENAME).resolve()
    )
    lineage = _validate_curation_lineage(
        conflict_report_path=report_source,
        cleaned_manifest_path=input_manifest,
    )
    split_ratios = _normalise_ratios(ratios)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed != OFFICIAL_SPLIT_SEED:
        raise SplitValidationError(
            f"official split seed is fixed at {OFFICIAL_SPLIT_SEED}; received {seed!r}"
        )
    frame = load_manifest_table(input_manifest)
    split_frame = create_group_splits(
        frame,
        ratios=split_ratios,
        seed=seed,
        restarts=restarts,
    )
    audit = audit_split(split_frame, enforce_official_balance=True)
    if not audit["valid"]:
        raise SplitValidationError("official split audit failed: " + "; ".join(audit["errors"]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-", dir=destination.parent
    ) as temporary_name:
        staging = Path(temporary_name) / "bundle"
        staging.mkdir()
        manifest_path = staging / SPLIT_MANIFEST_FILENAME
        split_frame.to_csv(manifest_path, index=False, lineterminator="\n")
        subset_frames: dict[str, pd.DataFrame] = {}
        for split_name, filename in SPLIT_SUBSET_FILENAMES.items():
            subset = split_frame.loc[split_frame["split"].eq(split_name)].copy()
            subset_frames[split_name] = subset
            subset.to_csv(staging / filename, index=False, lineterminator="\n")
        audit_path = staging / SPLIT_AUDIT_FILENAME
        _write_json_file(audit, audit_path)

        source_paths = lineage["resolved_paths"]
        shutil.copy2(source_paths["parent_manifest"], staging / PARENT_MANIFEST_SNAPSHOT_FILENAME)
        shutil.copy2(input_manifest, staging / CURATED_MANIFEST_SNAPSHOT_FILENAME)
        shutil.copy2(source_paths["conflict_rows"], staging / CONFLICT_ROWS_SNAPSHOT_FILENAME)
        shutil.copy2(report_source, staging / CONFLICT_REPORT_SNAPSHOT_FILENAME)

        artifact_entries: dict[str, dict[str, Any]] = {
            SPLIT_MANIFEST_FILENAME: _inventory_entry(
                manifest_path,
                rows=len(split_frame),
                canonical_sha256=manifest_sha256(split_frame),
            ),
            SPLIT_AUDIT_FILENAME: _inventory_entry(audit_path),
            PARENT_MANIFEST_SNAPSHOT_FILENAME: _inventory_entry(
                staging / PARENT_MANIFEST_SNAPSHOT_FILENAME,
                rows=lineage["parent_manifest"]["rows"],
                canonical_sha256=lineage["parent_manifest"]["canonical_sha256"],
            ),
            CURATED_MANIFEST_SNAPSHOT_FILENAME: _inventory_entry(
                staging / CURATED_MANIFEST_SNAPSHOT_FILENAME,
                rows=lineage["cleaned_manifest"]["rows"],
                canonical_sha256=lineage["cleaned_manifest"]["canonical_sha256"],
            ),
            CONFLICT_ROWS_SNAPSHOT_FILENAME: _inventory_entry(
                staging / CONFLICT_ROWS_SNAPSHOT_FILENAME,
                rows=lineage["conflict_rows"]["rows"],
            ),
            CONFLICT_REPORT_SNAPSHOT_FILENAME: _inventory_entry(
                staging / CONFLICT_REPORT_SNAPSHOT_FILENAME
            ),
        }
        for split_name, filename in SPLIT_SUBSET_FILENAMES.items():
            subset = subset_frames[split_name]
            artifact_entries[filename] = _inventory_entry(
                staging / filename,
                rows=len(subset),
                canonical_sha256=manifest_sha256(subset),
            )
        inventory = {
            "schema_version": SPLIT_BUNDLE_SCHEMA_VERSION,
            "artifact_type": "locked_official_split_inventory",
            "status": "LOCKED_SPLIT_INVENTORY",
            "created_utc": datetime.now(UTC).isoformat(),
            "immutable": True,
            "seed": OFFICIAL_SPLIT_SEED,
            "ratios": dict(split_ratios),
            "canonical_manifest_sha256": manifest_sha256(split_frame),
            "split_audit_valid": True,
            "protocol": audit["protocol"],
            "curation_lineage": {
                "conflict_report_file_sha256": lineage["conflict_report_file_sha256"],
                "parent_manifest": {
                    "filename": PARENT_MANIFEST_SNAPSHOT_FILENAME,
                    **lineage["parent_manifest"],
                },
                "cleaned_manifest": {
                    "filename": CURATED_MANIFEST_SNAPSHOT_FILENAME,
                    **lineage["cleaned_manifest"],
                },
                "conflict_rows": {
                    "filename": CONFLICT_ROWS_SNAPSHOT_FILENAME,
                    **lineage["conflict_rows"],
                },
                "conflict_report": {"filename": CONFLICT_REPORT_SNAPSHOT_FILENAME},
            },
            "files": artifact_entries,
            "immutable_protocol": (
                "Do not regenerate or mutate this directory after validation or test access."
            ),
        }
        inventory_path = staging / SPLIT_INVENTORY_FILENAME
        _write_json_file(inventory, inventory_path)
        completion_hashes = {
            name: sha256_file(staging / name)
            for name in (*SPLIT_BUNDLE_ARTIFACT_FILENAMES, SPLIT_INVENTORY_FILENAME)
        }
        completion = {
            "schema_version": SPLIT_BUNDLE_SCHEMA_VERSION,
            "artifact_type": "locked_official_split_completion",
            "status": "LOCKED_SPLIT_COMPLETE",
            "completed_utc": datetime.now(UTC).isoformat(),
            "immutable": True,
            "seed": OFFICIAL_SPLIT_SEED,
            "ratios": dict(split_ratios),
            "manifest_sha256": manifest_sha256(split_frame),
            "inventory_sha256": completion_hashes[SPLIT_INVENTORY_FILENAME],
            "artifact_sha256": completion_hashes,
        }
        # Deliberately the final file written.  Its absence makes a staged or
        # interrupted directory invalid even if every other artifact exists.
        _write_json_file(completion, staging / SPLIT_COMPLETION_FILENAME)
        verify_locked_split_bundle(staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable split directory: {destination}")
        try:
            staging.rename(destination)
        except OSError:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite immutable split directory: {destination}"
                ) from None
            raise
    return verify_locked_split_bundle(destination)


def verify_locked_split_bundle(
    bundle_dir: str | Path,
    *,
    conflict_report_path: str | Path | None = None,
) -> LockedSplitBundle:
    """Fail closed unless a split bundle and its curation lineage are intact."""

    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise SplitValidationError(f"locked split bundle does not exist: {root}")
    children = list(root.iterdir())
    unsafe = [path.name for path in children if path.is_symlink() or not path.is_file()]
    if unsafe:
        raise SplitValidationError(f"locked split bundle contains unsafe entries: {unsafe}")
    observed_names = {path.name for path in children}
    expected_names = set(SPLIT_BUNDLE_FILENAMES)
    if observed_names != expected_names:
        raise SplitValidationError(
            "locked split bundle file set mismatch: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )

    completion_path = root / SPLIT_COMPLETION_FILENAME
    inventory_path = root / SPLIT_INVENTORY_FILENAME
    completion = _read_json_object(completion_path, "split completion marker")
    inventory = _read_json_object(inventory_path, "split hash inventory")
    if (
        completion.get("schema_version") != SPLIT_BUNDLE_SCHEMA_VERSION
        or completion.get("artifact_type") != "locked_official_split_completion"
        or completion.get("status") != "LOCKED_SPLIT_COMPLETE"
        or completion.get("immutable") is not True
    ):
        raise SplitValidationError("split_complete.json is not a valid immutable completion marker")
    if (
        inventory.get("schema_version") != SPLIT_BUNDLE_SCHEMA_VERSION
        or inventory.get("artifact_type") != "locked_official_split_inventory"
        or inventory.get("status") != "LOCKED_SPLIT_INVENTORY"
        or inventory.get("immutable") is not True
    ):
        raise SplitValidationError("manifest_hashes.json is not a valid immutable inventory")
    _require_official_protocol(completion, "split_complete.json")
    _require_official_protocol(inventory, "manifest_hashes.json")

    actual_inventory_hash = sha256_file(inventory_path)
    if (
        _required_sha256(completion.get("inventory_sha256"), "inventory_sha256")
        != actual_inventory_hash
    ):
        raise SplitValidationError("manifest_hashes.json does not match split completion marker")
    completion_hashes = completion.get("artifact_sha256")
    if not isinstance(completion_hashes, Mapping):
        raise SplitValidationError("split_complete.json lacks artifact_sha256 inventory")
    expected_completion_names = set(SPLIT_BUNDLE_ARTIFACT_FILENAMES).union(
        {SPLIT_INVENTORY_FILENAME}
    )
    if set(completion_hashes) != expected_completion_names:
        raise SplitValidationError("split completion artifact inventory has an unexpected file set")
    for name, expected_hash in completion_hashes.items():
        if sha256_file(root / name) != _required_sha256(
            expected_hash, f"split_complete.artifact_sha256.{name}"
        ):
            raise SplitValidationError(f"locked split artifact hash mismatch: {name}")

    inventory_files = inventory.get("files")
    if not isinstance(inventory_files, Mapping) or set(inventory_files) != set(
        SPLIT_BUNDLE_ARTIFACT_FILENAMES
    ):
        raise SplitValidationError("manifest_hashes.json has an unexpected artifact file set")
    for name, record in inventory_files.items():
        if not isinstance(record, Mapping):
            raise SplitValidationError(f"manifest_hashes.files.{name} must be an object")
        if sha256_file(root / name) != _required_sha256(
            record.get("file_sha256"), f"manifest_hashes.files.{name}.file_sha256"
        ):
            raise SplitValidationError(f"split inventory hash mismatch: {name}")

    if conflict_report_path is not None:
        external_report = Path(conflict_report_path).resolve()
        if not external_report.is_file():
            raise SplitValidationError(
                f"explicit conflict report does not exist: {external_report}"
            )
        if sha256_file(external_report) != sha256_file(root / CONFLICT_REPORT_SNAPSHOT_FILENAME):
            raise SplitValidationError(
                "explicit conflict report differs from bundled lineage snapshot"
            )
    lineage = _validate_curation_lineage(
        conflict_report_path=root / CONFLICT_REPORT_SNAPSHOT_FILENAME,
        cleaned_manifest_path=root / CURATED_MANIFEST_SNAPSHOT_FILENAME,
        parent_manifest_path=root / PARENT_MANIFEST_SNAPSHOT_FILENAME,
        conflict_rows_path=root / CONFLICT_ROWS_SNAPSHOT_FILENAME,
    )
    recorded_lineage = inventory.get("curation_lineage")
    if not isinstance(recorded_lineage, Mapping):
        raise SplitValidationError("manifest_hashes.json lacks curation_lineage")
    if (
        recorded_lineage.get("conflict_report_file_sha256")
        != lineage["conflict_report_file_sha256"]
    ):
        raise SplitValidationError("curation conflict report hash does not match split inventory")
    for field in ("parent_manifest", "cleaned_manifest", "conflict_rows"):
        record = recorded_lineage.get(field)
        if not isinstance(record, Mapping):
            raise SplitValidationError(f"curation_lineage.{field} must be an object")
        for key, expected in lineage[field].items():
            if record.get(key) != expected:
                raise SplitValidationError(
                    f"curation_lineage.{field}.{key} does not match snapshot"
                )

    manifest_path = root / SPLIT_MANIFEST_FILENAME
    manifest = load_manifest_table(manifest_path)
    canonical_hash = manifest_sha256(manifest)
    for field, value in (
        ("completion.manifest_sha256", completion.get("manifest_sha256")),
        ("inventory.canonical_manifest_sha256", inventory.get("canonical_manifest_sha256")),
        (
            "inventory.files.manifest.csv.canonical_sha256",
            inventory_files[SPLIT_MANIFEST_FILENAME].get("canonical_sha256"),
        ),
    ):
        if _required_sha256(value, field) != canonical_hash:
            raise SplitValidationError(f"canonical split manifest hash mismatch: {field}")
    if _required_nonnegative_int(
        inventory_files[SPLIT_MANIFEST_FILENAME].get("rows"),
        "manifest_hashes.files.manifest.csv.rows",
    ) != len(manifest):
        raise SplitValidationError("split manifest row count does not match inventory")

    recomputed_audit = audit_split(manifest, enforce_official_balance=True)
    recorded_audit = _read_json_object(root / SPLIT_AUDIT_FILENAME, "split audit")
    if not recomputed_audit["valid"]:
        raise SplitValidationError(
            "locked split manifest fails official audit: " + "; ".join(recomputed_audit["errors"])
        )
    if recorded_audit != recomputed_audit:
        raise SplitValidationError("split_audit.json does not match recomputed official audit")
    if (
        inventory.get("split_audit_valid") is not True
        or inventory.get("protocol") != recomputed_audit["protocol"]
    ):
        raise SplitValidationError("split inventory protocol/audit status is inconsistent")

    for split_name, filename in SPLIT_SUBSET_FILENAMES.items():
        subset = load_manifest_table(root / filename)
        expected_subset = manifest.loc[manifest["split"].eq(split_name)].copy()
        record = inventory_files[filename]
        if len(subset) != len(expected_subset) or _required_nonnegative_int(
            record.get("rows"), f"manifest_hashes.files.{filename}.rows"
        ) != len(subset):
            raise SplitValidationError(f"{filename} row count does not match manifest.csv")
        subset_hash = manifest_sha256(subset)
        if subset_hash != manifest_sha256(expected_subset) or subset_hash != _required_sha256(
            record.get("canonical_sha256"),
            f"manifest_hashes.files.{filename}.canonical_sha256",
        ):
            raise SplitValidationError(f"{filename} is not the exact {split_name} subset")

    return LockedSplitBundle(
        directory=root,
        manifest_path=manifest_path,
        audit_path=root / SPLIT_AUDIT_FILENAME,
        inventory_path=inventory_path,
        completion_path=completion_path,
        manifest_sha256=canonical_hash,
        audit=recomputed_audit,
        inventory=inventory,
        completion=completion,
    )


def split_audit_json(frame: pd.DataFrame) -> str:
    """Serialise the stable split audit representation."""

    return json.dumps(audit_split(frame), ensure_ascii=False, indent=2, sort_keys=True)
