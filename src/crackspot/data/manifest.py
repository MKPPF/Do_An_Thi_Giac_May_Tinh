"""Build a reproducible, safety-audited SDNET2018 image manifest."""

from __future__ import annotations

import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, Final

import pandas as pd

from .audit import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_PIXELS,
    ImageValidationError,
    inspect_image,
)

FOLDER_SCHEMA: Final[dict[str, tuple[str, int]]] = {
    "CD": ("D", 1),
    "UD": ("D", 0),
    "CP": ("P", 1),
    "UP": ("P", 0),
    "CW": ("W", 1),
    "UW": ("W", 0),
}
CLASS_NAMES: Final[dict[int, str]] = {0: "Non-crack", 1: "Crack"}
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "relative_path",
    "label",
    "class_name",
    "surface",
    "source_group",
    "source_group_verified",
    "group_resolution_method",
    "sha256",
    "perceptual_hash",
    "width",
    "height",
    "source_width",
    "source_height",
    "source_mode",
    "image_format",
    "file_size_bytes",
    "exif_orientation",
    "audit_status",
    "audit_error",
    "split",
)
MANIFEST_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "relative_path",
    "absolute_path",
    "class_name",
    "surface",
    "source_group",
    "group_resolution_method",
    "sha256",
    "perceptual_hash",
    "source_mode",
    "image_format",
    "audit_status",
    "audit_error",
    "split",
)


class DatasetStructureError(ValueError):
    """Raised when a path does not follow SDNET2018's class-folder schema."""


class GroupResolutionError(ValueError):
    """Raised when a claimed source-group rule cannot be verified completely."""


def load_manifest_table(path: str | PathLike[str]) -> pd.DataFrame:
    """Load a CSV/Parquet manifest without corrupting textual identifiers.

    Pandas' default CSV inference turns identifiers such as ``source_group=001``
    into integer ``1``.  Besides losing provenance, that changes the canonical
    manifest hash between split creation and later training/evaluation loads.
    Known path, identifier and hash fields are therefore pinned to textual CSV
    values.  Numeric labels/dimensions and booleans retain normal inference.

    Parquet already stores a typed schema and is returned without coercion.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        text_dtypes = {column: str for column in MANIFEST_TEXT_COLUMNS}
        return pd.read_csv(source, dtype=text_dtypes)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"manifest must be CSV or Parquet: {source}")


def _normalise_relative_path(value: str | PathLike[str]) -> str:
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"relative path is unsafe: {value!r}")
    return path.as_posix()


def parse_sdnet_path(path: str | PathLike[str]) -> tuple[str, int]:
    """Return ``(surface, label)`` from a CD/UD/CP/UP/CW/UW path.

    ``Crack`` is always label 1 and ``Non-crack`` is always label 0.  A path
    containing zero or multiple class folders is rejected instead of guessed.
    """

    # Treat both separators explicitly so manifests remain portable between
    # Windows and Linux/Colab even before paths are materialised locally.
    parts = [part.upper() for part in PurePosixPath(str(path).replace("\\", "/")).parts]
    matches = [(index, part) for index, part in enumerate(parts) if part in FOLDER_SCHEMA]
    if len(matches) != 1:
        raise DatasetStructureError(
            f"expected exactly one SDNET class folder in {path!s}; found "
            f"{[match[1] for match in matches]}"
        )
    index, folder = matches[0]
    surface, label = FOLDER_SCHEMA[folder]
    if index > 0 and parts[index - 1] in {"D", "P", "W"} and parts[index - 1] != surface:
        raise DatasetStructureError(
            f"class folder {folder} is under inconsistent surface {parts[index - 1]}"
        )
    return surface, label


def _coerce_verified(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "verified"}


def _load_group_map(
    group_map: str | PathLike[str] | pd.DataFrame | Mapping[str, str],
) -> pd.DataFrame:
    if isinstance(group_map, pd.DataFrame):
        frame = group_map.copy()
    elif isinstance(group_map, Mapping):
        frame = pd.DataFrame(
            {
                "relative_path": list(group_map.keys()),
                "source_group": list(group_map.values()),
            }
        )
    else:
        frame = pd.read_csv(group_map, dtype=str, keep_default_na=False)
    required = {"relative_path", "source_group"}
    missing = required.difference(frame.columns)
    if missing:
        raise GroupResolutionError(f"group map is missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    try:
        frame["relative_path"] = frame["relative_path"].map(_normalise_relative_path)
    except ValueError as exc:
        raise GroupResolutionError(str(exc)) from exc
    frame["source_group"] = frame["source_group"].astype(str).str.strip()
    if frame["relative_path"].duplicated().any():
        duplicates = frame.loc[
            frame["relative_path"].duplicated(keep=False), "relative_path"
        ].tolist()
        raise GroupResolutionError(f"group map has duplicate paths: {duplicates[:5]}")
    return frame


def _resolve_groups(
    relative_paths: list[str],
    sha_by_path: Mapping[str, str],
    *,
    group_map: str | PathLike[str] | pd.DataFrame | Mapping[str, str] | None,
    group_regex: str | None,
    group_rule_verified: bool,
) -> dict[str, tuple[str, bool, str]]:
    if group_map is not None and group_regex is not None:
        raise GroupResolutionError("use either group_map or group_regex, not both")

    if group_map is not None:
        frame = _load_group_map(group_map)
        expected = set(relative_paths)
        provided = set(frame["relative_path"])
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing or extra:
            raise GroupResolutionError(
                "group map coverage does not match the audited dataset; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if (frame["source_group"] == "").any():
            rows = frame.loc[frame["source_group"] == "", "relative_path"].tolist()
            raise GroupResolutionError(f"blank source_group values: {rows[:5]}")

        declared_verified = False
        verification_column = next(
            (column for column in ("source_group_verified", "verified") if column in frame.columns),
            None,
        )
        if verification_column is not None:
            declared_verified = bool(frame[verification_column].map(_coerce_verified).all())
        verified = bool(group_rule_verified or declared_verified)

        if "sha256" in frame.columns:
            for row in frame.itertuples(index=False):
                expected_sha = str(getattr(row, "sha256", "")).strip().lower()
                observed_sha = sha_by_path.get(row.relative_path, "").lower()
                if expected_sha and observed_sha and expected_sha != observed_sha:
                    raise GroupResolutionError(
                        f"SHA-256 mismatch in group map for {row.relative_path}"
                    )
        method = "verified_group_map" if verified else "unverified_group_map"
        return {
            str(row.relative_path): (str(row.source_group), verified, method)
            for row in frame.itertuples(index=False)
        }

    if group_regex is not None:
        try:
            pattern = re.compile(group_regex)
        except re.error as exc:
            raise GroupResolutionError(f"invalid group regex: {exc}") from exc
        if "source_group" not in pattern.groupindex and pattern.groups < 1:
            raise GroupResolutionError(
                "group regex needs a named 'source_group' or first capture group"
            )
        result: dict[str, tuple[str, bool, str]] = {}
        unmatched: list[str] = []
        for relative_path in relative_paths:
            match = pattern.search(relative_path)
            group = ""
            if match is not None:
                group = (
                    match.group("source_group")
                    if "source_group" in pattern.groupindex
                    else match.group(1)
                ).strip()
            if not group:
                unmatched.append(relative_path)
            result[relative_path] = (
                group,
                bool(group_rule_verified and group),
                "verified_regex" if group_rule_verified else "unverified_regex",
            )
        if unmatched and group_rule_verified:
            raise GroupResolutionError(
                f"verified regex did not resolve {len(unmatched)} paths: {unmatched[:5]}"
            )
        return result

    return {path: ("", False, "unresolved") for path in relative_paths}


def _candidate_files(dataset_root: Path) -> list[Path]:
    root_resolved = dataset_root.resolve()
    candidates: list[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            parse_sdnet_path(path.relative_to(dataset_root))
        except DatasetStructureError:
            continue
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise DatasetStructureError(
                f"dataset file resolves outside dataset root: {path}"
            ) from exc
        candidates.append(path)
    return sorted(candidates, key=lambda item: item.relative_to(dataset_root).as_posix())


def build_manifest(
    dataset_root: str | PathLike[str],
    *,
    group_map: str | PathLike[str] | pd.DataFrame | Mapping[str, str] | None = None,
    group_regex: str | None = None,
    group_rule_verified: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    raise_on_image_error: bool = False,
) -> pd.DataFrame:
    """Audit files beneath the six SDNET class folders and build a manifest.

    Invalid files remain in the returned frame with ``audit_status='invalid'``;
    this makes data loss visible and causes split creation to fail fast.  Set
    ``raise_on_image_error`` for interactive callers that prefer the first error.
    Source groups are considered verified only when the map marks every row as
    verified or the caller explicitly confirms a reviewed rule.
    """

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    candidates = _candidate_files(root)
    if not candidates:
        raise DatasetStructureError(f"no files found beneath CD/UD/CP/UP/CW/UW in {root}")

    rows: list[dict[str, Any]] = []
    for path in candidates:
        relative_path = path.relative_to(root).as_posix()
        surface, label = parse_sdnet_path(relative_path)
        row: dict[str, Any] = {
            "relative_path": relative_path,
            "label": label,
            "class_name": CLASS_NAMES[label],
            "surface": surface,
            "source_group": "",
            "source_group_verified": False,
            "group_resolution_method": "unresolved",
            "sha256": "",
            "perceptual_hash": "",
            "width": pd.NA,
            "height": pd.NA,
            "source_width": pd.NA,
            "source_height": pd.NA,
            "source_mode": "",
            "image_format": "",
            "file_size_bytes": path.stat().st_size,
            "exif_orientation": pd.NA,
            "audit_status": "invalid",
            "audit_error": "",
            "split": "",
        }
        try:
            inspected = inspect_image(path, max_bytes=max_bytes, max_pixels=max_pixels)
        except ImageValidationError as exc:
            row["audit_error"] = f"{type(exc).__name__}: {exc}"
            if raise_on_image_error:
                raise
        except OSError as exc:
            row["audit_error"] = f"OSError: {exc}"
            if raise_on_image_error:
                raise
        else:
            row.update(
                {
                    "sha256": inspected.sha256,
                    "perceptual_hash": inspected.perceptual_hash,
                    "width": inspected.width,
                    "height": inspected.height,
                    "source_width": inspected.source_width,
                    "source_height": inspected.source_height,
                    "source_mode": inspected.source_mode,
                    "image_format": inspected.image_format,
                    "file_size_bytes": inspected.file_size_bytes,
                    "exif_orientation": (
                        inspected.exif_orientation
                        if inspected.exif_orientation is not None
                        else pd.NA
                    ),
                    "audit_status": "ok",
                }
            )
        rows.append(row)

    paths = [str(row["relative_path"]) for row in rows]
    sha_by_path = {str(row["relative_path"]): str(row["sha256"]) for row in rows}
    groups = _resolve_groups(
        paths,
        sha_by_path,
        group_map=group_map,
        group_regex=group_regex,
        group_rule_verified=group_rule_verified,
    )
    for row in rows:
        group, verified, method = groups[str(row["relative_path"])]
        row["source_group"] = group
        row["source_group_verified"] = verified
        row["group_resolution_method"] = method

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    integer_columns = [
        "label",
        "width",
        "height",
        "source_width",
        "source_height",
        "file_size_bytes",
        "exif_orientation",
    ]
    for column in integer_columns:
        frame[column] = frame[column].astype("Int64")
    frame["source_group_verified"] = frame["source_group_verified"].astype(bool)
    frame.attrs["dataset_root"] = str(root.resolve())
    return frame


def write_group_map_template(frame: pd.DataFrame, output_path: str | PathLike[str]) -> Path:
    """Write a review template without pretending an inferred group is proven."""

    required = {"relative_path", "sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template = frame.loc[:, ["relative_path", "sha256"]].copy()
    template["source_group"] = ""
    template["verified"] = False
    template["review_notes"] = ""
    template.to_csv(output, index=False, lineterminator="\n")
    return output
