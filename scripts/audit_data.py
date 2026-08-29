#!/usr/bin/env python3
"""Audit SDNET2018 files and emit a complete manifest plus evidence report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from crackspot.data import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_PIXELS,
    GroupResolutionError,
    build_manifest,
    manifest_sha256,
    write_group_map_template,
)


def _duplicate_summary(frame: pd.DataFrame, column: str, max_examples: int = 20) -> dict[str, Any]:
    usable = frame.loc[frame["audit_status"].eq("ok") & frame[column].astype(str).ne("")]
    groups = [
        {
            column: str(value),
            "count": len(rows),
            "relative_paths": rows["relative_path"].astype(str).tolist(),
        }
        for value, rows in usable.groupby(column, sort=True)
        if len(rows) > 1
    ]
    groups.sort(key=lambda item: (-item["count"], item[column]))
    return {
        "duplicate_groups": len(groups),
        "images_in_duplicate_groups": int(sum(item["count"] for item in groups)),
        "examples": groups[:max_examples],
        "examples_truncated": len(groups) > max_examples,
    }


def build_report(frame: pd.DataFrame, dataset_root: Path) -> dict[str, Any]:
    ok = frame["audit_status"].eq("ok")
    valid = frame.loc[ok]
    invalid = frame.loc[~ok]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "manifest_canonical_sha256": manifest_sha256(frame),
        "totals": {
            "discovered": len(frame),
            "valid": int(ok.sum()),
            "invalid": int((~ok).sum()),
            "source_groups": int(valid["source_group"].replace("", pd.NA).nunique()),
            "source_group_verified_rows": int(valid["source_group_verified"].sum()),
        },
        "class_distribution": {
            str(key): int(value) for key, value in valid.groupby("label").size().items()
        },
        "surface_distribution": {
            str(key): int(value) for key, value in valid.groupby("surface").size().items()
        },
        "surface_class_distribution": {
            f"{surface}|{label}": int(value)
            for (surface, label), value in valid.groupby(["surface", "label"]).size().items()
        },
        "dimensions": {
            f"{width}x{height}": int(value)
            for (width, height), value in valid.groupby(["width", "height"]).size().items()
        },
        "source_modes": {
            str(key): int(value) for key, value in valid.groupby("source_mode").size().items()
        },
        "formats": {
            str(key): int(value) for key, value in valid.groupby("image_format").size().items()
        },
        "exact_duplicates_sha256": _duplicate_summary(frame, "sha256"),
        "perceptual_hash_candidates": _duplicate_summary(frame, "perceptual_hash"),
        "invalid_files": invalid.loc[
            :, ["relative_path", "audit_error", "file_size_bytes"]
        ].to_dict(orient="records"),
        "notes": [
            "Equal dHash values are near-duplicate candidates for human review, not proof of common source.",
            "Perceptual hashing never substitutes for a verified source_group map/rule.",
        ],
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode and audit every SDNET2018 image.")
    parser.add_argument("dataset_root", type=Path, help="Directory containing D/P/W")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/audit_manifest.csv"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/manifests/data_audit.json"),
    )
    parser.add_argument("--group-map", type=Path)
    parser.add_argument("--group-regex")
    parser.add_argument(
        "--confirm-group-rule-verified",
        action="store_true",
        help="Assert that the supplied regex/map was reviewed against archive documentation",
    )
    parser.add_argument(
        "--group-template-out",
        type=Path,
        default=Path("data/manifests/group_map_template.csv"),
    )
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        frame = build_manifest(
            args.dataset_root,
            group_map=args.group_map,
            group_regex=args.group_regex,
            group_rule_verified=args.confirm_group_rule_verified,
            max_bytes=args.max_bytes,
            max_pixels=args.max_pixels,
        )
    except (FileNotFoundError, GroupResolutionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _atomic_csv(frame, args.manifest_out)
    report = build_report(frame, args.dataset_root)
    _atomic_json(report, args.report_out)
    print(f"Manifest: {args.manifest_out.resolve()}")
    print(f"Audit report: {args.report_out.resolve()}")
    print(f"Images: {report['totals']['valid']} valid, {report['totals']['invalid']} invalid")
    all_groups_verified = bool(frame["source_group_verified"].all())
    if not all_groups_verified:
        write_group_map_template(frame, args.group_template_out)
        print(
            "Source groups are unresolved/unverified. Template written to "
            f"{args.group_template_out.resolve()}. Split creation will fail until reviewed.",
            file=sys.stderr,
        )
    if report["totals"]["invalid"]:
        print(
            "Audit found invalid images; inspect data_audit.json before splitting.", file=sys.stderr
        )
        return 3
    return 0 if all_groups_verified else 4


if __name__ == "__main__":
    raise SystemExit(main())
