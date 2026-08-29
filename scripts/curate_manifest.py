#!/usr/bin/env python3
"""Create immutable evidence for contradictory exact-duplicate exclusions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from crackspot.data import ManifestCurationError, create_curation_bundle
except ModuleNotFoundError:  # Support direct use before editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.data import ManifestCurationError, create_curation_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Before splitting, exclude every byte-identical SHA-256 group whose audited "
            "labels conflict; preserve all other rows and emit immutable provenance."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Original audited pre-split CSV manifest (kept unchanged)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests/pre_split_curation_v1"),
        help=(
            "New immutable directory for pre_split_manifest.csv, conflict_rows.csv, "
            "and conflict_report.json"
        ),
    )
    return parser


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        bundle = create_curation_bundle(args.manifest, args.output_dir)
    except (
        FileExistsError,
        FileNotFoundError,
        ManifestCurationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Immutable curation artifacts: {bundle.output_dir}")
    print(
        f"Rows: {bundle.parent_rows} parent, {bundle.excluded_rows} excluded, "
        f"{bundle.cleaned_rows} retained"
    )
    print(f"Contradictory exact-hash groups: {bundle.conflicting_hash_groups}")
    print(f"Cleaned pre-split manifest: {bundle.cleaned_manifest_path}")
    print(f"Conflict evidence: {bundle.conflict_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
