#!/usr/bin/env python3
"""Create and verify one immutable official group-aware split bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from crackspot.data import (
        OFFICIAL_SPLIT_SEED,
        SplitValidationError,
        create_locked_split_bundle,
    )
except ModuleNotFoundError:  # Support direct use before editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.data import (
        OFFICIAL_SPLIT_SEED,
        SplitValidationError,
        create_locked_split_bundle,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an atomic, leakage-safe 70/15/15 SDNET2018 split bundle "
            "from the immutable pre-split curation manifest."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help=(
            "Immutable pre_split_manifest.csv produced by scripts/curate_manifest.py; "
            "raw audit manifests and implicit row dropping are rejected"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests/split_v1"),
        help="New create-once directory for the locked split and provenance bundle",
    )
    parser.add_argument(
        "--conflict-report",
        type=Path,
        help=(
            "Optional explicit conflict_report.json to bind; by default it must be "
            "beside pre_split_manifest.csv"
        ),
    )
    parser.add_argument("--seed", type=int, default=OFFICIAL_SPLIT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--restarts", type=int, default=64)
    return parser


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    args = build_parser().parse_args(argv)
    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    try:
        bundle = create_locked_split_bundle(
            args.manifest,
            args.output_dir,
            conflict_report_path=args.conflict_report,
            seed=args.seed,
            ratios=ratios,
            restarts=args.restarts,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        SplitValidationError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Locked split bundle: {bundle.directory}")
    print(f"Completion status: {bundle.completion['status']}")
    print(f"Canonical manifest SHA-256: {bundle.manifest_sha256}")
    for name in ("train", "validation", "test"):
        details = bundle.audit["counts"][name]
        print(
            f"  {name}: {details['images']} images "
            f"({details['fraction']:.3%}), {details['source_groups']} source groups"
        )
    print("Leakage and official-balance audit PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
