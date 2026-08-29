#!/usr/bin/env python3
"""Create immutable ``selection_complete.json`` from verified validation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:
    from crackspot.modeling.selection import lock_run_selection
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.modeling.selection import lock_run_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Khóa checkpoint/config/manifest/threshold trước final test"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment", help="Ví dụ E1, E2, E3 hoặc E5")
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--threshold", type=float, help="Ngưỡng protocol cố định")
    threshold_group.add_argument(
        "--threshold-result",
        type=Path,
        help="JSON tune threshold trên validation",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cho phép khóa run NOT_VALID_FOR_REPORT",
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
        contract = lock_run_selection(
            run_dir=args.run_dir,
            output=args.output,
            experiment=args.experiment,
            threshold=args.threshold,
            threshold_result=args.threshold_result,
            allow_smoke=args.smoke,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(contract), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
