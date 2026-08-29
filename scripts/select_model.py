#!/usr/bin/env python3
"""Select E2 or E3 strictly from validation ``best_val_loss`` evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from crackspot.reporting.aggregate import ArtifactValidationError, select_e2_e3
except ModuleNotFoundError:  # Support direct use before an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.reporting.aggregate import ArtifactValidationError, select_e2_e3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Chọn E2/E3 chỉ bằng best_val_loss validation và khóa model_selection.json bất biến."
        )
    )
    parser.add_argument("--e2-run", type=Path, required=True, help="Thư mục artifact run E2")
    parser.add_argument("--e3-run", type=Path, required=True, help="Thư mục artifact run E3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model_selection.json"),
        help="Tệp quyết định mới; không ghi đè",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Gốc dự án dùng sinh winner_config portable",
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
        result = select_e2_e3(
            e2_run=args.e2_run,
            e3_run=args.e3_run,
            output=args.output,
            project_root=args.project_root,
        )
    except (
        ArtifactValidationError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
