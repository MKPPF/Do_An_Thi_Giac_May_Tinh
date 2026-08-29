#!/usr/bin/env python3
"""Export inference metadata using the threshold from a locked selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from crackspot.modeling.selection import export_selected_metadata
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.modeling.selection import export_selected_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Xuất selected_model.metadata.json đã khóa threshold và provenance "
            "cho CLI/Streamlit/benchmark."
        )
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--metadata", type=Path, help="Mặc định: model.metadata.json cạnh checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, help="Mặc định: selected_model.metadata.json cạnh checkpoint"
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
        output = export_selected_metadata(
            args.selection,
            metadata_path=args.metadata,
            output=args.output,
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
    print(f"Selected metadata: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
