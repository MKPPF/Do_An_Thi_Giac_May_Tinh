#!/usr/bin/env python3
"""Evaluate labelled self-captured images separately from the SDNET test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from crackspot.reporting.real_images import RealImageEvaluationError, evaluate_real_images
except ModuleNotFoundError:  # Support direct use before editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.reporting.real_images import RealImageEvaluationError, evaluate_real_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Đánh giá ảnh tự chụp có nhãn, tách hoàn toàn khỏi test SDNET2018."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--confirm-self-captured",
        action="store_true",
        help="Xác nhận ảnh do nhóm tự chụp và manifest có capture_source=self_captured",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Chạy kiểm thử kỹ thuật và luôn gắn NOT_VALID_FOR_REPORT",
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
        result = evaluate_real_images(
            selection_path=args.selection,
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            metadata_path=args.metadata,
            confirm_self_captured=args.confirm_self_captured,
            smoke=args.smoke,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RealImageEvaluationError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Real-image evaluation: {result.status}")
    print(f"Artifacts: {result.output_dir}")
    print(f"Completion marker: {result.completion_path}")
    print(f"Sample count: {result.sample_count}")
    print(f"Accuracy: {result.accuracy:.6f}")
    print(f"F1 Crack: {result.f1_crack:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
