#!/usr/bin/env python3
"""Aggregate only measured final artifacts into report facts and tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from crackspot.reporting.aggregate import ArtifactValidationError, generate_report_assets
except ModuleNotFoundError:  # Support direct use before an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.reporting.aggregate import ArtifactValidationError, generate_report_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sinh comparison/report_facts từ final artifacts thật; từ chối smoke "
            "khi output nằm trong artifacts/report."
        )
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        action="append",
        required=True,
        help="Thư mục final evaluation; lặp flag để so sánh E1-E5",
    )
    parser.add_argument("--manifest", type=Path, help="Manifest đã khóa để sinh dataset summary")
    parser.add_argument(
        "--split-bundle-dir",
        type=Path,
        help="Thư mục official split chứa manifest_hashes.json và split_audit.json",
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        help="predictions_validation.csv của E4 để sinh E5 threshold curve",
    )
    parser.add_argument(
        "--threshold-result",
        type=Path,
        help="Threshold artifact validation có provenance/hash khóa với E4/E5",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        default=[],
        help="JSON benchmark; lặp flag nếu có nhiều môi trường/checkpoint",
    )
    parser.add_argument(
        "--training-run-dir",
        type=Path,
        action="append",
        default=[],
        help="Run E1-E4 hoàn tất; lặp flag đủ bốn run",
    )
    parser.add_argument(
        "--augmentation-sidecar",
        type=Path,
        help="JSON sidecar augmentation report-valid gắn với E4",
    )
    parser.add_argument(
        "--gradcam-sidecar",
        type=Path,
        help="JSON sidecar Grad-CAM report-valid gắn với E5",
    )
    parser.add_argument(
        "--real-evaluation-dir",
        type=Path,
        action="append",
        default=[],
        help="Artifact đánh giá ảnh tự chụp riêng; lặp flag nếu cần",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Gốc dự án dùng ghi path portable trong report_facts",
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
        result = generate_report_assets(
            evaluation_dirs=args.evaluation_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            split_bundle_dir=args.split_bundle_dir,
            validation_predictions=args.validation_predictions,
            threshold_result_path=args.threshold_result,
            benchmark_paths=args.benchmark,
            real_evaluation_dirs=args.real_evaluation_dir,
            training_run_dirs=args.training_run_dir,
            augmentation_sidecar=args.augmentation_sidecar,
            gradcam_sidecar=args.gradcam_sidecar,
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
    print(f"Report facts: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
