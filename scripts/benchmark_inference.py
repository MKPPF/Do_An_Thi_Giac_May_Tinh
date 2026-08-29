#!/usr/bin/env python3
"""Benchmark decode, preprocessing, and inference with raw timing evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from crackspot.inference import InferenceError, InferenceService
    from crackspot.reporting.aggregate import is_official_report_path
    from crackspot.reporting.benchmark import BenchmarkError, benchmark_inference_service
    from crackspot.reporting.export import write_json
except ModuleNotFoundError:  # Support direct use before an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.inference import InferenceError, InferenceService
    from crackspot.reporting.aggregate import is_official_report_path
    from crackspot.reporting.benchmark import BenchmarkError, benchmark_inference_service
    from crackspot.reporting.export import write_json


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    default_model = os.environ.get(
        "CRACKSPOT_MODEL_PATH", str(project_root / "models" / "crackspot.keras")
    )
    parser = argparse.ArgumentParser(
        description="Đo latency một ảnh, tách warm-up và xuất raw/mean/median/p50/p95."
    )
    parser.add_argument("image", type=Path, help="Ảnh JPEG/PNG dùng lặp benchmark")
    parser.add_argument("--model", type=Path, default=Path(default_model))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(os.environ["CRACKSPOT_METADATA_PATH"])
        if os.environ.get("CRACKSPOT_METADATA_PATH")
        else None,
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON mới; không ghi đè")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int, default=100)
    parser.add_argument("--target-seconds", type=float, default=5.0)
    parser.add_argument(
        "--include-gradcam",
        action="store_true",
        help="Đo decode + preprocess + inference + Grad-CAM thay vì classification thuần",
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
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        service = InferenceService.from_files(args.model, args.metadata, verify_hash=True)
        result = benchmark_inference_service(
            service,
            args.image,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            target_seconds=args.target_seconds,
            include_gradcam=args.include_gradcam,
        )
        if is_official_report_path(args.output) and not result["valid_for_report"]:
            raise BenchmarkError(
                "Benchmark smoke/NOT_VALID_FOR_REPORT không được ghi vào artifacts/report"
            )
        write_json(args.output, result, overwrite=False)
    except (
        BenchmarkError,
        FileExistsError,
        FileNotFoundError,
        InferenceError,
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
