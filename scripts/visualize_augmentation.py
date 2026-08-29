#!/usr/bin/env python3
"""Render before/after samples from CrackSpot's exact train augmentation path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from crackspot.config import load_config
    from crackspot.data import load_manifest_table, manifest_sha256
    from crackspot.reporting.aggregate import is_official_report_path
    from crackspot.reporting.qualitative import (
        QualitativeEvidenceError,
        generate_augmentation_audit_grid,
    )
    from crackspot.utils.hashing import sha256_file
except ModuleNotFoundError:  # Support direct use before editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.config import load_config
    from crackspot.data import load_manifest_table, manifest_sha256
    from crackspot.reporting.aggregate import is_official_report_path
    from crackspot.reporting.qualitative import (
        QualitativeEvidenceError,
        generate_augmentation_audit_grid,
    )
    from crackspot.utils.hashing import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit ảnh trước/sau bằng đúng tf.data augmentation chỉ-train."
    )
    parser.add_argument("--config", type=Path, required=True, help="Config E4 hoặc snapshot YAML")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run E4 hoàn tất; bắt buộc cho artifact report-valid",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cho phép manifest tiny và gắn NOT_VALID_FOR_REPORT",
    )
    return parser


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualitativeEvidenceError(f"JSON không hợp lệ: {path}") from exc
    if not isinstance(payload, dict):
        raise QualitativeEvidenceError(f"JSON phải là object: {path}")
    return payload


def _official_run_provenance(
    run_dir: Path,
    *,
    config_path: Path,
    config_sha256: str,
    manifest_path: Path,
) -> tuple[str, str]:
    directory = run_dir.resolve()
    expected_config = directory / "config_snapshot.json"
    if config_path.resolve() != expected_config:
        raise QualitativeEvidenceError(
            "Artifact augmentation chính thức phải dùng config_snapshot E4"
        )
    summary = _read_json_object(directory / "run_summary.json")
    completion = _read_json_object(directory / "training_complete.json")
    if str(summary.get("experiment", "")).strip().upper() != "E4":
        raise QualitativeEvidenceError("run_summary phải thuộc E4")
    if summary.get("valid_for_report") is not True or str(summary.get("status", "")) != (
        "VALIDATION_COMPLETE_TEST_LOCKED"
    ):
        raise QualitativeEvidenceError("Run E4 chưa hoàn tất report-valid")
    run_id = str(summary.get("run_id", "")).strip()
    if not run_id or str(completion.get("run_id", "")).strip() != run_id:
        raise QualitativeEvidenceError("run_id E4 không nhất quán")
    manifest_hash = manifest_sha256(load_manifest_table(manifest_path))
    for field, expected in (
        ("config_sha256", config_sha256),
        ("manifest_sha256", manifest_hash),
    ):
        if str(summary.get(field, "")).strip().lower() != expected:
            raise QualitativeEvidenceError(f"run_summary không khớp {field}")
        if str(completion.get(field, "")).strip().lower() != expected:
            raise QualitativeEvidenceError(f"training_complete không khớp {field}")
    artifact_hashes = completion.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise QualitativeEvidenceError("training_complete thiếu artifact_sha256")
    for filename in ("config_snapshot.json", "run_summary.json", "history.csv", "model.keras"):
        artifact = directory / filename
        if str(artifact_hashes.get(filename, "")).strip().lower() != sha256_file(artifact):
            raise QualitativeEvidenceError(f"Run E4 artifact hash không khớp: {filename}")
    return run_id, manifest_hash


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        loaded = load_config(args.config)
        pipeline = loaded.values.get("pipeline", {})
        if not isinstance(pipeline, dict):
            raise QualitativeEvidenceError("config.pipeline phải là mapping")
        augmentation = pipeline.get("augmentation", {})
        if not isinstance(augmentation, dict):
            raise QualitativeEvidenceError("config.pipeline.augmentation phải là mapping")
        if augmentation.get("enabled") is not True:
            raise QualitativeEvidenceError("Config phải bật augmentation (E4)")
        if augmentation.get("train_only") is not True:
            raise QualitativeEvidenceError("Config augmentation phải chỉ áp dụng cho train")
        if float(augmentation.get("horizontal_flip_probability", 0.5)) != 0.5:
            raise QualitativeEvidenceError(
                "Config phải khớp pipeline: horizontal flip probability=0.5"
            )
        experiment_id = str(loaded.values.get("experiment", {}).get("id", "")).upper()
        if not args.smoke and experiment_id != "E4":
            raise QualitativeEvidenceError("Artifact augmentation report-valid phải dùng config E4")
        if args.smoke and is_official_report_path(args.output):
            raise QualitativeEvidenceError(
                "Artifact smoke/NOT_VALID_FOR_REPORT không được ghi vào artifacts/report"
            )
        run_id: str | None = None
        if not args.smoke:
            if args.run_dir is None:
                raise QualitativeEvidenceError("Artifact report-valid bắt buộc có --run-dir E4")
            run_id, _ = _official_run_provenance(
                args.run_dir,
                config_path=args.config,
                config_sha256=loaded.sha256,
                manifest_path=args.manifest,
            )
        payload = generate_augmentation_audit_grid(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_path=args.output,
            sample_count=args.sample_count,
            seed=loaded.seed,
            image_size=tuple(pipeline.get("image_size", (224, 224))),
            rotation_degrees=float(augmentation.get("max_rotation_degrees", 15.0)),
            brightness_delta=float(augmentation.get("brightness_delta", 0.15)),
            contrast_delta=float(augmentation.get("contrast_delta", 0.15)),
            valid_for_report=not args.smoke,
            experiment=experiment_id if not args.smoke else None,
            run_id=run_id,
            config_path=args.config if not args.smoke else None,
            config_sha256=loaded.sha256 if not args.smoke else None,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        QualitativeEvidenceError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Augmentation audit: {payload['output']}")
    print(f"Status: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
