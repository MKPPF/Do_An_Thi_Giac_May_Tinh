#!/usr/bin/env python3
"""Generate a deterministic TP/TN/FP/FN Grad-CAM evidence grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from crackspot.inference import ModelMetadata
    from crackspot.modeling.selection import verify_selection_contract
    from crackspot.reporting.aggregate import is_official_report_path
    from crackspot.reporting.qualitative import (
        QualitativeEvidenceError,
        generate_gradcam_outcome_grid,
    )
    from crackspot.utils.hashing import sha256_file
except ModuleNotFoundError:  # Support direct use before editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.inference import ModelMetadata
    from crackspot.modeling.selection import verify_selection_contract
    from crackspot.reporting.aggregate import is_official_report_path
    from crackspot.reporting.qualitative import (
        QualitativeEvidenceError,
        generate_gradcam_outcome_grid,
    )
    from crackspot.utils.hashing import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sinh grid Grad-CAM TP/TN/FP/FN từ final predictions đã khóa."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Mặc định dùng model.metadata.json cạnh checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Gắn NOT_VALID_FOR_REPORT; dùng cho synthetic/tiny verification",
    )
    return parser


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _metadata_is_smoke(metadata: ModelMetadata) -> bool:
    status = str(metadata.extra.get("status", "")).upper()
    return (
        bool(metadata.extra.get("smoke_test", False))
        or metadata.extra.get("valid_for_report") is False
        or "NOT_VALID_FOR_REPORT" in status
    )


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


def _verify_official_predictions(
    predictions_path: Path,
    contract: object,
    *,
    selection_path: Path,
    metadata_path: Path,
) -> None:
    predictions = predictions_path.resolve()
    if predictions.name.casefold() != "predictions_test.csv":
        raise QualitativeEvidenceError(
            "Artifact report TP/TN/FP/FN phải dùng predictions_test.csv chính thức"
        )
    completion = _read_json_object(predictions.parent / "evaluation_complete.json")
    metadata = _read_json_object(predictions.parent / "evaluation_metadata.json")
    selection = _read_json_object(predictions.parent / "selection_contract_snapshot.json")
    if completion.get("status") != "FINAL_TEST_COMPLETE":
        raise QualitativeEvidenceError("Final evaluation chưa hoàn tất hợp lệ")
    artifact_hashes = completion.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise QualitativeEvidenceError("evaluation_complete.json thiếu artifact_sha256")
    for filename in (
        "predictions_test.csv",
        "evaluation_metadata.json",
        "selection_contract_snapshot.json",
    ):
        artifact = predictions.parent / filename
        expected_hash = str(artifact_hashes.get(filename, "")).lower()
        if sha256_file(artifact) != expected_hash:
            raise QualitativeEvidenceError(f"{filename} không khớp evaluation_complete.json")
    expected = {
        "experiment": contract.experiment,
        "run_id": contract.run_id,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "config_sha256": contract.config_sha256,
        "manifest_sha256": contract.manifest_sha256,
        "threshold": contract.threshold,
    }
    for source_name, source in (
        ("evaluation_metadata.json", metadata),
        ("selection_contract_snapshot.json", selection),
    ):
        for field, expected_value in expected.items():
            observed = source.get(field)
            if field == "threshold":
                try:
                    matches = float(observed) == float(expected_value)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = str(observed) == str(expected_value)
            if not matches:
                raise QualitativeEvidenceError(f"{source_name} không khớp selection: {field}")
    if (
        metadata.get("status") != "FINAL_TEST_COMPLETE"
        or metadata.get("valid_for_report") is not True
        or metadata.get("smoke_test") is not False
        or metadata.get("selection_selected_by") != "validation"
        or metadata.get("threshold_source") != "validation"
        or metadata.get("prediction_passes") != 1
        or selection.get("selected_by") != "validation"
    ):
        raise QualitativeEvidenceError("Final evaluation không đủ điều kiện report-valid")
    if metadata.get("selection_contract_sha256") != sha256_file(selection_path):
        raise QualitativeEvidenceError(
            "evaluation_metadata.json không khớp selection_complete.json"
        )
    if metadata.get("model_metadata_sha256") != sha256_file(metadata_path):
        raise QualitativeEvidenceError("evaluation_metadata.json không khớp model.metadata.json")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        contract = verify_selection_contract(args.selection)
        checkpoint = Path(contract.checkpoint).resolve()
        metadata_path = (args.metadata or checkpoint.with_name("model.metadata.json")).resolve()
        metadata = ModelMetadata.from_json(metadata_path)
        if metadata.model_sha256 != contract.checkpoint_sha256:
            raise QualitativeEvidenceError("Metadata/checkpoint không khớp selection contract")
        if metadata.manifest_sha256 != contract.manifest_sha256:
            raise QualitativeEvidenceError("Metadata/manifest không khớp selection contract")
        if metadata.extra.get("config_sha256") != contract.config_sha256:
            raise QualitativeEvidenceError("Metadata/config không khớp selection contract")
        smoke_metadata = _metadata_is_smoke(metadata)
        if smoke_metadata and not args.smoke:
            raise QualitativeEvidenceError("Checkpoint smoke bắt buộc dùng --smoke")
        if not args.smoke:
            _verify_official_predictions(
                args.predictions,
                contract,
                selection_path=args.selection.resolve(),
                metadata_path=metadata_path,
            )
        if args.smoke and is_official_report_path(args.output):
            raise QualitativeEvidenceError(
                "Artifact smoke/NOT_VALID_FOR_REPORT không được ghi vào artifacts/report"
            )
        payload = generate_gradcam_outcome_grid(
            predictions_path=args.predictions,
            dataset_root=args.dataset_root,
            model_path=checkpoint,
            metadata_path=metadata_path,
            output_path=args.output,
            threshold=contract.threshold,
            valid_for_report=not args.smoke and not smoke_metadata,
            experiment=contract.experiment if not args.smoke else None,
            run_id=contract.run_id if not args.smoke else None,
            config_sha256=contract.config_sha256 if not args.smoke else None,
            manifest_sha256=contract.manifest_sha256 if not args.smoke else None,
            selection_path=args.selection if not args.smoke else None,
            probability_tolerance=args.probability_tolerance,
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
    print(f"Grad-CAM grid: {payload['output']}")
    print(f"Status: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
