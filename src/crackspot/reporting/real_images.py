"""Separate evaluation for genuinely self-captured external images."""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from crackspot.inference import InferenceService, ModelMetadata
from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.modeling.selection import SelectionContract, verify_selection_contract
from crackspot.reporting.export import write_json
from crackspot.reporting.plots import plot_confusion_matrix
from crackspot.utils.environment import capture_environment
from crackspot.utils.hashing import sha256_file


class RealImageEvaluationError(RuntimeError):
    """Raised when external-image evidence violates separation/provenance rules."""


@dataclass(frozen=True)
class RealImageEvaluationResult:
    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    completion_path: Path
    sample_count: int
    accuracy: float
    f1_crack: float
    status: str


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Manifest ảnh thực tế phải là CSV hoặc Parquet")


def validate_real_manifest(
    frame: pd.DataFrame,
    *,
    require_self_captured_declaration: bool,
) -> pd.DataFrame:
    """Validate that the manifest is labelled and separate from SDNET splits."""

    required = {"relative_path", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise RealImageEvaluationError(f"Manifest ảnh thực tế thiếu cột: {sorted(missing)}")
    if frame.empty:
        raise RealImageEvaluationError("Manifest ảnh thực tế rỗng")
    normalized = frame.copy()
    if normalized["relative_path"].isna().any():
        raise RealImageEvaluationError("relative_path ảnh thực tế không được rỗng")
    normalized["relative_path"] = normalized["relative_path"].map(_normalize_relative_path)
    duplicate_key = normalized["relative_path"].str.casefold()
    if duplicate_key.duplicated().any():
        duplicates = sorted(
            normalized.loc[duplicate_key.duplicated(False), "relative_path"].unique()
        )
        raise RealImageEvaluationError(f"Manifest ảnh thực tế có relative_path trùng: {duplicates}")
    if normalized["label"].map(lambda value: isinstance(value, bool | np.bool_)).any():
        raise RealImageEvaluationError("label bắt buộc Non-crack=0, Crack=1; không nhận boolean")
    try:
        numeric_labels = pd.to_numeric(normalized["label"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise RealImageEvaluationError("label bắt buộc Non-crack=0, Crack=1") from exc
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, [0.0, 1.0]).all():
        raise RealImageEvaluationError("label bắt buộc Non-crack=0, Crack=1")
    normalized["label"] = numeric_labels.astype(np.int64)
    if "split" in normalized.columns:
        splits = set(normalized["split"].dropna().astype(str).str.strip().str.casefold())
        forbidden = splits.intersection({"train", "val", "validation", "test"})
        if forbidden:
            raise RealImageEvaluationError(
                f"Ảnh thực tế không được gộp vào split chuẩn: {sorted(forbidden)}"
            )
    normalized["split"] = "real_external"
    if require_self_captured_declaration:
        if "capture_source" not in normalized.columns:
            raise RealImageEvaluationError(
                "Artifact report yêu cầu cột capture_source=self_captured"
            )
        sources = normalized["capture_source"].astype(str).str.strip().str.casefold()
        if not sources.eq("self_captured").all():
            raise RealImageEvaluationError(
                "Mọi ảnh báo cáo phải khai báo capture_source=self_captured"
            )
        normalized["capture_source"] = "self_captured"
    return normalized


def _normalize_relative_path(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
    ):
        raise RealImageEvaluationError(f"relative_path không an toàn: {value!r}")
    normalized = PurePosixPath(*posix.parts).as_posix()
    if normalized in {"", "."}:
        raise RealImageEvaluationError(f"relative_path không an toàn: {value!r}")
    return normalized


def _resolve_path(root: Path, value: object) -> Path:
    text = _normalize_relative_path(value)
    posix = PurePosixPath(text)
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise RealImageEvaluationError(f"Path thoát khỏi external root: {value!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _metadata_is_smoke(metadata: ModelMetadata) -> bool:
    status = str(metadata.extra.get("status", "")).upper()
    return (
        bool(metadata.extra.get("smoke_test", False))
        or metadata.extra.get("valid_for_report") is False
        or "NOT_VALID_FOR_REPORT" in status
    )


def _verify_model_provenance(
    contract: SelectionContract,
    metadata: ModelMetadata,
) -> None:
    if metadata.model_sha256 != contract.checkpoint_sha256:
        raise RealImageEvaluationError("Metadata/checkpoint không khớp selection contract")
    if metadata.manifest_sha256 != contract.manifest_sha256:
        raise RealImageEvaluationError("Metadata/SDNET manifest không khớp selection contract")
    if str(metadata.extra.get("config_sha256", "")).lower() != contract.config_sha256.lower():
        raise RealImageEvaluationError("Metadata/config không khớp selection contract")
    if metadata.run_id != contract.run_id:
        raise RealImageEvaluationError("Metadata/run_id không khớp selection contract")


def _validate_contract(contract: SelectionContract) -> None:
    if not contract.experiment.strip() or not contract.run_id.strip():
        raise RealImageEvaluationError("Selection contract thiếu experiment/run_id")
    if not Path(contract.checkpoint).is_absolute():
        raise RealImageEvaluationError("Checkpoint trong selection contract phải là path tuyệt đối")
    if (
        isinstance(contract.threshold, bool)
        or not isinstance(contract.threshold, int | float)
        or not math.isfinite(float(contract.threshold))
        or not 0.0 <= float(contract.threshold) <= 1.0
    ):
        raise RealImageEvaluationError("Threshold đã khóa phải hữu hạn và nằm trong [0,1]")


def predict_real_frame(
    frame: pd.DataFrame,
    *,
    dataset_root: str | Path,
    service: Any,
    threshold: float,
) -> pd.DataFrame:
    """Predict a validated real-image manifest without touching standard test data."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise RealImageEvaluationError("threshold phải hữu hạn và nằm trong [0,1]")
    threshold_value = float(threshold)
    rows: list[dict[str, Any]] = []
    for record in frame.sort_values("relative_path", kind="stable").to_dict(orient="records"):
        image_path = _resolve_path(root, record["relative_path"])
        image_hash_before = sha256_file(image_path)
        result = service.predict_image(image_path, include_gradcam=False)
        image_hash_after = sha256_file(image_path)
        if image_hash_after != image_hash_before:
            raise RealImageEvaluationError(
                f"Ảnh thay đổi trong lúc suy luận: {record['relative_path']}"
            )
        try:
            probability = float(result.crack_probability)
            latency_ms = float(result.latency_ms)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RealImageEvaluationError(
                f"Service trả kết quả không hợp lệ cho {record['relative_path']}"
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RealImageEvaluationError(
                f"P(Crack) không hợp lệ cho {record['relative_path']}: {probability!r}"
            )
        if not math.isfinite(latency_ms) or latency_ms < 0.0:
            raise RealImageEvaluationError(
                f"Latency không hợp lệ cho {record['relative_path']}: {latency_ms!r}"
            )
        predicted = int(probability >= threshold_value)
        truth = int(record["label"])
        if truth == predicted == 1:
            outcome = "TP"
        elif truth == predicted == 0:
            outcome = "TN"
        elif truth == 0:
            outcome = "FP"
        else:
            outcome = "FN"
        row = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "image_sha256",
                "label",
                "latency_ms",
                "outcome",
                "probability_crack",
                "threshold",
                "y_pred",
                "y_true",
            }
        }
        row.update(
            {
                "y_true": truth,
                "image_sha256": image_hash_before,
                "probability_crack": probability,
                "threshold": threshold_value,
                "y_pred": predicted,
                "outcome": outcome,
                "latency_ms": latency_ms,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _latency_summary(values: pd.Series) -> dict[str, float]:
    timings = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if timings.size == 0 or not np.isfinite(timings).all() or np.any(timings < 0):
        raise RealImageEvaluationError("Latency ảnh thực tế không hợp lệ")
    return {
        "mean": float(np.mean(timings)),
        "median": float(np.median(timings)),
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
        "minimum": float(np.min(timings)),
        "maximum": float(np.max(timings)),
    }


def _write_artifacts(
    staging: Path,
    *,
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    selection_snapshot: dict[str, Any],
) -> None:
    manifest.to_csv(staging / "manifest_real_snapshot.csv", index=False, lineterminator="\n")
    predictions.to_csv(staging / "predictions_real.csv", index=False, lineterminator="\n")
    predictions.loc[predictions["outcome"].isin({"FP", "FN"})].to_csv(
        staging / "errors_real.csv", index=False, lineterminator="\n"
    )
    write_json(staging / "metrics_real.json", metrics, overwrite=False)
    write_json(
        staging / "classification_report_real.json",
        metrics["classification_report"],
        overwrite=False,
    )
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=["true_non_crack", "true_crack"],
        columns=["pred_non_crack", "pred_crack"],
    ).to_csv(staging / "confusion_matrix_real.csv", lineterminator="\n")
    pd.DataFrame(
        metrics["confusion_matrix_normalized"],
        index=["true_non_crack", "true_crack"],
        columns=["pred_non_crack", "pred_crack"],
    ).to_csv(staging / "confusion_matrix_real_normalized.csv", lineterminator="\n")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        staging / "confusion_matrix_real.png",
        title="Ma trận nhầm lẫn - ảnh tự chụp (tách test chuẩn)",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix_normalized"],
        staging / "confusion_matrix_real_normalized.png",
        normalized=True,
        title="Ma trận chuẩn hóa - ảnh tự chụp (tách test chuẩn)",
    )
    write_json(staging / "evaluation_metadata.json", metadata, overwrite=False)
    write_json(staging / "environment.json", capture_environment(), overwrite=False)
    write_json(
        staging / "selection_contract_snapshot.json",
        selection_snapshot,
        overwrite=False,
    )
    artifact_names = (
        "manifest_real_snapshot.csv",
        "predictions_real.csv",
        "errors_real.csv",
        "metrics_real.json",
        "classification_report_real.json",
        "confusion_matrix_real.csv",
        "confusion_matrix_real_normalized.csv",
        "confusion_matrix_real.png",
        "confusion_matrix_real_normalized.png",
        "evaluation_metadata.json",
        "environment.json",
        "selection_contract_snapshot.json",
    )
    write_json(
        staging / "evaluation_complete.json",
        {
            "schema_version": 1,
            "status": metadata["status"],
            "valid_for_report": metadata["valid_for_report"],
            "immutable": True,
            "sample_count": len(predictions),
            "artifact_sha256": {name: sha256_file(staging / name) for name in artifact_names},
        },
        overwrite=False,
    )


def _assert_input_unchanged(path: Path, expected_sha256: str, label: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise RealImageEvaluationError(f"{label} đã thay đổi trong lúc đánh giá")


def _assert_images_unchanged(predictions: pd.DataFrame, dataset_root: Path) -> None:
    for record in predictions[["relative_path", "image_sha256"]].to_dict(orient="records"):
        image_path = _resolve_path(dataset_root, record["relative_path"])
        _assert_input_unchanged(
            image_path,
            str(record["image_sha256"]),
            f"Ảnh {record['relative_path']}",
        )


def evaluate_real_images(
    *,
    selection_path: str | Path,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    metadata_path: str | Path | None = None,
    confirm_self_captured: bool = False,
    smoke: bool = False,
    service: Any | None = None,
) -> RealImageEvaluationResult:
    """Evaluate external photos separately using the validation-locked threshold."""

    selection_file = Path(selection_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Từ chối ghi đè real-image evaluation: {target}")
    selection_hash_before = sha256_file(selection_file)
    contract = verify_selection_contract(selection_file)
    _assert_input_unchanged(selection_file, selection_hash_before, "Selection contract")
    _validate_contract(contract)
    checkpoint = Path(contract.checkpoint).resolve()
    resolved_metadata = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else checkpoint.with_name("model.metadata.json")
    )
    metadata_hash_before = sha256_file(resolved_metadata)
    model_metadata = ModelMetadata.from_json(resolved_metadata)
    _assert_input_unchanged(resolved_metadata, metadata_hash_before, "Model metadata")
    _verify_model_provenance(contract, model_metadata)
    metadata_smoke = _metadata_is_smoke(model_metadata)
    if metadata_smoke and not smoke:
        raise RealImageEvaluationError("Checkpoint smoke bắt buộc dùng --smoke")
    valid_for_report = bool(confirm_self_captured and not smoke and not metadata_smoke)
    if not smoke and not confirm_self_captured:
        raise RealImageEvaluationError(
            "Cần --confirm-self-captured; không được gọi ảnh giả/tải mạng là ảnh tự chụp"
        )

    manifest_file_hash = sha256_file(manifest_file)
    manifest = validate_real_manifest(
        _read_manifest(manifest_file),
        require_self_captured_declaration=bool(confirm_self_captured),
    )
    _assert_input_unchanged(manifest_file, manifest_file_hash, "Manifest ảnh thực tế")
    inference = (
        service
        if service is not None
        else InferenceService.from_files(
            checkpoint,
            resolved_metadata,
            verify_hash=True,
        )
    )
    inference_metadata = getattr(inference, "metadata", None)
    if service is not None:
        if inference_metadata is None:
            raise RealImageEvaluationError(
                "Service được tiêm vào phải có metadata để xác minh provenance"
            )
        _verify_model_provenance(contract, inference_metadata)
    predictions = predict_real_frame(
        manifest,
        dataset_root=dataset_root,
        service=inference,
        threshold=contract.threshold,
    )
    metrics = compute_binary_metrics(
        predictions["y_true"],
        predictions["probability_crack"],
        contract.threshold,
    )
    status = "REAL_IMAGE_EVALUATION_COMPLETE" if valid_for_report else "NOT_VALID_FOR_REPORT"
    evaluation_scope = (
        "external_self_captured_images"
        if confirm_self_captured
        else "external_images_smoke_unverified_source"
    )
    metrics.update(
        {
            "status": status,
            "valid_for_report": valid_for_report,
            "evaluation_scope": evaluation_scope,
            "included_in_standard_test": False,
            "self_captured_confirmed": bool(confirm_self_captured),
            "manifest_file_sha256": manifest_file_hash,
            "selection_sha256": selection_hash_before,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "config_sha256": contract.config_sha256,
            "sdnet_manifest_sha256": contract.manifest_sha256,
            "experiment": contract.experiment,
            "run_id": contract.run_id,
            "latency_ms": _latency_summary(predictions["latency_ms"]),
        }
    )
    manifest_snapshot = manifest.sort_values("relative_path", kind="stable").reset_index(drop=True)
    image_hashes = predictions.set_index("relative_path")["image_sha256"]
    manifest_snapshot["image_sha256"] = manifest_snapshot["relative_path"].map(image_hashes)
    evaluation_metadata = {
        "schema_version": 1,
        "kind": "external_real_image_evaluation",
        "status": status,
        "valid_for_report": valid_for_report,
        "sample_count": len(predictions),
        "selection": str(selection_file),
        "selection_sha256": selection_hash_before,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.checkpoint_sha256,
        "config_sha256": contract.config_sha256,
        "experiment": contract.experiment,
        "run_id": contract.run_id,
        "model_metadata": str(resolved_metadata),
        "model_metadata_sha256": metadata_hash_before,
        "manifest": str(manifest_file),
        "manifest_file_sha256": manifest_file_hash,
        "sdnet_manifest_sha256": contract.manifest_sha256,
        "dataset_root": str(Path(dataset_root).resolve()),
        "threshold": contract.threshold,
        "threshold_selected_by": "validation",
        "evaluation_scope": evaluation_scope,
        "included_in_standard_test": False,
        "self_captured_confirmed": bool(confirm_self_captured),
        "immutable": True,
    }

    _assert_input_unchanged(selection_file, selection_hash_before, "Selection contract")
    _assert_input_unchanged(checkpoint, contract.checkpoint_sha256, "Checkpoint")
    _assert_input_unchanged(resolved_metadata, metadata_hash_before, "Model metadata")
    _assert_input_unchanged(manifest_file, manifest_file_hash, "Manifest ảnh thực tế")
    _assert_images_unchanged(predictions, Path(dataset_root).resolve())

    target.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        _write_artifacts(
            staging_path,
            manifest=manifest_snapshot,
            predictions=predictions,
            metrics=metrics,
            metadata=evaluation_metadata,
            selection_snapshot=asdict(contract),
        )
        _assert_input_unchanged(selection_file, selection_hash_before, "Selection contract")
        _assert_input_unchanged(checkpoint, contract.checkpoint_sha256, "Checkpoint")
        _assert_input_unchanged(resolved_metadata, metadata_hash_before, "Model metadata")
        _assert_input_unchanged(manifest_file, manifest_file_hash, "Manifest ảnh thực tế")
        _assert_images_unchanged(predictions, Path(dataset_root).resolve())
        result = RealImageEvaluationResult(
            output_dir=target,
            metrics_path=target / "metrics_real.json",
            predictions_path=target / "predictions_real.csv",
            completion_path=target / "evaluation_complete.json",
            sample_count=len(predictions),
            accuracy=float(metrics["accuracy"]),
            f1_crack=float(metrics["crack"]["f1"]),
            status=status,
        )
        if target.exists():
            raise FileExistsError(f"Từ chối ghi đè real-image evaluation: {target}")
        staging_path.rename(target)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    return result


__all__ = [
    "RealImageEvaluationError",
    "RealImageEvaluationResult",
    "evaluate_real_images",
    "predict_real_frame",
    "validate_real_manifest",
]
