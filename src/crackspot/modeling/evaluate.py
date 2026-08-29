"""One-shot final-test evaluation guarded by an immutable selection contract.

The public test rows are intentionally not selected from the manifest until the
validation selection contract, checkpoint hash, and canonical manifest hash all
match.  An exclusive marker keyed by checkpoint SHA-256 is then created before
test access, making a second final evaluation of the same checkpoint fail even
when a different output directory is requested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crackspot import __version__
from crackspot.constants import ARTIFACTS_DIR, IMAGE_SIZE, LABEL_MAPPING, PREPROCESSING_NAME
from crackspot.data import (
    DatasetIntegrityError,
    audit_split,
    build_tf_dataset,
    load_manifest_table,
    manifest_sha256,
    verify_official_dataset_preconditions,
)
from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.modeling.selection import SelectionContract, verify_selection_contract
from crackspot.reporting.export import write_json
from crackspot.reporting.plots import plot_confusion_matrix
from crackspot.utils.environment import capture_environment
from crackspot.utils.hashing import sha256_file

FINAL_EVALUATION_REGISTRY = ARTIFACTS_DIR / "final_evaluation_registry"
REPORT_EVALUATION_ROOT = ARTIFACTS_DIR / "report" / "final_evaluation"
SMOKE_EVALUATION_ROOT = ARTIFACTS_DIR / "smoke" / "final_evaluation"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class FinalEvaluationProtocolError(RuntimeError):
    """Raised before or during an evaluation that violates the test protocol."""


class FinalEvaluationAlreadyRunError(FinalEvaluationProtocolError):
    """Raised when a checkpoint hash already has a final-test access marker."""


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Locations and headline metrics from one completed evaluation."""

    output_dir: Path
    marker_path: Path
    metrics_path: Path
    predictions_path: Path
    accuracy: float
    f1_crack: float
    status: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_manifest(path: Path) -> pd.DataFrame:
    try:
        # Keep this identical to the training runner: the selection contract's
        # canonical hash must derive from the same lossless parsed representation.
        return load_manifest_table(path)
    except ValueError as exc:
        raise FinalEvaluationProtocolError(
            "Manifest final evaluation phải là CSV hoặc Parquet"
        ) from exc


def _validate_contract_fields(contract: SelectionContract) -> None:
    if (
        not isinstance(contract.experiment, str)
        or not contract.experiment.strip()
        or not isinstance(contract.run_id, str)
        or not contract.run_id.strip()
    ):
        raise FinalEvaluationProtocolError("selection_complete.json thiếu experiment/run_id")
    if not Path(contract.checkpoint).is_absolute():
        raise FinalEvaluationProtocolError(
            "Checkpoint trong selection contract phải là đường dẫn tuyệt đối"
        )
    for name in ("checkpoint_sha256", "config_sha256", "manifest_sha256"):
        value = str(getattr(contract, name))
        if not _SHA256_PATTERN.fullmatch(value):
            raise FinalEvaluationProtocolError(f"{name} không phải SHA-256 hợp lệ")
    if (
        isinstance(contract.threshold, bool)
        or not isinstance(contract.threshold, int | float)
        or not math.isfinite(contract.threshold)
        or not 0.0 <= contract.threshold <= 1.0
    ):
        raise FinalEvaluationProtocolError("Threshold đã khóa phải hữu hạn và nằm trong [0,1]")


def _verify_lock_before_test_access(
    selection_path: Path, manifest_path: Path
) -> tuple[SelectionContract, pd.DataFrame, str]:
    """Verify the lock and hashes without extracting any test rows."""

    if not selection_path.is_file():
        raise FinalEvaluationProtocolError(
            f"Thiếu selection_complete.json; test vẫn bị khóa: {selection_path}"
        )
    try:
        contract = verify_selection_contract(selection_path)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FinalEvaluationProtocolError(f"Selection contract không hợp lệ: {exc}") from exc
    _validate_contract_fields(contract)

    # Loading the complete immutable manifest is necessary to reproduce its canonical
    # hash.  No test-row filtering, labels, dataset, or prediction is accessed here.
    manifest = _load_manifest(manifest_path)
    actual_manifest_hash = manifest_sha256(manifest)
    if actual_manifest_hash.lower() != contract.manifest_sha256.lower():
        raise FinalEvaluationProtocolError(
            "Manifest hash không khớp selection_complete.json; test vẫn bị khóa"
        )
    return contract, manifest, actual_manifest_hash.lower()


def _metadata_candidates(checkpoint: Path) -> list[Path]:
    return list(
        dict.fromkeys(
            [
                checkpoint.with_suffix(".metadata.json"),
                checkpoint.parent / f"{checkpoint.stem}.metadata.json",
                checkpoint.parent / "model.metadata.json",
            ]
        )
    )


def _load_model_metadata(
    checkpoint: Path,
    metadata_path: Path | None,
    *,
    checkpoint_hash: str,
    manifest_hash: str,
    config_hash: str,
    run_id: str,
) -> tuple[Path | None, dict[str, Any]]:
    if metadata_path is not None:
        selected = metadata_path.resolve()
        if not selected.is_file():
            raise FileNotFoundError(selected)
    else:
        selected = next(
            (candidate for candidate in _metadata_candidates(checkpoint) if candidate.is_file()),
            None,
        )
    if selected is None:
        raise FinalEvaluationProtocolError(
            "Thiếu model.metadata.json; không thể xác minh config/model/manifest provenance"
        )
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise FinalEvaluationProtocolError(f"Model metadata không hợp lệ: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalEvaluationProtocolError("Model metadata phải là một JSON object")
    expected_model_hash = payload.get("model_sha256")
    if expected_model_hash and str(expected_model_hash).lower() != checkpoint_hash.lower():
        raise FinalEvaluationProtocolError("model_sha256 trong metadata không khớp checkpoint")
    expected_manifest_hash = payload.get("manifest_sha256")
    if str(expected_manifest_hash or "").lower() != manifest_hash.lower():
        raise FinalEvaluationProtocolError("manifest_sha256 trong metadata không khớp contract")
    expected_config_hash = payload.get("config_sha256")
    if str(expected_config_hash or "").lower() != config_hash.lower():
        raise FinalEvaluationProtocolError("config_sha256 trong metadata không khớp contract")
    if str(payload.get("run_id", "")).strip() != run_id:
        raise FinalEvaluationProtocolError("run_id trong metadata không khớp contract")
    required_fields = {
        "model_version",
        "threshold",
        "input_size",
        "preprocessing",
        "label_mapping",
        "gradcam_layer",
        "tensorflow_version",
    }
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise FinalEvaluationProtocolError(f"Model metadata thiếu trường bắt buộc: {missing}")
    mapping = payload.get("label_mapping")
    if mapping is not None and mapping != LABEL_MAPPING:
        raise FinalEvaluationProtocolError("Model metadata vi phạm quy ước Non-crack=0, Crack=1")
    preprocessing = payload.get("preprocessing")
    if preprocessing is not None and preprocessing != PREPROCESSING_NAME:
        raise FinalEvaluationProtocolError(
            "Model metadata dùng preprocessing không đúng MobileNetV2"
        )
    return selected, payload


def _resolve_image_size(
    metadata: dict[str, Any], override: tuple[int, int] | None
) -> tuple[int, int]:
    values = override or metadata.get("input_size") or IMAGE_SIZE
    if not isinstance(values, list | tuple) or len(values) != 2:
        raise FinalEvaluationProtocolError("input_size phải gồm [height, width]")
    height, width = (int(value) for value in values)
    if height <= 0 or width <= 0:
        raise FinalEvaluationProtocolError("input_size phải dương")
    return height, width


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return component or "run"


def _resolve_output_dir(
    output_dir: Path | None,
    contract: SelectionContract,
    *,
    smoke: bool,
) -> Path:
    root = SMOKE_EVALUATION_ROOT if smoke else REPORT_EVALUATION_ROOT
    name = f"{_safe_component(contract.run_id)}-{contract.checkpoint_sha256[:12]}"
    target = (output_dir or (root / name)).resolve()
    if smoke and _is_within(target, ARTIFACTS_DIR / "report"):
        raise FinalEvaluationProtocolError(
            "Smoke evaluation không được ghi vào artifacts/report; dùng artifacts/smoke"
        )
    if target.exists():
        raise FileExistsError(f"Không ghi đè final-evaluation artifact: {target}")
    return target


def _marker_path(checkpoint_hash: str) -> Path:
    return FINAL_EVALUATION_REGISTRY.resolve() / f"{checkpoint_hash.lower()}.json"


def _refuse_existing_marker(path: Path) -> None:
    if path.exists():
        raise FinalEvaluationAlreadyRunError(
            f"Checkpoint này đã được cấp quyền truy cập final test một lần; marker bất biến: {path}"
        )


def _claim_final_test_access(path: Path, payload: dict[str, Any]) -> None:
    """Create the immutable marker with an atomic create-if-absent operation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise FinalEvaluationAlreadyRunError(
            f"Checkpoint này vừa được một tiến trình khác cấp quyền final test; marker: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Never silently delete a claim: once test access was granted, that fact is evidence.
        raise


def _outcome_path(marker: Path) -> Path:
    return marker.with_name(f"{marker.stem}.outcome.json")


def _write_final_test_outcome(
    marker: Path,
    *,
    status: str,
    output_dir: Path,
    error: BaseException | None = None,
) -> Path:
    """Append an immutable outcome beside the immutable access claim.

    The claim itself is never edited.  If anything fails after test access was
    granted, a separate outcome explains why the checkpoint remains consumed.
    """

    outcome = _outcome_path(marker)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "event": "FINAL_TEST_OUTCOME",
        "status": status,
        "recorded_at_utc": _utc_now(),
        "access_marker": str(marker),
        "access_marker_sha256": sha256_file(marker),
        "output_dir": str(output_dir),
        "immutable": True,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    try:
        _claim_final_test_access(outcome, payload)
    except FinalEvaluationAlreadyRunError as exc:
        raise FinalEvaluationProtocolError(
            f"Outcome final-test đã tồn tại và không được ghi đè: {outcome}"
        ) from exc
    return outcome


def _select_test_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"relative_path", "label", "split"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise FinalEvaluationProtocolError(f"Manifest thiếu cột bắt buộc: {missing}")
    split_names = manifest["split"].astype(str).str.strip().str.lower()
    test_frame = manifest.loc[split_names.eq("test")].copy().reset_index(drop=True)
    if test_frame.empty:
        raise FinalEvaluationProtocolError("Manifest đã khóa không có mẫu test")
    return test_frame


def _load_model(checkpoint: Path) -> Any:
    try:
        import tensorflow as tf
    except (ImportError, RuntimeError, ValueError) as exc:
        raise RuntimeError("TensorFlow không sẵn sàng cho final evaluation") from exc
    return tf.keras.models.load_model(checkpoint, compile=False)


def _predict_probabilities(model: Any, dataset: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float64).reshape(-1)
    if probabilities.size == 0:
        raise FinalEvaluationProtocolError("Model không trả về prediction")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise FinalEvaluationProtocolError("Model phải trả về P(Crack) hữu hạn trong [0,1]")
    return probabilities


def _prediction_frame(
    test_frame: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    preferred = ["relative_path", "surface", "source_group", "sha256", "width", "height"]
    columns = [name for name in preferred if name in test_frame.columns]
    predictions = test_frame.loc[:, columns].copy()
    truth = pd.to_numeric(test_frame["label"], errors="raise").astype(int).to_numpy()
    predicted = (probabilities >= threshold).astype(int)
    predictions["y_true"] = truth
    predictions["probability_crack"] = probabilities
    predictions["threshold"] = float(threshold)
    predictions["y_pred"] = predicted
    outcomes = np.full(len(predictions), "TN", dtype=object)
    outcomes[(truth == 1) & (predicted == 1)] = "TP"
    outcomes[(truth == 0) & (predicted == 1)] = "FP"
    outcomes[(truth == 1) & (predicted == 0)] = "FN"
    predictions["outcome"] = outcomes
    fixed_predicted = (probabilities >= 0.5).astype(int)
    fixed_outcomes = np.full(len(predictions), "TN", dtype=object)
    fixed_outcomes[(truth == 1) & (fixed_predicted == 1)] = "TP"
    fixed_outcomes[(truth == 0) & (fixed_predicted == 1)] = "FP"
    fixed_outcomes[(truth == 1) & (fixed_predicted == 0)] = "FN"
    predictions["y_pred_threshold_0_5"] = fixed_predicted
    predictions["outcome_threshold_0_5"] = fixed_outcomes
    return predictions


def _classification_report_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"label": label, **values})
        else:
            rows.append({"label": label, "accuracy": values})
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, lineterminator="\n", mode="x")


def _write_threshold_evidence(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    suffix: str,
    outcome_column: str,
) -> None:
    report_frame = _classification_report_frame(metrics["classification_report"])
    write_json(
        output_dir / f"classification_report_{suffix}.json",
        metrics["classification_report"],
        overwrite=False,
    )
    _write_csv(report_frame, output_dir / f"classification_report_{suffix}.csv")
    with (output_dir / f"classification_report_{suffix}.txt").open("x", encoding="utf-8") as handle:
        handle.write(report_frame.to_string(index=False) + "\n")

    raw_matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["true_non_crack", "true_crack"],
        columns=["predicted_non_crack", "predicted_crack"],
    )
    normalized_matrix = pd.DataFrame(
        metrics["confusion_matrix_normalized"],
        index=["true_non_crack", "true_crack"],
        columns=["predicted_non_crack", "predicted_crack"],
    )
    _write_csv(raw_matrix, output_dir / f"confusion_matrix_{suffix}.csv", index=True)
    _write_csv(
        normalized_matrix,
        output_dir / f"confusion_matrix_{suffix}_normalized.csv",
        index=True,
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        output_dir / f"confusion_matrix_{suffix}.png",
        title=f"Confusion matrix ({metrics['threshold']:.6g})",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix_normalized"],
        output_dir / f"confusion_matrix_{suffix}_normalized.png",
        normalized=True,
        title=f"Normalized confusion matrix ({metrics['threshold']:.6g})",
    )
    _write_csv(
        predictions.loc[predictions[outcome_column].eq("FP")],
        output_dir / f"false_positives_{suffix}.csv",
    )
    _write_csv(
        predictions.loc[predictions[outcome_column].eq("FN")],
        output_dir / f"false_negatives_{suffix}.csv",
    )


def _write_artifacts(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    fixed_threshold_metrics: dict[str, Any],
    predictions: pd.DataFrame,
    split_audit: dict[str, Any],
    metadata: dict[str, Any],
    environment: dict[str, Any],
    selection_snapshot: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "metrics_test.json", metrics, overwrite=False)
    write_json(
        output_dir / "metrics_test_threshold_0_5.json",
        fixed_threshold_metrics,
        overwrite=False,
    )
    _write_csv(predictions, output_dir / "predictions_test.csv")
    _write_threshold_evidence(
        output_dir=output_dir,
        metrics=metrics,
        predictions=predictions,
        suffix="test",
        outcome_column="outcome",
    )
    _write_threshold_evidence(
        output_dir=output_dir,
        metrics=fixed_threshold_metrics,
        predictions=predictions,
        suffix="test_threshold_0_5",
        outcome_column="outcome_threshold_0_5",
    )
    write_json(output_dir / "split_audit.json", split_audit, overwrite=False)
    write_json(output_dir / "selection_contract_snapshot.json", selection_snapshot, overwrite=False)
    write_json(output_dir / "environment.json", environment, overwrite=False)
    write_json(output_dir / "evaluation_metadata.json", metadata, overwrite=False)

    evidence_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    evidence_hashes = {path.name: sha256_file(path) for path in evidence_files}
    write_json(
        output_dir / "evaluation_complete.json",
        {
            "schema_version": 1,
            "status": metadata["status"],
            "completed_at_utc": _utc_now(),
            "artifact_sha256": evidence_hashes,
        },
        overwrite=False,
    )


def run_final_evaluation(
    *,
    selection_path: str | Path,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path | None = None,
    metadata_path: str | Path | None = None,
    batch_size: int = 32,
    image_size: tuple[int, int] | None = None,
    smoke: bool = False,
) -> FinalEvaluationResult:
    """Evaluate the locked checkpoint exactly once on the immutable test split.

    The registry is deliberately project-global and is not caller-configurable,
    so changing the output directory cannot bypass the one-evaluation policy.
    """

    started = time.perf_counter()
    selection = Path(selection_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    dataset_directory = Path(dataset_root).resolve()
    contract, manifest, manifest_hash = _verify_lock_before_test_access(selection, manifest_file)
    checkpoint = Path(contract.checkpoint).resolve()
    checkpoint_hash = contract.checkpoint_sha256.lower()
    marker = _marker_path(checkpoint_hash)
    _refuse_existing_marker(marker)

    official_preconditions: dict[str, Any] | None = None
    if not smoke:
        try:
            official_preconditions = verify_official_dataset_preconditions(
                manifest,
                manifest_file,
                dataset_directory,
            ).to_dict()
        except (DatasetIntegrityError, FileNotFoundError, OSError, ValueError) as exc:
            raise FinalEvaluationProtocolError(
                f"Official dataset preconditions không hợp lệ; test vẫn bị khóa: {exc}"
            ) from exc

    target = _resolve_output_dir(
        Path(output_dir) if output_dir is not None else None,
        contract,
        smoke=smoke,
    )
    selected_metadata_path, model_metadata = _load_model_metadata(
        checkpoint,
        Path(metadata_path) if metadata_path is not None else None,
        checkpoint_hash=checkpoint_hash,
        manifest_hash=manifest_hash,
        config_hash=contract.config_sha256,
        run_id=contract.run_id,
    )
    resolved_image_size = _resolve_image_size(model_metadata, image_size)
    if batch_size <= 0:
        raise ValueError("batch_size phải dương")

    selection_hash = sha256_file(selection)
    marker_payload = {
        "schema_version": 1,
        "event": "FINAL_TEST_ACCESS_GRANTED",
        "created_at_utc": _utc_now(),
        "experiment": contract.experiment,
        "run_id": contract.run_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": contract.config_sha256.lower(),
        "manifest": str(manifest_file),
        "manifest_sha256": manifest_hash,
        "selection_contract": str(selection),
        "selection_contract_sha256": selection_hash,
        "threshold": float(contract.threshold),
        "threshold_selected_by": "validation",
        "output_dir": str(target),
        "smoke_test": bool(smoke),
        "valid_for_report": not smoke,
        "status": "NOT_VALID_FOR_REPORT" if smoke else "FINAL_TEST_ACCESS_GRANTED",
        "official_preconditions": official_preconditions,
        "immutable": True,
    }
    if sha256_file(checkpoint).lower() != checkpoint_hash:
        raise FinalEvaluationProtocolError(
            "Checkpoint đã thay đổi sau bước xác minh contract; test vẫn bị khóa"
        )
    _claim_final_test_access(marker, marker_payload)
    try:
        # This is the first operation that extracts or consumes final-test rows.
        test_frame = _select_test_rows(manifest)
        split_audit = audit_split(manifest)
        if not smoke and not split_audit.get("valid", False):
            raise FinalEvaluationProtocolError(
                "Split audit không hợp lệ cho báo cáo: " + "; ".join(split_audit.get("errors", []))
            )

        dataset = build_tf_dataset(
            test_frame,
            batch_size=int(batch_size),
            image_size=resolved_image_size,
            training=False,
            augment=False,
            dataset_root=dataset_directory,
        )
        model = _load_model(checkpoint)
        if sha256_file(checkpoint).lower() != checkpoint_hash:
            raise FinalEvaluationProtocolError(
                "Checkpoint đã thay đổi trong lúc load model; quyền final-test đã được ghi nhận"
            )
        prediction_started = time.perf_counter()
        probabilities = _predict_probabilities(model, dataset)
        prediction_seconds = time.perf_counter() - prediction_started
        if len(probabilities) != len(test_frame):
            raise FinalEvaluationProtocolError(
                f"Số prediction ({len(probabilities)}) không khớp số mẫu test ({len(test_frame)})"
            )

        metrics = compute_binary_metrics(test_frame["label"], probabilities, contract.threshold)
        fixed_threshold_metrics = compute_binary_metrics(test_frame["label"], probabilities, 0.5)
        status = "NOT_VALID_FOR_REPORT" if smoke else "FINAL_TEST_COMPLETE"
        shared_metric_metadata = {
            "status": status,
            "valid_for_report": not smoke,
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": manifest_hash,
            "config_sha256": contract.config_sha256.lower(),
            "prediction_passes": 1,
        }
        metrics.update(
            {
                **shared_metric_metadata,
                "threshold_role": "validation_locked",
                "fixed_threshold_0_5_metrics_file": "metrics_test_threshold_0_5.json",
            }
        )
        fixed_threshold_metrics.update(
            {
                **shared_metric_metadata,
                "threshold_role": "fixed_protocol_0_5",
                "validation_locked_threshold": float(contract.threshold),
                "validation_locked_metrics_file": "metrics_test.json",
            }
        )
        predictions = _prediction_frame(test_frame, probabilities, contract.threshold)
        elapsed_seconds = time.perf_counter() - started
        environment = capture_environment()
        metadata = {
            "schema_version": 1,
            "status": status,
            "valid_for_report": not smoke,
            "smoke_test": bool(smoke),
            "created_at_utc": _utc_now(),
            "project_version": __version__,
            "experiment": contract.experiment,
            "run_id": contract.run_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": contract.config_sha256.lower(),
            "manifest": str(manifest_file),
            "manifest_file_sha256": sha256_file(manifest_file),
            "manifest_sha256": manifest_hash,
            "selection_contract": str(selection),
            "selection_contract_sha256": selection_hash,
            "selection_selected_by": contract.selected_by,
            "threshold": float(contract.threshold),
            "threshold_source": "validation",
            "evaluated_thresholds": {
                "validation_locked": float(contract.threshold),
                "fixed_protocol_0_5": 0.5,
            },
            "prediction_passes": 1,
            "test_sample_count": len(test_frame),
            "batch_size": int(batch_size),
            "input_size": list(resolved_image_size),
            "preprocessing": PREPROCESSING_NAME,
            "label_mapping": LABEL_MAPPING,
            "model_metadata": str(selected_metadata_path) if selected_metadata_path else None,
            "model_metadata_sha256": (
                sha256_file(selected_metadata_path) if selected_metadata_path else None
            ),
            "prediction_seconds_total": float(prediction_seconds),
            "prediction_seconds_per_image": float(prediction_seconds / len(test_frame)),
            "evaluation_seconds_total": float(elapsed_seconds),
            "one_evaluation_marker": str(marker),
            "one_evaluation_marker_sha256": sha256_file(marker),
            "failure_outcome_if_any": str(_outcome_path(marker)),
            "official_preconditions": official_preconditions,
        }
        _write_artifacts(
            output_dir=target,
            metrics=metrics,
            fixed_threshold_metrics=fixed_threshold_metrics,
            predictions=predictions,
            split_audit=split_audit,
            metadata=metadata,
            environment=environment,
            selection_snapshot=asdict(contract),
        )
    except BaseException as exc:
        _write_final_test_outcome(
            marker,
            status="FINAL_TEST_FAILED_CHECKPOINT_CONSUMED",
            output_dir=target,
            error=exc,
        )
        raise
    return FinalEvaluationResult(
        output_dir=target,
        marker_path=marker,
        metrics_path=target / "metrics_test.json",
        predictions_path=target / "predictions_test.csv",
        accuracy=float(metrics["accuracy"]),
        f1_crack=float(metrics["crack"]["f1"]),
        status=status,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Đánh giá final test đúng một lần sau khi checkpoint, manifest và threshold "
            "đã được khóa bằng validation."
        )
    )
    parser.add_argument("--selection", type=Path, required=True, help="selection_complete.json")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest split CSV/Parquet")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Thư mục gốc ảnh")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Mặc định: artifacts/report/final_evaluation/<run>; smoke dùng artifacts/smoke",
    )
    parser.add_argument("--metadata", type=Path, help="Model metadata JSON tùy chọn")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Override kích thước input; mặc định đọc metadata hoặc 224 224",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Đánh giá kiểm thử kỹ thuật, luôn gắn NOT_VALID_FOR_REPORT. "
            "Vẫn tiêu thụ quyền final-test một lần của checkpoint."
        ),
    )
    return parser


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_final_evaluation(
            selection_path=args.selection,
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            metadata_path=args.metadata,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size) if args.image_size else None,
            smoke=args.smoke,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Final evaluation: {result.status}")
    print(f"Artifacts: {result.output_dir}")
    print(f"Accuracy: {result.accuracy:.6f}")
    print(f"F1 Crack: {result.f1_crack:.6f}")
    print(f"Immutable marker: {result.marker_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
