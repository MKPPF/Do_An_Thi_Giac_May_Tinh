"""Immutable validation-selection contract required before final test evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from crackspot.modeling.threshold import optimize_threshold
from crackspot.utils.hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class SelectionContract:
    experiment: str
    run_id: str
    checkpoint: str
    checkpoint_sha256: str
    config_sha256: str
    manifest_sha256: str
    threshold: float
    selected_by: str = "validation"
    schema_version: int = 2
    trained_experiment: str | None = None
    threshold_source: str = "fixed_protocol"
    threshold_result: str | None = None
    threshold_result_sha256: str | None = None
    training_complete_sha256: str | None = None


def _require_sha256(value: str, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} phải là SHA-256 64 ký tự hex")
    return normalized


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON phải là object: {path}")
    return payload


def create_selection_contract(
    *,
    experiment: str,
    run_id: str,
    checkpoint: str | Path,
    config_sha256: str,
    manifest_sha256: str,
    threshold: float,
    output: str | Path,
    trained_experiment: str | None = None,
    threshold_source: str = "fixed_protocol",
    threshold_result: str | Path | None = None,
    training_complete_sha256: str | None = None,
) -> SelectionContract:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if (
        isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold phải trong [0,1]")
    normalized_source = str(threshold_source).strip().lower()
    if normalized_source not in {"fixed_protocol", "validation"}:
        raise ValueError("threshold_source phải là fixed_protocol hoặc validation")
    threshold_result_path: Path | None = None
    threshold_result_hash: str | None = None
    if threshold_result is not None:
        threshold_result_path = Path(threshold_result).resolve()
        if not threshold_result_path.is_file():
            raise FileNotFoundError(threshold_result_path)
        threshold_result_hash = sha256_file(threshold_result_path)
    if normalized_source == "validation" and threshold_result_path is None:
        raise ValueError("threshold validation bắt buộc có threshold_result provenance")
    if normalized_source == "fixed_protocol" and threshold_result_path is not None:
        raise ValueError("fixed_protocol không nhận threshold_result")
    if normalized_source == "fixed_protocol" and not math.isclose(
        float(threshold), 0.5, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("fixed_protocol threshold phải bằng 0.5")
    contract = SelectionContract(
        experiment=experiment,
        run_id=run_id,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        config_sha256=_require_sha256(config_sha256, "config_sha256"),
        manifest_sha256=_require_sha256(manifest_sha256, "manifest_sha256"),
        threshold=float(threshold),
        trained_experiment=(str(trained_experiment).strip() if trained_experiment else experiment),
        threshold_source=normalized_source,
        threshold_result=str(threshold_result_path) if threshold_result_path else None,
        threshold_result_sha256=threshold_result_hash,
        training_complete_sha256=(
            _require_sha256(training_complete_sha256, "training_complete_sha256")
            if training_complete_sha256 is not None
            else None
        ),
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Không ghi đè selection contract: {target}")
    target.write_text(json.dumps(asdict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    return contract


def _verify_training_completion(
    directory: Path,
    *,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    config_path: Path,
    checkpoint: Path,
) -> tuple[str, str, str, str, str, bool]:
    completion_path = directory / "training_complete.json"
    completion = _read_json_object(completion_path)
    config_hash = _require_sha256(str(summary.get("config_sha256", "")), "config_sha256")
    manifest_hash = _require_sha256(str(summary.get("manifest_sha256", "")), "manifest_sha256")
    checkpoint_hash = _require_sha256(str(summary.get("model_sha256", "")), "model_sha256")
    if sha256_json(_read_json_object(config_path)) != config_hash:
        raise ValueError("config_snapshot.json không khớp config_sha256")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("model.keras không khớp model_sha256")
    for payload_name, payload in (("metadata", metadata), ("completion", completion)):
        for field, expected in (
            ("config_sha256", config_hash),
            ("manifest_sha256", manifest_hash),
            ("model_sha256", checkpoint_hash),
        ):
            if str(payload.get(field, "")).strip().lower() != expected:
                raise ValueError(f"{payload_name}.{field} không khớp run summary")
    run_id = str(summary.get("run_id", "")).strip()
    if not run_id or str(completion.get("run_id", "")).strip() != run_id:
        raise ValueError("run_id thiếu hoặc không nhất quán")
    artifacts = completion.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("training_complete.json thiếu artifact_sha256")
    for name in (
        "config_snapshot.json",
        "model.keras",
        "model.metadata.json",
        "run_summary.json",
        "predictions_validation.csv",
    ):
        expected = _require_sha256(artifacts.get(name), f"artifact_sha256.{name}")
        candidate = directory / name
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"Training artifact hash không khớp: {candidate}")
    status = str(summary.get("status", "")).strip().upper()
    completion_status = str(completion.get("status", "")).strip().upper()
    smoke = (
        bool(metadata.get("smoke_test", False))
        or summary.get("valid_for_report") is False
        or "NOT_VALID_FOR_REPORT" in status
    )
    if status != completion_status:
        raise ValueError("run summary/training completion status không nhất quán")
    if not smoke and (
        status != "VALIDATION_COMPLETE_TEST_LOCKED" or summary.get("valid_for_report") is not True
    ):
        raise ValueError("Run chưa hoàn tất validation hợp lệ cho báo cáo")
    return (
        run_id,
        config_hash,
        manifest_hash,
        checkpoint_hash,
        sha256_file(completion_path),
        smoke,
    )


def _verify_threshold_result(
    path: Path,
    *,
    directory: Path,
    run_id: str,
    config_hash: str,
    manifest_hash: str,
    checkpoint_hash: str,
    completion_hash: str,
    smoke: bool,
) -> float:
    result = _read_json_object(path)
    if (
        result.get("schema_version") != 2
        or result.get("artifact_type") != "validation_threshold_selection"
    ):
        raise ValueError("Threshold result không đúng schema validation đã khóa")
    if str(result.get("source_split", "")).strip().lower() != "validation":
        raise ValueError("Threshold result phải được tạo từ validation")
    if str(result.get("selected_by", "")).strip().lower() != "validation":
        raise ValueError("Threshold result không được chọn bằng validation")
    expected_pairs = (
        ("run_id", run_id),
        ("config_sha256", config_hash),
        ("manifest_sha256", manifest_hash),
        ("checkpoint_sha256", checkpoint_hash),
        ("training_complete_sha256", completion_hash),
    )
    for field, expected in expected_pairs:
        if str(result.get(field, "")).strip().lower() != expected.lower():
            raise ValueError(f"Threshold result không khớp {field}")
    if bool(result.get("valid_for_report")) == smoke:
        raise ValueError("Threshold result valid_for_report không khớp run")
    predictions = Path(str(result.get("source_predictions", ""))).resolve()
    expected_predictions = (directory / "predictions_validation.csv").resolve()
    if predictions != expected_predictions or not predictions.is_file():
        raise ValueError("Threshold result không trỏ tới predictions_validation.csv của run")
    predictions_hash = _require_sha256(result.get("predictions_sha256"), "predictions_sha256")
    if sha256_file(predictions) != predictions_hash:
        raise ValueError("predictions_validation.csv đã thay đổi sau threshold tuning")
    frame = pd.read_csv(predictions)
    if int(result.get("sample_count", -1)) != len(frame):
        raise ValueError("Threshold result sample_count không khớp predictions")
    recomputed = optimize_threshold(frame["y_true"], frame["probability_crack"])
    scalar_fields = {
        "threshold": recomputed.threshold,
        "f1_crack": recomputed.f1_crack,
        "recall_crack": recomputed.recall_crack,
        "precision_crack": recomputed.precision_crack,
        "accuracy": recomputed.accuracy,
    }
    for field, expected in scalar_fields.items():
        try:
            observed = float(result[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Threshold result thiếu {field}") from exc
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Threshold result {field} không khớp predictions")
    if int(result.get("evaluated_candidates", -1)) != recomputed.evaluated_candidates:
        raise ValueError("Threshold result evaluated_candidates không khớp")
    return recomputed.threshold


def lock_run_selection(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
    experiment: str | None = None,
    threshold: float | None = None,
    threshold_result: str | Path | None = None,
    allow_smoke: bool = False,
) -> SelectionContract:
    """Lock a validation-complete run after verifying every local provenance hash."""

    if (threshold is None) == (threshold_result is None):
        raise ValueError("Cần đúng một trong threshold hoặc threshold_result")
    directory = Path(run_dir).resolve()
    summary = _read_json_object(directory / "run_summary.json")
    metadata = _read_json_object(directory / "model.metadata.json")
    config_path = directory / "config_snapshot.json"
    config = _read_json_object(config_path)
    checkpoint = directory / "model.keras"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    (
        run_id,
        config_hash,
        manifest_hash,
        checkpoint_hash,
        completion_hash,
        smoke,
    ) = _verify_training_completion(
        directory,
        summary=summary,
        metadata=metadata,
        config_path=config_path,
        checkpoint=checkpoint,
    )
    if smoke and not allow_smoke:
        raise ValueError("Run smoke/NOT_VALID_FOR_REPORT cần xác nhận --smoke")

    threshold_result_path: Path | None = None
    if threshold_result is not None:
        threshold_result_path = Path(threshold_result).resolve()
        selected_threshold = _verify_threshold_result(
            threshold_result_path,
            directory=directory,
            run_id=run_id,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            checkpoint_hash=checkpoint_hash,
            completion_hash=completion_hash,
            smoke=smoke,
        )
        threshold_source = "validation"
    else:
        selected_threshold = float(threshold)
        if not math.isclose(selected_threshold, 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Threshold protocol cố định cho E1-E4 phải bằng 0.5")
        threshold_source = "fixed_protocol"
    experiment_config = config.get("experiment", {})
    selected_experiment = str(
        experiment or experiment_config.get("id") or experiment_config.get("name") or ""
    ).strip()
    if not selected_experiment:
        raise ValueError("Không xác định được experiment")
    target = Path(output) if output is not None else directory / "selection_complete.json"
    return create_selection_contract(
        experiment=selected_experiment,
        run_id=run_id,
        checkpoint=checkpoint,
        config_sha256=config_hash,
        manifest_sha256=manifest_hash,
        threshold=selected_threshold,
        output=target,
        trained_experiment=str(experiment_config.get("id") or experiment_config.get("name")),
        threshold_source=threshold_source,
        threshold_result=threshold_result_path,
        training_complete_sha256=completion_hash,
    )


def verify_selection_contract(
    path: str | Path, *, expected_manifest_sha256: str | None = None
) -> SelectionContract:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    contract = SelectionContract(**payload)
    if contract.selected_by != "validation":
        raise ValueError("Model/threshold phải được chọn bằng validation")
    if expected_manifest_sha256 and contract.manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Manifest hash không khớp selection contract")
    _require_sha256(contract.checkpoint_sha256, "checkpoint_sha256")
    _require_sha256(contract.config_sha256, "config_sha256")
    _require_sha256(contract.manifest_sha256, "manifest_sha256")
    checkpoint = Path(contract.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != contract.checkpoint_sha256:
        raise ValueError("Checkpoint hash không khớp selection contract")
    if contract.threshold_source not in {"fixed_protocol", "validation"}:
        raise ValueError("threshold_source không hợp lệ")
    if contract.threshold_source == "validation":
        if not contract.threshold_result or not contract.threshold_result_sha256:
            raise ValueError("Selection validation thiếu threshold result provenance")
        result_path = Path(contract.threshold_result)
        if (
            not result_path.is_file()
            or sha256_file(result_path) != contract.threshold_result_sha256
        ):
            raise ValueError("Threshold result hash không khớp selection contract")
        if not contract.training_complete_sha256:
            raise ValueError("Selection validation thiếu training completion provenance")
        result = _read_json_object(result_path)
        source_predictions = Path(str(result.get("source_predictions", ""))).resolve()
        if not source_predictions.is_file():
            raise ValueError("Threshold result thiếu predictions_validation.csv")
        recomputed_threshold = _verify_threshold_result(
            result_path,
            directory=source_predictions.parent,
            run_id=contract.run_id,
            config_hash=contract.config_sha256,
            manifest_hash=contract.manifest_sha256,
            checkpoint_hash=contract.checkpoint_sha256,
            completion_hash=_require_sha256(
                contract.training_complete_sha256, "training_complete_sha256"
            ),
            smoke=bool(result.get("valid_for_report") is False),
        )
        if not math.isclose(recomputed_threshold, contract.threshold, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Threshold result không khớp selection threshold")
    elif contract.threshold_result is not None or contract.threshold_result_sha256 is not None:
        raise ValueError("fixed_protocol không được chứa threshold result")
    elif not math.isclose(contract.threshold, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fixed_protocol threshold phải bằng 0.5")
    return contract


def export_selected_metadata(
    selection_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    """Export immutable inference metadata bound to a locked selection threshold."""

    selection_source = Path(selection_path).resolve()
    contract = verify_selection_contract(selection_source)
    checkpoint = Path(contract.checkpoint).resolve()
    run_dir = checkpoint.parent
    source = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else run_dir / "model.metadata.json"
    )
    metadata = _read_json_object(source)

    expected_metadata = (
        ("run_id", contract.run_id),
        ("model_sha256", contract.checkpoint_sha256),
        ("config_sha256", contract.config_sha256),
        ("manifest_sha256", contract.manifest_sha256),
    )
    for field, expected in expected_metadata:
        if str(metadata.get(field, "")).strip().lower() != str(expected).lower():
            raise ValueError(f"model metadata {field} không khớp selection contract")

    config_path = run_dir / "config_snapshot.json"
    if sha256_json(_read_json_object(config_path)) != contract.config_sha256:
        raise ValueError("config_snapshot.json không khớp selection contract")

    completion_path = run_dir / "training_complete.json"
    completion = _read_json_object(completion_path)
    if not contract.training_complete_sha256:
        raise ValueError("selection contract thiếu training_complete_sha256")
    if sha256_file(completion_path) != contract.training_complete_sha256:
        raise ValueError("training_complete.json không khớp selection contract")
    artifact_hashes = completion.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("training_complete.json thiếu artifact_sha256")
    for name, path in (
        ("model.keras", checkpoint),
        ("model.metadata.json", source),
        ("config_snapshot.json", config_path),
    ):
        recorded = _require_sha256(artifact_hashes.get(name), f"artifact_sha256.{name}")
        if sha256_file(path) != recorded:
            raise ValueError(f"Training artifact hash không khớp: {path}")

    target = (
        Path(output).resolve() if output is not None else run_dir / "selected_model.metadata.json"
    )
    if target.exists():
        raise FileExistsError(f"Không ghi đè selected metadata: {target}")
    if target == source:
        raise ValueError("Không được ghi đè model.metadata.json gốc")

    payload = dict(metadata)
    payload.update(
        {
            "threshold": contract.threshold,
            "selection_experiment": contract.experiment,
            "selected_by": contract.selected_by,
            "threshold_source": contract.threshold_source,
            "selection_contract": str(selection_source),
            "selection_contract_sha256": sha256_file(selection_source),
            "source_metadata": str(source),
            "source_metadata_sha256": sha256_file(source),
            "deployment_metadata_schema_version": 1,
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
