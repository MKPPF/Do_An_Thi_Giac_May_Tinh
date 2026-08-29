"""Strict aggregation of measured selection, evaluation, and report artifacts."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crackspot.data import (
    CONFLICT_REPORT_SNAPSHOT_FILENAME,
    CURATED_MANIFEST_SNAPSHOT_FILENAME,
    SplitValidationError,
    audit_split,
    load_manifest_table,
    manifest_sha256,
    verify_locked_split_bundle,
)
from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.reporting.export import write_json
from crackspot.reporting.plots import (
    plot_dataset_distribution,
    plot_threshold_curve,
    plot_training_history,
)
from crackspot.utils.hashing import sha256_file, sha256_json


class ArtifactValidationError(RuntimeError):
    """Raised when evidence is absent, inconsistent, smoke-only, or tampered."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"JSON không hợp lệ: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"JSON phải là object: {path}")
    return payload


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ArtifactValidationError(f"{field} phải là SHA-256 64 ký tự hex")
    return text


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _path_inside_project(path: Path, project_root: Path, field: str) -> tuple[str, str]:
    resolved = path.resolve()
    try:
        portable = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{field} phải nằm trong project root để có path portable: {resolved}"
        ) from exc
    return portable, str(resolved)


def _is_not_valid(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status", "")).upper()
    return (
        payload.get("valid_for_report") is False
        or bool(payload.get("smoke_test", False))
        or "NOT_VALID_FOR_REPORT" in status
        or "SMOKE" in status
    )


def _is_smoke_path(path: Path) -> bool:
    parts = [part.casefold() for part in path.resolve().parts]
    return any(
        parts[index] == "artifacts" and parts[index + 1] == "smoke"
        for index in range(len(parts) - 1)
    )


def is_official_report_path(path: str | Path) -> bool:
    """Return true for ``artifacts/report`` and every descendant."""

    parts = [part.casefold() for part in Path(path).resolve().parts]
    return any(
        parts[index] == "artifacts" and parts[index + 1] == "report"
        for index in range(len(parts) - 1)
    )


@dataclass(frozen=True)
class _SelectionCandidate:
    experiment: str
    experiment_name: str
    run_id: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    completion_path: Path
    best_val_loss: float
    config_sha256: str
    checkpoint_sha256: str
    completion_sha256: str
    manifest_sha256: str
    valid_for_report: bool


def _load_selection_candidate(
    run_dir: str | Path,
    *,
    expected_experiment: str,
) -> _SelectionCandidate:
    directory = Path(run_dir).resolve()
    summary_path = directory / "run_summary.json"
    config_path = directory / "config_snapshot.json"
    checkpoint_path = directory / "model.keras"
    metadata_path = directory / "model.metadata.json"
    completion_path = directory / "training_complete.json"
    summary = _read_json(summary_path)
    config = _read_json(config_path)
    metadata = _read_json(metadata_path)
    completion = _read_json(completion_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, dict):
        raise ArtifactValidationError(f"config snapshot thiếu experiment: {config_path}")
    experiment_id = str(experiment_config.get("id", "")).strip().upper()
    if experiment_id != expected_experiment.upper():
        raise ArtifactValidationError(
            f"Run {directory.name} phải là {expected_experiment}, nhận {experiment_id or 'trống'}"
        )

    try:
        best_val_loss = float(summary["best_val_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"run_summary thiếu best_val_loss: {summary_path}") from exc
    if not math.isfinite(best_val_loss) or best_val_loss < 0:
        raise ArtifactValidationError("best_val_loss phải hữu hạn và không âm")

    config_hash = _require_sha256(summary.get("config_sha256"), "config_sha256")
    if sha256_json(config) != config_hash:
        raise ArtifactValidationError(f"config_snapshot hash không khớp: {directory}")
    checkpoint_hash = _require_sha256(summary.get("model_sha256"), "model_sha256")
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise ArtifactValidationError(f"checkpoint hash không khớp: {directory}")
    manifest_hash = _require_sha256(summary.get("manifest_sha256"), "manifest_sha256")

    for source_name, source in (("metadata", metadata), ("completion", completion)):
        pairs = (
            ("config_sha256", config_hash),
            ("model_sha256", checkpoint_hash),
            ("manifest_sha256", manifest_hash),
        )
        for key, expected in pairs:
            if source.get(key) is not None and str(source[key]).lower() != expected:
                raise ArtifactValidationError(f"{source_name}.{key} không khớp run summary")

    run_id = str(summary.get("run_id", "")).strip()
    if not run_id:
        raise ArtifactValidationError(f"run_summary thiếu run_id: {summary_path}")
    if str(completion.get("run_id", "")).strip() != run_id:
        raise ArtifactValidationError(f"training_complete.run_id không khớp: {directory}")
    valid_for_report = not (
        _is_not_valid(summary) or _is_not_valid(metadata) or _is_smoke_path(directory)
    )
    expected_status = (
        "VALIDATION_COMPLETE_TEST_LOCKED" if valid_for_report else "NOT_VALID_FOR_REPORT"
    )
    if (
        summary.get("status") != expected_status
        or metadata.get("status") != expected_status
        or completion.get("status") != expected_status
        or completion.get("immutable") is not True
    ):
        raise ArtifactValidationError(
            f"Run {directory.name} chưa có completion/status validation nhất quán"
        )
    artifact_hashes = completion.get("artifact_sha256")
    required_artifacts = {
        "config_snapshot.json": config_path,
        "model.keras": checkpoint_path,
        "model.metadata.json": metadata_path,
        "run_summary.json": summary_path,
    }
    if not isinstance(artifact_hashes, dict):
        raise ArtifactValidationError(f"training_complete thiếu artifact_sha256: {directory}")
    for name, path in required_artifacts.items():
        expected_hash = _require_sha256(
            artifact_hashes.get(name), f"training_complete.artifact_sha256.{name}"
        )
        if sha256_file(path) != expected_hash:
            raise ArtifactValidationError(f"Training artifact hash không khớp: {path}")
    return _SelectionCandidate(
        experiment=experiment_id,
        experiment_name=str(
            experiment_config.get("name") or experiment_config.get("slug") or experiment_id
        ),
        run_id=run_id,
        run_dir=directory,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        completion_path=completion_path,
        best_val_loss=best_val_loss,
        config_sha256=config_hash,
        checkpoint_sha256=checkpoint_hash,
        completion_sha256=sha256_file(completion_path),
        manifest_sha256=manifest_hash,
        valid_for_report=valid_for_report,
    )


def select_e2_e3(
    *,
    e2_run: str | Path,
    e3_run: str | Path,
    output: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Select E2/E3 using only their validation ``best_val_loss`` evidence."""

    candidates = [
        _load_selection_candidate(e2_run, expected_experiment="E2"),
        _load_selection_candidate(e3_run, expected_experiment="E3"),
    ]
    manifest_hashes = {candidate.manifest_sha256 for candidate in candidates}
    if len(manifest_hashes) != 1:
        raise ArtifactValidationError("E2 và E3 phải dùng cùng manifest_sha256")
    validity = {candidate.valid_for_report for candidate in candidates}
    if len(validity) != 1:
        raise ArtifactValidationError("Không được trộn run smoke và run report-valid")

    # Ties are resolved by the fixed experiment ID, never by a test metric.
    winner = min(candidates, key=lambda item: (item.best_val_loss, item.experiment))
    root = Path(project_root).resolve()
    winner_config, winner_config_resolved = _path_inside_project(
        winner.config_path, root, "winner_config"
    )
    winner_checkpoint, winner_checkpoint_resolved = _path_inside_project(
        winner.checkpoint_path, root, "winner_checkpoint"
    )
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        config_portable, config_resolved = _path_inside_project(
            candidate.config_path, root, f"{candidate.experiment}.config"
        )
        checkpoint_portable, checkpoint_resolved = _path_inside_project(
            candidate.checkpoint_path, root, f"{candidate.experiment}.checkpoint"
        )
        completion_portable, completion_resolved = _path_inside_project(
            candidate.completion_path, root, f"{candidate.experiment}.training_complete"
        )
        candidate_rows.append(
            {
                "experiment": candidate.experiment,
                "experiment_name": candidate.experiment_name,
                "run_id": candidate.run_id,
                "best_val_loss": candidate.best_val_loss,
                "config": config_portable,
                "config_resolved": config_resolved,
                "config_sha256": candidate.config_sha256,
                "checkpoint": checkpoint_portable,
                "checkpoint_resolved": checkpoint_resolved,
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "training_complete": completion_portable,
                "training_complete_resolved": completion_resolved,
                "training_complete_sha256": candidate.completion_sha256,
                "manifest_sha256": candidate.manifest_sha256,
            }
        )

    report_valid = next(iter(validity))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_by": "validation",
        "selection_split": "validation",
        "metric": "val_loss",
        "mode": "min",
        "tie_break": "experiment_id_ascending",
        "winner_experiment": winner.experiment,
        "winner_experiment_name": winner.experiment_name,
        "winner_run_id": winner.run_id,
        "winner_best_val_loss": winner.best_val_loss,
        "winner_config": winner_config,
        "winner_config_resolved": winner_config_resolved,
        "winner_config_sha256": winner.config_sha256,
        "winner_checkpoint": winner_checkpoint,
        "winner_checkpoint_resolved": winner_checkpoint_resolved,
        "winner_checkpoint_sha256": winner.checkpoint_sha256,
        "manifest_sha256": winner.manifest_sha256,
        "candidates": candidate_rows,
        "valid_for_report": report_valid,
        "status": "MODEL_SELECTED" if report_valid else "NOT_VALID_FOR_REPORT",
    }
    write_json(output, payload, overwrite=False)
    return payload


@dataclass(frozen=True)
class _EvaluationEvidence:
    directory: Path
    experiment: str
    run_id: str
    threshold: float
    metrics: dict[str, Any]
    fixed_threshold_metrics: dict[str, Any]
    checkpoint_sha256: str
    config_sha256: str
    manifest_sha256: str
    metrics_path: Path
    fixed_threshold_metrics_path: Path
    predictions_path: Path
    metadata_path: Path
    selection_snapshot_path: Path
    completion_path: Path
    valid_for_report: bool
    status: str


@dataclass(frozen=True)
class _RealImageEvidence:
    directory: Path
    metrics: dict[str, Any]
    status: str
    valid_for_report: bool
    metrics_path: Path
    predictions_path: Path
    metadata_path: Path
    selection_snapshot_path: Path
    completion_path: Path


@dataclass(frozen=True)
class _TrainingEvidence:
    directory: Path
    experiment: str
    run_id: str
    checkpoint_sha256: str
    config_sha256: str
    manifest_sha256: str
    history: pd.DataFrame
    history_path: Path
    config_path: Path
    checkpoint_path: Path
    summary_path: Path
    completion_path: Path
    valid_for_report: bool


@dataclass(frozen=True)
class _QualitativeEvidence:
    sidecar_path: Path
    image_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class _LockedManifestBundle:
    directory: Path
    manifest_path: Path
    manifest_hashes_path: Path
    split_audit_path: Path
    canonical_sha256: str
    input_manifest_path: Path
    curation_report_path: Path


def _consistent_value(
    payloads: Sequence[Mapping[str, Any]], keys: Sequence[str], field: str
) -> Any:
    values: list[Any] = []
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value is not None and value != "":
                values.append(value)
                break
    if not values:
        raise ArtifactValidationError(f"Thiếu {field} trong final evaluation artifacts")
    normalized = {str(value) for value in values}
    if len(normalized) != 1:
        raise ArtifactValidationError(f"{field} không nhất quán: {sorted(normalized)}")
    return values[0]


def _assert_metrics_match(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any], source: Path
) -> None:
    scalar_paths = (
        ("threshold",),
        ("sample_count",),
        ("accuracy",),
        ("tn",),
        ("fp",),
        ("fn",),
        ("tp",),
        ("crack", "precision"),
        ("crack", "recall"),
        ("crack", "f1"),
        ("macro", "precision"),
        ("macro", "recall"),
        ("macro", "f1"),
    )
    for path in scalar_paths:
        left: Any = recorded
        right: Any = recomputed
        try:
            for key in path:
                left = left[key]
                right = right[key]
        except (KeyError, TypeError) as exc:
            raise ArtifactValidationError(f"metrics_test thiếu {'.'.join(path)}: {source}") from exc
        if isinstance(right, int):
            matches = int(left) == right
        else:
            matches = math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
        if not matches:
            raise ArtifactValidationError(
                f"metrics_test không khớp predictions tại {'.'.join(path)}: {source}"
            )
    if recorded.get("confusion_matrix") != recomputed.get("confusion_matrix"):
        raise ArtifactValidationError(f"confusion_matrix không khớp predictions: {source}")


def _load_evaluation(directory: str | Path) -> _EvaluationEvidence:
    root = Path(directory).resolve()
    metrics_path = root / "metrics_test.json"
    fixed_threshold_metrics_path = root / "metrics_test_threshold_0_5.json"
    predictions_path = root / "predictions_test.csv"
    metadata_path = root / "evaluation_metadata.json"
    selection_snapshot_path = root / "selection_contract_snapshot.json"
    completion_path = root / "evaluation_complete.json"
    metrics = _read_json(metrics_path)
    fixed_threshold_metrics = _read_json(fixed_threshold_metrics_path)
    metadata = _read_json(metadata_path)
    selection_snapshot = _read_json(selection_snapshot_path)
    completion = _read_json(completion_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    frame = pd.read_csv(predictions_path)
    required_prediction_columns = {"y_true", "probability_crack"}
    missing = required_prediction_columns.difference(frame.columns)
    if missing:
        raise ArtifactValidationError(
            f"predictions_test thiếu cột: {sorted(missing)} ({predictions_path})"
        )
    if "split" in frame.columns:
        splits = set(frame["split"].dropna().astype(str).str.strip().str.lower())
        if splits != {"test"}:
            raise ArtifactValidationError(f"predictions_test chứa split khác test: {splits}")

    try:
        threshold = float(metrics["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"metrics_test thiếu threshold: {metrics_path}") from exc
    recomputed = compute_binary_metrics(frame["y_true"], frame["probability_crack"], threshold)
    _assert_metrics_match(metrics, recomputed, metrics_path)
    fixed_recomputed = compute_binary_metrics(frame["y_true"], frame["probability_crack"], 0.5)
    _assert_metrics_match(
        fixed_threshold_metrics,
        fixed_recomputed,
        fixed_threshold_metrics_path,
    )
    if metrics.get("threshold_role") not in (None, "validation_locked"):
        raise ArtifactValidationError("metrics_test.json phải có threshold_role=validation_locked")
    if fixed_threshold_metrics.get("threshold_role") not in (None, "fixed_protocol_0_5"):
        raise ArtifactValidationError(
            "metrics_test_threshold_0_5.json phải có threshold_role=fixed_protocol_0_5"
        )

    payloads = (metrics, metadata)
    checkpoint_hash = _require_sha256(
        _consistent_value(payloads, ("checkpoint_sha256", "model_sha256"), "checkpoint_sha256"),
        "checkpoint_sha256",
    )
    config_hash = _require_sha256(
        _consistent_value(payloads, ("config_sha256",), "config_sha256"),
        "config_sha256",
    )
    manifest_hash = _require_sha256(
        _consistent_value(payloads, ("manifest_sha256",), "manifest_sha256"),
        "manifest_sha256",
    )
    fixed_provenance = (
        ("checkpoint_sha256", checkpoint_hash),
        ("config_sha256", config_hash),
        ("manifest_sha256", manifest_hash),
    )
    for key, expected in fixed_provenance:
        if str(fixed_threshold_metrics.get(key, "")).strip().lower() != expected:
            raise ArtifactValidationError(
                f"metrics_test_threshold_0_5.{key} không khớp final evaluation"
            )
    experiment = str(
        _consistent_value(payloads, ("experiment", "experiment_id"), "experiment")
    ).strip()
    run_id = str(_consistent_value(payloads, ("run_id",), "run_id")).strip()
    if not experiment or not run_id:
        raise ArtifactValidationError("experiment/run_id không được trống")

    if str(selection_snapshot.get("selected_by", "")).strip().lower() != "validation":
        raise ArtifactValidationError("Final evaluation không được khóa bằng validation")
    contract_pairs = (
        ("checkpoint_sha256", checkpoint_hash),
        ("config_sha256", config_hash),
        ("manifest_sha256", manifest_hash),
    )
    for key, expected in contract_pairs:
        if str(selection_snapshot.get(key, "")).strip().lower() != expected:
            raise ArtifactValidationError(f"selection_contract_snapshot.{key} không khớp")
    if str(selection_snapshot.get("experiment", "")).strip() != experiment:
        raise ArtifactValidationError("selection contract experiment không khớp evaluation")
    if str(selection_snapshot.get("run_id", "")).strip() != run_id:
        raise ArtifactValidationError("selection contract run_id không khớp evaluation")
    if not math.isclose(
        float(selection_snapshot.get("threshold", math.nan)),
        threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ArtifactValidationError("selection contract threshold không khớp evaluation")

    artifact_hashes = completion.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ArtifactValidationError("evaluation_complete.json thiếu artifact_sha256")
    required_integrity_files = {
        metrics_path.name,
        fixed_threshold_metrics_path.name,
        predictions_path.name,
        metadata_path.name,
        selection_snapshot_path.name,
    }
    if not required_integrity_files.issubset(artifact_hashes):
        raise ArtifactValidationError("evaluation_complete.json thiếu hash artifact bắt buộc")
    for name, expected_value in artifact_hashes.items():
        candidate = (root / str(name)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactValidationError("evaluation_complete chứa path không an toàn") from exc
        expected_hash = _require_sha256(expected_value, f"artifact_sha256.{name}")
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise ArtifactValidationError(f"Final artifact hash không khớp: {candidate}")

    statuses = {
        str(metrics.get("status", "")).strip(),
        str(fixed_threshold_metrics.get("status", "")).strip(),
        str(metadata.get("status", "")).strip(),
        str(completion.get("status", "")).strip(),
    }
    if len(statuses) != 1:
        raise ArtifactValidationError(f"Final evaluation status không nhất quán: {statuses}")
    status = next(iter(statuses))
    valid = bool(
        metrics.get("valid_for_report") is True
        and fixed_threshold_metrics.get("valid_for_report") is True
        and metadata.get("valid_for_report") is True
        and status == "FINAL_TEST_COMPLETE"
        and not bool(metadata.get("smoke_test", False))
        and not _is_smoke_path(root)
    )
    return _EvaluationEvidence(
        directory=root,
        experiment=experiment,
        run_id=run_id,
        threshold=threshold,
        metrics=metrics,
        fixed_threshold_metrics=fixed_threshold_metrics,
        checkpoint_sha256=checkpoint_hash,
        config_sha256=config_hash,
        manifest_sha256=manifest_hash,
        metrics_path=metrics_path,
        fixed_threshold_metrics_path=fixed_threshold_metrics_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        selection_snapshot_path=selection_snapshot_path,
        completion_path=completion_path,
        valid_for_report=valid,
        status=status,
    )


def _comparison_row(
    evidence: _EvaluationEvidence,
    metrics: Mapping[str, Any],
    *,
    threshold_role: str,
    logical_experiment: str | None = None,
) -> dict[str, Any]:
    try:
        return {
            "experiment": logical_experiment or evidence.experiment,
            "selection_experiment": evidence.experiment,
            "run_id": evidence.run_id,
            "split": "test",
            "threshold_role": threshold_role,
            "threshold": float(metrics["threshold"]),
            "sample_count": int(metrics["sample_count"]),
            "accuracy": float(metrics["accuracy"]),
            "precision_crack": float(metrics["crack"]["precision"]),
            "recall_crack": float(metrics["crack"]["recall"]),
            "f1_crack": float(metrics["crack"]["f1"]),
            "macro_precision": float(metrics["macro"]["precision"]),
            "macro_recall": float(metrics["macro"]["recall"]),
            "macro_f1": float(metrics["macro"]["f1"]),
            "tn": int(metrics["tn"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tp": int(metrics["tp"]),
            "checkpoint_sha256": evidence.checkpoint_sha256,
            "config_sha256": evidence.config_sha256,
            "manifest_sha256": evidence.manifest_sha256,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Metric bắt buộc bị thiếu: {evidence.metrics_path}") from exc


def _experiment_sort_key(evidence: _EvaluationEvidence) -> tuple[int, str, float, str]:
    text = evidence.experiment.strip().upper()
    number = 999
    if text.startswith("E"):
        digits = "".join(character for character in text[1:] if character.isdigit())
        if digits:
            number = int(digits)
    return number, text, evidence.threshold, evidence.run_id


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return format(float(value), ".10g")
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _write_text_immutable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def _write_csv_immutable(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if frame.isna().any(axis=None):
        raise ArtifactValidationError(f"Từ chối CSV có ô trống: {path.name}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_manifest(path: Path) -> pd.DataFrame:
    return load_manifest_table(path)


def _dataset_summary(
    manifest_path: Path,
    *,
    expected_manifest_hash: str,
    official: bool,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    manifest = _load_manifest(manifest_path)
    required = {"relative_path", "label", "surface", "source_group", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ArtifactValidationError(f"Manifest thiếu cột: {sorted(missing)}")
    actual_hash = manifest_sha256(manifest)
    if actual_hash != expected_manifest_hash:
        raise ArtifactValidationError(
            "Manifest dùng sinh dataset summary không khớp final evaluation"
        )
    split_audit = audit_split(manifest)
    if official and not split_audit.get("valid"):
        raise ArtifactValidationError(
            f"Manifest report không pass split audit: {split_audit.get('errors')}"
        )
    labels = pd.to_numeric(manifest["label"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ArtifactValidationError("Manifest chỉ được có label 0/1")
    work = manifest.assign(label=labels)
    summary = (
        work.groupby(["split", "surface", "label"], dropna=False, sort=True)
        .agg(image_count=("relative_path", "size"), source_group_count=("source_group", "nunique"))
        .reset_index()
    )
    summary.insert(
        3,
        "label_name",
        summary["label"].map({0: "Non-crack", 1: "Crack"}),
    )
    if summary.isna().any(axis=None):
        raise ArtifactValidationError("Dataset summary có giá trị trống")
    payload = {
        "manifest_sha256": actual_hash,
        "manifest_file_sha256": sha256_file(manifest_path),
        "row_count": len(work),
        "source_group_count": int(work["source_group"].nunique()),
        "split_audit": split_audit,
        "distribution": summary.to_dict(orient="records"),
    }
    return summary, payload, manifest


def _verify_evaluations_against_manifest(
    evidence: Sequence[_EvaluationEvidence], manifest: pd.DataFrame
) -> None:
    split = manifest["split"].astype(str).str.strip().str.lower()
    test_rows = manifest.loc[split.eq("test")].copy()
    if test_rows.empty:
        raise ArtifactValidationError("Manifest không có test rows")
    if test_rows["relative_path"].astype(str).duplicated().any():
        raise ArtifactValidationError("Manifest test có relative_path trùng")
    expected_labels = test_rows.set_index(test_rows["relative_path"].astype(str))["label"]
    expected_paths = set(expected_labels.index)
    expected_sha: pd.Series | None = None
    if "sha256" in test_rows.columns:
        expected_sha = test_rows.set_index(test_rows["relative_path"].astype(str))["sha256"]

    for item in evidence:
        predictions = pd.read_csv(item.predictions_path)
        if "relative_path" not in predictions.columns:
            raise ArtifactValidationError(
                f"predictions_test thiếu relative_path: {item.predictions_path}"
            )
        paths = predictions["relative_path"].astype(str)
        if paths.duplicated().any() or set(paths) != expected_paths:
            raise ArtifactValidationError(
                f"predictions_test không phủ đúng test manifest: {item.predictions_path}"
            )
        observed_labels = pd.to_numeric(predictions["y_true"], errors="raise").astype(int)
        aligned_labels = paths.map(expected_labels).astype(int)
        if not observed_labels.reset_index(drop=True).equals(aligned_labels.reset_index(drop=True)):
            raise ArtifactValidationError(
                f"y_true trong predictions không khớp manifest: {item.predictions_path}"
            )
        if expected_sha is not None:
            if "sha256" not in predictions.columns:
                raise ArtifactValidationError(
                    f"predictions_test thiếu SHA-256 manifest: {item.predictions_path}"
                )
            observed_sha = predictions["sha256"].astype(str).str.lower()
            aligned_sha = paths.map(expected_sha).astype(str).str.lower()
            if not observed_sha.reset_index(drop=True).equals(aligned_sha.reset_index(drop=True)):
                raise ArtifactValidationError(
                    f"SHA-256 trong predictions không khớp manifest: {item.predictions_path}"
                )


def _load_training_evidence(directory: str | Path) -> _TrainingEvidence:
    root = Path(directory).resolve()
    config_path = root / "config_snapshot.json"
    checkpoint_path = root / "model.keras"
    history_path = root / "history.csv"
    summary_path = root / "run_summary.json"
    completion_path = root / "training_complete.json"
    config = _read_json(config_path)
    summary = _read_json(summary_path)
    completion = _read_json(completion_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not history_path.is_file():
        raise FileNotFoundError(history_path)

    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, dict):
        raise ArtifactValidationError(f"config snapshot thiếu experiment: {config_path}")
    experiment = str(experiment_config.get("id", "")).strip().upper()
    if experiment not in {"E1", "E2", "E3", "E4"}:
        raise ArtifactValidationError(f"Training evidence không thuộc E1-E4: {root}")
    summary_experiment = str(summary.get("experiment", "")).strip().upper()
    if summary_experiment != experiment:
        raise ArtifactValidationError(f"run_summary experiment không khớp config: {root}")
    run_id = str(summary.get("run_id", "")).strip()
    if not run_id or str(completion.get("run_id", "")).strip() != run_id:
        raise ArtifactValidationError(f"training run_id không nhất quán: {root}")

    config_hash = _require_sha256(summary.get("config_sha256"), "config_sha256")
    checkpoint_hash = _require_sha256(summary.get("model_sha256"), "model_sha256")
    manifest_hash = _require_sha256(summary.get("manifest_sha256"), "manifest_sha256")
    if sha256_json(config) != config_hash:
        raise ArtifactValidationError(f"config_snapshot hash không khớp: {root}")
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise ArtifactValidationError(f"training checkpoint hash không khớp: {root}")
    for field, expected in (
        ("config_sha256", config_hash),
        ("model_sha256", checkpoint_hash),
        ("manifest_sha256", manifest_hash),
    ):
        if str(completion.get(field, "")).strip().lower() != expected:
            raise ArtifactValidationError(f"training_complete.{field} không khớp: {root}")

    artifact_hashes = completion.get("artifact_sha256")
    required = {
        config_path.name,
        checkpoint_path.name,
        history_path.name,
        summary_path.name,
    }
    if not isinstance(artifact_hashes, dict) or not required.issubset(artifact_hashes):
        raise ArtifactValidationError(f"training_complete thiếu hash artifact bắt buộc: {root}")
    for name, expected_value in artifact_hashes.items():
        candidate = (root / str(name)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactValidationError("training_complete chứa path không an toàn") from exc
        expected = _require_sha256(expected_value, f"artifact_sha256.{name}")
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ArtifactValidationError(f"Training artifact hash không khớp: {candidate}")

    history = pd.read_csv(history_path)
    required_history = {"loss", "val_loss"}
    missing = required_history.difference(history.columns)
    if missing or history.empty:
        raise ArtifactValidationError(
            f"history.csv rỗng hoặc thiếu cột {sorted(missing)}: {history_path}"
        )
    metric_columns = ["loss", "val_loss"]
    if "accuracy" in history.columns or "val_accuracy" in history.columns:
        if not {"accuracy", "val_accuracy"}.issubset(history.columns):
            raise ArtifactValidationError("history phải có đồng thời accuracy và val_accuracy")
        metric_columns.extend(("accuracy", "val_accuracy"))
    numeric = history.loc[:, metric_columns].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ArtifactValidationError(f"history có metric không hữu hạn: {history_path}")
    if (numeric[["loss", "val_loss"]] < 0).any(axis=None):
        raise ArtifactValidationError(f"history có loss âm: {history_path}")
    if {"accuracy", "val_accuracy"}.issubset(numeric.columns) and (
        (numeric[["accuracy", "val_accuracy"]] < 0).any(axis=None)
        or (numeric[["accuracy", "val_accuracy"]] > 1).any(axis=None)
    ):
        raise ArtifactValidationError(f"history có accuracy ngoài [0,1]: {history_path}")

    expected_status = "VALIDATION_COMPLETE_TEST_LOCKED"
    valid = bool(
        summary.get("valid_for_report") is True
        and str(summary.get("status", "")) == expected_status
        and str(completion.get("status", "")) == expected_status
        and not _is_smoke_path(root)
    )
    return _TrainingEvidence(
        directory=root,
        experiment=experiment,
        run_id=run_id,
        checkpoint_sha256=checkpoint_hash,
        config_sha256=config_hash,
        manifest_sha256=manifest_hash,
        history=history,
        history_path=history_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        completion_path=completion_path,
        valid_for_report=valid,
    )


def _load_locked_manifest_bundle(
    *,
    manifest_path: Path,
    bundle_dir: Path,
    expected_manifest_hash: str,
    official: bool,
) -> _LockedManifestBundle:
    directory = bundle_dir.resolve()
    locked_manifest = (directory / "manifest.csv").resolve()
    if manifest_path.resolve() != locked_manifest:
        raise ArtifactValidationError("--manifest phải là manifest.csv trong split bundle đã khóa")
    del official  # Locked bundles always use the official split protocol.
    try:
        verified = verify_locked_split_bundle(directory)
    except (SplitValidationError, FileNotFoundError, OSError, ValueError) as exc:
        raise ArtifactValidationError(f"Split bundle không hợp lệ: {exc}") from exc
    canonical_hash = verified.manifest_sha256
    if canonical_hash != expected_manifest_hash:
        raise ArtifactValidationError("Split bundle không khớp manifest của final evaluation")

    return _LockedManifestBundle(
        directory=directory,
        manifest_path=locked_manifest,
        manifest_hashes_path=verified.inventory_path,
        split_audit_path=verified.audit_path,
        canonical_sha256=canonical_hash,
        input_manifest_path=directory / CURATED_MANIFEST_SNAPSHOT_FILENAME,
        curation_report_path=directory / CONFLICT_REPORT_SNAPSHOT_FILENAME,
    )


def _verify_png(path: Path, *, field: str) -> None:
    if path.suffix.casefold() != ".png" or not path.is_file() or path.stat().st_size <= 8:
        raise ArtifactValidationError(f"{field} phải là PNG tồn tại và không rỗng")
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise ArtifactValidationError(f"{field} không phải PNG hợp lệ")
            image.verify()
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"{field} không decode được: {path}") from exc


def _load_qualitative_sidecar(
    path: str | Path,
    *,
    expected_kind: str,
    official: bool,
) -> _QualitativeEvidence:
    sidecar = Path(path).resolve()
    payload = _read_json(sidecar)
    if str(payload.get("kind", "")) != expected_kind:
        raise ArtifactValidationError(f"Qualitative sidecar sai kind: {sidecar}")
    output_value = str(payload.get("output", "")).strip()
    if not output_value:
        raise ArtifactValidationError(f"Qualitative sidecar thiếu output: {sidecar}")
    image_path = Path(output_value).resolve()
    if image_path != sidecar.with_suffix(".png").resolve():
        raise ArtifactValidationError("Qualitative sidecar/output PNG phải cùng basename")
    _verify_png(image_path, field=expected_kind)
    expected_hash = _require_sha256(payload.get("output_sha256"), "output_sha256")
    if sha256_file(image_path) != expected_hash:
        raise ArtifactValidationError(f"Qualitative PNG hash không khớp sidecar: {image_path}")
    if official and (
        payload.get("valid_for_report") is not True
        or str(payload.get("status", "")) != "REPORT_ARTIFACT"
        or _is_smoke_path(sidecar)
        or _is_smoke_path(image_path)
    ):
        raise ArtifactValidationError("Từ chối qualitative smoke/chưa report-valid")
    return _QualitativeEvidence(
        sidecar_path=sidecar,
        image_path=image_path,
        payload=payload,
    )


def _load_threshold_evidence(
    path: str | Path,
    *,
    predictions_path: Path,
    e5: _EvaluationEvidence,
    e4_training: _TrainingEvidence | None,
    official: bool,
) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = _read_json(source)
    kind = str(payload.get("kind", payload.get("artifact_type", ""))).strip()
    if kind not in {"validation_threshold_selection", "validation_threshold_tuning"}:
        raise ArtifactValidationError("Threshold artifact thiếu kind validation rõ ràng")
    split = str(payload.get("source_split", "")).strip().casefold()
    if split not in {"val", "validation"}:
        raise ArtifactValidationError("Threshold artifact phải có source_split=validation")
    threshold_value = payload.get("threshold", payload.get("selected_threshold"))
    try:
        threshold = float(threshold_value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("Threshold artifact thiếu threshold hợp lệ") from exc
    if not math.isclose(threshold, e5.threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ArtifactValidationError("Threshold artifact không khớp E5 final evaluation")
    predictions_hash = _require_sha256(payload.get("predictions_sha256"), "predictions_sha256")
    if sha256_file(predictions_path) != predictions_hash:
        raise ArtifactValidationError("Threshold artifact không khớp validation predictions")
    validation_frame = pd.read_csv(predictions_path)
    if int(payload.get("sample_count", -1)) != len(validation_frame):
        raise ArtifactValidationError("Threshold artifact sample_count không khớp predictions")
    for field, expected in (
        ("checkpoint_sha256", e5.checkpoint_sha256),
        ("config_sha256", e5.config_sha256),
        ("manifest_sha256", e5.manifest_sha256),
        ("run_id", e5.run_id),
    ):
        if str(payload.get(field, "")).strip().lower() != str(expected).lower():
            raise ArtifactValidationError(f"Threshold artifact provenance không khớp {field}")
    if e4_training is not None and (
        e4_training.checkpoint_sha256 != e5.checkpoint_sha256
        or e4_training.config_sha256 != e5.config_sha256
        or e4_training.run_id != e5.run_id
    ):
        raise ArtifactValidationError("E5 không dùng đúng checkpoint/config/run của E4")
    if e4_training is not None and _require_sha256(
        payload.get("training_complete_sha256"), "training_complete_sha256"
    ) != sha256_file(e4_training.completion_path):
        raise ArtifactValidationError("Threshold artifact không khớp training_complete E4")
    if e4_training is not None:
        training_completion = _read_json(e4_training.completion_path)
        training_hashes = training_completion.get("artifact_sha256")
        if (
            not isinstance(training_hashes, dict)
            or _require_sha256(
                training_hashes.get("predictions_validation.csv"),
                "training_complete.predictions_validation.csv",
            )
            != predictions_hash
        ):
            raise ArtifactValidationError("Validation predictions không khớp training_complete E4")
    selection = _read_json(e5.selection_snapshot_path)
    if str(selection.get("trained_experiment", "")).strip().upper() != "E4":
        raise ArtifactValidationError("E5 selection phải khai báo trained_experiment=E4")
    if str(selection.get("threshold_source", "")).strip().casefold() != "validation":
        raise ArtifactValidationError("E5 selection phải có threshold_source=validation")
    recorded_threshold_hash = selection.get("threshold_result_sha256")
    if recorded_threshold_hash is not None and _require_sha256(
        recorded_threshold_hash, "selection.threshold_result_sha256"
    ) != sha256_file(source):
        raise ArtifactValidationError("E5 selection không khớp threshold result")
    if official and (
        payload.get("valid_for_report") is not True
        or _is_not_valid(payload)
        or _is_smoke_path(source)
        or str(payload.get("status", ""))
        not in {
            "VALIDATION_THRESHOLD_SELECTED",
            "VALIDATION_THRESHOLD_LOCKED",
            "THRESHOLD_SELECTION_COMPLETE",
        }
    ):
        raise ArtifactValidationError("Threshold artifact chưa đủ điều kiện report-valid")
    result = dict(payload)
    result["artifact_path"] = str(source)
    result["artifact_sha256"] = sha256_file(source)
    return result


def _validate_augmentation_binding(
    evidence: _QualitativeEvidence,
    *,
    e4: _TrainingEvidence,
    manifest_path: Path,
) -> None:
    payload = evidence.payload
    expected = (
        ("experiment", "E4"),
        ("run_id", e4.run_id),
        ("config_sha256", e4.config_sha256),
        ("manifest_sha256", e4.manifest_sha256),
    )
    for field, value in expected:
        if str(payload.get(field, "")).strip().lower() != str(value).lower():
            raise ArtifactValidationError(f"Augmentation sidecar không khớp E4 {field}")
    if _require_sha256(payload.get("manifest_file_sha256"), "manifest_file_sha256") != sha256_file(
        manifest_path
    ):
        raise ArtifactValidationError("Augmentation sidecar không khớp file manifest")
    config_path = Path(str(payload.get("config_path", ""))).resolve()
    if config_path != e4.config_path or _require_sha256(
        payload.get("config_file_sha256"), "config_file_sha256"
    ) != sha256_file(e4.config_path):
        raise ArtifactValidationError("Augmentation sidecar không khớp config snapshot E4")
    split_audit = payload.get("split_audit")
    if not isinstance(split_audit, dict) or split_audit.get("valid") is not True:
        raise ArtifactValidationError("Augmentation sidecar thiếu split audit hợp lệ")
    augmentation = payload.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ArtifactValidationError("Augmentation sidecar thiếu tham số augmentation")
    required_values = {
        "horizontal_flip_probability": 0.5,
        "max_rotation_degrees": 15.0,
        "brightness_delta": 0.15,
        "contrast_delta": 0.15,
    }
    for field, expected_value in required_values.items():
        if not math.isclose(
            float(augmentation.get(field, math.nan)),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactValidationError(f"Augmentation sidecar sai tham số {field}")


def _validate_gradcam_binding(
    evidence: _QualitativeEvidence,
    *,
    e5: _EvaluationEvidence,
) -> None:
    payload = evidence.payload
    expected = (
        ("experiment", "E5"),
        ("run_id", e5.run_id),
        ("model_sha256", e5.checkpoint_sha256),
        ("config_sha256", e5.config_sha256),
        ("manifest_sha256", e5.manifest_sha256),
        ("predictions_sha256", sha256_file(e5.predictions_path)),
    )
    for field, value in expected:
        if str(payload.get(field, "")).strip().lower() != str(value).lower():
            raise ArtifactValidationError(f"Grad-CAM sidecar không khớp E5 {field}")
    if not math.isclose(
        float(payload.get("threshold", math.nan)),
        e5.threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ArtifactValidationError("Grad-CAM sidecar không khớp threshold E5")
    predictions_path = Path(str(payload.get("predictions_path", ""))).resolve()
    if predictions_path != e5.predictions_path:
        raise ArtifactValidationError("Grad-CAM sidecar không trỏ đúng predictions E5")
    selection_path = Path(str(payload.get("selection_path", ""))).resolve()
    selection_hash = _require_sha256(payload.get("selection_sha256"), "selection_sha256")
    if not selection_path.is_file() or sha256_file(selection_path) != selection_hash:
        raise ArtifactValidationError("Grad-CAM selection contract hash không khớp")
    selection = _read_json(selection_path)
    for field, expected_value in (
        ("experiment", "E5"),
        ("run_id", e5.run_id),
        ("checkpoint_sha256", e5.checkpoint_sha256),
        ("config_sha256", e5.config_sha256),
        ("manifest_sha256", e5.manifest_sha256),
    ):
        if str(selection.get(field, "")).strip().lower() != str(expected_value).lower():
            raise ArtifactValidationError(f"Grad-CAM selection không khớp {field}")
    if not math.isclose(
        float(selection.get("threshold", math.nan)),
        e5.threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ArtifactValidationError("Grad-CAM selection không khớp threshold")
    selected = payload.get("selected")
    if not isinstance(selected, dict) or set(selected) != {"TP", "TN", "FP", "FN"}:
        raise ArtifactValidationError("Grad-CAM sidecar phải khai báo đủ TP/TN/FP/FN")


def _threshold_curve(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "test" in path.name.casefold():
        raise ArtifactValidationError("Từ chối tune/vẽ threshold từ predictions test")
    frame = pd.read_csv(path)
    required = {"y_true", "probability_crack"}
    missing = required.difference(frame.columns)
    if missing:
        raise ArtifactValidationError(f"Validation predictions thiếu cột: {sorted(missing)}")
    if "split" in frame.columns:
        splits = set(frame["split"].dropna().astype(str).str.strip().str.lower())
        if not splits.issubset({"val", "validation"}):
            raise ArtifactValidationError("Threshold curve chỉ được dùng validation")

    truth = pd.to_numeric(frame["y_true"], errors="raise").to_numpy(dtype=np.int64)
    scores = pd.to_numeric(frame["probability_crack"], errors="raise").to_numpy(dtype=np.float64)
    if truth.size == 0 or truth.shape != scores.shape:
        raise ArtifactValidationError("Validation predictions rỗng hoặc khác shape")
    if not np.isin(truth, [0, 1]).all():
        raise ArtifactValidationError("y_true validation chỉ được 0/1")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ArtifactValidationError("probability_crack validation phải trong [0,1]")

    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    order = np.argsort(scores, kind="stable")
    ordered_truth = truth[order]
    ordered_scores = scores[order]
    positive_prefix = np.concatenate(([0], np.cumsum(ordered_truth == 1)))
    negative_prefix = np.concatenate(([0], np.cumsum(ordered_truth == 0)))
    indices = np.searchsorted(ordered_scores, thresholds, side="left")
    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))
    tp = positives - positive_prefix[indices]
    fp = negatives - negative_prefix[indices]
    fn = positives - tp
    tn = negatives - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )
    accuracy = (tp + tn) / truth.size
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
    )
    ranked = max(
        curve.itertuples(index=False),
        key=lambda row: (
            row.f1,
            row.recall,
            -abs(row.threshold - 0.5),
            -row.threshold,
        ),
    )
    result = {
        "source_split": "validation",
        "prediction_count": int(truth.size),
        "candidate_count": len(curve),
        "selected_threshold": float(ranked.threshold),
        "f1_crack": float(ranked.f1),
        "recall_crack": float(ranked.recall),
        "precision_crack": float(ranked.precision),
        "accuracy": float(ranked.accuracy),
        "predictions_sha256": sha256_file(path),
    }
    return curve, result


def _load_benchmark(path: Path, *, official: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("kind") != "inference_latency_benchmark":
        raise ArtifactValidationError(f"Không phải benchmark CrackSpot: {path}")
    if official and (_is_not_valid(payload) or _is_smoke_path(path)):
        raise ArtifactValidationError(f"Từ chối smoke benchmark trong report: {path}")
    latency = payload.get("latency_ms")
    target = payload.get("target")
    if not isinstance(latency, dict) or not isinstance(target, dict):
        raise ArtifactValidationError(f"Benchmark thiếu latency/target: {path}")
    raw = latency.get("raw")
    if not isinstance(raw, list) or len(raw) != int(payload.get("measured_runs", -1)):
        raise ArtifactValidationError(f"Benchmark raw timings không khớp measured_runs: {path}")
    values = np.asarray(raw, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ArtifactValidationError(f"Benchmark timings không hợp lệ: {path}")
    expected = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }
    for key, value in expected.items():
        if not math.isclose(float(latency.get(key, math.nan)), value, rel_tol=1e-9, abs_tol=1e-9):
            raise ArtifactValidationError(f"Benchmark summary {key} không khớp raw timings")
    checkpoint_hash = _require_sha256(payload.get("checkpoint_sha256"), "checkpoint_sha256")
    manifest_hash = _require_sha256(payload.get("manifest_sha256"), "manifest_sha256")
    row = {
        "run_id": str(payload.get("run_id", "")).strip(),
        "model_version": str(payload.get("model_version", "")).strip(),
        "warmup_runs": int(payload["warmup_runs"]),
        "measured_runs": int(payload["measured_runs"]),
        "mean_ms": expected["mean"],
        "median_ms": expected["median"],
        "p50_ms": expected["p50"],
        "p95_ms": expected["p95"],
        "target_ms": float(target["milliseconds_per_image"]),
        "meets_target_p95": bool(target["meets_target"]),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
    }
    if any(value == "" for value in row.values()):
        raise ArtifactValidationError(f"Benchmark có trường provenance trống: {path}")
    fact = dict(payload)
    fact["artifact_sha256"] = sha256_file(path)
    return row, fact


def _load_real_image_evaluation(
    directory: str | Path,
    *,
    official: bool,
    expected_manifest_hash: str,
    allowed_provenance: set[tuple[str, str, str, str]],
) -> _RealImageEvidence:
    root = Path(directory).resolve()
    metrics_path = root / "metrics_real.json"
    predictions_path = root / "predictions_real.csv"
    metadata_path = root / "evaluation_metadata.json"
    selection_snapshot_path = root / "selection_contract_snapshot.json"
    completion_path = root / "evaluation_complete.json"
    metrics = _read_json(metrics_path)
    metadata = _read_json(metadata_path)
    selection = _read_json(selection_snapshot_path)
    completion = _read_json(completion_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    frame = pd.read_csv(predictions_path)
    required = {
        "relative_path",
        "y_true",
        "probability_crack",
        "threshold",
        "split",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ArtifactValidationError(f"predictions_real thiếu cột: {sorted(missing)}")
    if frame.empty or frame["relative_path"].astype(str).duplicated().any():
        raise ArtifactValidationError("predictions_real rỗng hoặc trùng relative_path")
    splits = set(frame["split"].dropna().astype(str).str.strip().str.casefold())
    if splits != {"real_external"}:
        raise ArtifactValidationError(f"predictions_real có split không hợp lệ: {splits}")
    threshold = float(metrics.get("threshold", math.nan))
    observed_thresholds = pd.to_numeric(frame["threshold"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not math.isfinite(threshold) or not np.allclose(
        observed_thresholds, threshold, rtol=0.0, atol=1e-12
    ):
        raise ArtifactValidationError("Threshold ảnh thực tế không nhất quán")
    recomputed = compute_binary_metrics(frame["y_true"], frame["probability_crack"], threshold)
    _assert_metrics_match(metrics, recomputed, metrics_path)

    checkpoint_hash = _require_sha256(
        _consistent_value((metrics, metadata), ("checkpoint_sha256",), "checkpoint_sha256"),
        "checkpoint_sha256",
    )
    config_hash = _require_sha256(
        _consistent_value((metrics, metadata), ("config_sha256",), "config_sha256"),
        "config_sha256",
    )
    manifest_hash = _require_sha256(
        _consistent_value(
            (metrics, metadata),
            ("sdnet_manifest_sha256",),
            "sdnet_manifest_sha256",
        ),
        "sdnet_manifest_sha256",
    )
    experiment = str(_consistent_value((metrics, metadata), ("experiment",), "experiment")).strip()
    run_id = str(_consistent_value((metrics, metadata), ("run_id",), "run_id")).strip()
    if manifest_hash != expected_manifest_hash:
        raise ArtifactValidationError("Real-image evaluation không cùng manifest SDNET đã khóa")
    if (checkpoint_hash, config_hash, experiment, run_id) not in allowed_provenance:
        raise ArtifactValidationError("Real-image evaluation không khớp checkpoint final")
    if str(selection.get("selected_by", "")).strip().casefold() != "validation":
        raise ArtifactValidationError("Threshold ảnh thực tế không được khóa bằng validation")
    for field, expected in (
        ("checkpoint_sha256", checkpoint_hash),
        ("config_sha256", config_hash),
        ("manifest_sha256", manifest_hash),
        ("experiment", experiment),
        ("run_id", run_id),
    ):
        if str(selection.get(field, "")).strip() != expected:
            raise ArtifactValidationError(f"selection_contract_snapshot.{field} không khớp")
    if not math.isclose(
        float(selection.get("threshold", math.nan)), threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ArtifactValidationError("Threshold selection/real metrics không khớp")

    artifact_hashes = completion.get("artifact_sha256")
    required_integrity = {
        metrics_path.name,
        predictions_path.name,
        metadata_path.name,
        selection_snapshot_path.name,
    }
    if not isinstance(artifact_hashes, dict) or not required_integrity.issubset(artifact_hashes):
        raise ArtifactValidationError("Real-image completion thiếu hash artifact bắt buộc")
    for name, expected_value in artifact_hashes.items():
        candidate = (root / str(name)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactValidationError("Real-image completion chứa path không an toàn") from exc
        expected = _require_sha256(expected_value, f"artifact_sha256.{name}")
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ArtifactValidationError(f"Real-image artifact hash không khớp: {candidate}")

    statuses = {
        str(metrics.get("status", "")).strip(),
        str(metadata.get("status", "")).strip(),
        str(completion.get("status", "")).strip(),
    }
    if len(statuses) != 1:
        raise ArtifactValidationError(f"Real-image status không nhất quán: {statuses}")
    status = next(iter(statuses))
    valid = bool(
        metrics.get("valid_for_report") is True
        and metadata.get("valid_for_report") is True
        and completion.get("valid_for_report") is True
        and metrics.get("included_in_standard_test") is False
        and metadata.get("included_in_standard_test") is False
        and metrics.get("self_captured_confirmed") is True
        and metadata.get("self_captured_confirmed") is True
        and metrics.get("evaluation_scope") == "external_self_captured_images"
        and status == "REAL_IMAGE_EVALUATION_COMPLETE"
        and not _is_smoke_path(root)
    )
    if official and not valid:
        raise ArtifactValidationError(
            "Từ chối ảnh thực tế smoke/chưa xác nhận self_captured trong report"
        )
    return _RealImageEvidence(
        directory=root,
        metrics=metrics,
        status=status,
        valid_for_report=valid,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        selection_snapshot_path=selection_snapshot_path,
        completion_path=completion_path,
    )


def generate_report_assets(
    *,
    evaluation_dirs: Iterable[str | Path],
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    split_bundle_dir: str | Path | None = None,
    validation_predictions: str | Path | None = None,
    threshold_result_path: str | Path | None = None,
    benchmark_paths: Iterable[str | Path] = (),
    real_evaluation_dirs: Iterable[str | Path] = (),
    training_run_dirs: Iterable[str | Path] = (),
    augmentation_sidecar: str | Path | None = None,
    gradcam_sidecar: str | Path | None = None,
    project_root: str | Path = ".",
) -> Path:
    """Generate an atomic report bundle; official output is fail-closed."""

    target = Path(output_dir).resolve()
    official = is_official_report_path(target)
    if target.exists():
        raise FileExistsError(f"Không ghi đè report bundle bất biến: {target}")
    root = Path(project_root).resolve()

    evidence = [_load_evaluation(path) for path in evaluation_dirs]
    if not evidence:
        raise ArtifactValidationError("Cần ít nhất một final evaluation artifact")
    evidence.sort(key=_experiment_sort_key)
    if official:
        invalid = [item.directory for item in evidence if not item.valid_for_report]
        if invalid:
            raise ArtifactValidationError(
                "Từ chối smoke/NOT_VALID_FOR_REPORT trong artifacts/report: "
                + ", ".join(str(path) for path in invalid)
            )
    manifest_hashes = {item.manifest_sha256 for item in evidence}
    if len(manifest_hashes) != 1:
        raise ArtifactValidationError("Mọi final evaluation trong comparison phải cùng manifest")
    expected_manifest_hash = next(iter(manifest_hashes))
    evaluation_by_experiment: dict[str, _EvaluationEvidence] = {}
    for item in evidence:
        experiment = item.experiment.strip().upper()
        if experiment in evaluation_by_experiment:
            raise ArtifactValidationError(f"Có nhiều final evaluation cho {experiment}")
        evaluation_by_experiment[experiment] = item
    if official:
        unexpected = set(evaluation_by_experiment).difference({"E1", "E2", "E3", "E5"})
        if unexpected:
            raise ArtifactValidationError(
                f"Official comparison chỉ nhận E1,E2,E3,E5; nhận thêm {sorted(unexpected)}"
            )

    training = [_load_training_evidence(path) for path in training_run_dirs]
    training_by_experiment: dict[str, _TrainingEvidence] = {}
    for item in training:
        if item.experiment in training_by_experiment:
            raise ArtifactValidationError(f"Có nhiều training run cho {item.experiment}")
        training_by_experiment[item.experiment] = item
        if item.manifest_sha256 != expected_manifest_hash:
            raise ArtifactValidationError(f"Training {item.experiment} không cùng manifest final")
        if official and not item.valid_for_report:
            raise ArtifactValidationError(f"Training {item.experiment} không report-valid")

    comparison_entries: list[tuple[_EvaluationEvidence, dict[str, Any], Path, str, str]] = []
    for experiment in ("E1", "E2", "E3"):
        item = evaluation_by_experiment.get(experiment)
        if item is not None:
            comparison_entries.append(
                (
                    item,
                    item.fixed_threshold_metrics,
                    item.fixed_threshold_metrics_path,
                    "fixed_protocol_0_5",
                    experiment,
                )
            )
    e5 = evaluation_by_experiment.get("E5")
    if e5 is not None:
        comparison_entries.extend(
            (
                (
                    e5,
                    e5.fixed_threshold_metrics,
                    e5.fixed_threshold_metrics_path,
                    "fixed_protocol_0_5",
                    "E4",
                ),
                (e5, e5.metrics, e5.metrics_path, "validation_locked", "E5"),
            )
        )
    for experiment, item in sorted(evaluation_by_experiment.items()):
        if experiment not in {"E1", "E2", "E3", "E5"}:
            comparison_entries.append(
                (item, item.metrics, item.metrics_path, "provided", experiment)
            )
    comparison = pd.DataFrame(
        [
            _comparison_row(
                item,
                metrics,
                threshold_role=role,
                logical_experiment=logical_experiment,
            )
            for item, metrics, _, role, logical_experiment in comparison_entries
        ]
    )

    dataset_frame: pd.DataFrame | None = None
    dataset_fact: dict[str, Any] | None = None
    locked_manifest: pd.DataFrame | None = None
    manifest_source = Path(manifest_path).resolve() if manifest_path is not None else None
    manifest_bundle: _LockedManifestBundle | None = None
    if manifest_source is not None:
        dataset_frame, dataset_fact, locked_manifest = _dataset_summary(
            manifest_source,
            expected_manifest_hash=expected_manifest_hash,
            official=official,
        )
        _verify_evaluations_against_manifest(evidence, locked_manifest)
    if split_bundle_dir is not None:
        if manifest_source is None:
            raise ArtifactValidationError("--split-bundle-dir bắt buộc đi cùng --manifest")
        manifest_bundle = _load_locked_manifest_bundle(
            manifest_path=manifest_source,
            bundle_dir=Path(split_bundle_dir),
            expected_manifest_hash=expected_manifest_hash,
            official=official,
        )

    curve_frame: pd.DataFrame | None = None
    threshold_fact: dict[str, Any] | None = None
    threshold_evidence: dict[str, Any] | None = None
    validation_source = (
        Path(validation_predictions).resolve() if validation_predictions is not None else None
    )
    if validation_source is not None:
        curve_frame, threshold_fact = _threshold_curve(validation_source)
        if e5 is not None and not math.isclose(
            e5.threshold,
            float(threshold_fact["selected_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactValidationError(
                "E5 final evaluation không dùng threshold tối ưu từ validation predictions"
            )
    if threshold_result_path is not None:
        if validation_source is None or e5 is None:
            raise ArtifactValidationError(
                "Threshold artifact bắt buộc có validation predictions và E5 evaluation"
            )
        threshold_evidence = _load_threshold_evidence(
            threshold_result_path,
            predictions_path=validation_source,
            e5=e5,
            e4_training=training_by_experiment.get("E4"),
            official=official,
        )
        if threshold_fact is not None and not math.isclose(
            float(
                threshold_evidence.get("threshold", threshold_evidence.get("selected_threshold"))
            ),
            float(threshold_fact["selected_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactValidationError("Threshold artifact không khớp threshold recompute")

    benchmarks: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for source_value in benchmark_paths:
        source = Path(source_value).resolve()
        row, fact = _load_benchmark(source, official=official)
        if row["manifest_sha256"] != expected_manifest_hash:
            raise ArtifactValidationError(f"Benchmark không cùng manifest: {source}")
        if official and (e5 is None or row["checkpoint_sha256"] != e5.checkpoint_sha256):
            raise ArtifactValidationError(f"Benchmark không khớp checkpoint E5: {source}")
        benchmark_rows.append(row)
        benchmarks.append(fact)
    benchmark_frame = pd.DataFrame(benchmark_rows) if benchmark_rows else None

    allowed_real_provenance = (
        {(e5.checkpoint_sha256, e5.config_sha256, e5.experiment, e5.run_id)}
        if official and e5 is not None
        else {
            (item.checkpoint_sha256, item.config_sha256, item.experiment, item.run_id)
            for item in evidence
        }
    )
    real_evidence = [
        _load_real_image_evaluation(
            source,
            official=official,
            expected_manifest_hash=expected_manifest_hash,
            allowed_provenance=allowed_real_provenance,
        )
        for source in real_evaluation_dirs
    ]

    augmentation_evidence: _QualitativeEvidence | None = None
    if augmentation_sidecar is not None:
        augmentation_evidence = _load_qualitative_sidecar(
            augmentation_sidecar,
            expected_kind="train_augmentation_before_after",
            official=official,
        )
        e4_training = training_by_experiment.get("E4")
        if e4_training is None or manifest_source is None:
            raise ArtifactValidationError(
                "Augmentation sidecar bắt buộc có training E4 và locked manifest"
            )
        _validate_augmentation_binding(
            augmentation_evidence,
            e4=e4_training,
            manifest_path=manifest_source,
        )
    gradcam_evidence: _QualitativeEvidence | None = None
    if gradcam_sidecar is not None:
        gradcam_evidence = _load_qualitative_sidecar(
            gradcam_sidecar,
            expected_kind="gradcam_tp_tn_fp_fn_grid",
            official=official,
        )
        if e5 is None:
            raise ArtifactValidationError("Grad-CAM sidecar bắt buộc có E5 evaluation")
        _validate_gradcam_binding(gradcam_evidence, e5=e5)

    missing: list[str] = []
    for experiment in ("E1", "E2", "E3", "E5"):
        if experiment not in evaluation_by_experiment:
            missing.append(f"official_evaluation:{experiment}")
    for experiment in ("E1", "E2", "E3"):
        item = evaluation_by_experiment.get(experiment)
        if item is not None and not math.isclose(item.threshold, 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise ArtifactValidationError(
                f"{experiment} official evaluation bắt buộc threshold=0.5"
            )
    for experiment in ("E1", "E2", "E3", "E4"):
        if experiment not in training_by_experiment:
            missing.append(f"training_curve:{experiment}")
    for experiment in ("E1", "E2", "E3"):
        evaluation_item = evaluation_by_experiment.get(experiment)
        training_item = training_by_experiment.get(experiment)
        if (
            evaluation_item is not None
            and training_item is not None
            and (
                evaluation_item.checkpoint_sha256 != training_item.checkpoint_sha256
                or evaluation_item.config_sha256 != training_item.config_sha256
                or evaluation_item.run_id != training_item.run_id
            )
        ):
            raise ArtifactValidationError(f"{experiment} evaluation không khớp training run")
    e4_training = training_by_experiment.get("E4")
    if (
        e5 is not None
        and e4_training is not None
        and (
            e5.checkpoint_sha256 != e4_training.checkpoint_sha256
            or e5.config_sha256 != e4_training.config_sha256
            or e5.run_id != e4_training.run_id
        )
    ):
        raise ArtifactValidationError("E4 fixed/E5 tuned không dùng cùng checkpoint E4")
    if manifest_source is None or manifest_bundle is None:
        missing.append("locked_manifest_bundle")
    if validation_source is None or threshold_evidence is None:
        missing.append("provenance_bound_validation_threshold")
    if not benchmarks:
        missing.append("official_benchmark")
    if not any(item.valid_for_report for item in real_evidence):
        missing.append("confirmed_self_captured_evaluation")
    if augmentation_evidence is None:
        missing.append("report_valid_augmentation_e4")
    if gradcam_evidence is None:
        missing.append("report_valid_gradcam_e5")
    if not official:
        missing.append("official_output_path")
    missing = sorted(set(missing))
    if official and missing:
        raise ArtifactValidationError("Official report evidence chưa đầy đủ: " + ", ".join(missing))

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    generated: dict[str, dict[str, str]] = {}

    def register_generated(key: str, staged_path: Path) -> None:
        relative = staged_path.relative_to(staging)
        generated[key] = {
            "path": _portable_path(target / relative, root),
            "sha256": sha256_file(staged_path),
        }

    try:
        comparison_csv = staging / "comparison_table.csv"
        comparison_md = staging / "comparison_table.md"
        _write_csv_immutable(comparison_csv, comparison)
        _write_text_immutable(comparison_md, _markdown_table(comparison))
        register_generated("comparison_csv", comparison_csv)
        register_generated("comparison_markdown", comparison_md)

        if dataset_frame is not None and dataset_fact is not None:
            dataset_csv = staging / "dataset_summary.csv"
            dataset_json = staging / "dataset_summary.json"
            dataset_figure = staging / "fig_dataset_distribution.png"
            _write_csv_immutable(dataset_csv, dataset_frame)
            write_json(dataset_json, dataset_fact, overwrite=False)
            plot_dataset_distribution(dataset_frame, dataset_figure)
            register_generated("dataset_summary_csv", dataset_csv)
            register_generated("dataset_summary_json", dataset_json)
            register_generated("dataset_distribution_figure", dataset_figure)

        if curve_frame is not None and threshold_fact is not None:
            curve_csv = staging / "threshold_curve_validation.csv"
            curve_png = staging / "fig_threshold_curves_e5.png"
            _write_csv_immutable(curve_csv, curve_frame)
            plot_threshold_curve(curve_frame.to_dict(orient="records"), curve_png)
            register_generated("threshold_curve_csv", curve_csv)
            register_generated("threshold_curve_figure", curve_png)

        if benchmark_frame is not None:
            benchmark_csv = staging / "table_latency_benchmark.csv"
            benchmark_md = staging / "table_latency_benchmark.md"
            _write_csv_immutable(benchmark_csv, benchmark_frame)
            _write_text_immutable(benchmark_md, _markdown_table(benchmark_frame))
            register_generated("latency_benchmark_csv", benchmark_csv)
            register_generated("latency_benchmark_markdown", benchmark_md)

        for item in sorted(training, key=lambda value: value.experiment):
            curve_path = staging / f"fig_training_curves_{item.experiment.lower()}.png"
            history_payload = {
                "loss": item.history["loss"].tolist(),
                "val_loss": item.history["val_loss"].tolist(),
            }
            if {"accuracy", "val_accuracy"}.issubset(item.history.columns):
                history_payload.update(
                    {
                        "accuracy": item.history["accuracy"].tolist(),
                        "val_accuracy": item.history["val_accuracy"].tolist(),
                    }
                )
            plot_training_history(history_payload, curve_path)
            register_generated(f"training_curves_{item.experiment.lower()}", curve_path)

        experiment_facts: list[dict[str, Any]] = []
        for (item, _, metric_source, _, _), row in zip(
            comparison_entries,
            comparison.to_dict(orient="records"),
            strict=True,
        ):
            experiment_facts.append(
                {
                    **row,
                    "status": item.status,
                    "source_artifacts": {
                        "metrics_test": {
                            "path": _portable_path(metric_source, root),
                            "sha256": sha256_file(metric_source),
                        },
                        "predictions_test": {
                            "path": _portable_path(item.predictions_path, root),
                            "sha256": sha256_file(item.predictions_path),
                        },
                        "evaluation_metadata": {
                            "path": _portable_path(item.metadata_path, root),
                            "sha256": sha256_file(item.metadata_path),
                        },
                        "selection_contract_snapshot": {
                            "path": _portable_path(item.selection_snapshot_path, root),
                            "sha256": sha256_file(item.selection_snapshot_path),
                        },
                        "evaluation_complete": {
                            "path": _portable_path(item.completion_path, root),
                            "sha256": sha256_file(item.completion_path),
                        },
                    },
                }
            )

        facts: dict[str, Any] = {
            "schema_version": 2,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "FINAL_REPORT_FACTS" if official else "REPORT_INCOMPLETE",
            "valid_for_report": official,
            "missing_evidence": missing,
            "manifest_sha256": expected_manifest_hash,
            "experiments": experiment_facts,
            "generated_assets": generated,
        }
        if dataset_fact is not None and manifest_source is not None:
            facts["dataset"] = {
                **dataset_fact,
                "source_path": _portable_path(manifest_source, root),
            }
        if manifest_bundle is not None:
            facts["locked_manifest_bundle"] = {
                "path": _portable_path(manifest_bundle.directory, root),
                "manifest_hashes_sha256": sha256_file(manifest_bundle.manifest_hashes_path),
                "split_audit_sha256": sha256_file(manifest_bundle.split_audit_path),
                "input_manifest_sha256": sha256_file(manifest_bundle.input_manifest_path),
                "curation_report_sha256": sha256_file(manifest_bundle.curation_report_path),
            }
        if threshold_fact is not None and validation_source is not None:
            facts["threshold_tuning"] = {
                **threshold_fact,
                "source_path": _portable_path(validation_source, root),
                "threshold_artifact": threshold_evidence,
            }
        if benchmarks:
            facts["latency_benchmarks"] = benchmarks
        if training:
            facts["training_evidence"] = [
                {
                    "experiment": item.experiment,
                    "run_id": item.run_id,
                    "checkpoint_sha256": item.checkpoint_sha256,
                    "config_sha256": item.config_sha256,
                    "manifest_sha256": item.manifest_sha256,
                    "history_path": _portable_path(item.history_path, root),
                    "history_sha256": sha256_file(item.history_path),
                    "training_complete_path": _portable_path(item.completion_path, root),
                    "training_complete_sha256": sha256_file(item.completion_path),
                }
                for item in sorted(training, key=lambda value: value.experiment)
            ]
        if real_evidence:
            facts["real_image_evaluations"] = [
                {
                    "status": item.status,
                    "valid_for_report": item.valid_for_report,
                    "included_in_standard_test": False,
                    "sample_count": int(item.metrics["sample_count"]),
                    "threshold": float(item.metrics["threshold"]),
                    "accuracy": float(item.metrics["accuracy"]),
                    "precision_crack": float(item.metrics["crack"]["precision"]),
                    "recall_crack": float(item.metrics["crack"]["recall"]),
                    "f1_crack": float(item.metrics["crack"]["f1"]),
                    "fp": int(item.metrics["fp"]),
                    "fn": int(item.metrics["fn"]),
                    "source_artifacts": {
                        name: {
                            "path": _portable_path(path, root),
                            "sha256": sha256_file(path),
                        }
                        for name, path in (
                            ("metrics_real", item.metrics_path),
                            ("predictions_real", item.predictions_path),
                            ("evaluation_metadata", item.metadata_path),
                            ("selection_contract_snapshot", item.selection_snapshot_path),
                            ("evaluation_complete", item.completion_path),
                        )
                    },
                }
                for item in real_evidence
            ]
        qualitative: dict[str, Any] = {}
        for name, item in (
            ("augmentation_e4", augmentation_evidence),
            ("gradcam_e5", gradcam_evidence),
        ):
            if item is not None:
                qualitative[name] = {
                    "sidecar_path": _portable_path(item.sidecar_path, root),
                    "sidecar_sha256": sha256_file(item.sidecar_path),
                    "image_path": _portable_path(item.image_path, root),
                    "image_sha256": sha256_file(item.image_path),
                }
        if qualitative:
            facts["qualitative_evidence"] = qualitative

        facts_path = staging / "report_facts.json"
        write_json(facts_path, facts, overwrite=False)
        completion_files = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        completion = {
            "schema_version": 1,
            "status": "FINAL_REPORT_COMPLETE" if official else "REPORT_INCOMPLETE",
            "valid_for_report": official,
            "missing_evidence": missing,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "artifact_sha256": completion_files,
            "immutable": True,
        }
        write_json(staging / "report_complete.json", completion, overwrite=False)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "report_facts.json"


__all__ = [
    "ArtifactValidationError",
    "generate_report_assets",
    "is_official_report_path",
    "select_e2_e3",
]
