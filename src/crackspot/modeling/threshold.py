"""Validation-only threshold tuning with deterministic tie breaking."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.utils.hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1_crack: float
    recall_crack: float
    precision_crack: float
    accuracy: float
    evaluated_candidates: int
    source_split: str = "validation"
    schema_version: int = 2
    artifact_type: str = "validation_threshold_selection"
    selected_by: str = "validation"
    status: str = "UNBOUND_VALIDATION_RESULT"
    valid_for_report: bool = False
    source_predictions: str | None = None
    predictions_sha256: str | None = None
    training_complete_sha256: str | None = None
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    manifest_sha256: str | None = None
    run_id: str | None = None
    sample_count: int | None = None


def _candidate_thresholds(probabilities: np.ndarray) -> np.ndarray:
    unique = np.unique(probabilities)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique)))
    return candidates[(candidates >= 0.0) & (candidates <= 1.0)]


def optimize_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    source_split: str = "val",
) -> ThresholdResult:
    """Maximize Crack F1, then Recall, then closeness to 0.5, then lower threshold."""

    normalized_split = source_split.strip().lower()
    if normalized_split not in {"val", "validation"}:
        raise ValueError("Threshold chỉ được tune trên validation, không phải test")
    truth = np.asarray(list(y_true), dtype=np.int64).reshape(-1)
    scores = np.asarray(list(probabilities), dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape or truth.size == 0:
        raise ValueError("Validation labels/probabilities phải cùng shape và không rỗng")

    candidates = _candidate_thresholds(scores)
    ranked: list[tuple[tuple[float, float, float, float], float, dict]] = []
    for threshold in candidates:
        metrics = compute_binary_metrics(truth, scores, float(threshold))
        key = (
            metrics["crack"]["f1"],
            metrics["crack"]["recall"],
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        ranked.append((key, float(threshold), metrics))
    _, threshold, metrics = max(ranked, key=lambda item: item[0])
    return ThresholdResult(
        threshold=threshold,
        f1_crack=float(metrics["crack"]["f1"]),
        recall_crack=float(metrics["crack"]["recall"]),
        precision_crack=float(metrics["crack"]["precision"]),
        accuracy=float(metrics["accuracy"]),
        evaluated_candidates=len(candidates),
    )


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _require_sha256(value: object, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a 64-character SHA-256")
    return text


def _load_run_provenance(
    source: Path,
    *,
    allow_smoke: bool,
) -> dict[str, Any]:
    run_dir = source.parent
    summary_path = run_dir / "run_summary.json"
    metadata_path = run_dir / "model.metadata.json"
    config_path = run_dir / "config_snapshot.json"
    checkpoint_path = run_dir / "model.keras"
    completion_path = run_dir / "training_complete.json"
    summary = _read_json_object(summary_path, "run summary")
    metadata = _read_json_object(metadata_path, "model metadata")
    config = _read_json_object(config_path, "config snapshot")
    completion = _read_json_object(completion_path, "training completion marker")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config_hash = _require_sha256(summary.get("config_sha256"), "config_sha256")
    manifest_hash = _require_sha256(summary.get("manifest_sha256"), "manifest_sha256")
    checkpoint_hash = _require_sha256(summary.get("model_sha256"), "model_sha256")
    if sha256_json(config) != config_hash:
        raise ValueError("config_snapshot.json does not match config_sha256")
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise ValueError("model.keras does not match model_sha256")
    for payload_name, payload in (("metadata", metadata), ("completion", completion)):
        for field, expected in (
            ("config_sha256", config_hash),
            ("manifest_sha256", manifest_hash),
            ("model_sha256", checkpoint_hash),
        ):
            if str(payload.get(field, "")).strip().lower() != expected:
                raise ValueError(f"{payload_name}.{field} does not match run summary")

    run_id = str(summary.get("run_id", "")).strip()
    if not run_id or str(completion.get("run_id", "")).strip() != run_id:
        raise ValueError("run_id is missing or inconsistent")
    artifact_hashes = completion.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("training_complete.json is missing artifact_sha256")
    required_artifacts = {
        source.name: source,
        "config_snapshot.json": config_path,
        "model.keras": checkpoint_path,
        "model.metadata.json": metadata_path,
        "run_summary.json": summary_path,
    }
    for name, artifact_path in required_artifacts.items():
        expected = _require_sha256(artifact_hashes.get(name), f"artifact_sha256.{name}")
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected:
            raise ValueError(f"Training artifact hash does not match: {artifact_path}")

    summary_status = str(summary.get("status", "")).strip().upper()
    completion_status = str(completion.get("status", "")).strip().upper()
    smoke = (
        summary.get("valid_for_report") is False
        or bool(metadata.get("smoke_test", False))
        or "NOT_VALID_FOR_REPORT" in summary_status
    )
    if summary_status != completion_status:
        raise ValueError("Training summary/completion status is inconsistent")
    if smoke and not allow_smoke:
        raise ValueError("Smoke/NOT_VALID_FOR_REPORT threshold tuning requires allow_smoke=True")
    if not smoke and (
        summary.get("valid_for_report") is not True
        or summary_status != "VALIDATION_COMPLETE_TEST_LOCKED"
    ):
        raise ValueError("Run is not a completed report-valid validation run")
    return {
        "run_id": run_id,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "training_complete_sha256": sha256_file(completion_path),
        "valid_for_report": not smoke,
        "status": "VALIDATION_THRESHOLD_SELECTED" if not smoke else "NOT_VALID_FOR_REPORT",
    }


def tune_from_predictions(
    path: str | Path,
    output: str | Path,
    *,
    allow_smoke: bool = False,
) -> ThresholdResult:
    import pandas as pd

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if "test" in source.stem.casefold():
        raise ValueError("Từ chối tune threshold từ file có tên test")
    if source.name.casefold() != "predictions_validation.csv":
        raise ValueError("Threshold source must be the immutable predictions_validation.csv")
    source_hash_before = sha256_file(source)
    provenance = _load_run_provenance(source, allow_smoke=allow_smoke)
    frame = pd.read_csv(source)
    required = {"y_true", "probability_crack"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Thiếu cột: {sorted(required - set(frame.columns))}")
    if "split" in frame.columns:
        splits = set(frame["split"].dropna().astype(str).str.strip().str.casefold())
        if not splits or not splits.issubset({"val", "validation"}):
            raise ValueError("Threshold chỉ được tune từ predictions validation")
    elif "validation" not in source.stem.casefold() and "val" not in source.stem.casefold():
        raise ValueError("Predictions không có cột split; tên file phải xác định rõ validation")
    if "split" in frame.columns:
        splits = set(frame["split"].dropna().astype(str).str.strip().str.casefold())
        if splits != {"validation"}:
            raise ValueError("Threshold predictions must contain only split=validation")
    if sha256_file(source) != source_hash_before:
        raise ValueError("predictions_validation.csv changed while threshold was tuned")
    basic = optimize_threshold(frame["y_true"], frame["probability_crack"])
    result = ThresholdResult(
        threshold=basic.threshold,
        f1_crack=basic.f1_crack,
        recall_crack=basic.recall_crack,
        precision_crack=basic.precision_crack,
        accuracy=basic.accuracy,
        evaluated_candidates=basic.evaluated_candidates,
        status=str(provenance["status"]),
        valid_for_report=bool(provenance["valid_for_report"]),
        source_predictions=str(source),
        predictions_sha256=source_hash_before,
        training_complete_sha256=str(provenance["training_complete_sha256"]),
        checkpoint_sha256=str(provenance["checkpoint_sha256"]),
        config_sha256=str(provenance["config_sha256"]),
        manifest_sha256=str(provenance["manifest_sha256"]),
        run_id=str(provenance["run_id"]),
        sample_count=len(frame),
    )
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Từ chối ghi đè threshold result: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if sha256_file(source) != source_hash_before:
        target.unlink(missing_ok=True)
        raise ValueError("predictions_validation.csv changed while threshold artifact was written")
    return result


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Tune Crack threshold on validation predictions only"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a completed NOT_VALID_FOR_REPORT smoke run",
    )
    args = parser.parse_args()
    result = tune_from_predictions(args.predictions, args.output, allow_smoke=args.smoke)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
