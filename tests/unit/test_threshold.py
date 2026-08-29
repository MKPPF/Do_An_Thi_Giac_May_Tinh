from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crackspot.modeling.threshold import optimize_threshold, tune_from_predictions
from crackspot.utils.hashing import sha256_file, sha256_json


def test_threshold_tuning_maximizes_f1() -> None:
    result = optimize_threshold([0, 0, 1, 1], [0.1, 0.4, 0.45, 0.9])
    assert result.threshold == pytest.approx(0.45)
    assert result.f1_crack == pytest.approx(1.0)
    assert result.source_split == "validation"


def test_threshold_tie_break_is_deterministic_and_close_to_half() -> None:
    result = optimize_threshold([0, 1], [0.1, 0.9])
    assert result.threshold == pytest.approx(0.5)


def test_threshold_refuses_test_split() -> None:
    with pytest.raises(ValueError, match="validation"):
        optimize_threshold([0, 1], [0.1, 0.9], source_split="test")


def test_threshold_file_requires_validation_provenance_and_is_immutable(
    tmp_path: Path,
) -> None:
    ambiguous = tmp_path / "predictions.csv"
    pd.DataFrame({"y_true": [0, 1], "probability_crack": [0.1, 0.9]}).to_csv(ambiguous, index=False)
    with pytest.raises(ValueError, match="validation"):
        tune_from_predictions(ambiguous, tmp_path / "threshold.json")

    run = tmp_path / "run"
    run.mkdir()
    validation = run / "predictions_validation.csv"
    pd.DataFrame(
        {
            "y_true": [0, 1],
            "probability_crack": [0.1, 0.9],
            "split": ["validation", "validation"],
        }
    ).to_csv(validation, index=False)
    checkpoint = run / "model.keras"
    checkpoint.write_bytes(b"model")
    config = {"experiment": {"id": "E4"}}
    config_path = run / "config_snapshot.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_hash = sha256_json(config)
    manifest_hash = "b" * 64
    checkpoint_hash = sha256_file(checkpoint)
    summary = {
        "run_id": "e4-run",
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "model_sha256": checkpoint_hash,
        "valid_for_report": True,
        "status": "VALIDATION_COMPLETE_TEST_LOCKED",
    }
    metadata = {
        "run_id": "e4-run",
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "model_sha256": checkpoint_hash,
        "smoke_test": False,
    }
    (run / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "model.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    artifact_names = (
        "predictions_validation.csv",
        "config_snapshot.json",
        "model.keras",
        "model.metadata.json",
        "run_summary.json",
    )
    completion = {
        "run_id": "e4-run",
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "model_sha256": checkpoint_hash,
        "status": "VALIDATION_COMPLETE_TEST_LOCKED",
        "artifact_sha256": {name: sha256_file(run / name) for name in artifact_names},
    }
    (run / "training_complete.json").write_text(json.dumps(completion), encoding="utf-8")
    output = tmp_path / "threshold.json"
    result = tune_from_predictions(validation, output)
    assert result.run_id == "e4-run"
    assert result.predictions_sha256 == sha256_file(validation)
    assert result.valid_for_report is True
    with pytest.raises(FileExistsError, match="ghi đè"):
        tune_from_predictions(validation, output)

    frame = pd.read_csv(validation)
    frame.loc[0, "probability_crack"] = 0.8
    frame.to_csv(validation, index=False)
    with pytest.raises(ValueError, match="artifact hash"):
        tune_from_predictions(validation, tmp_path / "tampered-threshold.json")
