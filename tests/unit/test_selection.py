from __future__ import annotations

import json
from pathlib import Path

import pytest

from crackspot.modeling.selection import (
    create_selection_contract,
    export_selected_metadata,
    lock_run_selection,
    verify_selection_contract,
)
from crackspot.modeling.threshold import tune_from_predictions
from crackspot.utils.hashing import sha256_file, sha256_json


def test_selection_contract_detects_checkpoint_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"valid-model")
    contract_path = tmp_path / "selection_complete.json"
    created = create_selection_contract(
        experiment="e4_augmentation",
        run_id="run-1",
        checkpoint=checkpoint,
        config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        threshold=0.5,
        output=contract_path,
    )
    assert verify_selection_contract(contract_path, expected_manifest_sha256="b" * 64) == created
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        verify_selection_contract(contract_path)


def test_selection_contract_requires_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"model")
    path = tmp_path / "selection.json"
    create_selection_contract(
        experiment="e4",
        run_id="r",
        checkpoint=checkpoint,
        config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        threshold=0.5,
        output=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_by"] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        verify_selection_contract(path)


def test_lock_run_selection_verifies_run_and_validation_threshold(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model.keras"
    checkpoint.write_bytes(b"model")
    config = {"experiment": {"id": "E4", "name": "e4_augmentation"}}
    config_hash = sha256_json(config)
    manifest_hash = "b" * 64
    model_hash = sha256_file(checkpoint)
    (run_dir / "config_snapshot.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "e4-v1",
                "config_sha256": config_hash,
                "manifest_sha256": manifest_hash,
                "model_sha256": model_hash,
                "valid_for_report": True,
                "status": "VALIDATION_COMPLETE_TEST_LOCKED",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model.metadata.json").write_text(
        json.dumps(
            {
                "run_id": "e4-v1",
                "config_sha256": config_hash,
                "manifest_sha256": manifest_hash,
                "model_sha256": model_hash,
            }
        ),
        encoding="utf-8",
    )
    predictions = run_dir / "predictions_validation.csv"
    predictions.write_text(
        "y_true,probability_crack,split\n"
        "0,0.1,validation\n0,0.4,validation\n"
        "1,0.42,validation\n1,0.9,validation\n",
        encoding="utf-8",
    )
    artifact_names = (
        "config_snapshot.json",
        "model.keras",
        "model.metadata.json",
        "run_summary.json",
        "predictions_validation.csv",
    )
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "run_id": "e4-v1",
                "config_sha256": config_hash,
                "manifest_sha256": manifest_hash,
                "model_sha256": model_hash,
                "status": "VALIDATION_COMPLETE_TEST_LOCKED",
                "artifact_sha256": {name: sha256_file(run_dir / name) for name in artifact_names},
            }
        ),
        encoding="utf-8",
    )
    threshold_path = run_dir / "threshold_validation.json"
    tune_from_predictions(predictions, threshold_path)

    contract = lock_run_selection(
        run_dir=run_dir,
        threshold_result=threshold_path,
        experiment="E5",
    )

    assert contract.threshold == pytest.approx(0.42)
    assert contract.experiment == "E5"
    assert contract.trained_experiment == "E4"
    assert contract.threshold_source == "validation"
    assert contract.threshold_result_sha256 == sha256_file(threshold_path)
    assert verify_selection_contract(run_dir / "selection_complete.json") == contract

    selected_metadata = export_selected_metadata(run_dir / "selection_complete.json")
    payload = json.loads(selected_metadata.read_text(encoding="utf-8"))
    assert selected_metadata == run_dir / "selected_model.metadata.json"
    assert payload["threshold"] == pytest.approx(0.42)
    assert payload["selection_experiment"] == "E5"
    assert payload["threshold_source"] == "validation"
    assert payload["selection_contract_sha256"] == sha256_file(run_dir / "selection_complete.json")
    assert payload["source_metadata_sha256"] == sha256_file(run_dir / "model.metadata.json")

    with pytest.raises(FileExistsError, match="Không ghi đè"):
        export_selected_metadata(run_dir / "selection_complete.json")


def test_lock_run_selection_refuses_tampered_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model.keras"
    checkpoint.write_bytes(b"model")
    model_hash = sha256_file(checkpoint)
    (run_dir / "config_snapshot.json").write_text('{"tampered": true}', encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "e1-v1",
                "config_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "model_sha256": model_hash,
                "valid_for_report": True,
                "status": "VALIDATION_COMPLETE_TEST_LOCKED",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model.metadata.json").write_text(
        json.dumps(
            {
                "config_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "model_sha256": model_hash,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions_validation.csv").write_text(
        "y_true,probability_crack,split\n0,0.1,validation\n1,0.9,validation\n",
        encoding="utf-8",
    )
    artifact_names = (
        "config_snapshot.json",
        "model.keras",
        "model.metadata.json",
        "run_summary.json",
        "predictions_validation.csv",
    )
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "run_id": "e1-v1",
                "config_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "model_sha256": model_hash,
                "status": "VALIDATION_COMPLETE_TEST_LOCKED",
                "artifact_sha256": {name: sha256_file(run_dir / name) for name in artifact_names},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config_snapshot"):
        lock_run_selection(run_dir=run_dir, threshold=0.5)
