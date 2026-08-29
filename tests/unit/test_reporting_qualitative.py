from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from crackspot.reporting.qualitative import (
    QualitativeEvidenceError,
    generate_augmentation_audit_grid,
    generate_gradcam_outcome_grid,
    select_outcome_examples,
    validate_prediction_frame,
)
from crackspot.utils.hashing import sha256_file
from scripts.generate_gradcam_grid import _verify_official_predictions


def _outcome_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "relative_path": [
                "z/tp.png",
                "a/tp.png",
                "z/tn.png",
                "a/tn.png",
                "z/fp.png",
                "a/fp.png",
                "z/fn.png",
                "a/fn.png",
            ],
            "y_true": [1, 1, 0, 0, 0, 0, 1, 1],
            "probability_crack": [0.9, 0.9, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2],
            "threshold": [0.5] * 8,
        }
    )


def test_select_outcome_examples_is_deterministic_with_explicit_tie_breaks() -> None:
    expected_paths = {
        "TP": "a/tp.png",
        "TN": "a/tn.png",
        "FP": "a/fp.png",
        "FN": "a/fn.png",
    }

    for random_state in range(5):
        shuffled = _outcome_frame().sample(frac=1.0, random_state=random_state)
        selected = select_outcome_examples(shuffled)

        assert {name: record["relative_path"] for name, record in selected.items()} == (
            expected_paths
        )


def test_validate_prediction_frame_normalizes_paths_and_threshold_boundary() -> None:
    source = pd.DataFrame(
        {
            "relative_path": [r"D\UD\non_crack.png", "D/CD/crack.png"],
            "y_true": [0, 1],
            "probability_crack": [0.5, 0.499],
            "threshold": [0.5, 0.5],
            "y_pred": [1, 0],
            "outcome": [" fp ", "fn"],
        }
    )

    normalized = validate_prediction_frame(source)

    assert normalized["relative_path"].tolist() == [
        "D/UD/non_crack.png",
        "D/CD/crack.png",
    ]
    assert normalized["y_pred"].tolist() == [1, 0]
    assert normalized["outcome"].tolist() == ["FP", "FN"]
    assert source.loc[0, "relative_path"] == r"D\UD\non_crack.png"


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("y_true", [0.25, 1], "y_true"),
        ("probability_crack", [float("nan"), 0.2], "probability_crack"),
        ("threshold", [0.5, 0.6], "threshold"),
        ("y_pred", [0.25, 0], "y_pred"),
        ("outcome", ["TP", "TN"], "outcome"),
    ],
)
def test_validate_prediction_frame_rejects_inconsistent_evidence(
    column: str,
    replacement: list[Any],
    message: str,
) -> None:
    frame = pd.DataFrame(
        {
            "relative_path": ["a.png", "b.png"],
            "y_true": [0, 1],
            "probability_crack": [0.1, 0.9],
            "threshold": [0.5, 0.5],
            "y_pred": [0, 1],
            "outcome": ["TN", "TP"],
        }
    )
    frame[column] = replacement

    with pytest.raises(QualitativeEvidenceError, match=message):
        validate_prediction_frame(frame)


@pytest.mark.parametrize("path", ["../escape.png", "C:/absolute.png", "/absolute.png", " "])
def test_validate_prediction_frame_rejects_unsafe_paths(path: str) -> None:
    frame = pd.DataFrame(
        {
            "relative_path": [path],
            "y_true": [0],
            "probability_crack": [0.1],
            "threshold": [0.5],
        }
    )

    with pytest.raises(QualitativeEvidenceError, match="relative_path"):
        validate_prediction_frame(frame)


def test_validate_prediction_frame_detects_duplicates_after_path_normalization() -> None:
    frame = pd.DataFrame(
        {
            "relative_path": ["D/UD/a.png", r"D\UD\a.png"],
            "y_true": [0, 0],
            "probability_crack": [0.1, 0.2],
            "threshold": [0.5, 0.5],
        }
    )

    with pytest.raises(QualitativeEvidenceError, match="trùng"):
        validate_prediction_frame(frame)


def test_validate_prediction_frame_rejects_boolean_as_numeric_evidence() -> None:
    frame = pd.DataFrame(
        {
            "relative_path": ["a.png"],
            "y_true": [False],
            "probability_crack": [0.1],
            "threshold": [0.5],
        }
    )

    with pytest.raises(QualitativeEvidenceError, match="bool"):
        validate_prediction_frame(frame)


class _FakeGradcamService:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[str] = []

    def predict_image(self, image: Path, *, include_gradcam: bool) -> SimpleNamespace:
        assert include_gradcam is True
        relative_name = image.name
        self.calls.append(relative_name)
        return SimpleNamespace(
            crack_probability=self.probabilities[relative_name],
            overlay=Image.new("RGB", (12, 8), (40, 80, 120)),
            heatmap=np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
        )


def _write_gradcam_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, float]]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rows = (
        ("tp.png", 1, 0.8, "TP"),
        ("tn.png", 0, 0.2, "TN"),
        ("fp.png", 0, 0.7, "FP"),
        ("fn.png", 1, 0.3, "FN"),
    )
    probabilities: dict[str, float] = {}
    prediction_rows: list[dict[str, Any]] = []
    for index, (name, truth, probability, outcome) in enumerate(rows):
        image_path = dataset_root / name
        Image.new("RGB", (12, 8), (index * 30, 50, 100)).save(image_path)
        predicted = int(probability >= 0.5)
        probabilities[name] = probability
        prediction_rows.append(
            {
                "relative_path": name,
                "sha256": sha256_file(image_path),
                "y_true": truth,
                "probability_crack": probability,
                "threshold": 0.5,
                "y_pred": predicted,
                "outcome": outcome,
            }
        )
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions, index=False, lineterminator="\n")
    model = tmp_path / "model.keras"
    metadata = tmp_path / "model.metadata.json"
    model.write_bytes(b"fake-model")
    metadata.write_text("{}", encoding="utf-8")
    return predictions, dataset_root, model, probabilities


def test_generate_gradcam_grid_writes_hashed_deterministic_evidence(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    predictions, dataset_root, model, probabilities = _write_gradcam_fixture(tmp_path)
    metadata = tmp_path / "model.metadata.json"
    service = _FakeGradcamService(probabilities)
    output = tmp_path / "evidence" / "gradcam.png"

    payload = generate_gradcam_outcome_grid(
        predictions_path=predictions,
        dataset_root=dataset_root,
        model_path=model,
        metadata_path=metadata,
        output_path=output,
        threshold=0.5,
        valid_for_report=False,
        service=service,
    )

    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert payload["status"] == "NOT_VALID_FOR_REPORT"
    assert payload["output_sha256"] == sha256_file(output)
    assert list(payload["selected"]) == ["TP", "TN", "FP", "FN"]
    assert service.calls == ["tp.png", "tn.png", "fp.png", "fn.png"]
    assert all(record["probability_delta"] == 0.0 for record in payload["selected"].values())

    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved == payload


def test_generate_gradcam_grid_rejects_tampered_selected_image_before_inference(
    tmp_path: Path,
) -> None:
    predictions, dataset_root, model, probabilities = _write_gradcam_fixture(tmp_path)
    metadata = tmp_path / "model.metadata.json"
    Image.new("RGB", (12, 8), (255, 255, 255)).save(dataset_root / "tp.png")
    service = _FakeGradcamService(probabilities)
    output = tmp_path / "gradcam.png"

    with pytest.raises(QualitativeEvidenceError, match="SHA-256"):
        generate_gradcam_outcome_grid(
            predictions_path=predictions,
            dataset_root=dataset_root,
            model_path=model,
            metadata_path=metadata,
            output_path=output,
            threshold=0.5,
            valid_for_report=False,
            service=service,
        )

    assert service.calls == []
    assert not output.exists()


def test_report_gradcam_grid_rejects_relaxed_probability_tolerance(tmp_path: Path) -> None:
    predictions, dataset_root, model, probabilities = _write_gradcam_fixture(tmp_path)

    with pytest.raises(QualitativeEvidenceError, match="probability_tolerance"):
        generate_gradcam_outcome_grid(
            predictions_path=predictions,
            dataset_root=dataset_root,
            model_path=model,
            metadata_path=tmp_path / "model.metadata.json",
            output_path=tmp_path / "gradcam.png",
            threshold=0.5,
            valid_for_report=True,
            probability_tolerance=0.01,
            service=_FakeGradcamService(probabilities),
        )


def _write_official_evaluation_fixture(tmp_path: Path) -> tuple[Path, Any, Path, Path]:
    evaluation = tmp_path / "final_evaluation"
    evaluation.mkdir()
    contract = SimpleNamespace(
        experiment="E5",
        run_id="e5-run",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        manifest_sha256="c" * 64,
        threshold=0.45,
    )
    selection_path = tmp_path / "selection_complete.json"
    selection_path.write_text('{"selected_by":"validation"}', encoding="utf-8")
    model_metadata_path = tmp_path / "model.metadata.json"
    model_metadata_path.write_text('{"model_version":"test"}', encoding="utf-8")
    predictions = evaluation / "predictions_test.csv"
    predictions.write_text(
        "relative_path,y_true,probability_crack\na.png,0,0.1\n", encoding="utf-8"
    )
    common = {
        "experiment": contract.experiment,
        "run_id": contract.run_id,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "config_sha256": contract.config_sha256,
        "manifest_sha256": contract.manifest_sha256,
        "threshold": contract.threshold,
    }
    evaluation_metadata = {
        **common,
        "status": "FINAL_TEST_COMPLETE",
        "valid_for_report": True,
        "smoke_test": False,
        "selection_selected_by": "validation",
        "threshold_source": "validation",
        "prediction_passes": 1,
        "selection_contract_sha256": sha256_file(selection_path),
        "model_metadata_sha256": sha256_file(model_metadata_path),
    }
    selection_snapshot = {**common, "selected_by": "validation"}
    metadata_file = evaluation / "evaluation_metadata.json"
    snapshot_file = evaluation / "selection_contract_snapshot.json"
    metadata_file.write_text(json.dumps(evaluation_metadata), encoding="utf-8")
    snapshot_file.write_text(json.dumps(selection_snapshot), encoding="utf-8")
    completion = {
        "status": "FINAL_TEST_COMPLETE",
        "artifact_sha256": {
            path.name: sha256_file(path) for path in (predictions, metadata_file, snapshot_file)
        },
    }
    (evaluation / "evaluation_complete.json").write_text(json.dumps(completion), encoding="utf-8")
    return predictions, contract, selection_path, model_metadata_path


def test_official_gradcam_provenance_detects_predictions_tampering(tmp_path: Path) -> None:
    predictions, contract, selection_path, metadata_path = _write_official_evaluation_fixture(
        tmp_path
    )
    _verify_official_predictions(
        predictions,
        contract,
        selection_path=selection_path,
        metadata_path=metadata_path,
    )

    predictions.write_text(
        "relative_path,y_true,probability_crack\na.png,0,0.99\n", encoding="utf-8"
    )

    with pytest.raises(QualitativeEvidenceError, match="evaluation_complete"):
        _verify_official_predictions(
            predictions,
            contract,
            selection_path=selection_path,
            metadata_path=metadata_path,
        )


def test_report_augmentation_rejects_unverified_group_outside_train(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "relative_path": ["train.png", "validation.png", "test.png"],
            "label": [0, 1, 0],
            "surface": ["D", "D", "D"],
            "source_group": ["train-group", "validation-group", "test-group"],
            "source_group_verified": [True, False, True],
            "sha256": ["a" * 64, "b" * 64, "c" * 64],
            "split": ["train", "validation", "test"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")

    with pytest.raises(QualitativeEvidenceError, match="nguồn nhóm"):
        generate_augmentation_audit_grid(
            manifest_path=manifest_path,
            dataset_root=tmp_path / "dataset",
            output_path=tmp_path / "augmentation.png",
            valid_for_report=True,
        )


def test_augmentation_rejects_fractional_manifest_label_before_tensorflow(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "relative_path": ["train.png"],
            "label": [0.25],
            "split": ["train"],
        }
    ).to_csv(manifest_path, index=False, lineterminator="\n")

    with pytest.raises(QualitativeEvidenceError, match="label"):
        generate_augmentation_audit_grid(
            manifest_path=manifest_path,
            dataset_root=tmp_path / "dataset",
            output_path=tmp_path / "augmentation.png",
            valid_for_report=False,
        )


def test_augmentation_rejects_non_finite_transform_before_tensorflow(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "relative_path": ["train.png"],
            "label": [0],
            "split": ["train"],
        }
    ).to_csv(manifest_path, index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="brightness_delta"):
        generate_augmentation_audit_grid(
            manifest_path=manifest_path,
            dataset_root=tmp_path / "dataset",
            output_path=tmp_path / "augmentation.png",
            brightness_delta=float("nan"),
            valid_for_report=False,
        )


@pytest.mark.integration
def test_augmentation_smoke_is_reproducible_and_non_reportable(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    pytest.importorskip("matplotlib")
    dataset_root = tmp_path / "dataset"
    rows: list[dict[str, Any]] = []
    for index, (relative, label) in enumerate(
        (("train/a.png", 0), ("train/b.png", 1), ("train/c.png", 0))
    ):
        image_path = dataset_root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (14, 10), (40 + index * 50, 80, 120)).save(image_path)
        rows.append({"relative_path": relative, "label": label, "split": "train"})
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False, lineterminator="\n")

    first = generate_augmentation_audit_grid(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_path=tmp_path / "first.png",
        sample_count=2,
        seed=42,
        image_size=(16, 12),
        rotation_degrees=5.0,
        brightness_delta=0.1,
        contrast_delta=0.1,
        valid_for_report=False,
    )
    second = generate_augmentation_audit_grid(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_path=tmp_path / "second.png",
        sample_count=2,
        seed=42,
        image_size=(16, 12),
        rotation_degrees=5.0,
        brightness_delta=0.1,
        contrast_delta=0.1,
        valid_for_report=False,
    )

    assert first["status"] == "NOT_VALID_FOR_REPORT"
    assert first["valid_for_report"] is False
    assert first["sample_count"] == 2
    assert first["samples"] == second["samples"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["split_audit"]["valid"] is False
    assert (tmp_path / "first.json").is_file()
