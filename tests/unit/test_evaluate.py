from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import crackspot.modeling.evaluate as evaluate
from crackspot.data import manifest_sha256
from crackspot.modeling.selection import create_selection_contract
from crackspot.utils.hashing import sha256_file


def _manifest() -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        for surface in ("D", "P", "W"):
            for label in (0, 1):
                class_folder = ("C" if label else "U") + surface
                rows.append(
                    (
                        f"{surface}/{class_folder}/{split}-{surface}-{label}.jpg",
                        label,
                        surface,
                        f"{split}-{surface}-{label}",
                        split,
                    )
                )
    return pd.DataFrame(
        {
            "relative_path": [row[0] for row in rows],
            "label": [row[1] for row in rows],
            "surface": [row[2] for row in rows],
            "source_group": [row[3] for row in rows],
            "source_group_verified": [True] * len(rows),
            "sha256": [f"a{index:063x}" for index in range(1, len(rows) + 1)],
            "width": [256] * len(rows),
            "height": [256] * len(rows),
            "split": [row[4] for row in rows],
            "audit_status": ["ok"] * len(rows),
        }
    )


def _write_contract(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    manifest_hash: str | None = None,
    threshold: float = 0.5,
) -> tuple[Path, Path, Path]:
    manifest_path = tmp_path / "manifest.csv"
    frame.to_csv(manifest_path, index=False, lineterminator="\n")
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"mock-keras-checkpoint")
    selection = tmp_path / "selection_complete.json"
    create_selection_contract(
        experiment="E4",
        run_id="e4-test-run",
        checkpoint=checkpoint,
        config_sha256="a" * 64,
        manifest_sha256=manifest_hash or manifest_sha256(frame),
        threshold=threshold,
        output=selection,
    )
    (tmp_path / "model.metadata.json").write_text(
        json.dumps(
            {
                "run_id": "e4-test-run",
                "model_version": "test",
                "threshold": threshold,
                "input_size": [224, 224],
                "preprocessing": "mobilenet_v2.preprocess_input",
                "label_mapping": {"Non-crack": 0, "Crack": 1},
                "gradcam_layer": "out_relu",
                "tensorflow_version": "test",
                "model_sha256": sha256_file(checkpoint),
                "manifest_sha256": manifest_hash or manifest_sha256(frame),
                "config_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return selection, manifest_path, checkpoint


def _mock_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    probabilities = np.asarray([0.1, 0.9, 0.6, 0.4, 0.2, 0.8], dtype=np.float64)

    def fake_dataset(frame: pd.DataFrame, **kwargs: Any) -> object:
        assert set(frame["split"]) == {"test"}
        assert kwargs["training"] is False
        assert kwargs["augment"] is False
        return object()

    def fake_plot(matrix: Any, output: str | Path, **kwargs: Any) -> Path:
        del matrix, kwargs
        target = Path(output)
        target.write_bytes(b"mock-png")
        return target

    monkeypatch.setattr(evaluate, "build_tf_dataset", fake_dataset)
    monkeypatch.setattr(evaluate, "_load_model", lambda checkpoint: object())
    monkeypatch.setattr(evaluate, "_predict_probabilities", lambda model, dataset: probabilities)
    monkeypatch.setattr(evaluate, "plot_confusion_matrix", fake_plot)
    monkeypatch.setattr(evaluate, "capture_environment", lambda: {"test_environment": True})
    _mock_official_preconditions(monkeypatch)


def _mock_official_preconditions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluate,
        "verify_official_dataset_preconditions",
        lambda *args, **kwargs: SimpleNamespace(
            to_dict=lambda: {"status": "OFFICIAL_DATASET_PRECONDITIONS_VERIFIED"}
        ),
    )


def _isolate_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "registry"
) -> Path:
    registry = tmp_path / name
    monkeypatch.setattr(evaluate, "FINAL_EVALUATION_REGISTRY", registry)
    return registry


def test_final_evaluation_refuses_missing_selection_before_manifest_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed = False

    def forbidden_manifest(path: Path) -> pd.DataFrame:
        nonlocal accessed
        accessed = True
        raise AssertionError(f"manifest must stay locked: {path}")

    monkeypatch.setattr(evaluate, "_load_manifest", forbidden_manifest)
    _isolate_registry(tmp_path, monkeypatch)
    with pytest.raises(evaluate.FinalEvaluationProtocolError, match="Thiếu selection_complete"):
        evaluate.run_final_evaluation(
            selection_path=tmp_path / "selection_complete.json",
            manifest_path=tmp_path / "manifest.csv",
            dataset_root=tmp_path,
            output_dir=tmp_path / "output",
        )
    assert accessed is False


def test_manifest_hash_mismatch_never_selects_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, manifest_path, _ = _write_contract(
        tmp_path,
        _manifest(),
        manifest_hash="f" * 64,
    )
    selected = False

    def forbidden_selection(frame: pd.DataFrame) -> pd.DataFrame:
        nonlocal selected
        selected = True
        raise AssertionError(f"test rows must stay locked: {len(frame)}")

    monkeypatch.setattr(evaluate, "_select_test_rows", forbidden_selection)
    _isolate_registry(tmp_path, monkeypatch)
    with pytest.raises(evaluate.FinalEvaluationProtocolError, match="Manifest hash"):
        evaluate.run_final_evaluation(
            selection_path=selection,
            manifest_path=manifest_path,
            dataset_root=tmp_path,
            output_dir=tmp_path / "output",
        )
    assert selected is False
    assert not (tmp_path / "registry").exists()


def test_checkpoint_hash_mismatch_never_reads_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, manifest_path, checkpoint = _write_contract(tmp_path, _manifest())
    checkpoint.write_bytes(b"tampered-after-selection")
    manifest_accessed = False

    def forbidden_manifest(path: Path) -> pd.DataFrame:
        nonlocal manifest_accessed
        manifest_accessed = True
        raise AssertionError(f"manifest must stay locked: {path}")

    monkeypatch.setattr(evaluate, "_load_manifest", forbidden_manifest)
    _isolate_registry(tmp_path, monkeypatch)
    with pytest.raises(evaluate.FinalEvaluationProtocolError, match="Checkpoint hash"):
        evaluate.run_final_evaluation(
            selection_path=selection,
            manifest_path=manifest_path,
            dataset_root=tmp_path,
            output_dir=tmp_path / "output",
        )
    assert manifest_accessed is False


def test_failure_after_test_access_records_outcome_and_consumes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, manifest_path, checkpoint = _write_contract(tmp_path, _manifest())
    registry = _isolate_registry(tmp_path, monkeypatch)
    _mock_official_preconditions(monkeypatch)
    monkeypatch.setattr(
        evaluate,
        "build_tf_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic pipeline failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic pipeline failure"):
        evaluate.run_final_evaluation(
            selection_path=selection,
            manifest_path=manifest_path,
            dataset_root=tmp_path,
            output_dir=tmp_path / "failed-output",
        )

    checkpoint_hash = evaluate.sha256_file(checkpoint)
    marker = registry / f"{checkpoint_hash}.json"
    outcome = registry / f"{checkpoint_hash}.outcome.json"
    assert marker.is_file()
    payload = json.loads(outcome.read_text(encoding="utf-8"))
    assert payload["status"] == "FINAL_TEST_FAILED_CHECKPOINT_CONSUMED"
    assert payload["error_type"] == "RuntimeError"
    assert not (tmp_path / "failed-output").exists()
    with pytest.raises(evaluate.FinalEvaluationAlreadyRunError):
        evaluate.run_final_evaluation(
            selection_path=selection,
            manifest_path=manifest_path,
            dataset_root=tmp_path,
            output_dir=tmp_path / "retry-output",
        )


def test_dataset_precondition_failure_never_claims_final_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, manifest_path, _ = _write_contract(tmp_path, _manifest())
    registry = _isolate_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(
        evaluate,
        "verify_official_dataset_preconditions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            evaluate.DatasetIntegrityError("synthetic byte mismatch")
        ),
    )

    with pytest.raises(evaluate.FinalEvaluationProtocolError, match="test vẫn bị khóa"):
        evaluate.run_final_evaluation(
            selection_path=selection,
            manifest_path=manifest_path,
            dataset_root=tmp_path,
            output_dir=tmp_path / "output",
        )

    assert not registry.exists()
    assert not (tmp_path / "output").exists()


def test_happy_path_writes_evidence_then_refuses_second_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _manifest()
    selection, manifest_path, checkpoint = _write_contract(tmp_path, frame)
    _mock_runtime(monkeypatch)
    output = tmp_path / "report-output"
    registry = _isolate_registry(tmp_path, monkeypatch)

    result = evaluate.run_final_evaluation(
        selection_path=selection,
        manifest_path=manifest_path,
        dataset_root=tmp_path,
        output_dir=output,
    )

    assert result.status == "FINAL_TEST_COMPLETE"
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.f1_crack == pytest.approx(2 / 3)
    assert result.marker_path == registry / f"{evaluate.sha256_file(checkpoint)}.json"
    assert result.marker_path.is_file()
    outcome_path = registry / f"{evaluate.sha256_file(checkpoint)}.outcome.json"
    assert not outcome_path.exists()
    marker_before = result.marker_path.read_bytes()
    required = {
        "metrics_test.json",
        "metrics_test_threshold_0_5.json",
        "predictions_test.csv",
        "classification_report_test.json",
        "classification_report_test.csv",
        "classification_report_test.txt",
        "confusion_matrix_test.csv",
        "confusion_matrix_test.png",
        "confusion_matrix_test_normalized.csv",
        "confusion_matrix_test_normalized.png",
        "false_positives_test.csv",
        "false_negatives_test.csv",
        "classification_report_test_threshold_0_5.json",
        "confusion_matrix_test_threshold_0_5.csv",
        "confusion_matrix_test_threshold_0_5.png",
        "confusion_matrix_test_threshold_0_5_normalized.csv",
        "confusion_matrix_test_threshold_0_5_normalized.png",
        "false_positives_test_threshold_0_5.csv",
        "false_negatives_test_threshold_0_5.csv",
        "evaluation_metadata.json",
        "environment.json",
        "evaluation_complete.json",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    metrics = json.loads((output / "metrics_test.json").read_text(encoding="utf-8"))
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["valid_for_report"] is True
    assert metrics["prediction_passes"] == 1
    fixed_metrics = json.loads(
        (output / "metrics_test_threshold_0_5.json").read_text(encoding="utf-8")
    )
    assert fixed_metrics["fp"] == 1
    assert fixed_metrics["fn"] == 1
    assert fixed_metrics["threshold_role"] == "fixed_protocol_0_5"
    assert len(pd.read_csv(output / "false_positives_test.csv")) == 1
    assert len(pd.read_csv(output / "false_negatives_test.csv")) == 1
    predictions = pd.read_csv(output / "predictions_test.csv")
    assert predictions["y_pred"].tolist() == [0, 1, 1, 0, 0, 1]
    assert predictions["y_pred_threshold_0_5"].tolist() == [0, 1, 1, 0, 0, 1]

    with pytest.raises(evaluate.FinalEvaluationAlreadyRunError, match="đã được cấp quyền"):
        evaluate.run_final_evaluation(
            selection_path=selection,
            manifest_path=manifest_path,
            dataset_root=tmp_path,
            output_dir=tmp_path / "different-output",
        )
    assert result.marker_path.read_bytes() == marker_before
    assert not (tmp_path / "different-output").exists()


def test_smoke_evaluation_is_explicitly_not_report_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _manifest()
    selection, manifest_path, _ = _write_contract(tmp_path, frame)
    _mock_runtime(monkeypatch)
    output = tmp_path / "smoke-output"
    _isolate_registry(tmp_path, monkeypatch, "registry-smoke")

    result = evaluate.run_final_evaluation(
        selection_path=selection,
        manifest_path=manifest_path,
        dataset_root=tmp_path,
        output_dir=output,
        smoke=True,
    )

    assert result.status == "NOT_VALID_FOR_REPORT"
    metrics = json.loads((output / "metrics_test.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "evaluation_metadata.json").read_text(encoding="utf-8"))
    assert metrics["valid_for_report"] is False
    assert metadata["valid_for_report"] is False
    assert metadata["smoke_test"] is True
