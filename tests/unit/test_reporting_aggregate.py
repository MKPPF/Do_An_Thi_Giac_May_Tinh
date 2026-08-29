from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crackspot.data import manifest_sha256
from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.reporting.aggregate import (
    ArtifactValidationError,
    generate_report_assets,
    select_e2_e3,
)
from crackspot.reporting.benchmark import benchmark_callable
from crackspot.reporting.export import write_json
from crackspot.utils.hashing import sha256_file, sha256_json


def _write_candidate(
    root: Path,
    experiment: str,
    val_loss: float,
    *,
    manifest_hash: str = "b" * 64,
    smoke: bool = False,
) -> Path:
    run = root / f"{experiment.lower()}-run"
    run.mkdir(parents=True)
    config = {
        "experiment": {
            "id": experiment,
            "name": f"{experiment.lower()}_candidate",
        },
        "training": {
            "fine_tune": {
                "unfreeze_from": "block_14_expand",
                "learning_rate": 0.0001,
            }
        },
    }
    write_json(run / "config_snapshot.json", config)
    (run / "model.keras").write_bytes(f"model-{experiment}".encode())
    model_hash = sha256_file(run / "model.keras")
    summary = {
        "run_id": run.name,
        "experiment": experiment.lower(),
        "best_val_loss": val_loss,
        "config_sha256": sha256_json(config),
        "manifest_sha256": manifest_hash,
        "model_sha256": model_hash,
        "valid_for_report": not smoke,
        "status": "NOT_VALID_FOR_REPORT" if smoke else "VALIDATION_COMPLETE_TEST_LOCKED",
    }
    write_json(run / "run_summary.json", summary)
    write_json(
        run / "model.metadata.json",
        {
            "config_sha256": summary["config_sha256"],
            "manifest_sha256": manifest_hash,
            "model_sha256": model_hash,
            "smoke_test": smoke,
            "status": "NOT_VALID_FOR_REPORT" if smoke else "VALIDATION_COMPLETE_TEST_LOCKED",
        },
    )
    write_json(
        run / "training_complete.json",
        {
            "schema_version": 2,
            "status": summary["status"],
            "run_id": summary["run_id"],
            "config_sha256": summary["config_sha256"],
            "manifest_sha256": manifest_hash,
            "model_sha256": model_hash,
            "artifact_sha256": {
                name: sha256_file(run / name)
                for name in (
                    "config_snapshot.json",
                    "model.keras",
                    "model.metadata.json",
                    "run_summary.json",
                )
            },
            "immutable": True,
        },
    )
    return run


def test_select_e2_e3_uses_only_val_loss_and_writes_portable_immutable_path(
    tmp_path: Path,
) -> None:
    e2 = _write_candidate(tmp_path, "E2", 0.31)
    e3 = _write_candidate(tmp_path, "E3", 0.22)
    output = tmp_path / "artifacts" / "model_selection.json"

    result = select_e2_e3(
        e2_run=e2,
        e3_run=e3,
        output=output,
        project_root=tmp_path,
    )

    assert result["winner_experiment"] == "E3"
    assert result["winner_best_val_loss"] == pytest.approx(0.22)
    assert result["metric"] == "val_loss"
    assert result["selected_by"] == "validation"
    assert result["winner_config"] == "e3-run/config_snapshot.json"
    assert Path(result["winner_config_resolved"]).is_absolute()
    assert result["manifest_sha256"] == "b" * 64
    with pytest.raises(FileExistsError):
        select_e2_e3(
            e2_run=e2,
            e3_run=e3,
            output=output,
            project_root=tmp_path,
        )


def test_select_e2_e3_refuses_different_manifest_hashes(tmp_path: Path) -> None:
    e2 = _write_candidate(tmp_path, "E2", 0.2, manifest_hash="a" * 64)
    e3 = _write_candidate(tmp_path, "E3", 0.1, manifest_hash="b" * 64)
    with pytest.raises(ArtifactValidationError, match="cùng manifest"):
        select_e2_e3(
            e2_run=e2,
            e3_run=e3,
            output=tmp_path / "selection.json",
            project_root=tmp_path,
        )


def test_select_e2_e3_refuses_candidate_without_training_completion(tmp_path: Path) -> None:
    e2 = _write_candidate(tmp_path, "E2", 0.2)
    e3 = _write_candidate(tmp_path, "E3", 0.1)
    (e3 / "training_complete.json").unlink()

    with pytest.raises(FileNotFoundError):
        select_e2_e3(
            e2_run=e2,
            e3_run=e3,
            output=tmp_path / "selection.json",
            project_root=tmp_path,
        )


def _write_final_evaluation(
    root: Path,
    *,
    status: str = "FINAL_TEST_COMPLETE",
    valid_for_report: bool = True,
    experiment: str = "E5",
    threshold: float = 0.45,
    manifest_hash: str = "b" * 64,
) -> Path:
    root.mkdir(parents=True)
    truth = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.45, 0.9]
    predictions = pd.DataFrame(
        {
            "relative_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "y_true": truth,
            "probability_crack": scores,
        }
    )
    predictions.to_csv(root / "predictions_test.csv", index=False, lineterminator="\n")
    checkpoint_hash = "a" * 64
    config_hash = "c" * 64
    metrics: dict[str, Any] = compute_binary_metrics(truth, scores, threshold)
    metrics.update(
        {
            "status": status,
            "valid_for_report": valid_for_report,
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
        }
    )
    write_json(root / "metrics_test.json", metrics)
    fixed_metrics: dict[str, Any] = compute_binary_metrics(truth, scores, 0.5)
    fixed_metrics.update(
        {
            "status": status,
            "valid_for_report": valid_for_report,
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "threshold_role": "fixed_protocol_0_5",
        }
    )
    write_json(root / "metrics_test_threshold_0_5.json", fixed_metrics)
    write_json(
        root / "evaluation_metadata.json",
        {
            "status": status,
            "valid_for_report": valid_for_report,
            "smoke_test": not valid_for_report,
            "experiment": experiment,
            "run_id": f"{experiment.lower()}-run",
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "threshold": threshold,
        },
    )
    write_json(
        root / "selection_contract_snapshot.json",
        {
            "experiment": experiment,
            "run_id": f"{experiment.lower()}-run",
            "checkpoint": "model.keras",
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "threshold": threshold,
            "selected_by": "validation",
        },
    )
    files = (
        "metrics_test.json",
        "metrics_test_threshold_0_5.json",
        "predictions_test.csv",
        "evaluation_metadata.json",
        "selection_contract_snapshot.json",
    )
    write_json(
        root / "evaluation_complete.json",
        {
            "status": status,
            "artifact_sha256": {name: sha256_file(root / name) for name in files},
        },
    )
    return root


def _write_real_evaluation(root: Path, *, manifest_hash: str = "b" * 64) -> Path:
    root.mkdir(parents=True)
    truth = [0, 1]
    scores = [0.1, 0.8]
    threshold = 0.45
    predictions = pd.DataFrame(
        {
            "relative_path": ["real-n.jpg", "real-c.jpg"],
            "split": ["real_external", "real_external"],
            "y_true": truth,
            "probability_crack": scores,
            "threshold": [threshold, threshold],
        }
    )
    predictions.to_csv(root / "predictions_real.csv", index=False, lineterminator="\n")
    metrics: dict[str, Any] = compute_binary_metrics(truth, scores, threshold)
    metrics.update(
        {
            "status": "REAL_IMAGE_EVALUATION_COMPLETE",
            "valid_for_report": True,
            "included_in_standard_test": False,
            "self_captured_confirmed": True,
            "evaluation_scope": "external_self_captured_images",
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "c" * 64,
            "sdnet_manifest_sha256": manifest_hash,
            "experiment": "E5",
            "run_id": "e5-run",
        }
    )
    write_json(root / "metrics_real.json", metrics)
    write_json(
        root / "evaluation_metadata.json",
        {
            "status": "REAL_IMAGE_EVALUATION_COMPLETE",
            "valid_for_report": True,
            "included_in_standard_test": False,
            "self_captured_confirmed": True,
            "evaluation_scope": "external_self_captured_images",
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "c" * 64,
            "sdnet_manifest_sha256": manifest_hash,
            "experiment": "E5",
            "run_id": "e5-run",
        },
    )
    write_json(
        root / "selection_contract_snapshot.json",
        {
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "c" * 64,
            "manifest_sha256": manifest_hash,
            "experiment": "E5",
            "run_id": "e5-run",
            "threshold": threshold,
            "selected_by": "validation",
        },
    )
    files = (
        "metrics_real.json",
        "predictions_real.csv",
        "evaluation_metadata.json",
        "selection_contract_snapshot.json",
    )
    write_json(
        root / "evaluation_complete.json",
        {
            "status": "REAL_IMAGE_EVALUATION_COMPLETE",
            "valid_for_report": True,
            "artifact_sha256": {name: sha256_file(root / name) for name in files},
        },
    )
    return root


def test_generate_report_assets_recomputes_metrics_and_records_hashes(tmp_path: Path) -> None:
    evaluation = _write_final_evaluation(
        tmp_path / "artifacts" / "report" / "final_evaluation" / "e5"
    )
    validation_predictions = tmp_path / "predictions_validation.csv"
    pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "probability_crack": [0.1, 0.4, 0.45, 0.9],
        }
    ).to_csv(validation_predictions, index=False, lineterminator="\n")
    output = tmp_path / "generated"

    facts_path = generate_report_assets(
        evaluation_dirs=[evaluation],
        output_dir=output,
        validation_predictions=validation_predictions,
        project_root=tmp_path,
    )

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    assert facts["valid_for_report"] is False
    assert facts["status"] == "REPORT_INCOMPLETE"
    assert "official_output_path" in facts["missing_evidence"]
    assert [row["experiment"] for row in facts["experiments"]] == ["E4", "E5"]
    assert facts["experiments"][1]["accuracy"] == pytest.approx(1.0)
    assert facts["threshold_tuning"]["selected_threshold"] == pytest.approx(0.45)
    assert len(facts["experiments"][0]["checkpoint_sha256"]) == 64
    assert (output / "comparison_table.csv").is_file()
    assert (output / "comparison_table.md").is_file()
    assert (output / "fig_threshold_curves_e5.png").is_file()
    assert facts["generated_assets"]["comparison_csv"]["sha256"] == sha256_file(
        output / "comparison_table.csv"
    )
    with pytest.raises(FileExistsError):
        generate_report_assets(
            evaluation_dirs=[evaluation],
            output_dir=output,
            validation_predictions=validation_predictions,
            project_root=tmp_path,
        )


def test_generate_report_assets_refuses_smoke_in_official_report(tmp_path: Path) -> None:
    smoke = _write_final_evaluation(
        tmp_path / "artifacts" / "smoke" / "final_evaluation" / "smoke",
        status="NOT_VALID_FOR_REPORT",
        valid_for_report=False,
    )
    with pytest.raises(ArtifactValidationError, match="smoke/NOT_VALID_FOR_REPORT"):
        generate_report_assets(
            evaluation_dirs=[smoke],
            output_dir=tmp_path / "artifacts" / "report",
            project_root=tmp_path,
        )


def test_generate_report_assets_detects_tampered_predictions(tmp_path: Path) -> None:
    evaluation = _write_final_evaluation(tmp_path / "evaluation")
    predictions = pd.read_csv(evaluation / "predictions_test.csv")
    predictions.loc[0, "probability_crack"] = 0.99
    predictions.to_csv(evaluation / "predictions_test.csv", index=False, lineterminator="\n")
    with pytest.raises(ArtifactValidationError, match="metrics_test không khớp"):
        generate_report_assets(
            evaluation_dirs=[evaluation],
            output_dir=tmp_path / "generated",
            project_root=tmp_path,
        )


def test_generate_report_assets_builds_dataset_and_latency_tables(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    definitions = (
        ("train", "train-n.jpg", 0),
        ("train", "train-c.jpg", 1),
        ("validation", "val-n.jpg", 0),
        ("validation", "val-c.jpg", 1),
        ("test", "a.jpg", 0),
        ("test", "b.jpg", 0),
        ("test", "c.jpg", 1),
        ("test", "d.jpg", 1),
    )
    for index, (split, relative_path, label) in enumerate(definitions):
        rows.append(
            {
                "relative_path": relative_path,
                "label": label,
                "surface": "D",
                "source_group": f"group-{index}",
                "source_group_verified": True,
                "sha256": format(index + 10, "x").ljust(64, "a"),
                "audit_status": "ok",
                "split": split,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    canonical_hash = manifest_sha256(manifest)
    evaluation = _write_final_evaluation(
        tmp_path / "artifacts" / "report" / "final_evaluation" / "e5",
        manifest_hash=canonical_hash,
    )
    predictions = pd.read_csv(evaluation / "predictions_test.csv")
    sha_by_path = manifest.set_index("relative_path")["sha256"]
    predictions["sha256"] = predictions["relative_path"].map(sha_by_path)
    predictions.to_csv(evaluation / "predictions_test.csv", index=False, lineterminator="\n")
    completion_path = evaluation / "evaluation_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifact_sha256"]["predictions_test.csv"] = sha256_file(
        evaluation / "predictions_test.csv"
    )
    write_json(completion_path, completion)

    benchmark = benchmark_callable(
        lambda: None,
        warmup_runs=0,
        measured_runs=2,
        target_seconds=5.0,
        environment={"device": "test-cpu"},
    )
    benchmark.update(
        {
            "run_id": "e5-run",
            "model_version": "0.1.0",
            "checkpoint_sha256": "a" * 64,
            "manifest_sha256": canonical_hash,
            "threshold": 0.45,
            "include_gradcam": False,
            "timing_scope": "decode_preprocess_inference",
            "valid_for_report": True,
            "status": "MEASURED",
        }
    )
    benchmark_path = tmp_path / "benchmark.json"
    write_json(benchmark_path, benchmark)
    output = tmp_path / "generated"

    facts_path = generate_report_assets(
        evaluation_dirs=[evaluation],
        output_dir=output,
        manifest_path=manifest_path,
        benchmark_paths=[benchmark_path],
        project_root=tmp_path,
    )

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    assert facts["dataset"]["row_count"] == 8
    assert facts["dataset"]["split_audit"]["valid"] is False
    assert facts["latency_benchmarks"][0]["target"]["seconds_per_image"] == 5.0
    assert (output / "dataset_summary.csv").is_file()
    assert (output / "dataset_summary.json").is_file()
    assert (output / "fig_dataset_distribution.png").is_file()
    assert (output / "table_latency_benchmark.csv").is_file()
    assert (output / "table_latency_benchmark.md").is_file()


def test_generate_report_assets_registers_separate_real_image_metrics(tmp_path: Path) -> None:
    evaluation = _write_final_evaluation(
        tmp_path / "artifacts" / "report" / "final_evaluation" / "e5"
    )
    real = _write_real_evaluation(tmp_path / "artifacts" / "report" / "real_images" / "e5")
    output = tmp_path / "generated"

    facts_path = generate_report_assets(
        evaluation_dirs=[evaluation],
        output_dir=output,
        real_evaluation_dirs=[real],
        project_root=tmp_path,
    )

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    result = facts["real_image_evaluations"][0]
    assert result["valid_for_report"] is True
    assert result["included_in_standard_test"] is False
    assert result["sample_count"] == 2
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["source_artifacts"]["evaluation_complete"]["sha256"] == sha256_file(
        real / "evaluation_complete.json"
    )

    predictions = pd.read_csv(real / "predictions_real.csv")
    predictions.loc[0, "probability_crack"] = 0.99
    predictions.to_csv(real / "predictions_real.csv", index=False, lineterminator="\n")
    with pytest.raises(ArtifactValidationError, match="metrics_test không khớp"):
        generate_report_assets(
            evaluation_dirs=[evaluation],
            output_dir=tmp_path / "other-report",
            real_evaluation_dirs=[real],
            project_root=tmp_path,
        )
