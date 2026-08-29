from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from crackspot.data import (
    audit_split,
    create_curation_bundle,
    create_locked_split_bundle,
    manifest_sha256,
)
from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.reporting.aggregate import ArtifactValidationError, generate_report_assets
from crackspot.reporting.benchmark import benchmark_callable
from crackspot.reporting.export import write_json
from crackspot.utils.hashing import sha256_file, sha256_json


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 12), color).save(path)


def _write_locked_bundle(root: Path) -> tuple[Path, Path, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    index = 0
    for surface in ("D", "P", "W"):
        for label in (0, 1):
            for group_index in range(20):
                index += 1
                class_folder = ("C" if label else "U") + surface
                rows.append(
                    {
                        "relative_path": f"{surface}/{class_folder}/{index:03d}-1.jpg",
                        "label": label,
                        "surface": surface,
                        "source_group": f"{surface}-{label}-{group_index}",
                        "source_group_verified": True,
                        "sha256": format(index, "x").zfill(64),
                        "audit_status": "ok",
                        "split": "",
                    }
                )
    pre_split = pd.DataFrame(rows)
    curation_dir = root / "data" / "manifests" / "pre_split_curation_v1"
    audit_path = root / "data" / "manifests" / "audit_manifest.csv"
    audit_path.parent.mkdir(parents=True)
    pre_split.to_csv(audit_path, index=False, lineterminator="\n")
    curation = create_curation_bundle(audit_path, curation_dir)
    bundle = root / "data" / "manifests" / "split_v1"
    create_locked_split_bundle(
        curation.cleaned_manifest_path,
        bundle,
        conflict_report_path=curation.conflict_report_path,
        seed=42,
        restarts=64,
    )
    manifest_path = bundle / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    return bundle, manifest_path, manifest


def _write_training_run(
    root: Path,
    experiment: str,
    *,
    manifest_hash: str,
    validation_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    run = root / "artifacts" / "runs" / experiment.lower()
    run.mkdir(parents=True)
    config = {"experiment": {"id": experiment, "name": experiment.lower()}}
    write_json(run / "config_snapshot.json", config)
    (run / "model.keras").write_bytes(f"official-{experiment}".encode())
    history = pd.DataFrame(
        {
            "loss": [0.8, 0.5],
            "val_loss": [0.9, 0.6],
            "accuracy": [0.6, 0.8],
            "val_accuracy": [0.55, 0.75],
        }
    )
    history.to_csv(run / "history.csv", index=False, lineterminator="\n")
    if validation_predictions is not None:
        validation_predictions.to_csv(
            run / "predictions_validation.csv", index=False, lineterminator="\n"
        )
    summary = {
        "run_id": f"{experiment.lower()}-run",
        "experiment": experiment,
        "best_val_loss": 0.6,
        "config_sha256": sha256_json(config),
        "manifest_sha256": manifest_hash,
        "model_sha256": sha256_file(run / "model.keras"),
        "valid_for_report": True,
        "status": "VALIDATION_COMPLETE_TEST_LOCKED",
    }
    write_json(run / "run_summary.json", summary)
    artifact_names = ["config_snapshot.json", "model.keras", "history.csv", "run_summary.json"]
    if validation_predictions is not None:
        artifact_names.append("predictions_validation.csv")
    write_json(
        run / "training_complete.json",
        {
            "schema_version": 1,
            "status": "VALIDATION_COMPLETE_TEST_LOCKED",
            "run_id": summary["run_id"],
            "config_sha256": summary["config_sha256"],
            "manifest_sha256": manifest_hash,
            "model_sha256": summary["model_sha256"],
            "artifact_sha256": {name: sha256_file(run / name) for name in artifact_names},
            "immutable": True,
        },
    )
    return {
        "directory": run,
        "experiment": experiment,
        "run_id": summary["run_id"],
        "checkpoint_sha256": summary["model_sha256"],
        "config_sha256": summary["config_sha256"],
        "manifest_sha256": manifest_hash,
    }


def _write_final_evaluation(
    root: Path,
    *,
    selection_experiment: str,
    trained_experiment: str,
    training: dict[str, Any],
    test_manifest: pd.DataFrame,
    threshold: float,
    threshold_result_sha256: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    truth = pd.to_numeric(test_manifest["label"], errors="raise").astype(int)
    scores = truth.map({0: 0.1, 1: 0.9}).astype(float)
    predictions = pd.DataFrame(
        {
            "relative_path": test_manifest["relative_path"].astype(str),
            "sha256": test_manifest["sha256"].astype(str),
            "split": "test",
            "y_true": truth,
            "probability_crack": scores,
        }
    )
    predictions.to_csv(root / "predictions_test.csv", index=False, lineterminator="\n")
    common = {
        "status": "FINAL_TEST_COMPLETE",
        "valid_for_report": True,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "config_sha256": training["config_sha256"],
        "manifest_sha256": training["manifest_sha256"],
    }
    metrics = compute_binary_metrics(truth, scores, threshold)
    metrics.update({**common, "threshold_role": "validation_locked"})
    write_json(root / "metrics_test.json", metrics)
    fixed = compute_binary_metrics(truth, scores, 0.5)
    fixed.update({**common, "threshold_role": "fixed_protocol_0_5"})
    write_json(root / "metrics_test_threshold_0_5.json", fixed)
    write_json(
        root / "evaluation_metadata.json",
        {
            **common,
            "smoke_test": False,
            "experiment": selection_experiment,
            "run_id": training["run_id"],
            "threshold": threshold,
        },
    )
    selection = {
        "experiment": selection_experiment,
        "trained_experiment": trained_experiment,
        "run_id": training["run_id"],
        "checkpoint": str(training["directory"] / "model.keras"),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "config_sha256": training["config_sha256"],
        "manifest_sha256": training["manifest_sha256"],
        "threshold": threshold,
        "selected_by": "validation",
        "threshold_source": ("validation" if selection_experiment == "E5" else "fixed_protocol"),
    }
    if threshold_result_sha256 is not None:
        selection["threshold_result_sha256"] = threshold_result_sha256
    write_json(root / "selection_contract_snapshot.json", selection)
    artifact_names = (
        "metrics_test.json",
        "metrics_test_threshold_0_5.json",
        "predictions_test.csv",
        "evaluation_metadata.json",
        "selection_contract_snapshot.json",
    )
    write_json(
        root / "evaluation_complete.json",
        {
            "status": "FINAL_TEST_COMPLETE",
            "artifact_sha256": {name: sha256_file(root / name) for name in artifact_names},
        },
    )
    return root


def _write_real_evaluation(root: Path, e5: dict[str, Any], threshold: float) -> Path:
    root.mkdir(parents=True)
    truth = [0, 1]
    scores = [0.1, 0.9]
    pd.DataFrame(
        {
            "relative_path": ["real/negative.jpg", "real/crack.jpg"],
            "split": ["real_external", "real_external"],
            "y_true": truth,
            "probability_crack": scores,
            "threshold": [threshold, threshold],
        }
    ).to_csv(root / "predictions_real.csv", index=False, lineterminator="\n")
    common = {
        "status": "REAL_IMAGE_EVALUATION_COMPLETE",
        "valid_for_report": True,
        "included_in_standard_test": False,
        "self_captured_confirmed": True,
        "evaluation_scope": "external_self_captured_images",
        "checkpoint_sha256": e5["checkpoint_sha256"],
        "config_sha256": e5["config_sha256"],
        "sdnet_manifest_sha256": e5["manifest_sha256"],
        "experiment": "E5",
        "run_id": e5["run_id"],
    }
    metrics = compute_binary_metrics(truth, scores, threshold)
    metrics.update(common)
    write_json(root / "metrics_real.json", metrics)
    write_json(root / "evaluation_metadata.json", common)
    write_json(
        root / "selection_contract_snapshot.json",
        {
            "checkpoint_sha256": e5["checkpoint_sha256"],
            "config_sha256": e5["config_sha256"],
            "manifest_sha256": e5["manifest_sha256"],
            "experiment": "E5",
            "run_id": e5["run_id"],
            "threshold": threshold,
            "selected_by": "validation",
        },
    )
    names = (
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
            "artifact_sha256": {name: sha256_file(root / name) for name in names},
        },
    )
    return root


@pytest.fixture(scope="module")
def official_suite(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("official-report-suite")
    bundle, manifest_path, manifest = _write_locked_bundle(root)
    manifest_hash = manifest_sha256(manifest)
    validation_rows = manifest.loc[manifest["split"].eq("validation")].copy()
    negatives = validation_rows.loc[validation_rows["label"].eq("0")].head(2)
    positives = validation_rows.loc[validation_rows["label"].eq("1")].head(2)
    selected_validation = pd.concat((negatives, positives), ignore_index=True)
    validation_predictions = pd.DataFrame(
        {
            "relative_path": selected_validation["relative_path"],
            "split": "validation",
            "y_true": [0, 0, 1, 1],
            "probability_crack": [0.1, 0.4, 0.45, 0.9],
        }
    )
    training = {
        experiment: _write_training_run(
            root,
            experiment,
            manifest_hash=manifest_hash,
            validation_predictions=(validation_predictions if experiment == "E4" else None),
        )
        for experiment in ("E1", "E2", "E3", "E4")
    }
    validation_path = training["E4"]["directory"] / "predictions_validation.csv"
    threshold_path = training["E4"]["directory"] / "threshold_validation.json"
    threshold_payload = {
        "schema_version": 2,
        "artifact_type": "validation_threshold_selection",
        "selected_by": "validation",
        "status": "VALIDATION_THRESHOLD_SELECTED",
        "valid_for_report": True,
        "source_split": "validation",
        "source_predictions": str(validation_path.resolve()),
        "threshold": 0.45,
        "sample_count": len(validation_predictions),
        "predictions_sha256": sha256_file(validation_path),
        "training_complete_sha256": sha256_file(
            training["E4"]["directory"] / "training_complete.json"
        ),
        "checkpoint_sha256": training["E4"]["checkpoint_sha256"],
        "config_sha256": training["E4"]["config_sha256"],
        "manifest_sha256": manifest_hash,
        "run_id": training["E4"]["run_id"],
    }
    write_json(threshold_path, threshold_payload)
    test_manifest = manifest.loc[manifest["split"].eq("test")].copy()
    evaluations = {
        experiment: _write_final_evaluation(
            root / "official-evaluations" / experiment.lower(),
            selection_experiment=experiment,
            trained_experiment=experiment,
            training=training[experiment],
            test_manifest=test_manifest,
            threshold=0.5,
        )
        for experiment in ("E1", "E2", "E3")
    }
    evaluations["E5"] = _write_final_evaluation(
        root / "official-evaluations" / "e5",
        selection_experiment="E5",
        trained_experiment="E4",
        training=training["E4"],
        test_manifest=test_manifest,
        threshold=0.45,
        threshold_result_sha256=sha256_file(threshold_path),
    )
    benchmark = benchmark_callable(
        lambda: None,
        warmup_runs=0,
        measured_runs=2,
        target_seconds=5.0,
        environment={"device": "official-test-cpu"},
    )
    benchmark.update(
        {
            "run_id": training["E4"]["run_id"],
            "model_version": "test",
            "checkpoint_sha256": training["E4"]["checkpoint_sha256"],
            "manifest_sha256": manifest_hash,
            "valid_for_report": True,
            "status": "MEASURED",
        }
    )
    benchmark_path = root / "official-benchmark.json"
    write_json(benchmark_path, benchmark)
    real = _write_real_evaluation(root / "official-real-evaluation", training["E4"], threshold=0.45)

    augmentation_png = root / "qualitative" / "augmentation.png"
    _write_png(augmentation_png, (40, 80, 120))
    augmentation_sidecar = augmentation_png.with_suffix(".json")
    write_json(
        augmentation_sidecar,
        {
            "schema_version": 1,
            "kind": "train_augmentation_before_after",
            "status": "REPORT_ARTIFACT",
            "valid_for_report": True,
            "output": str(augmentation_png.resolve()),
            "output_sha256": sha256_file(augmentation_png),
            "experiment": "E4",
            "run_id": training["E4"]["run_id"],
            "config_path": str(training["E4"]["directory"] / "config_snapshot.json"),
            "config_file_sha256": sha256_file(training["E4"]["directory"] / "config_snapshot.json"),
            "config_sha256": training["E4"]["config_sha256"],
            "manifest_sha256": manifest_hash,
            "manifest_file_sha256": sha256_file(manifest_path),
            "split_audit": audit_split(manifest),
            "augmentation": {
                "horizontal_flip_probability": 0.5,
                "max_rotation_degrees": 15.0,
                "brightness_delta": 0.15,
                "contrast_delta": 0.15,
            },
        },
    )
    e5_predictions = evaluations["E5"] / "predictions_test.csv"
    e5_selection = evaluations["E5"] / "selection_contract_snapshot.json"
    gradcam_png = root / "qualitative" / "gradcam.png"
    _write_png(gradcam_png, (120, 60, 20))
    gradcam_sidecar = gradcam_png.with_suffix(".json")
    write_json(
        gradcam_sidecar,
        {
            "schema_version": 1,
            "kind": "gradcam_tp_tn_fp_fn_grid",
            "status": "REPORT_ARTIFACT",
            "valid_for_report": True,
            "output": str(gradcam_png.resolve()),
            "output_sha256": sha256_file(gradcam_png),
            "experiment": "E5",
            "run_id": training["E4"]["run_id"],
            "model_sha256": training["E4"]["checkpoint_sha256"],
            "config_sha256": training["E4"]["config_sha256"],
            "manifest_sha256": manifest_hash,
            "threshold": 0.45,
            "predictions_path": str(e5_predictions.resolve()),
            "predictions_sha256": sha256_file(e5_predictions),
            "selection_path": str(e5_selection.resolve()),
            "selection_sha256": sha256_file(e5_selection),
            "selected": {"TP": {}, "TN": {}, "FP": None, "FN": None},
        },
    )
    return {
        "root": root,
        "evaluations": evaluations,
        "training": training,
        "kwargs": {
            "evaluation_dirs": list(evaluations.values()),
            "manifest_path": manifest_path,
            "split_bundle_dir": bundle,
            "validation_predictions": validation_path,
            "threshold_result_path": threshold_path,
            "benchmark_paths": [benchmark_path],
            "real_evaluation_dirs": [real],
            "training_run_dirs": [item["directory"] for item in training.values()],
            "augmentation_sidecar": augmentation_sidecar,
            "gradcam_sidecar": gradcam_sidecar,
            "project_root": root,
        },
    }


def test_official_report_requires_complete_suite_and_is_atomic(
    official_suite: dict[str, Any],
) -> None:
    output = official_suite["root"] / "artifacts" / "report" / "complete"
    facts_path = generate_report_assets(output_dir=output, **official_suite["kwargs"])
    facts = json.loads(facts_path.read_text(encoding="utf-8"))

    assert facts["status"] == "FINAL_REPORT_FACTS"
    assert facts["valid_for_report"] is True
    assert facts["missing_evidence"] == []
    assert [row["experiment"] for row in facts["experiments"]] == ["E1", "E2", "E3", "E4", "E5"]
    assert (
        facts["experiments"][3]["checkpoint_sha256"] == facts["experiments"][4]["checkpoint_sha256"]
    )
    completion = json.loads((output / "report_complete.json").read_text(encoding="utf-8"))
    assert completion["status"] == "FINAL_REPORT_COMPLETE"
    assert completion["artifact_sha256"]["report_facts.json"] == sha256_file(facts_path)
    assert all(
        (output / f"fig_training_curves_{experiment}.png").is_file()
        for experiment in ("e1", "e2", "e3", "e4")
    )


@pytest.mark.parametrize(
    "missing_key",
    [
        "evaluation:E1",
        "evaluation:E2",
        "evaluation:E3",
        "evaluation:E5",
        "locked_manifest_bundle",
        "threshold",
        "benchmark",
        "real",
        "training:E1",
        "training:E2",
        "training:E3",
        "training:E4",
        "augmentation",
        "gradcam",
    ],
)
def test_official_report_refuses_every_missing_evidence_without_partial_output(
    official_suite: dict[str, Any],
    tmp_path: Path,
    missing_key: str,
) -> None:
    kwargs = {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in official_suite["kwargs"].items()
    }
    if missing_key.startswith("evaluation:"):
        experiment = missing_key.split(":", 1)[1]
        omitted = official_suite["evaluations"][experiment]
        kwargs["evaluation_dirs"] = [path for path in kwargs["evaluation_dirs"] if path != omitted]
    elif missing_key.startswith("training:"):
        experiment = missing_key.split(":", 1)[1]
        omitted = official_suite["training"][experiment]["directory"]
        kwargs["training_run_dirs"] = [
            path for path in kwargs["training_run_dirs"] if path != omitted
        ]
    elif missing_key == "locked_manifest_bundle":
        kwargs["split_bundle_dir"] = None
    elif missing_key == "threshold":
        kwargs["threshold_result_path"] = None
    elif missing_key == "benchmark":
        kwargs["benchmark_paths"] = []
    elif missing_key == "real":
        kwargs["real_evaluation_dirs"] = []
    elif missing_key == "augmentation":
        kwargs["augmentation_sidecar"] = None
    elif missing_key == "gradcam":
        kwargs["gradcam_sidecar"] = None
    output = tmp_path / "artifacts" / "report" / missing_key.replace(":", "-")

    with pytest.raises(ArtifactValidationError):
        generate_report_assets(output_dir=output, **kwargs)

    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


@pytest.mark.parametrize("artifact", ["augmentation", "gradcam"])
def test_official_report_rejects_tampered_qualitative_png(
    official_suite: dict[str, Any],
    tmp_path: Path,
    artifact: str,
) -> None:
    source_sidecar = Path(official_suite["kwargs"][f"{artifact}_sidecar"])
    copied_png = tmp_path / f"{artifact}.png"
    copied_sidecar = copied_png.with_suffix(".json")
    copied_png.write_bytes(source_sidecar.with_suffix(".png").read_bytes() + b"tamper")
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload["output"] = str(copied_png.resolve())
    write_json(copied_sidecar, payload)
    kwargs = dict(official_suite["kwargs"])
    kwargs[f"{artifact}_sidecar"] = copied_sidecar

    with pytest.raises(ArtifactValidationError, match="PNG hash"):
        generate_report_assets(
            output_dir=tmp_path / "artifacts" / "report" / f"tampered-{artifact}",
            **kwargs,
        )


def test_partial_output_never_claims_final_even_with_complete_evidence(
    official_suite: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "partial"
    facts_path = generate_report_assets(output_dir=output, **official_suite["kwargs"])
    facts = json.loads(facts_path.read_text(encoding="utf-8"))

    assert facts["status"] == "REPORT_INCOMPLETE"
    assert facts["valid_for_report"] is False
    assert facts["missing_evidence"] == ["official_output_path"]
    completion = json.loads((output / "report_complete.json").read_text(encoding="utf-8"))
    assert completion["status"] == "REPORT_INCOMPLETE"
