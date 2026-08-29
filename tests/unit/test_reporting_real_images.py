from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from crackspot.inference import ModelMetadata
from crackspot.modeling.selection import create_selection_contract
from crackspot.reporting import real_images
from crackspot.reporting.export import write_json
from crackspot.reporting.real_images import (
    RealImageEvaluationError,
    evaluate_real_images,
    predict_real_frame,
    validate_real_manifest,
)
from crackspot.utils.hashing import sha256_file


class FakeService:
    def __init__(
        self,
        probabilities: dict[str, float],
        *,
        latency_ms: float = 2.0,
        metadata: ModelMetadata | None = None,
    ) -> None:
        self.probabilities = probabilities
        self.latency_ms = latency_ms
        self.metadata = metadata
        self.calls: list[tuple[str, bool]] = []

    def predict_image(self, image: Path, *, include_gradcam: bool) -> SimpleNamespace:
        self.calls.append((image.name, include_gradcam))
        return SimpleNamespace(
            crack_probability=self.probabilities[image.name],
            latency_ms=self.latency_ms,
        )


@pytest.mark.parametrize("split", ["train", "VAL", " validation ", "test"])
def test_real_manifest_refuses_standard_dataset_splits(split: str) -> None:
    frame = pd.DataFrame(
        {
            "relative_path": ["photo.jpg"],
            "label": [1],
            "split": [split],
            "capture_source": ["self_captured"],
        }
    )

    with pytest.raises(RealImageEvaluationError, match="không được gộp"):
        validate_real_manifest(frame, require_self_captured_declaration=True)


@pytest.mark.parametrize("source", [None, "downloaded", "synthetic", "self captured", ""])
def test_real_manifest_requires_explicit_self_captured_per_row(source: object) -> None:
    frame = pd.DataFrame({"relative_path": ["photo.jpg"], "label": [1], "capture_source": [source]})

    with pytest.raises(RealImageEvaluationError, match="capture_source=self_captured"):
        validate_real_manifest(frame, require_self_captured_declaration=True)


@pytest.mark.parametrize("label", [0.1, 0.9, True, False, float("nan")])
def test_real_manifest_refuses_labels_that_would_be_lossily_cast(label: object) -> None:
    frame = pd.DataFrame({"relative_path": ["photo.jpg"], "label": [label]})

    with pytest.raises(RealImageEvaluationError, match="Non-crack=0, Crack=1"):
        validate_real_manifest(frame, require_self_captured_declaration=False)


@pytest.mark.parametrize(
    "paths",
    [
        [None],
        ["../photo.jpg"],
        ["C:photo.jpg"],
        ["/photo.jpg"],
        ["PHOTO.jpg", "photo.jpg"],
    ],
)
def test_real_manifest_refuses_unsafe_or_ambiguous_paths(paths: list[object]) -> None:
    frame = pd.DataFrame({"relative_path": paths, "label": [0] * len(paths)})

    with pytest.raises(RealImageEvaluationError, match="relative_path|trùng"):
        validate_real_manifest(frame, require_self_captured_declaration=False)


def test_predict_real_frame_is_sorted_deterministic_and_uses_locked_threshold(
    tmp_path: Path,
) -> None:
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    frame = validate_real_manifest(
        pd.DataFrame(
            {
                "relative_path": ["d.jpg", "b.jpg", "a.jpg", "c.jpg"],
                "label": [1, 0, 0, 1],
                "capture_source": ["self_captured"] * 4,
            }
        ),
        require_self_captured_declaration=True,
    )
    service = FakeService({"a.jpg": 0.2, "b.jpg": 0.5, "c.jpg": 0.5, "d.jpg": 0.1})

    first = predict_real_frame(frame, dataset_root=tmp_path, service=service, threshold=0.5)
    second = predict_real_frame(frame, dataset_root=tmp_path, service=service, threshold=0.5)

    pd.testing.assert_frame_equal(first, second)
    assert first["relative_path"].tolist() == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert first["outcome"].tolist() == ["TN", "FP", "TP", "FN"]
    assert first["y_pred"].tolist() == [0, 1, 1, 0]
    assert first["image_sha256"].tolist() == [
        sha256_file(tmp_path / name) for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")
    ]
    assert all(include_gradcam is False for _, include_gradcam in service.calls)


@pytest.mark.parametrize(
    ("probability", "latency", "message"),
    [(-0.01, 1.0, "P\\(Crack\\)"), (1.01, 1.0, "P\\(Crack\\)"), (0.5, -1.0, "Latency")],
)
def test_predict_real_frame_refuses_invalid_service_values(
    tmp_path: Path,
    probability: float,
    latency: float,
    message: str,
) -> None:
    (tmp_path / "photo.jpg").write_bytes(b"photo")
    frame = pd.DataFrame({"relative_path": ["photo.jpg"], "label": [0]})

    with pytest.raises(RealImageEvaluationError, match=message):
        predict_real_frame(
            frame,
            dataset_root=tmp_path,
            service=FakeService({"photo.jpg": probability}, latency_ms=latency),
            threshold=0.5,
        )


def _evaluation_inputs(tmp_path: Path, *, metadata_smoke: bool = False) -> dict[str, Path]:
    checkpoint = tmp_path / "run" / "model.keras"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fake-keras-model")
    config_hash = "c" * 64
    manifest_hash = "d" * 64
    selection = checkpoint.parent / "selection_complete.json"
    create_selection_contract(
        experiment="E5",
        run_id="e5-run",
        checkpoint=checkpoint,
        config_sha256=config_hash,
        manifest_sha256=manifest_hash,
        threshold=0.5,
        output=selection,
    )
    metadata = checkpoint.parent / "model.metadata.json"
    write_json(
        metadata,
        {
            "run_id": "e5-run",
            "model_version": "0.1.0",
            "threshold": 0.5,
            "input_size": [224, 224],
            "gradcam_layer": "out_relu",
            "preprocessing": "mobilenet_v2.preprocess_input",
            "label_mapping": {"Non-crack": 0, "Crack": 1},
            "model_sha256": sha256_file(checkpoint),
            "manifest_sha256": manifest_hash,
            "config_sha256": config_hash,
            "smoke_test": metadata_smoke,
            "status": "NOT_VALID_FOR_REPORT"
            if metadata_smoke
            else "VALIDATION_COMPLETE_TEST_LOCKED",
        },
    )
    image_root = tmp_path / "real"
    image_root.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (image_root / name).write_bytes(name.encode())
    manifest = image_root / "manifest.csv"
    pd.DataFrame(
        {
            "relative_path": ["d.jpg", "b.jpg", "a.jpg", "c.jpg"],
            "label": [1, 0, 0, 1],
            "capture_source": ["self_captured"] * 4,
        }
    ).to_csv(manifest, index=False, lineterminator="\n")
    return {
        "checkpoint": checkpoint,
        "selection": selection,
        "metadata": metadata,
        "image_root": image_root,
        "manifest": manifest,
    }


def _run_evaluation(
    paths: dict[str, Path],
    output: Path,
    *,
    smoke: bool = False,
    confirm_self_captured: bool = True,
) -> real_images.RealImageEvaluationResult:
    return evaluate_real_images(
        selection_path=paths["selection"],
        manifest_path=paths["manifest"],
        dataset_root=paths["image_root"],
        output_dir=output,
        metadata_path=paths["metadata"],
        confirm_self_captured=confirm_self_captured,
        smoke=smoke,
        service=FakeService(
            {"a.jpg": 0.2, "b.jpg": 0.5, "c.jpg": 0.5, "d.jpg": 0.1},
            metadata=ModelMetadata.from_json(paths["metadata"]),
        ),
    )


def test_evaluate_real_images_writes_deterministic_metrics_and_integrity_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _evaluation_inputs(tmp_path)
    monkeypatch.setattr(real_images, "capture_environment", lambda: {"device": "fake"})
    monkeypatch.setattr(
        real_images,
        "plot_confusion_matrix",
        lambda _matrix, output, **_kwargs: Path(output).write_bytes(b"deterministic-png"),
    )
    output = tmp_path / "artifacts" / "real-evaluation"

    result = _run_evaluation(paths, output)

    assert result.output_dir == output.resolve()
    assert result.completion_path.is_file()
    assert result.status == "REAL_IMAGE_EVALUATION_COMPLETE"
    assert result.sample_count == 4
    assert result.accuracy == pytest.approx(0.5)
    assert result.f1_crack == pytest.approx(0.5)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["included_in_standard_test"] is False
    assert metrics["self_captured_confirmed"] is True
    assert metrics["valid_for_report"] is True
    completion = json.loads(result.completion_path.read_text(encoding="utf-8"))
    assert completion["immutable"] is True
    assert completion["sample_count"] == 4
    for name, expected_hash in completion["artifact_sha256"].items():
        assert sha256_file(output / name) == expected_hash
    snapshot = pd.read_csv(output / "manifest_real_snapshot.csv")
    assert snapshot["relative_path"].tolist() == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert snapshot["split"].eq("real_external").all()
    assert snapshot["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    with pytest.raises(FileExistsError, match="ghi đè"):
        _run_evaluation(paths, output)


def test_smoke_checkpoint_requires_smoke_and_remains_non_reportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _evaluation_inputs(tmp_path, metadata_smoke=True)
    output = tmp_path / "smoke-real"
    with pytest.raises(RealImageEvaluationError, match="bắt buộc dùng --smoke"):
        _run_evaluation(paths, output)
    assert not output.exists()

    monkeypatch.setattr(real_images, "capture_environment", lambda: {})
    monkeypatch.setattr(
        real_images,
        "plot_confusion_matrix",
        lambda _matrix, destination, **_kwargs: Path(destination).write_bytes(b"png"),
    )
    result = _run_evaluation(paths, output, smoke=True)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert result.status == "NOT_VALID_FOR_REPORT"
    assert metrics["valid_for_report"] is False


def test_unconfirmed_source_is_refused_for_report_and_named_unverified_in_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _evaluation_inputs(tmp_path)
    report_output = tmp_path / "report-real"
    with pytest.raises(RealImageEvaluationError, match="--confirm-self-captured"):
        _run_evaluation(paths, report_output, confirm_self_captured=False)
    assert not report_output.exists()

    monkeypatch.setattr(real_images, "capture_environment", lambda: {})
    monkeypatch.setattr(
        real_images,
        "plot_confusion_matrix",
        lambda _matrix, destination, **_kwargs: Path(destination).write_bytes(b"png"),
    )
    smoke_output = tmp_path / "smoke-real"
    result = _run_evaluation(
        paths,
        smoke_output,
        smoke=True,
        confirm_self_captured=False,
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["status"] == "NOT_VALID_FOR_REPORT"
    assert metrics["self_captured_confirmed"] is False
    assert metrics["evaluation_scope"] == "external_images_smoke_unverified_source"


def test_failed_staging_never_publishes_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _evaluation_inputs(tmp_path)
    target = tmp_path / "artifacts" / "real-evaluation"

    def fail_after_partial_write(staging: Path, **_kwargs: Any) -> None:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated artifact failure")

    monkeypatch.setattr(real_images, "_write_artifacts", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated artifact failure"):
        _run_evaluation(paths, target)

    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))


def test_image_changed_during_staging_prevents_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _evaluation_inputs(tmp_path)
    target = tmp_path / "artifacts" / "real-evaluation"
    monkeypatch.setattr(real_images, "capture_environment", lambda: {})
    monkeypatch.setattr(
        real_images,
        "plot_confusion_matrix",
        lambda _matrix, destination, **_kwargs: Path(destination).write_bytes(b"png"),
    )
    write_artifacts = real_images._write_artifacts

    def write_then_mutate(staging: Path, **kwargs: Any) -> None:
        write_artifacts(staging, **kwargs)
        (paths["image_root"] / "a.jpg").write_bytes(b"changed-after-prediction")

    monkeypatch.setattr(real_images, "_write_artifacts", write_then_mutate)

    with pytest.raises(RealImageEvaluationError, match="đã thay đổi"):
        _run_evaluation(paths, target)

    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))


def test_provenance_rejects_wrong_run_or_config_before_inference(tmp_path: Path) -> None:
    paths = _evaluation_inputs(tmp_path)
    payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    payload["run_id"] = "different-run"
    write_json(paths["metadata"], payload)
    service = FakeService(
        {"a.jpg": 0.1, "b.jpg": 0.1, "c.jpg": 0.1, "d.jpg": 0.1},
        metadata=ModelMetadata.from_json(paths["metadata"]),
    )

    with pytest.raises(RealImageEvaluationError, match="run_id"):
        evaluate_real_images(
            selection_path=paths["selection"],
            manifest_path=paths["manifest"],
            dataset_root=paths["image_root"],
            output_dir=tmp_path / "output",
            metadata_path=paths["metadata"],
            confirm_self_captured=True,
            service=service,
        )

    assert service.calls == []
    assert not (tmp_path / "output").exists()
