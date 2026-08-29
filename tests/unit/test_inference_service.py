from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crackspot.inference.service import (
    InferenceError,
    InferenceService,
    MetadataError,
    ModelLoadError,
    ModelMetadata,
    discover_metadata_path,
)


class ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.last_batch: np.ndarray | None = None

    def __call__(self, batch: np.ndarray, *, training: bool) -> np.ndarray:
        assert training is False
        self.last_batch = np.asarray(batch)
        return np.asarray([[self.probability]], dtype=np.float32)


def valid_metadata(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "threshold": 0.6,
        "model_version": "e4-v1",
        "run_id": "E4_20260829_120000",
        "input_size": [224, 224],
        "gradcam_layer": "out_relu",
        "preprocessing": "mobilenet_v2.preprocess_input",
        "label_mapping": {"Non-crack": 0, "Crack": 1},
    }
    payload.update(overrides)
    return payload


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), (64, 128, 192)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("probability", "expected_class", "expected_label"),
    [(0.599, 0, "Non-crack"), (0.6, 1, "Crack"), (0.9, 1, "Crack")],
)
def test_prediction_contract_and_threshold(
    probability: float, expected_class: int, expected_label: str
) -> None:
    model = ConstantModel(probability)
    service = InferenceService(model, valid_metadata())

    result = service.predict_image(png_bytes())

    assert result.predicted_class == expected_class
    assert result.predicted_label == expected_label
    assert result.crack_probability == pytest.approx(probability)
    assert result.threshold == 0.6
    assert result.model_version == "e4-v1"
    assert result.run_id == "E4_20260829_120000"
    assert result.latency_ms >= 0.0
    assert model.last_batch is not None
    assert model.last_batch.shape == (1, 224, 224, 3)
    assert model.last_batch.dtype == np.float32
    assert result.to_dict()["display_label_vi"] in {
        "Có vết nứt",
        "Không phát hiện vết nứt",
    }


def test_metadata_rejects_reversed_label_semantics() -> None:
    with pytest.raises(MetadataError, match="Nhãn bị đảo"):
        ModelMetadata.from_mapping(valid_metadata(label_mapping={"Non-crack": 1, "Crack": 0}))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True, "bad"])
def test_metadata_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises(MetadataError):
        ModelMetadata.from_mapping(valid_metadata(threshold=threshold))


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_metadata_accepts_closed_interval_threshold_endpoints(threshold: float) -> None:
    assert ModelMetadata.from_mapping(valid_metadata(threshold=threshold)).threshold == threshold


def test_service_rejects_non_probability_model_output() -> None:
    service = InferenceService(ConstantModel(1.5), valid_metadata())

    with pytest.raises(InferenceError, match=r"P\(Crack\)"):
        service.predict_image(png_bytes())


def test_metadata_discovery_prefers_named_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "candidate.keras"
    model.write_bytes(b"not loaded by this test")
    named = tmp_path / "candidate.metadata.json"
    named.write_text("{}", encoding="utf-8")
    (tmp_path / "candidate.json").write_text("{}", encoding="utf-8")

    assert discover_metadata_path(model) == named


def test_model_hash_mismatch_fails_before_tensorflow_load(tmp_path: Path) -> None:
    model = tmp_path / "candidate.keras"
    model.write_bytes(b"checkpoint bytes")
    metadata = tmp_path / "candidate.metadata.json"
    metadata.write_text(
        json.dumps(valid_metadata(model_sha256="0" * 64)),
        encoding="utf-8",
    )

    with pytest.raises(ModelLoadError, match="SHA-256"):
        InferenceService.from_files(model, metadata)


def test_from_files_reports_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="Không tìm thấy"):
        InferenceService.from_files(tmp_path / "missing.keras")
