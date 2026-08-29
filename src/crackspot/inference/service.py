"""Checkpoint loading and one-image inference for all CrackSpot front ends."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from crackspot.constants import (
    CRACK_LABEL,
    DEFAULT_GRADCAM_LAYER,
    DEFAULT_THRESHOLD,
    IMAGE_SIZE,
    LABEL_MAPPING,
    PREPROCESSING_NAME,
)
from crackspot.inference.preprocessing import ImageLimits, ImageSource, prepare_image


class InferenceError(RuntimeError):
    """Base error for model loading and prediction."""


class MetadataError(InferenceError):
    """Raised when model metadata violates the locked inference contract."""


class ModelLoadError(InferenceError):
    """Raised when a checkpoint is missing, corrupt, or fails integrity checks."""


@dataclass(frozen=True)
class ModelMetadata:
    """Validated metadata saved beside the full Keras checkpoint."""

    threshold: float = DEFAULT_THRESHOLD
    model_version: str = "unknown"
    run_id: str = "unknown"
    input_size: tuple[int, int] = IMAGE_SIZE
    gradcam_layer: str = DEFAULT_GRADCAM_LAYER
    preprocessing: str = PREPROCESSING_NAME
    label_mapping: dict[str, int] = field(default_factory=lambda: dict(LABEL_MAPPING))
    model_sha256: str | None = None
    manifest_sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def display_version(self) -> str:
        if self.model_version != "unknown":
            return self.model_version
        return self.run_id

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ModelMetadata:
        if not isinstance(payload, Mapping):
            raise MetadataError("Metadata phải là một JSON object.")

        try:
            threshold_value = payload.get("threshold", DEFAULT_THRESHOLD)
            if isinstance(threshold_value, bool):
                raise TypeError
            threshold = float(threshold_value)
        except (TypeError, ValueError) as exc:
            raise MetadataError("metadata.threshold phải là số.") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise MetadataError("metadata.threshold phải nằm trong [0, 1].")

        raw_size = payload.get("input_size", list(IMAGE_SIZE))
        if (
            not isinstance(raw_size, list | tuple)
            or len(raw_size) != 2
            or any(isinstance(value, bool) for value in raw_size)
        ):
            raise MetadataError("metadata.input_size phải là [height, width].")
        try:
            input_size = tuple(int(value) for value in raw_size)
        except (TypeError, ValueError) as exc:
            raise MetadataError("metadata.input_size phải chứa hai số nguyên.") from exc
        if any(float(raw) != parsed for raw, parsed in zip(raw_size, input_size, strict=True)):
            raise MetadataError("metadata.input_size phải chứa hai số nguyên.")
        if input_size != IMAGE_SIZE:
            raise MetadataError(
                f"CrackSpot yêu cầu input_size={list(IMAGE_SIZE)}, nhận {list(input_size)}."
            )

        preprocessing = str(payload.get("preprocessing", PREPROCESSING_NAME))
        if preprocessing != PREPROCESSING_NAME:
            raise MetadataError(
                f"Preprocessing không tương thích: '{preprocessing}', cần '{PREPROCESSING_NAME}'."
            )

        mapping_payload = payload.get("label_mapping", LABEL_MAPPING)
        if not isinstance(mapping_payload, Mapping):
            raise MetadataError("metadata.label_mapping phải là JSON object.")
        try:
            mapping = {str(key): int(value) for key, value in mapping_payload.items()}
        except (TypeError, ValueError) as exc:
            raise MetadataError("metadata.label_mapping chứa giá trị không hợp lệ.") from exc
        if mapping != LABEL_MAPPING:
            raise MetadataError(f"Nhãn bị đảo: CrackSpot bắt buộc {LABEL_MAPPING}.")

        model_version = _normalise_identifier(payload.get("model_version"), "model_version")
        run_id = _normalise_identifier(payload.get("run_id"), "run_id")
        if model_version is None and run_id is None:
            model_version = "unknown"
            run_id = "unknown"
        elif model_version is None:
            model_version = "unknown"
        elif run_id is None:
            run_id = "unknown"

        gradcam_layer = payload.get("gradcam_layer", DEFAULT_GRADCAM_LAYER)
        if not isinstance(gradcam_layer, str) or not gradcam_layer.strip():
            raise MetadataError("metadata.gradcam_layer phải là chuỗi không rỗng.")

        model_sha256 = _validate_optional_sha256(payload.get("model_sha256"), "model_sha256")
        manifest_sha256 = _validate_optional_sha256(
            payload.get("manifest_sha256"), "manifest_sha256"
        )
        known = {
            "threshold",
            "model_version",
            "run_id",
            "input_size",
            "gradcam_layer",
            "preprocessing",
            "label_mapping",
            "model_sha256",
            "manifest_sha256",
        }
        extra = {str(key): value for key, value in payload.items() if key not in known}
        return cls(
            threshold=threshold,
            model_version=model_version,
            run_id=run_id,
            input_size=(input_size[0], input_size[1]),
            gradcam_layer=gradcam_layer.strip(),
            preprocessing=preprocessing,
            label_mapping=mapping,
            model_sha256=model_sha256,
            manifest_sha256=manifest_sha256,
            extra=extra,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelMetadata:
        metadata_path = Path(path).expanduser()
        if not metadata_path.is_file():
            raise MetadataError(f"Không tìm thấy metadata: {metadata_path}")
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MetadataError(f"Metadata JSON bị hỏng: {metadata_path}") from exc
        return cls.from_mapping(payload)


def _normalise_identifier(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"metadata.{field_name} phải là chuỗi không rỗng.")
    return value.strip()


def _validate_optional_sha256(value: Any, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise MetadataError(f"metadata.{field_name} phải là SHA-256 64 ký tự hex.")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise MetadataError(f"metadata.{field_name} phải là SHA-256 64 ký tự hex.")
    return lowered


@dataclass(frozen=True)
class PredictionResult:
    """Classification, provenance, timing, and optional visual explanation."""

    crack_probability: float
    predicted_class: int
    predicted_label: str
    threshold: float
    latency_ms: float
    model_version: str
    run_id: str
    original_image: Image.Image = field(repr=False, compare=False)
    heatmap: np.ndarray | None = field(default=None, repr=False, compare=False)
    overlay: Image.Image | None = field(default=None, repr=False, compare=False)
    gradcam_latency_ms: float | None = None

    @property
    def is_crack(self) -> bool:
        return self.predicted_class == CRACK_LABEL

    @property
    def probability(self) -> float:
        """Concise alias for the explicitly named Crack probability."""

        return self.crack_probability

    @property
    def label(self) -> str:
        return self.predicted_label

    @property
    def inference_time_ms(self) -> float:
        return self.latency_ms

    @property
    def display_label_vi(self) -> str:
        return "Có vết nứt" if self.is_crack else "Không phát hiện vết nứt"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe portion of a result (images remain in memory)."""

        return {
            "crack_probability": self.crack_probability,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "display_label_vi": self.display_label_vi,
            "threshold": self.threshold,
            "latency_ms": self.latency_ms,
            "gradcam_latency_ms": self.gradcam_latency_ms,
            "model_version": self.model_version,
            "run_id": self.run_id,
        }


def discover_metadata_path(model_path: str | Path) -> Path:
    """Find the conventional JSON sidecar for a Keras checkpoint."""

    path = Path(model_path).expanduser()
    candidates = (
        path.with_suffix(".metadata.json"),
        path.with_suffix(".json"),
        path.parent / "metadata.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MetadataError(
        "Không tìm thấy metadata checkpoint. Đã tìm: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ModelLoadError(f"Không thể đọc checkpoint: {path}") from exc
    return digest.hexdigest()


def _load_keras_model(path: Path) -> Any:
    try:
        import tensorflow as tf
    except (ImportError, OSError, ValueError) as exc:
        raise ModelLoadError(
            "Không thể khởi tạo TensorFlow. Hãy cài bộ dependency demo tương thích."
        ) from exc
    try:
        return tf.keras.models.load_model(path, compile=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"Checkpoint Keras bị hỏng hoặc không tương thích: {path}") from exc


class InferenceService:
    """Thread-safe, reusable service around one loaded model and its metadata."""

    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata | Mapping[str, Any],
        *,
        image_limits: ImageLimits | None = None,
    ) -> None:
        if model is None:
            raise ModelLoadError("Mô hình suy luận không được để trống.")
        self.model = model
        self.metadata = (
            metadata
            if isinstance(metadata, ModelMetadata)
            else ModelMetadata.from_mapping(metadata)
        )
        self.image_limits = image_limits or ImageLimits()
        self._lock = threading.RLock()

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        verify_hash: bool = True,
        image_limits: ImageLimits | None = None,
    ) -> InferenceService:
        checkpoint = Path(model_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise ModelLoadError(f"Không tìm thấy checkpoint .keras: {checkpoint}")
        if checkpoint.suffix.lower() != ".keras":
            raise ModelLoadError(f"Checkpoint phải là full model .keras: {checkpoint}")

        sidecar = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else discover_metadata_path(checkpoint).resolve()
        )
        metadata = ModelMetadata.from_json(sidecar)
        if verify_hash and metadata.model_sha256:
            actual_hash = _sha256_file(checkpoint)
            if actual_hash != metadata.model_sha256:
                raise ModelLoadError(
                    "SHA-256 checkpoint không khớp metadata: "
                    f"expected={metadata.model_sha256}, actual={actual_hash}."
                )
        model = _load_keras_model(checkpoint)
        return cls(model, metadata, image_limits=image_limits)

    def _predict_probability(self, batch: np.ndarray) -> float:
        try:
            raw = (
                self.model(batch, training=False)
                if callable(self.model)
                else self.model.predict(batch)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InferenceError("Mô hình không thể suy luận trên ảnh này.") from exc
        if isinstance(raw, Mapping):
            if len(raw) != 1:
                raise InferenceError("Mô hình trả nhiều đầu ra, mong đợi một P(Crack).")
            raw = next(iter(raw.values()))
        if isinstance(raw, list | tuple):
            if len(raw) != 1:
                raise InferenceError("Mô hình trả nhiều đầu ra, mong đợi một P(Crack).")
            raw = raw[0]
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        values = np.asarray(raw, dtype=np.float64)
        if values.size != 1:
            raise InferenceError(f"Đầu ra mô hình phải có một phần tử, nhận shape {values.shape}.")
        probability = float(values.reshape(-1)[0])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise InferenceError(
                f"Đầu ra {probability!r} không phải xác suất sigmoid P(Crack) trong [0,1]."
            )
        return probability

    def predict_image(
        self,
        image_or_bytes: ImageSource,
        *,
        include_gradcam: bool = False,
        gradcam_alpha: float = 0.4,
    ) -> PredictionResult:
        """Decode and classify one image, optionally creating an in-memory overlay."""

        started = time.perf_counter()
        prepared = prepare_image(
            image_or_bytes,
            input_size=self.metadata.input_size,
            limits=self.image_limits,
        )
        with self._lock:
            probability = self._predict_probability(prepared.batch)
        latency_ms = (time.perf_counter() - started) * 1000.0

        predicted_class = int(probability >= self.metadata.threshold)
        predicted_label = "Crack" if predicted_class == CRACK_LABEL else "Non-crack"
        heatmap: np.ndarray | None = None
        overlay: Image.Image | None = None
        gradcam_latency_ms: float | None = None
        if include_gradcam:
            from crackspot.modeling.gradcam import generate_gradcam, overlay_heatmap

            gradcam_started = time.perf_counter()
            with self._lock:
                heatmap = generate_gradcam(
                    self.model,
                    prepared.batch,
                    layer_name=self.metadata.gradcam_layer,
                )
            overlay = overlay_heatmap(prepared.original, heatmap, alpha=gradcam_alpha)
            gradcam_latency_ms = (time.perf_counter() - gradcam_started) * 1000.0

        return PredictionResult(
            crack_probability=probability,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            threshold=self.metadata.threshold,
            latency_ms=latency_ms,
            model_version=self.metadata.display_version,
            run_id=self.metadata.run_id,
            original_image=prepared.original,
            heatmap=heatmap,
            overlay=overlay,
            gradcam_latency_ms=gradcam_latency_ms,
        )


__all__ = [
    "InferenceError",
    "InferenceService",
    "MetadataError",
    "ModelLoadError",
    "ModelMetadata",
    "PredictionResult",
    "discover_metadata_path",
]
