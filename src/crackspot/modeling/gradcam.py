"""Grad-CAM for the Crack probability produced by a binary Keras classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from crackspot.constants import DEFAULT_GRADCAM_LAYER


class GradCAMError(RuntimeError):
    """Raised when a valid Grad-CAM cannot be produced."""


@dataclass(frozen=True)
class LayerResolution:
    """A named target layer and the nested model that owns its graph."""

    layer: Any
    owner: Any
    path: tuple[str, ...]


def _child_layers(container: Any) -> list[Any]:
    layers = getattr(container, "layers", None)
    if layers is None:
        return []
    return list(layers)


def resolve_target_layer(model: Any, layer_name: str = DEFAULT_GRADCAM_LAYER) -> LayerResolution:
    """Resolve a layer by name, including layers inside a nested backbone.

    A duplicate name in separate nested models is rejected instead of silently
    choosing a possibly incorrect convolutional feature map.
    """

    if not isinstance(layer_name, str) or not layer_name.strip():
        raise GradCAMError("Tên layer Grad-CAM không hợp lệ.")

    matches: list[LayerResolution] = []
    visited: set[int] = set()

    def visit(container: Any, path: tuple[str, ...]) -> None:
        identity = id(container)
        if identity in visited:
            return
        visited.add(identity)
        for layer in _child_layers(container):
            layer_path = (*path, str(getattr(layer, "name", "<unnamed>")))
            if getattr(layer, "name", None) == layer_name:
                matches.append(LayerResolution(layer=layer, owner=container, path=layer_path))
            if _child_layers(layer):
                visit(layer, layer_path)

    visit(model, ())
    if not matches:
        available = sorted(
            {
                str(getattr(layer, "name", ""))
                for layer in _child_layers(model)
                if getattr(layer, "name", None)
            }
        )
        summary = ", ".join(available[-12:]) if available else "không có"
        raise GradCAMError(
            f"Không tìm thấy layer Grad-CAM '{layer_name}'. Các layer cấp cao nhất: {summary}."
        )
    if len(matches) > 1:
        paths = ", ".join("/".join(match.path) for match in matches)
        raise GradCAMError(f"Layer '{layer_name}' không duy nhất: {paths}")
    return matches[0]


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except (ImportError, OSError, ValueError) as exc:
        raise GradCAMError(
            "Không thể khởi tạo TensorFlow để tính Grad-CAM. Hãy kiểm tra bộ dependency demo."
        ) from exc
    return tf


def _as_single_tensor(value: Any, *, description: str) -> Any:
    if isinstance(value, list | tuple):
        if len(value) != 1:
            raise GradCAMError(f"{description} có nhiều tensor; Grad-CAM yêu cầu một tensor.")
        return value[0]
    if isinstance(value, dict):
        if len(value) != 1:
            raise GradCAMError(f"{description} có nhiều tensor; Grad-CAM yêu cầu một tensor.")
        return next(iter(value.values()))
    return value


def _single_model_input(model: Any) -> Any:
    inputs = getattr(model, "inputs", None)
    if not isinstance(inputs, list | tuple) or len(inputs) != 1:
        raise GradCAMError("Grad-CAM yêu cầu mô hình có đúng một đầu vào ảnh.")
    return inputs[0]


def _build_nested_runner(tf: Any, model: Any, resolution: LayerResolution) -> Any:
    """Build a differentiable target/prediction runner for a nested backbone.

    Keras application backbones are commonly stored as one nested Model layer.
    Their internal symbolic tensors are not always directly connected to the
    outer classifier graph.  This runner bridges the selected inbound node with
    a prefix, an internal feature extractor, and the original classifier tail.
    """

    owner = resolution.owner
    feature_model = tf.keras.Model(
        inputs=_single_model_input(owner),
        outputs=[resolution.layer.output, owner.output],
        name="crackspot_gradcam_features",
    )

    inbound_nodes = list(getattr(owner, "_inbound_nodes", ()))
    for node in reversed(inbound_nodes):
        node_input = _as_single_tensor(
            getattr(node, "input_tensors", None), description="Node input"
        )
        node_output = _as_single_tensor(
            getattr(node, "output_tensors", None), description="Node output"
        )
        if node_input is None or node_output is None:
            continue
        try:
            prefix_model = tf.keras.Model(
                inputs=_single_model_input(model),
                outputs=node_input,
                name="crackspot_gradcam_prefix",
            )
            tail_model = tf.keras.Model(
                inputs=node_output,
                outputs=model.output,
                name="crackspot_gradcam_tail",
            )
        except (TypeError, ValueError):
            continue

        def run(
            batch: Any,
            prefix: Any = prefix_model,
            tail: Any = tail_model,
        ) -> tuple[Any, Any]:
            owner_input = prefix(batch, training=False)
            convolution, owner_output = feature_model(owner_input, training=False)
            prediction = tail(owner_output, training=False)
            return convolution, prediction

        return run

    raise GradCAMError("Không thể nối layer Grad-CAM trong backbone nested với đầu ra classifier.")


def _build_runner(tf: Any, model: Any, resolution: LayerResolution) -> Any:
    if resolution.owner is model:
        try:
            grad_model = tf.keras.Model(
                inputs=_single_model_input(model),
                outputs=[resolution.layer.output, model.output],
                name="crackspot_gradcam",
            )
        except (TypeError, ValueError) as exc:
            raise GradCAMError("Không thể tạo graph Grad-CAM cho mô hình.") from exc

        def run(batch: Any) -> tuple[Any, Any]:
            convolution, prediction = grad_model(batch, training=False)
            return convolution, prediction

        return run
    return _build_nested_runner(tf, model, resolution)


def generate_gradcam(
    model: Any,
    input_batch: np.ndarray,
    *,
    layer_name: str = DEFAULT_GRADCAM_LAYER,
) -> np.ndarray:
    """Return a normalized 2-D heatmap for the model's ``P(Crack)`` score."""

    batch = np.asarray(input_batch, dtype=np.float32)
    if batch.ndim != 4 or batch.shape[0] != 1:
        raise GradCAMError(f"Grad-CAM yêu cầu batch shape (1,H,W,C), nhận {batch.shape}.")

    tf = _import_tensorflow()
    resolution = resolve_target_layer(model, layer_name)
    runner = _build_runner(tf, model, resolution)

    try:
        with tf.GradientTape() as tape:
            convolution, prediction = runner(batch)
            prediction = _as_single_tensor(prediction, description="Model output")
            if prediction.shape.rank == 1:
                crack_score = prediction[0]
            elif prediction.shape.rank == 2 and prediction.shape[-1] == 1:
                crack_score = prediction[0, 0]
            else:
                raise GradCAMError("Mô hình phải có một đầu ra sigmoid duy nhất P(Crack).")
        gradients = tape.gradient(crack_score, convolution)
    except GradCAMError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GradCAMError("Không thể tính gradient cho P(Crack).") from exc

    if gradients is None:
        raise GradCAMError("Gradient của P(Crack) tới layer đích bằng None.")
    if convolution.shape.rank != 4:
        raise GradCAMError(
            f"Layer '{layer_name}' phải có activation 4 chiều, nhận rank {convolution.shape.rank}."
        )

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap_tensor = tf.reduce_sum(weights * convolution, axis=-1)
    heatmap_tensor = tf.nn.relu(heatmap_tensor)
    heatmap = np.asarray(heatmap_tensor[0].numpy(), dtype=np.float32)
    if not np.all(np.isfinite(heatmap)):
        raise GradCAMError("Heatmap Grad-CAM chứa NaN hoặc infinity.")

    maximum = float(np.max(heatmap)) if heatmap.size else 0.0
    if maximum > np.finfo(np.float32).eps:
        heatmap = heatmap / np.float32(maximum)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32, copy=False)


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a normalized heatmap to Pillow ``(width, height)`` coordinates."""

    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise GradCAMError("Heatmap phải là ma trận 2 chiều không rỗng.")
    if not np.all(np.isfinite(values)):
        raise GradCAMError("Heatmap chứa NaN hoặc infinity.")
    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise GradCAMError("Kích thước overlay không hợp lệ.")
    clipped = np.clip(values, 0.0, 1.0)
    image = Image.fromarray(clipped)
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def _turbo_like_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Small dependency-free blue/cyan/yellow/red activation colour map."""

    x = np.asarray(heatmap, dtype=np.float32)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def overlay_heatmap(
    image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay Grad-CAM on a copy of the original; the input is never mutated."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    if isinstance(image, Image.Image):
        original = image.convert("RGB").copy()
    else:
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
            raise ValueError("image array must have shape (height, width, 3|4)")
        if np.issubdtype(pixels.dtype, np.floating):
            pixels = np.clip(pixels, 0.0, 255.0)
        original = Image.fromarray(pixels.astype(np.uint8, copy=True)).convert("RGB")

    resized = resize_heatmap(heatmap, original.size)
    colours = (_turbo_like_colormap(resized) * np.float32(255.0)).round().astype(np.uint8)
    colour_image = Image.fromarray(colours)
    return Image.blend(original, colour_image, alpha=float(alpha))


# Familiar alias used by the official Keras Grad-CAM example and older notebooks.
make_gradcam_heatmap = generate_gradcam


__all__ = [
    "GradCAMError",
    "LayerResolution",
    "generate_gradcam",
    "make_gradcam_heatmap",
    "overlay_heatmap",
    "resize_heatmap",
    "resolve_target_layer",
]
