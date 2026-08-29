"""MobileNetV2 model factory and fine-tuning policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from crackspot.constants import IMAGE_SIZE


def _tf():
    try:
        import tensorflow as tf
    except (ImportError, RuntimeError, ValueError) as exc:
        raise RuntimeError("TensorFlow không sẵn sàng. Hãy cài môi trường theo README.") from exc
    return tf


def build_mobilenetv2_classifier(
    *,
    input_size: tuple[int, int] = IMAGE_SIZE,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
    name: str = "crackspot_mobilenetv2",
):
    """Build one-sigmoid MobileNetV2 classifier where output means P(Crack)."""

    if tuple(input_size) != IMAGE_SIZE:
        raise ValueError(f"CrackSpot yêu cầu input_size={IMAGE_SIZE}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout phải nằm trong [0,1)")
    tf = _tf()
    inputs = tf.keras.Input(shape=(*input_size, 3), dtype=tf.float32, name="image")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*input_size, 3), include_top=False, weights=weights
    )
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    dropped = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(pooled)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="crack_probability")(dropped)
    return tf.keras.Model(inputs, outputs, name=name)


def find_backbone(model):
    """Return the nested MobileNetV2 model from a saved or freshly built classifier."""

    tf = _tf()
    candidates = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower().replace("_", "")
    ]
    if not candidates:
        candidates = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and any(item.name == "out_relu" for item in layer.layers)
        ]
    if len(candidates) != 1:
        raise ValueError(f"Cần đúng một MobileNetV2 backbone, tìm thấy {len(candidates)}")
    return candidates[0]


def configure_fine_tuning(model, boundary: str | None) -> dict[str, Any]:
    """Freeze backbone or unfreeze non-BatchNorm layers from an exact boundary."""

    tf = _tf()
    backbone = find_backbone(model)
    if boundary is None:
        backbone.trainable = False
        return trainability_summary(model, boundary=None)

    names = [layer.name for layer in backbone.layers]
    if boundary not in names:
        raise ValueError(f"Không tìm thấy fine-tune boundary '{boundary}' trong backbone")
    boundary_index = names.index(boundary)
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= boundary_index and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    return trainability_summary(model, boundary=boundary)


def trainability_summary(model, boundary: str | None = None) -> dict[str, Any]:
    backbone = find_backbone(model)
    trainable_backbone = [layer.name for layer in backbone.layers if layer.trainable]
    batch_norm_trainable = [
        layer.name
        for layer in backbone.layers
        if layer.__class__.__name__ == "BatchNormalization" and layer.trainable
    ]
    return {
        "boundary": boundary,
        "backbone_layers": len(backbone.layers),
        "trainable_backbone_layers": len(trainable_backbone),
        "trainable_backbone_layer_names": trainable_backbone,
        "trainable_batch_norm_layers": batch_norm_trainable,
        "trainable_parameters": int(
            sum(int(np.prod(tuple(value.shape))) for value in model.trainable_weights)
        ),
        "total_parameters": int(model.count_params()),
    }


def compile_binary_model(model, learning_rate: float):
    if learning_rate <= 0:
        raise ValueError("learning_rate phải dương")
    tf = _tf()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def build_from_config(config: Mapping[str, Any], *, weights_override: str | None = None):
    model_config = config.get("model", {})
    weights = (
        model_config.get("weights", "imagenet") if weights_override is None else weights_override
    )
    input_size = config.get("data", {}).get(
        "image_size", config.get("pipeline", {}).get("image_size", IMAGE_SIZE)
    )
    return build_mobilenetv2_classifier(
        input_size=tuple(input_size),
        dropout=float(model_config.get("dropout", 0.3)),
        weights=weights,
    )
