from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from crackspot.modeling.gradcam import (
    GradCAMError,
    overlay_heatmap,
    resize_heatmap,
    resolve_target_layer,
)


class FakeLayer:
    def __init__(self, name: str, layers: list[FakeLayer] | None = None) -> None:
        self.name = name
        self.layers = layers or []


def test_resolve_target_layer_inside_nested_backbone() -> None:
    target = FakeLayer("out_relu")
    backbone = FakeLayer("mobilenetv2_1.00_224", [FakeLayer("block_16"), target])
    model = FakeLayer("crackspot", [FakeLayer("input"), backbone, FakeLayer("classifier")])

    resolution = resolve_target_layer(model, "out_relu")

    assert resolution.layer is target
    assert resolution.owner is backbone
    assert resolution.path == ("mobilenetv2_1.00_224", "out_relu")


def test_resolve_target_layer_rejects_ambiguous_name() -> None:
    model = FakeLayer(
        "root",
        [FakeLayer("first", [FakeLayer("out_relu")]), FakeLayer("second", [FakeLayer("out_relu")])],
    )

    with pytest.raises(GradCAMError, match="không duy nhất"):
        resolve_target_layer(model, "out_relu")


def test_resize_heatmap_preserves_range_and_shape() -> None:
    heatmap = np.asarray([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

    resized = resize_heatmap(heatmap, (11, 7))

    assert resized.shape == (7, 11)
    assert resized.dtype == np.float32
    assert np.isfinite(resized).all()
    assert float(resized.min()) >= 0.0
    assert float(resized.max()) <= 1.0


def test_overlay_does_not_mutate_original_image_or_heatmap() -> None:
    image = Image.new("RGB", (12, 8), (100, 120, 140))
    before = np.asarray(image).copy()
    heatmap = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    heatmap_before = heatmap.copy()

    overlay = overlay_heatmap(image, heatmap, alpha=0.45)

    assert overlay.mode == "RGB"
    assert overlay.size == image.size
    assert np.array_equal(np.asarray(image), before)
    assert np.array_equal(heatmap, heatmap_before)
    assert not np.array_equal(np.asarray(overlay), before)


def test_overlay_alpha_zero_returns_visual_copy() -> None:
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    overlay = overlay_heatmap(image, np.ones((2, 2), dtype=np.float32), alpha=0.0)

    assert overlay is not image
    assert np.array_equal(np.asarray(overlay), np.asarray(image))


def test_generate_gradcam_with_real_nested_keras_backbone() -> None:
    tf = pytest.importorskip("tensorflow")
    from crackspot.modeling.gradcam import generate_gradcam

    tf.keras.utils.set_random_seed(42)
    inner_input = tf.keras.Input((12, 12, 3), name="inner_input")
    features = tf.keras.layers.Conv2D(
        4,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="ones",
        name="out_relu",
    )(inner_input)
    backbone = tf.keras.Model(inner_input, features, name="mobilenetv2_test")
    outer_input = tf.keras.Input((12, 12, 3), name="image")
    output = backbone(outer_input)
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.Constant(0.01),
        bias_initializer="zeros",
    )(output)
    model = tf.keras.Model(outer_input, output)
    batch = np.ones((1, 12, 12, 3), dtype=np.float32)

    heatmap = generate_gradcam(model, batch, layer_name="out_relu")

    assert heatmap.shape == (12, 12)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) == pytest.approx(1.0)
