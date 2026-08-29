from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crackspot.modeling.model import (
    build_mobilenetv2_classifier,
    configure_fine_tuning,
    find_backbone,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)


@pytest.fixture(scope="module")
def classifier():
    return build_mobilenetv2_classifier(weights=None)


def test_model_outputs_one_valid_crack_probability(classifier) -> None:
    batch = np.zeros((2, 224, 224, 3), dtype=np.float32)

    probabilities = np.asarray(classifier(batch, training=False))

    assert probabilities.shape == (2, 1)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_fine_tuning_respects_boundary_and_freezes_batch_norm(classifier) -> None:
    summary = configure_fine_tuning(classifier, "block_14_expand")
    backbone = find_backbone(classifier)
    names = [layer.name for layer in backbone.layers]
    boundary_index = names.index("block_14_expand")

    assert summary["boundary"] == "block_14_expand"
    assert summary["trainable_backbone_layers"] > 0
    assert summary["trainable_batch_norm_layers"] == []
    assert all(not layer.trainable for layer in backbone.layers[:boundary_index])
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if layer.__class__.__name__ == "BatchNormalization"
    )


def test_unknown_fine_tune_boundary_fails_fast(classifier) -> None:
    with pytest.raises(ValueError, match="boundary"):
        configure_fine_tuning(classifier, "not_a_real_layer")


def test_full_keras_save_load_preserves_prediction(classifier, tmp_path: Path) -> None:
    import tensorflow as tf

    configure_fine_tuning(classifier, None)
    sample = np.linspace(-1.0, 1.0, 224 * 224 * 3, dtype=np.float32).reshape(1, 224, 224, 3)
    expected = np.asarray(classifier(sample, training=False))
    checkpoint = tmp_path / "model.keras"

    classifier.save(checkpoint)
    restored = tf.keras.models.load_model(checkpoint, compile=False)
    observed = np.asarray(restored(sample, training=False))

    assert np.allclose(observed, expected, rtol=1e-6, atol=1e-7)
