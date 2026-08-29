from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from crackspot.data import build_tf_dataset, compute_balanced_class_weights


def _tiny_manifest(root: Path, *, split: str = "validation") -> pd.DataFrame:
    paths = ["D/CD/rgb.png", "D/UD/gray.png", "W/CW/rgba.png"]
    modes = ["RGB", "L", "RGBA"]
    colors: list[object] = [(255, 128, 0), 64, (0, 128, 255, 100)]
    for relative, mode, color in zip(paths, modes, colors, strict=True):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, (12, 8), color=color).save(path)
    frame = pd.DataFrame(
        {
            "relative_path": paths,
            "label": [1, 0, 1],
            "split": [split] * 3,
        }
    )
    frame.attrs["dataset_root"] = str(root)
    return frame


def test_compute_balanced_class_weights_uses_expected_formula() -> None:
    frame = pd.DataFrame({"label": [0, 0, 0, 1], "split": ["train"] * 4})

    weights = compute_balanced_class_weights(frame)

    assert weights[0] == pytest.approx(4 / 6)
    assert weights[1] == pytest.approx(2.0)


def test_class_weights_reject_non_train_rows() -> None:
    frame = pd.DataFrame({"label": [0, 1], "split": ["train", "validation"]})

    with pytest.raises(ValueError, match="train rows only"):
        compute_balanced_class_weights(frame)


def test_pipeline_decodes_resizes_and_uses_mobilenet_preprocessing(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    frame = _tiny_manifest(tmp_path)

    dataset = build_tf_dataset(frame, batch_size=3, image_size=(16, 14))
    images, labels = next(iter(dataset))
    values = images.numpy()

    assert values.shape == (3, 16, 14, 3)
    assert values.dtype == np.float32
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    assert labels.numpy().tolist() == [1.0, 0.0, 1.0]
    # First RGB fixture is (255, 128, 0): exact MobileNetV2 x/127.5 - 1.
    assert values[0, 0, 0].tolist() == pytest.approx([1.0, 128 / 127.5 - 1, -1.0], abs=1e-6)


def test_augmentation_is_rejected_outside_training(tmp_path: Path) -> None:
    frame = _tiny_manifest(tmp_path)

    with pytest.raises(ValueError, match="train-only"):
        build_tf_dataset(frame, batch_size=2, training=False, augment=True)


def test_training_rejects_validation_or_test_rows(tmp_path: Path) -> None:
    frame = _tiny_manifest(tmp_path, split="test")

    with pytest.raises(ValueError, match="non-train"):
        build_tf_dataset(frame, batch_size=2, training=True)


def test_seeded_training_augmentation_is_reproducible(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    frame = _tiny_manifest(tmp_path, split="train")

    first = build_tf_dataset(frame, batch_size=3, training=True, augment=True, seed=42)
    second = build_tf_dataset(frame, batch_size=3, training=True, augment=True, seed=42)
    first_images, first_labels = next(iter(first))
    second_images, second_labels = next(iter(second))

    np.testing.assert_allclose(first_images.numpy(), second_images.numpy(), atol=1e-6)
    np.testing.assert_array_equal(first_labels.numpy(), second_labels.numpy())


def test_pipeline_can_return_paths_without_changing_preprocessing(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    frame = _tiny_manifest(tmp_path)

    plain_images, plain_labels = next(
        iter(build_tf_dataset(frame, batch_size=3, include_paths=False))
    )
    path_images, path_labels, paths = next(
        iter(build_tf_dataset(frame, batch_size=3, include_paths=True))
    )

    np.testing.assert_allclose(plain_images.numpy(), path_images.numpy())
    np.testing.assert_array_equal(plain_labels.numpy(), path_labels.numpy())
    assert [Path(value.decode()).name for value in paths.numpy()] == [
        "rgb.png",
        "gray.png",
        "rgba.png",
    ]
