"""The single ``tf.data`` input pipeline used by training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import DEFAULT_MAX_BYTES, DEFAULT_MAX_PIXELS, load_image_rgb


def mobilenet_v2_preprocess(pixels: Any) -> Any:
    """Apply the canonical MobileNetV2 ``[-1, 1]`` transform everywhere.

    TensorFlow tensors are passed directly to Keras.  A NumPy fallback keeps
    image-validation utilities importable in a lightweight runtime, while
    remaining algebraically identical for float RGB pixels.
    """

    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except (ImportError, OSError, ValueError):
        array = np.asarray(pixels, dtype=np.float32)
        return array / np.float32(127.5) - np.float32(1.0)
    return preprocess_input(pixels)


def _tensorflow() -> Any:
    """Import TensorFlow only when a dataset is actually requested."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional environment.
        raise RuntimeError(
            "TensorFlow is required to build the input pipeline. Install the "
            "training dependencies first."
        ) from exc
    return tf


def _validate_image_size(image_size: Sequence[int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must contain (height, width)")
    height, width = (int(value) for value in image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive")
    return height, width


def _resolve_paths(frame: pd.DataFrame, dataset_root: str | PathLike[str] | None) -> list[str]:
    if "relative_path" not in frame.columns:
        raise ValueError("manifest is missing relative_path")
    if "absolute_path" in frame.columns:
        paths = [Path(value) for value in frame["absolute_path"].astype(str)]
    else:
        root_value = dataset_root or frame.attrs.get("dataset_root")
        if root_value is None:
            raise ValueError(
                "dataset_root is required when the manifest has no absolute_path column "
                "or dataset_root attribute"
            )
        root = Path(root_value).resolve()
        paths = []
        for value in frame["relative_path"].astype(str):
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe relative_path: {value!r}")
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"relative_path escapes dataset root: {value!r}") from exc
            paths.append(candidate)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest images do not exist: {missing[:5]}")
    return [str(path) for path in paths]


def _rotation_transform(tf: Any, angle: Any, height: int, width: int) -> Any:
    cosine = tf.math.cos(angle)
    sine = tf.math.sin(angle)
    center_x = tf.cast(width - 1, tf.float32) / 2.0
    center_y = tf.cast(height - 1, tf.float32) / 2.0
    return tf.stack(
        [
            cosine,
            sine,
            center_x - cosine * center_x - sine * center_y,
            -sine,
            cosine,
            center_y + sine * center_x - cosine * center_y,
            0.0,
            0.0,
        ]
    )


def _augment_stateless(
    tf: Any,
    image: Any,
    index: Any,
    *,
    seed: int,
    height: int,
    width: int,
    rotation_degrees: float,
    brightness_delta: float,
    contrast_delta: float,
) -> Any:
    base_seed = tf.stack(
        [
            tf.cast(seed, tf.int32),
            tf.cast(tf.math.floormod(index, 2**31 - 1), tf.int32),
        ]
    )
    seeds = tf.random.experimental.stateless_split(base_seed, num=4)
    image = tf.image.stateless_random_flip_left_right(image, seed=seeds[0])
    angle_limit = tf.constant(np.deg2rad(rotation_degrees), dtype=tf.float32)
    angle = tf.random.stateless_uniform((), seed=seeds[1], minval=-angle_limit, maxval=angle_limit)
    transform = _rotation_transform(tf, angle, height, width)
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=transform[None, ...],
        output_shape=tf.constant([height, width], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    image = tf.image.stateless_random_brightness(
        image, max_delta=255.0 * brightness_delta, seed=seeds[2]
    )
    image = tf.image.stateless_random_contrast(
        image,
        lower=1.0 - contrast_delta,
        upper=1.0 + contrast_delta,
        seed=seeds[3],
    )
    return tf.clip_by_value(image, 0.0, 255.0)


def build_tf_dataset(
    frame: pd.DataFrame,
    batch_size: int,
    image_size: Sequence[int] = (224, 224),
    training: bool = False,
    seed: int = 42,
    augment: bool = False,
    *,
    dataset_root: str | PathLike[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    include_paths: bool = False,
    shuffle_buffer: int | None = None,
    rotation_degrees: float = 15.0,
    brightness_delta: float = 0.15,
    contrast_delta: float = 0.15,
) -> Any:
    """Create a safe, deterministic ``tf.data.Dataset`` from manifest rows.

    Images are decoded through :func:`load_image_rgb`, then resized and passed
    through MobileNetV2's exact ``preprocess_input`` implementation.  Augmentation
    is stateless and rejected for validation/test mode.  TensorFlow is imported
    lazily, so audit and split CLIs remain usable on machines without it.
    """

    if frame.empty:
        raise ValueError("cannot build a dataset from an empty manifest")
    if "label" not in frame.columns:
        raise ValueError("manifest is missing label")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if augment and not training:
        raise ValueError("augmentation is train-only; set training=True")
    if training and "split" in frame.columns:
        observed = set(frame["split"].dropna().astype(str).str.lower())
        if observed and observed != {"train"}:
            raise ValueError(f"training dataset contains non-train split rows: {sorted(observed)}")
    if rotation_degrees < 0 or rotation_degrees > 15:
        raise ValueError("rotation_degrees must be between 0 and 15")
    if not 0 <= brightness_delta <= 0.20:
        raise ValueError("brightness_delta must be between 0 and 0.20")
    if not 0 <= contrast_delta <= 0.20:
        raise ValueError("contrast_delta must be between 0 and 0.20")
    height, width = _validate_image_size(image_size)
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=np.float32)
    if not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("labels must use Non-crack=0 and Crack=1")
    paths = _resolve_paths(frame, dataset_root)

    tf = _tensorflow()
    path_tensor = tf.constant(paths, dtype=tf.string)
    label_tensor = tf.constant(labels, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if training:
        buffer_size = shuffle_buffer or len(frame)
        if buffer_size <= 0:
            raise ValueError("shuffle_buffer must be positive")
        dataset = dataset.shuffle(
            min(int(buffer_size), len(frame)),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )

    def decode_path(path_tensor_value: Any) -> np.ndarray:
        value = path_tensor_value.numpy()
        path_text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        image = load_image_rgb(path_text, max_bytes=max_bytes, max_pixels=max_pixels)
        return np.asarray(image, dtype=np.uint8)

    def decode_map(path: Any, label: Any) -> tuple[Any, Any, Any]:
        image = tf.py_function(decode_path, [path], Tout=tf.uint8)
        image.set_shape((None, None, 3))
        image = tf.image.resize(
            tf.cast(image, tf.float32),
            [height, width],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=True,
        )
        image.set_shape((height, width, 3))
        return image, tf.cast(label, tf.float32), path

    dataset = dataset.map(
        decode_map,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    if augment:
        dataset = dataset.enumerate()

        def augment_map(index: Any, values: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            image, label, path = values
            image = _augment_stateless(
                tf,
                image,
                index,
                seed=int(seed),
                height=height,
                width=width,
                rotation_degrees=float(rotation_degrees),
                brightness_delta=float(brightness_delta),
                contrast_delta=float(contrast_delta),
            )
            return image, label, path

        dataset = dataset.map(
            augment_map,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    def preprocess_map(image: Any, label: Any, path: Any) -> Any:
        processed = mobilenet_v2_preprocess(image)
        processed = tf.ensure_shape(processed, (height, width, 3))
        if include_paths:
            return processed, label, path
        return processed, label

    dataset = dataset.map(
        preprocess_map,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def compute_balanced_class_weights(frame: pd.DataFrame) -> dict[int, float]:
    """Compute inverse-frequency binary class weights from train rows only."""

    if "label" not in frame.columns:
        raise ValueError("manifest is missing label")
    if "split" in frame.columns:
        observed = set(frame["split"].dropna().astype(str).str.lower())
        if observed and observed != {"train"}:
            raise ValueError("class weights must be computed from train rows only")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    counts = labels.value_counts().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError("both labels 0 and 1 are required for balanced class weights")
    total = len(labels)
    return {label: float(total / (2 * counts[label])) for label in (0, 1)}
