from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from crackspot.data import load_image_rgb, mobilenet_v2_preprocess
from crackspot.inference.preprocessing import (
    ImageLimits,
    ImageValidationError,
    decode_image,
    prepare_image,
    preprocess_image,
)


def image_bytes(mode: str = "RGB", *, size: tuple[int, int] = (12, 8), fmt: str = "PNG") -> bytes:
    if mode == "RGBA":
        colour: int | tuple[int, ...] = (20, 80, 160, 128)
    elif mode == "RGB":
        colour = (20, 80, 160)
    else:
        colour = 120
    image = Image.new(mode, size, colour)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_decode_converts_valid_modes_to_rgb(mode: str) -> None:
    decoded = decode_image(image_bytes(mode))

    assert decoded.mode == "RGB"
    assert decoded.size == (12, 8)


def test_decode_honours_exif_orientation() -> None:
    image = Image.new("RGB", (8, 4), (50, 100, 150))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    decoded = decode_image(buffer.getvalue())

    assert decoded.size == (4, 8)


def test_decode_rejects_corrupt_fake_image() -> None:
    with pytest.raises(ImageValidationError, match="JPEG/PNG") as error:
        decode_image(b"this is not an image")

    assert error.value.code == "decode_failed"


def test_decode_enforces_real_byte_and_pixel_limits() -> None:
    payload = image_bytes(size=(20, 20))

    with pytest.raises(ImageValidationError) as bytes_error:
        decode_image(payload, limits=ImageLimits(max_bytes=10, max_pixels=1_000))
    assert bytes_error.value.code == "file_too_large"

    with pytest.raises(ImageValidationError) as pixels_error:
        decode_image(payload, limits=ImageLimits(max_bytes=10_000, max_pixels=399))
    assert pixels_error.value.code == "too_many_pixels"


def test_preprocess_shape_dtype_range_and_known_endpoints() -> None:
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[0, 0] = 255
    image = Image.fromarray(pixels)

    batch = preprocess_image(image, input_size=(224, 224))

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.isfinite(batch).all()
    assert float(batch.min()) >= -1.0
    assert float(batch.max()) <= 1.0
    assert np.allclose(batch[0, 0, 0], 1.0)


def test_prepare_image_returns_independent_original_and_batch() -> None:
    prepared = prepare_image(image_bytes("RGBA"))

    assert prepared.original.mode == "RGB"
    assert prepared.batch.shape == (1, 224, 224, 3)
    assert not np.shares_memory(np.asarray(prepared.original), prepared.batch)


def test_inference_uses_same_decode_and_preprocess_contract_as_data_pipeline() -> None:
    payload = image_bytes("RGBA", size=(17, 9))
    inference_image = decode_image(payload)
    data_image = load_image_rgb(payload)

    assert np.array_equal(np.asarray(inference_image), np.asarray(data_image))
    pixels = np.asarray(
        inference_image.resize((224, 224), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    assert np.allclose(
        preprocess_image(inference_image)[0],
        np.asarray(mobilenet_v2_preprocess(pixels), dtype=np.float32),
    )
