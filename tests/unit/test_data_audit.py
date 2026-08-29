from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from crackspot.data import (
    CorruptImageError,
    FileTooLargeError,
    ImageTooLargeError,
    UnsupportedImageError,
    inspect_image,
    load_image_rgb,
)


def image_bytes(
    mode: str,
    size: tuple[int, int] = (7, 5),
    *,
    image_format: str = "PNG",
    exif: Image.Exif | None = None,
) -> bytes:
    color: object
    if mode == "L":
        color = 120
    elif mode == "RGBA":
        color = (10, 20, 30, 100)
    else:
        color = (10, 20, 30)
    image = Image.new(mode, size, color=color)
    output = BytesIO()
    image.save(output, format=image_format, exif=exif)
    return output.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_load_image_normalises_valid_modes_to_rgb(mode: str) -> None:
    result = load_image_rgb(image_bytes(mode))

    assert result.mode == "RGB"
    assert result.size == (7, 5)


def test_load_image_honours_exif_orientation() -> None:
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    data = image_bytes("RGB", (2, 3), image_format="JPEG", exif=exif)

    result = load_image_rgb(data)
    inspection = inspect_image(data)

    assert result.size == (3, 2)
    assert inspection.source_width == 2
    assert inspection.source_height == 3
    assert inspection.width == 3
    assert inspection.height == 2
    assert inspection.exif_orientation == 6


def test_inspection_hashes_encoded_bytes_and_reports_properties() -> None:
    data = image_bytes("RGB")

    first = inspect_image(data)
    second = inspect_image(BytesIO(data))

    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert len(first.perceptual_hash) == 16
    assert first.image_format == "PNG"
    assert first.source_mode == "RGB"
    assert first.file_size_bytes == len(data)


def test_load_image_rejects_encoded_file_above_limit() -> None:
    data = image_bytes("RGB")

    with pytest.raises(FileTooLargeError):
        load_image_rgb(data, max_bytes=len(data) - 1)


def test_load_image_rejects_decoded_pixel_count_above_limit() -> None:
    with pytest.raises(ImageTooLargeError):
        load_image_rgb(image_bytes("RGB", (10, 10)), max_pixels=99)


def test_pillow_decompression_bomb_warning_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)

    with pytest.raises(ImageTooLargeError, match="decompression-bomb"):
        load_image_rgb(image_bytes("RGB", (10, 10)), max_pixels=1_000)


@pytest.mark.parametrize("payload", [b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_load_image_rejects_corrupt_or_truncated_bytes(payload: bytes) -> None:
    with pytest.raises(CorruptImageError):
        load_image_rgb(payload)


def test_load_image_rejects_valid_but_unsupported_format() -> None:
    with pytest.raises(UnsupportedImageError):
        load_image_rgb(image_bytes("RGB", image_format="BMP"))


def test_file_limit_is_checked_before_opening(tmp_path: Path) -> None:
    path = tmp_path / "oversize.png"
    path.write_bytes(image_bytes("RGB"))

    with pytest.raises(FileTooLargeError):
        load_image_rgb(path, max_bytes=1)
