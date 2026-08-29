"""Safe Pillow decoding and the canonical MobileNetV2 inference preprocessing.

The functions in this module deliberately accept bytes instead of trusting an
uploaded filename or MIME type.  Decoded images are EXIF-transposed, converted
to RGB, resized, and transformed using the MobileNetV2 ``[-1, 1]`` convention.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TypeAlias

import numpy as np
from PIL import Image, ImageOps

from crackspot.constants import IMAGE_SIZE, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from crackspot.data import (
    CorruptImageError as DataCorruptImageError,
)
from crackspot.data import (
    FileTooLargeError as DataFileTooLargeError,
)
from crackspot.data import (
    ImageTooLargeError as DataImageTooLargeError,
)
from crackspot.data import (
    UnsupportedImageError as DataUnsupportedImageError,
)
from crackspot.data import (
    load_image_rgb,
)
from crackspot.data import (
    mobilenet_v2_preprocess as shared_mobilenet_v2_preprocess,
)

ImageSource: TypeAlias = str | Path | bytes | bytearray | memoryview | BinaryIO | Image.Image
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG"})


class ImageValidationError(ValueError):
    """Raised when an input cannot safely be used as a CrackSpot image."""

    def __init__(self, message: str, *, code: str = "invalid_image") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ImageLimits:
    """Upload and decoded-image limits used by every inference entrypoint."""

    max_bytes: int = MAX_UPLOAD_BYTES
    max_pixels: int = MAX_IMAGE_PIXELS

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")


@dataclass(frozen=True)
class PreparedImage:
    """An oriented RGB original and its preprocessed rank-4 model tensor."""

    original: Image.Image
    batch: np.ndarray


def _read_source_bytes(
    source: str | Path | bytes | bytearray | memoryview | BinaryIO,
    *,
    max_bytes: int,
) -> bytes:
    if isinstance(source, str | Path):
        path = Path(source).expanduser()
        try:
            file_size = path.stat().st_size
        except (OSError, ValueError) as exc:
            raise ImageValidationError(
                f"Không thể đọc tệp ảnh: {path}", code="unreadable_file"
            ) from exc
        if not path.is_file():
            raise ImageValidationError(f"Không tìm thấy tệp ảnh: {path}", code="missing_file")
        if file_size > max_bytes:
            raise ImageValidationError(
                f"Tệp có dung lượng {file_size:,} byte, vượt giới hạn {max_bytes:,} byte.",
                code="file_too_large",
            )
        try:
            with path.open("rb") as handle:
                return handle.read(max_bytes + 1)
        except OSError as exc:
            raise ImageValidationError(
                f"Không thể đọc tệp ảnh: {path}", code="unreadable_file"
            ) from exc

    if isinstance(source, bytes | bytearray | memoryview):
        return bytes(source)

    if not hasattr(source, "read"):
        raise TypeError("image source must be a path, bytes, binary stream, or PIL image")

    stream = source
    original_position: int | None = None
    try:
        if hasattr(stream, "tell"):
            original_position = stream.tell()
        if hasattr(stream, "seek"):
            stream.seek(0)
        payload = stream.read(max_bytes + 1)
    except (OSError, ValueError, TypeError) as exc:
        raise ImageValidationError("Không thể đọc dữ liệu ảnh.", code="unreadable_file") from exc
    finally:
        if original_position is not None and hasattr(stream, "seek"):
            with suppress(OSError, ValueError):
                stream.seek(original_position)

    if not isinstance(payload, bytes):
        raise ImageValidationError("Luồng tệp phải trả về bytes.", code="invalid_stream")
    return payload


def _check_pixel_limit(image: Image.Image, max_pixels: int) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageValidationError("Kích thước ảnh không hợp lệ.", code="invalid_dimensions")
    pixels = width * height
    if pixels > max_pixels:
        raise ImageValidationError(
            f"Ảnh có {pixels:,} pixel, vượt giới hạn {max_pixels:,} pixel.",
            code="too_many_pixels",
        )


def _normalise_pil_image(image: Image.Image, *, max_pixels: int) -> Image.Image:
    _check_pixel_limit(image, max_pixels)
    try:
        image.load()
        oriented = ImageOps.exif_transpose(image)
        _check_pixel_limit(oriented, max_pixels)
        rgb = oriented.convert("RGB")
        rgb.load()
        return rgb.copy()
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "Ảnh bị hỏng hoặc không thể giải mã.", code="decode_failed"
        ) from exc


def decode_image(
    source: ImageSource,
    *,
    limits: ImageLimits | None = None,
    allowed_formats: frozenset[str] = ALLOWED_IMAGE_FORMATS,
) -> Image.Image:
    """Decode an input into an independent, correctly oriented RGB image.

    Filenames and caller-provided MIME types are intentionally ignored.  JPEG
    and PNG are recognised from their bytes by Pillow.
    """

    resolved_limits = limits or ImageLimits()
    if isinstance(source, Image.Image):
        return _normalise_pil_image(source.copy(), max_pixels=resolved_limits.max_pixels)

    if allowed_formats != ALLOWED_IMAGE_FORMATS:
        raise ValueError("CrackSpot chỉ hỗ trợ chính sách định dạng JPEG/PNG dùng chung.")
    payload = _read_source_bytes(source, max_bytes=resolved_limits.max_bytes)
    try:
        return load_image_rgb(
            payload,
            max_bytes=resolved_limits.max_bytes,
            max_pixels=resolved_limits.max_pixels,
        )
    except DataFileTooLargeError as exc:
        raise ImageValidationError(str(exc), code="file_too_large") from exc
    except DataImageTooLargeError as exc:
        code = "decompression_bomb" if "decompression-bomb" in str(exc) else "too_many_pixels"
        raise ImageValidationError(str(exc), code=code) from exc
    except DataUnsupportedImageError as exc:
        raise ImageValidationError(
            "Chỉ chấp nhận ảnh JPEG hoặc PNG hợp lệ.", code="unsupported_format"
        ) from exc
    except DataCorruptImageError as exc:
        code = "empty_file" if not payload else "decode_failed"
        raise ImageValidationError(
            "Tệp không phải ảnh JPEG/PNG hợp lệ hoặc đã bị hỏng.", code=code
        ) from exc


def mobilenet_v2_preprocess(rgb_pixels: np.ndarray) -> np.ndarray:
    """Apply Keras MobileNetV2 preprocessing, with an equivalent NumPy fallback.

    TensorFlow is imported lazily so image validation and application imports
    still work in environments where the optional ML runtime is not installed.
    The fallback is algebraically identical to Keras for float RGB pixels:
    ``x / 127.5 - 1``.
    """

    pixels = np.asarray(rgb_pixels, dtype=np.float32)
    return np.asarray(shared_mobilenet_v2_preprocess(pixels), dtype=np.float32)


def preprocess_image(
    image: Image.Image,
    *,
    input_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Resize an RGB image and return ``(1, height, width, 3)`` float32 data."""

    if len(input_size) != 2 or any(isinstance(value, bool) for value in input_size):
        raise ValueError("input_size must contain (height, width)")
    height, width = (int(value) for value in input_size)
    if height <= 0 or width <= 0:
        raise ValueError("input_size dimensions must be positive")

    rgb = image.convert("RGB")
    resized = rgb.resize((width, height), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    processed = mobilenet_v2_preprocess(pixels)
    return np.expand_dims(processed, axis=0).astype(np.float32, copy=False)


def prepare_image(
    source: ImageSource,
    *,
    input_size: tuple[int, int] = IMAGE_SIZE,
    limits: ImageLimits | None = None,
) -> PreparedImage:
    """Run the complete canonical decode/preprocess pipeline."""

    original = decode_image(source, limits=limits)
    batch = preprocess_image(original, input_size=input_size)
    return PreparedImage(original=original, batch=batch)
