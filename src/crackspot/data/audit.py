"""Safe image decoding and dataset-audit primitives.

The functions in this module deliberately use Pillow instead of trusting a file
extension.  They are shared by manifest generation and the TensorFlow input
pipeline so EXIF orientation and colour conversion do not drift between code
paths.
"""

from __future__ import annotations

import warnings
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Final, TypeAlias

from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MAX_BYTES: Final[int] = 10 * 1024 * 1024
DEFAULT_MAX_PIXELS: Final[int] = 25_000_000
ALLOWED_IMAGE_FORMATS: Final[frozenset[str]] = frozenset({"JPEG", "PNG"})

ImageSource: TypeAlias = str | PathLike[str] | bytes | bytearray | memoryview | BinaryIO


class ImageValidationError(ValueError):
    """Base class for an image rejected by the safety policy."""


class FileTooLargeError(ImageValidationError):
    """Raised when encoded image bytes exceed the configured limit."""


class ImageTooLargeError(ImageValidationError):
    """Raised when decoded image dimensions exceed the configured limit."""


class CorruptImageError(ImageValidationError):
    """Raised when Pillow cannot fully decode an image."""


class UnsupportedImageError(ImageValidationError):
    """Raised when decoded content is not an accepted still JPEG/PNG image."""


@dataclass(frozen=True, slots=True)
class ImageInspection:
    """Machine-readable properties obtained from real decoded image bytes."""

    sha256: str
    width: int
    height: int
    source_width: int
    source_height: int
    source_mode: str
    image_format: str
    file_size_bytes: int
    exif_orientation: int | None
    perceptual_hash: str


def _read_limited(source: ImageSource, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    if isinstance(source, str | PathLike):
        path = Path(source)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise CorruptImageError(f"cannot stat image: {exc}") from exc
        if size > max_bytes:
            raise FileTooLargeError(f"encoded image is {size} bytes; limit is {max_bytes} bytes")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CorruptImageError(f"cannot read image: {exc}") from exc
    elif isinstance(source, bytes | bytearray | memoryview):
        data = bytes(source)
    elif hasattr(source, "read"):
        stream = source
        original_position: int | None = None
        with suppress(AttributeError, OSError):
            original_position = stream.tell()
        try:
            data = stream.read(max_bytes + 1)
        except (OSError, ValueError) as exc:
            raise CorruptImageError(f"cannot read image stream: {exc}") from exc
        finally:
            if original_position is not None:
                with suppress(AttributeError, OSError):
                    stream.seek(original_position)
        if not isinstance(data, bytes):
            data = bytes(data)
    else:  # pragma: no cover - protected by the public type, kept for callers.
        raise TypeError(f"unsupported image source type: {type(source)!r}")

    if len(data) > max_bytes:
        raise FileTooLargeError(f"encoded image is greater than {max_bytes} bytes")
    if not data:
        raise CorruptImageError("image is empty")
    return data


def _decode_image(
    data: bytes,
    *,
    max_pixels: int,
) -> tuple[Image.Image, str, str, tuple[int, int], int | None]:
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                image_format = (probe.format or "").upper()
                source_mode = probe.mode
                source_size = probe.size
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    raise UnsupportedImageError(
                        f"unsupported image format {image_format or 'unknown'}; "
                        "only JPEG and PNG are accepted"
                    )
                if bool(getattr(probe, "is_animated", False)):
                    raise UnsupportedImageError("animated images are not accepted")
                width, height = source_size
                if width <= 0 or height <= 0:
                    raise CorruptImageError("image dimensions must be positive")
                if width * height > max_pixels:
                    raise ImageTooLargeError(
                        f"decoded image has {width * height} pixels; limit is {max_pixels} pixels"
                    )
            # Some plugins close ``fp`` when metadata is inspected.  Verification
            # therefore gets its own fresh decoder and is always the first action.
            with Image.open(BytesIO(data)) as verification_probe:
                verification_probe.verify()

            # Re-open after verify(), which intentionally invalidates the decoder.
            with Image.open(BytesIO(data)) as decoded:
                orientation_value = decoded.getexif().get(274)
                orientation = int(orientation_value) if orientation_value is not None else None
                decoded.load()
                transposed = ImageOps.exif_transpose(decoded)
                transposed.load()
                rgb = transposed.convert("RGB")
                rgb.load()
                result = rgb.copy()
    except (FileTooLargeError, ImageTooLargeError, UnsupportedImageError):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError(f"Pillow decompression-bomb protection: {exc}") from exc
    except (UnidentifiedImageError, OSError, RuntimeError, SyntaxError, ValueError) as exc:
        raise CorruptImageError(f"invalid or truncated image: {exc}") from exc

    return result, image_format, source_mode, source_size, orientation


def load_image_rgb(
    source: ImageSource,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Image.Image:
    """Decode a still JPEG/PNG safely, honour EXIF orientation and return RGB.

    The returned image is detached from the input stream and can therefore be
    used after this function returns.  Grayscale, palette and alpha modes are
    normalised to exactly three RGB channels.
    """

    data = _read_limited(source, max_bytes=max_bytes)
    image, _, _, _, _ = _decode_image(data, max_pixels=max_pixels)
    return image


def _difference_hash(image: Image.Image) -> str:
    """Return a dependency-free 64-bit dHash for near-duplicate triage."""

    grayscale = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:016x}"


def inspect_image(
    source: ImageSource,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> ImageInspection:
    """Decode and hash an image for inclusion in an audit manifest."""

    data = _read_limited(source, max_bytes=max_bytes)
    image, image_format, source_mode, source_size, orientation = _decode_image(
        data, max_pixels=max_pixels
    )
    return ImageInspection(
        sha256=sha256(data).hexdigest(),
        width=image.width,
        height=image.height,
        source_width=source_size[0],
        source_height=source_size[1],
        source_mode=source_mode,
        image_format=image_format,
        file_size_bytes=len(data),
        exif_orientation=orientation,
        perceptual_hash=_difference_hash(image),
    )


def sha256_file(path: str | PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Hash a file using bounded memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
