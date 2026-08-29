"""Project-wide constants and the immutable class contract."""

from __future__ import annotations

from pathlib import Path

SEED = 42
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

NON_CRACK_LABEL = 0
CRACK_LABEL = 1
LABEL_MAPPING = {"Non-crack": NON_CRACK_LABEL, "Crack": CRACK_LABEL}
INDEX_TO_LABEL = {value: key for key, value in LABEL_MAPPING.items()}

DEFAULT_THRESHOLD = 0.5
DEFAULT_GRADCAM_LAYER = "out_relu"
PREPROCESSING_NAME = "mobilenet_v2.preprocess_input"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_MANIFEST_COLUMNS = (
    "relative_path",
    "label",
    "surface",
    "source_group",
    "sha256",
    "width",
    "height",
    "split",
)
