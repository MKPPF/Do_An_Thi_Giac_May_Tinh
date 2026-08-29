"""Best-effort deterministic execution with explicit evidence."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_determinism(seed: int = 42, enable_ops: bool = True) -> dict[str, Any]:
    """Seed Python/NumPy/TensorFlow and return an auditable status record."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    evidence: dict[str, Any] = {
        "seed": seed,
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "tensorflow_available": False,
        "op_determinism_requested": enable_ops,
        "op_determinism_enabled": False,
        "warning": None,
    }
    try:
        import tensorflow as tf

        evidence["tensorflow_available"] = True
        tf.keras.utils.set_random_seed(seed)
        if enable_ops:
            tf.config.experimental.enable_op_determinism()
            evidence["op_determinism_enabled"] = True
    except (ImportError, RuntimeError, AttributeError) as exc:
        evidence["warning"] = f"{type(exc).__name__}: {exc}"
    return evidence
