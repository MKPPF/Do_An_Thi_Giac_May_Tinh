#!/usr/bin/env python3
"""Run CrackSpot's guarded, one-shot final-test evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from crackspot.modeling.evaluate import main
except ModuleNotFoundError:  # Support direct use before an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.modeling.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())
