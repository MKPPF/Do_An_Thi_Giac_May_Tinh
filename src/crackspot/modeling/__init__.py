"""Model construction, training, evaluation, threshold tuning, and Grad-CAM."""

from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.modeling.threshold import ThresholdResult, optimize_threshold

__all__ = ["ThresholdResult", "compute_binary_metrics", "optimize_threshold"]
