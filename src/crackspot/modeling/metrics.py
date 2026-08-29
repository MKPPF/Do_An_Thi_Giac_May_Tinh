"""Single source of truth for CrackSpot binary-classification metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def _arrays(y_true: Iterable[int], probabilities: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(list(y_true), dtype=np.int64).reshape(-1)
    scores = np.asarray(list(probabilities), dtype=np.float64).reshape(-1)
    if truth.size == 0:
        raise ValueError("Không thể tính metric trên tập rỗng")
    if truth.shape != scores.shape:
        raise ValueError(f"y_true và probabilities khác shape: {truth.shape} != {scores.shape}")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true chỉ được chứa Non-crack=0 và Crack=1")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities phải hữu hạn và nằm trong [0,1]")
    return truth, scores


def compute_binary_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5
) -> dict[str, Any]:
    """Compute report-ready metrics with Crack fixed as the positive class."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold phải nằm trong [0,1]")
    truth, scores = _arrays(y_true, probabilities)
    predicted = (scores >= threshold).astype(np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1], zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    return {
        "threshold": float(threshold),
        "sample_count": int(truth.size),
        "accuracy": float(accuracy_score(truth, predicted)),
        "non_crack": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "crack": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "classification_report": classification_report(
            truth,
            predicted,
            labels=[0, 1],
            target_names=["Non-crack", "Crack"],
            output_dict=True,
            zero_division=0,
        ),
    }
