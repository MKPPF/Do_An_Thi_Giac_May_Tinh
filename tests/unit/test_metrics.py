from __future__ import annotations

import pytest

from crackspot.modeling.metrics import compute_binary_metrics


def test_binary_metrics_known_example() -> None:
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2], threshold=0.5)
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["tp"] == metrics["tn"] == metrics["fp"] == metrics["fn"] == 1
    assert metrics["crack"]["precision"] == 0.5
    assert metrics["crack"]["recall"] == 0.5
    assert metrics["crack"]["f1"] == 0.5


def test_metric_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.2, 1.1])


def test_metric_keeps_fixed_matrix_shape_when_class_missing() -> None:
    metrics = compute_binary_metrics([0, 0], [0.1, 0.2])
    assert metrics["confusion_matrix"] == [[2, 0], [0, 0]]
