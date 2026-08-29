"""Publication-ready plots derived from real metrics and histories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(
    matrix: list[list[float]] | np.ndarray,
    output: str | Path,
    *,
    normalized: bool = False,
    title: str | None = None,
) -> Path:
    plt = _plt()
    values = np.asarray(matrix)
    if values.shape != (2, 2):
        raise ValueError("Confusion matrix phải có shape (2,2)")
    figure, axis = plt.subplots(figsize=(5.4, 4.5), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    labels = ["Non-crack", "Crack"]
    axis.set_xticks([0, 1], labels=labels)
    axis.set_yticks([0, 1], labels=labels)
    axis.set_xlabel("Dự đoán")
    axis.set_ylabel("Nhãn thật")
    axis.set_title(title or ("Ma trận nhầm lẫn chuẩn hóa" if normalized else "Ma trận nhầm lẫn"))
    maximum = float(values.max()) if values.size else 0.0
    for row in range(2):
        for column in range(2):
            value = values[row, column]
            text = f"{value:.3f}" if normalized else str(int(value))
            color = "white" if maximum and value > maximum / 2 else "black"
            axis.text(column, row, text, ha="center", va="center", color=color)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_training_history(history: Any, output: str | Path) -> Path:
    plt = _plt()
    if hasattr(history, "history"):
        history = history.history
    if not isinstance(history, dict):
        raise TypeError("history phải là dict hoặc Keras History")
    required = {"loss", "val_loss"}
    if not required.issubset(history):
        raise ValueError(f"History thiếu {sorted(required - set(history))}")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    epochs = np.arange(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[0].legend()
    if "accuracy" in history and "val_accuracy" in history:
        axes[1].plot(epochs, history["accuracy"], label="Train")
        axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_threshold_curve(rows: list[dict[str, float]], output: str | Path) -> Path:
    plt = _plt()
    if not rows:
        raise ValueError("Không có dữ liệu threshold")
    ordered = sorted(rows, key=lambda row: row["threshold"])
    thresholds = [row["threshold"] for row in ordered]
    figure, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        axis.plot(thresholds, [row[key] for row in ordered], label=label)
    axis.set(xlabel="Threshold", ylabel="Score", ylim=(0, 1), title="Validation threshold tuning")
    axis.grid(alpha=0.25)
    axis.legend()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_dataset_distribution(summary: Any, output: str | Path) -> Path:
    """Plot class counts by split and by surface from a measured summary table."""

    import pandas as pd

    frame = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary)
    required = {"split", "surface", "label_name", "image_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset summary thiếu {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset summary rỗng")
    counts = pd.to_numeric(frame["image_count"], errors="raise")
    if counts.isna().any() or (counts < 0).any():
        raise ValueError("image_count không hợp lệ")
    work = frame.assign(image_count=counts)
    split_table = work.pivot_table(
        index="split",
        columns="label_name",
        values="image_count",
        aggfunc="sum",
        fill_value=0,
    )
    surface_table = work.pivot_table(
        index="surface",
        columns="label_name",
        values="image_count",
        aggfunc="sum",
        fill_value=0,
    )
    class_order = [name for name in ("Non-crack", "Crack") if name in work["label_name"].values]
    split_order = [
        name for name in ("train", "validation", "val", "test") if name in split_table.index
    ]
    split_order.extend(name for name in split_table.index if name not in split_order)
    surface_order = [name for name in ("D", "P", "W") if name in surface_table.index]
    surface_order.extend(name for name in surface_table.index if name not in surface_order)
    split_table = split_table.reindex(index=split_order, columns=class_order, fill_value=0)
    surface_table = surface_table.reindex(index=surface_order, columns=class_order, fill_value=0)

    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    colors = ["#4C78A8", "#E45756"][: len(class_order)]
    split_table.plot(kind="bar", ax=axes[0], color=colors)
    surface_table.plot(kind="bar", ax=axes[1], color=colors)
    axes[0].set(title="Phân bố lớp theo split", xlabel="Split", ylabel="Số ảnh")
    axes[1].set(title="Phân bố lớp theo bề mặt", xlabel="Bề mặt", ylabel="Số ảnh")
    for axis in axes:
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Nhãn")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target
