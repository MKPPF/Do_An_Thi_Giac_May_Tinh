"""Deterministic qualitative evidence for augmentation and Grad-CAM."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from crackspot.data import (
    audit_split,
    build_tf_dataset,
    load_image_rgb,
    load_manifest_table,
    manifest_sha256,
)
from crackspot.inference import InferenceService
from crackspot.reporting.export import write_json
from crackspot.utils.hashing import sha256_file

OUTCOME_ORDER = ("TP", "TN", "FP", "FN")
MAX_REPORT_PROBABILITY_TOLERANCE = 1e-4


class QualitativeEvidenceError(RuntimeError):
    """Raised when qualitative evidence cannot be tied to its inputs."""


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    try:
        return load_manifest_table(source)
    except ValueError as exc:
        raise ValueError(f"Chỉ hỗ trợ manifest/predictions CSV hoặc Parquet: {source}") from exc


def _canonical_relative_path(value: object) -> str:
    if value is None or value is pd.NA:
        raise QualitativeEvidenceError(f"relative_path không an toàn: {value!r}")
    try:
        if bool(pd.isna(value)):
            raise QualitativeEvidenceError(f"relative_path không an toàn: {value!r}")
    except (TypeError, ValueError):
        pass
    text = str(value).strip().replace("\\", "/")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or not posix.parts
        or posix.as_posix() == "."
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
    ):
        raise QualitativeEvidenceError(f"relative_path không an toàn: {value!r}")
    return posix.as_posix()


def _resolve_relative_path(root: Path, value: object) -> Path:
    text = _canonical_relative_path(value)
    posix = PurePosixPath(text)
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise QualitativeEvidenceError(f"relative_path thoát khỏi dataset root: {value!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if frame[column].map(lambda value: isinstance(value, bool | np.bool_)).any():
        raise QualitativeEvidenceError(f"{column} không được chứa bool")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QualitativeEvidenceError(f"{column} phải là cột số hợp lệ") from exc
    if not np.isfinite(values).all():
        raise QualitativeEvidenceError(f"{column} phải chứa giá trị hữu hạn")
    return values


def _validated_threshold(value: object) -> float:
    if isinstance(value, bool):
        raise QualitativeEvidenceError("threshold không hợp lệ")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise QualitativeEvidenceError("threshold không hợp lệ") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise QualitativeEvidenceError("threshold phải hữu hạn trong [0,1]")
    return threshold


def _require_boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} phải là bool")
    return value


def _augmentation_parameters(
    *,
    rotation_degrees: object,
    brightness_delta: object,
    contrast_delta: object,
) -> tuple[float, float, float]:
    values: dict[str, float] = {}
    for name, raw in (
        ("rotation_degrees", rotation_degrees),
        ("brightness_delta", brightness_delta),
        ("contrast_delta", contrast_delta),
    ):
        if isinstance(raw, bool):
            raise ValueError(f"{name} không hợp lệ")
        try:
            values[name] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} không hợp lệ") from exc
        if not math.isfinite(values[name]):
            raise ValueError(f"{name} phải hữu hạn")
    if not 0.0 <= values["rotation_degrees"] <= 15.0:
        raise ValueError("rotation_degrees phải trong [0,15]")
    if not 0.0 <= values["brightness_delta"] <= 0.20:
        raise ValueError("brightness_delta phải trong [0,0.20]")
    if not 0.0 <= values["contrast_delta"] <= 0.20:
        raise ValueError("contrast_delta phải trong [0,0.20]")
    return (
        values["rotation_degrees"],
        values["brightness_delta"],
        values["contrast_delta"],
    )


def _expected_outcomes(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    outcomes = np.full(len(truth), "TN", dtype=object)
    outcomes[(truth == 1) & (predicted == 1)] = "TP"
    outcomes[(truth == 0) & (predicted == 1)] = "FP"
    outcomes[(truth == 1) & (predicted == 0)] = "FN"
    return outcomes


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Validate labels/probabilities/outcomes and return a normalized copy."""

    required = {"relative_path", "y_true", "probability_crack"}
    missing = required.difference(frame.columns)
    if missing:
        raise QualitativeEvidenceError(f"Predictions thiếu cột: {sorted(missing)}")
    normalized = frame.copy()
    if normalized.empty:
        raise QualitativeEvidenceError("Predictions rỗng")
    normalized["relative_path"] = normalized["relative_path"].map(_canonical_relative_path)
    if normalized["relative_path"].duplicated().any():
        raise QualitativeEvidenceError("Predictions có relative_path trùng")
    truth_values = _numeric_values(normalized, "y_true")
    scores = _numeric_values(normalized, "probability_crack")
    if not np.isin(truth_values, [0.0, 1.0]).all():
        raise QualitativeEvidenceError("y_true chỉ được chứa 0/1")
    truth = truth_values.astype(np.int64)
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise QualitativeEvidenceError("probability_crack phải trong [0,1]")

    if threshold is None:
        if "threshold" not in normalized.columns:
            raise QualitativeEvidenceError("Cần threshold đã khóa để xác minh outcome")
        thresholds = _numeric_values(normalized, "threshold")
        if not np.allclose(thresholds, thresholds[0], rtol=0.0, atol=1e-12):
            raise QualitativeEvidenceError("Predictions phải dùng đúng một threshold")
        threshold = _validated_threshold(thresholds[0])
    else:
        threshold = _validated_threshold(threshold)
    if "threshold" in normalized.columns:
        observed = _numeric_values(normalized, "threshold")
        if not np.allclose(observed, threshold, rtol=0.0, atol=1e-12):
            raise QualitativeEvidenceError("Threshold trong predictions không khớp selection")

    predicted = (scores >= threshold).astype(np.int64)
    if "y_pred" in normalized.columns:
        recorded_values = _numeric_values(normalized, "y_pred")
        if not np.isin(recorded_values, [0.0, 1.0]).all():
            raise QualitativeEvidenceError("y_pred chỉ được chứa 0/1")
        recorded = recorded_values.astype(np.int64)
        if not np.array_equal(recorded, predicted):
            raise QualitativeEvidenceError("y_pred không khớp probability/threshold")
    expected = _expected_outcomes(truth, predicted)
    if "outcome" in normalized.columns:
        recorded_outcomes = (
            normalized["outcome"].astype("string").str.strip().str.upper().to_numpy()
        )
        if not np.array_equal(recorded_outcomes, expected):
            raise QualitativeEvidenceError("outcome không khớp y_true/y_pred")
    if "sha256" in normalized.columns:
        hashes = normalized["sha256"].astype("string").str.strip().str.lower()
        if hashes.isna().any() or not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
            raise QualitativeEvidenceError("sha256 trong predictions không hợp lệ")
        normalized["sha256"] = hashes.astype(str)
    normalized["y_true"] = truth
    normalized["probability_crack"] = scores
    normalized["threshold"] = threshold
    normalized["y_pred"] = predicted
    normalized["outcome"] = expected
    return normalized


def select_outcome_examples(frame: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    """Select one reproducible, high-confidence example for every outcome."""

    normalized = validate_prediction_frame(frame)
    selected: dict[str, dict[str, Any] | None] = {}
    for outcome in OUTCOME_ORDER:
        candidates = normalized.loc[normalized["outcome"].eq(outcome)].copy()
        if candidates.empty:
            selected[outcome] = None
            continue
        descending = outcome in {"TP", "FP"}
        candidates = candidates.sort_values(
            ["probability_crack", "relative_path"],
            ascending=[not descending, True],
            kind="stable",
        )
        row = candidates.iloc[0]
        record: dict[str, Any] = {
            "relative_path": str(row["relative_path"]),
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "probability_crack": float(row["probability_crack"]),
            "threshold": float(row["threshold"]),
            "outcome": outcome,
        }
        if "sha256" in normalized.columns:
            record["sha256"] = str(row["sha256"])
        selected[outcome] = record
    return selected


def generate_gradcam_outcome_grid(
    *,
    predictions_path: str | Path,
    dataset_root: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    threshold: float,
    valid_for_report: bool,
    experiment: str | None = None,
    run_id: str | None = None,
    config_sha256: str | None = None,
    manifest_sha256: str | None = None,
    selection_path: str | Path | None = None,
    probability_tolerance: float = 1e-4,
    service: Any | None = None,
) -> dict[str, Any]:
    """Create a deterministic TP/TN/FP/FN Grad-CAM grid and JSON sidecar."""

    report_valid = _require_boolean(valid_for_report, "valid_for_report")
    locked_threshold = _validated_threshold(threshold)
    if isinstance(probability_tolerance, bool):
        raise ValueError("probability_tolerance không hợp lệ")
    try:
        tolerance = float(probability_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability_tolerance không hợp lệ") from exc
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("probability_tolerance phải hữu hạn và không âm")
    if report_valid and tolerance > MAX_REPORT_PROBABILITY_TOLERANCE:
        raise QualitativeEvidenceError(
            "Artifact report không được nới probability_tolerance quá "
            f"{MAX_REPORT_PROBABILITY_TOLERANCE:g}"
        )
    if report_valid and service is not None:
        raise QualitativeEvidenceError(
            "Artifact report phải tự load checkpoint đã khóa; không nhận service tiêm vào"
        )
    provenance_values = {
        "experiment": str(experiment or "").strip().upper(),
        "run_id": str(run_id or "").strip(),
        "config_sha256": str(config_sha256 or "").strip().lower(),
        "manifest_sha256": str(manifest_sha256 or "").strip().lower(),
    }
    if report_valid:
        if provenance_values["experiment"] != "E5":
            raise QualitativeEvidenceError("Grad-CAM report-valid phải thuộc selection E5")
        if not provenance_values["run_id"]:
            raise QualitativeEvidenceError("Grad-CAM report-valid thiếu run_id")
        for field in ("config_sha256", "manifest_sha256"):
            value = provenance_values[field]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise QualitativeEvidenceError(f"Grad-CAM report-valid thiếu {field} hợp lệ")
        if selection_path is None:
            raise QualitativeEvidenceError("Grad-CAM report-valid thiếu selection contract")
    source = Path(predictions_path).resolve()
    output = Path(output_path)
    if output.suffix.casefold() != ".png":
        raise ValueError("Grad-CAM grid phải xuất PNG")
    sidecar = output.with_suffix(".json")
    conflicts = [path for path in (output, sidecar) if path.exists()]
    if conflicts:
        raise FileExistsError(f"Từ chối ghi đè artifact: {conflicts}")

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    model_file = Path(model_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    model_hash = sha256_file(model_file)
    metadata_hash = sha256_file(metadata_file)
    predictions_hash = sha256_file(source)
    selection_file = Path(selection_path).resolve() if selection_path is not None else None
    if selection_file is not None and not selection_file.is_file():
        raise FileNotFoundError(selection_file)
    frame = validate_prediction_frame(_read_table(source), threshold=locked_threshold)
    if report_valid and "sha256" not in frame.columns:
        raise QualitativeEvidenceError("Predictions report-valid phải có SHA-256 từ manifest")
    selected = select_outcome_examples(frame)
    selected_paths = {record["relative_path"] for record in selected.values() if record is not None}
    resolved_images = {
        relative_path: _resolve_relative_path(root, relative_path)
        for relative_path in selected_paths
    }
    selected_image_hashes = {
        relative_path: sha256_file(image_path)
        for relative_path, image_path in resolved_images.items()
    }
    for record in selected.values():
        if (
            record is not None
            and "sha256" in record
            and selected_image_hashes[record["relative_path"]] != record["sha256"]
        ):
            raise QualitativeEvidenceError(
                f"SHA-256 ảnh không khớp predictions: {record['relative_path']}"
            )
    inference = (
        service
        if service is not None
        else InferenceService.from_files(model_file, metadata_file, verify_hash=True)
    )
    rendered: dict[str, dict[str, Any] | None] = {}

    plt = _plt()
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    try:
        for axis, outcome in zip(axes.ravel(), OUTCOME_ORDER, strict=True):
            record = selected[outcome]
            axis.axis("off")
            if record is None:
                axis.text(
                    0.5,
                    0.5,
                    f"{outcome}\nKhông có mẫu trong predictions",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                rendered[outcome] = None
                continue
            image_path = resolved_images[record["relative_path"]]
            image_hash = selected_image_hashes[record["relative_path"]]
            result = inference.predict_image(image_path, include_gradcam=True)
            if result.overlay is None or result.heatmap is None:
                raise QualitativeEvidenceError(f"Grad-CAM không được tạo cho {image_path}")
            heatmap = np.asarray(result.heatmap, dtype=np.float64)
            if (
                heatmap.ndim != 2
                or heatmap.size == 0
                or not np.isfinite(heatmap).all()
                or np.any((heatmap < 0.0) | (heatmap > 1.0))
            ):
                raise QualitativeEvidenceError(
                    f"Grad-CAM không phải heatmap 2D hữu hạn trong [0,1]: {image_path}"
                )
            probability = float(result.crack_probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise QualitativeEvidenceError(f"P(Crack) suy luận không hợp lệ: {image_path}")
            delta = abs(probability - record["probability_crack"])
            if delta > tolerance:
                raise QualitativeEvidenceError(
                    "P(Crack) khi dựng Grad-CAM không khớp predictions: "
                    f"{record['relative_path']} delta={delta:.8g}"
                )
            axis.imshow(result.overlay)
            truth_name = "Crack" if record["y_true"] == 1 else "Non-crack"
            prediction_name = "Crack" if record["y_pred"] == 1 else "Non-crack"
            axis.set_title(
                f"{outcome} | true={truth_name} | pred={prediction_name}\n"
                f"P(Crack)={record['probability_crack']:.4f} | {record['relative_path']}",
                fontsize=9,
            )
            rendered[outcome] = {
                **record,
                "image_sha256": image_hash,
                "inference_probability": probability,
                "probability_delta": delta,
                "heatmap_min": float(np.min(heatmap)),
                "heatmap_max": float(np.max(heatmap)),
            }
        figure.suptitle(
            "Grad-CAM: vùng kích hoạt score Crack (không phải mask phân đoạn)",
            fontsize=14,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    output_hash = sha256_file(output)

    payload = {
        "schema_version": 1,
        "kind": "gradcam_tp_tn_fp_fn_grid",
        "status": "REPORT_ARTIFACT" if report_valid else "NOT_VALID_FOR_REPORT",
        "valid_for_report": report_valid,
        "selection_rule": {
            "TP_FP": "highest_probability_then_relative_path",
            "TN_FN": "lowest_probability_then_relative_path",
            "missing_outcome": "render_explicit_empty_panel",
        },
        "threshold": locked_threshold,
        "predictions_path": str(source),
        "predictions_sha256": predictions_hash,
        "model_path": str(model_file),
        "model_sha256": model_hash,
        "metadata_path": str(metadata_file),
        "metadata_sha256": metadata_hash,
        "experiment": provenance_values["experiment"] or None,
        "run_id": provenance_values["run_id"] or None,
        "config_sha256": provenance_values["config_sha256"] or None,
        "manifest_sha256": provenance_values["manifest_sha256"] or None,
        "selection_path": str(selection_file) if selection_file is not None else None,
        "selection_sha256": sha256_file(selection_file) if selection_file is not None else None,
        "probability_tolerance": tolerance,
        "selected": rendered,
        "output": str(output.resolve()),
        "output_sha256": output_hash,
        "caveat": "Grad-CAM là vùng mô hình chú ý cho score Crack, không phải mask pixel.",
    }
    write_json(sidecar, payload, overwrite=False)
    return payload


def generate_augmentation_audit_grid(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    sample_count: int = 6,
    seed: int = 42,
    image_size: tuple[int, int] = (224, 224),
    rotation_degrees: float = 15.0,
    brightness_delta: float = 0.15,
    contrast_delta: float = 0.15,
    valid_for_report: bool = True,
    experiment: str | None = None,
    run_id: str | None = None,
    config_path: str | Path | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Render originals beside outputs from the exact train-only ``tf.data`` path."""

    report_valid = _require_boolean(valid_for_report, "valid_for_report")
    experiment_id = str(experiment or "").strip().upper()
    bound_run_id = str(run_id or "").strip()
    semantic_config_hash = str(config_sha256 or "").strip().lower()
    config_file = Path(config_path).resolve() if config_path is not None else None
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count phải là số nguyên")
    if sample_count <= 0 or sample_count > 12:
        raise ValueError("sample_count phải trong [1,12]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed phải là số nguyên không âm")
    rotation, brightness, contrast = _augmentation_parameters(
        rotation_degrees=rotation_degrees,
        brightness_delta=brightness_delta,
        contrast_delta=contrast_delta,
    )
    output = Path(output_path)
    if output.suffix.casefold() != ".png":
        raise ValueError("Augmentation audit phải xuất PNG")
    sidecar = output.with_suffix(".json")
    conflicts = [path for path in (output, sidecar) if path.exists()]
    if conflicts:
        raise FileExistsError(f"Từ chối ghi đè artifact: {conflicts}")
    manifest_file = Path(manifest_path).resolve()
    frame = _read_table(manifest_file)
    required = {"relative_path", "label", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise QualitativeEvidenceError(f"Manifest thiếu cột: {sorted(missing)}")
    manifest_hash = manifest_sha256(frame)
    normalized_paths = frame["relative_path"].map(_canonical_relative_path)
    if normalized_paths.duplicated().any():
        raise QualitativeEvidenceError("Manifest có relative_path trùng")
    frame = frame.copy()
    frame["relative_path"] = normalized_paths
    label_values = _numeric_values(frame, "label")
    if not np.isin(label_values, [0.0, 1.0]).all():
        raise QualitativeEvidenceError("Manifest label chỉ được chứa 0/1")
    frame["label"] = label_values.astype(np.int64)
    if "sha256" in frame.columns:
        hashes = frame["sha256"].astype("string").str.strip().str.lower()
        if hashes.isna().any() or not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
            raise QualitativeEvidenceError("Manifest có SHA-256 không hợp lệ")
        frame["sha256"] = hashes.astype(str)
    train = frame.loc[frame["split"].astype("string").str.strip().str.casefold().eq("train")].copy()
    if train.empty:
        raise QualitativeEvidenceError("Manifest không có split train")
    split_audit = audit_split(frame)
    if report_valid and not split_audit["valid"]:
        raise QualitativeEvidenceError(
            f"Từ chối artifact report khi split/nguồn nhóm không hợp lệ: {split_audit['errors']}"
        )
    if report_valid:
        if experiment_id != "E4":
            raise QualitativeEvidenceError("Augmentation report-valid phải thuộc run E4")
        if not bound_run_id:
            raise QualitativeEvidenceError("Augmentation report-valid thiếu run_id")
        if len(semantic_config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in semantic_config_hash
        ):
            raise QualitativeEvidenceError("Augmentation report-valid thiếu config_sha256 hợp lệ")
        if config_file is None:
            raise QualitativeEvidenceError("Augmentation report-valid thiếu config snapshot")
    if config_file is not None and not config_file.is_file():
        raise FileNotFoundError(config_file)

    ordered = train.sort_values("relative_path", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    count = min(sample_count, len(ordered))
    indices = np.sort(rng.choice(len(ordered), size=count, replace=False))
    sampled = ordered.iloc[indices].copy().reset_index(drop=True)
    dataset_frame = sampled.drop(columns=["absolute_path"], errors="ignore")
    dataset = build_tf_dataset(
        dataset_frame,
        batch_size=count,
        image_size=image_size,
        training=True,
        seed=seed,
        augment=True,
        dataset_root=dataset_root,
        include_paths=True,
        shuffle_buffer=count,
        rotation_degrees=rotation,
        brightness_delta=brightness,
        contrast_delta=contrast,
    )
    batch_iterator = iter(dataset.take(1).as_numpy_iterator())
    try:
        processed, labels, paths = next(batch_iterator)
    except StopIteration as exc:
        raise QualitativeEvidenceError("Không đọc được augmentation batch") from exc
    if next(batch_iterator, None) is not None:
        raise QualitativeEvidenceError("Không đọc được augmentation batch")
    processed_array = np.asarray(processed, dtype=np.float32)
    if (
        processed_array.shape != (count, int(image_size[0]), int(image_size[1]), 3)
        or not np.isfinite(processed_array).all()
        or np.any((processed_array < -1.0) | (processed_array > 1.0))
    ):
        raise QualitativeEvidenceError("Augmentation batch không phải RGB MobileNetV2 trong [-1,1]")
    augmented = np.clip((processed_array + 1.0) * 127.5, 0, 255).round().astype(np.uint8)

    plt = _plt()
    figure, axes = plt.subplots(count, 2, figsize=(8, 3.6 * count), squeeze=False)
    records: list[dict[str, Any]] = []
    sampled_by_path = sampled.set_index("relative_path", verify_integrity=True)
    root = Path(dataset_root).resolve()
    try:
        for row_index, (pixels, label, raw_path) in enumerate(
            zip(augmented, labels, paths, strict=True)
        ):
            path_text = raw_path.decode("utf-8") if isinstance(raw_path, bytes) else str(raw_path)
            image_path = Path(path_text).resolve()
            try:
                relative = image_path.relative_to(root).as_posix()
            except ValueError as exc:
                raise QualitativeEvidenceError(
                    f"Augmentation path thoát khỏi dataset root: {path_text}"
                ) from exc
            relative = _canonical_relative_path(relative)
            if relative not in sampled_by_path.index:
                raise QualitativeEvidenceError(
                    f"Augmentation trả về ảnh không thuộc mẫu đã chọn: {relative}"
                )
            expected_label = int(sampled_by_path.loc[relative, "label"])
            observed_label = float(label)
            if not math.isfinite(observed_label) or observed_label != expected_label:
                raise QualitativeEvidenceError(f"Nhãn augmentation không khớp manifest: {relative}")
            original = load_image_rgb(image_path)
            image_hash = sha256_file(image_path)
            if "sha256" in sampled_by_path.columns:
                expected_hash = str(sampled_by_path.loc[relative, "sha256"]).strip().lower()
                if image_hash != expected_hash:
                    raise QualitativeEvidenceError(f"SHA-256 ảnh không khớp manifest: {relative}")
            axes[row_index, 0].imshow(original)
            axes[row_index, 0].set_title(f"Gốc | label={expected_label} | {relative}", fontsize=9)
            axes[row_index, 1].imshow(pixels)
            axes[row_index, 1].set_title("Sau augmentation train", fontsize=9)
            for axis in axes[row_index]:
                axis.axis("off")
            records.append(
                {
                    "relative_path": relative,
                    "label": expected_label,
                    "image_sha256": image_hash,
                }
            )
        figure.suptitle("Audit trước/sau augmentation (chỉ pipeline train)", fontsize=14)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    output_hash = sha256_file(output)

    payload = {
        "schema_version": 1,
        "kind": "train_augmentation_before_after",
        "status": "REPORT_ARTIFACT" if report_valid else "NOT_VALID_FOR_REPORT",
        "valid_for_report": report_valid,
        "manifest_path": str(manifest_file),
        "manifest_file_sha256": sha256_file(manifest_file),
        "manifest_sha256": manifest_hash,
        "experiment": experiment_id or None,
        "run_id": bound_run_id or None,
        "config_path": str(config_file) if config_file is not None else None,
        "config_file_sha256": sha256_file(config_file) if config_file is not None else None,
        "config_sha256": semantic_config_hash or None,
        "split_audit": split_audit,
        "seed": seed,
        "sample_count": count,
        "image_size": list(image_size),
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "max_rotation_degrees": rotation,
            "brightness_delta": brightness,
            "contrast_delta": contrast,
        },
        "samples": records,
        "output": str(output.resolve()),
        "output_sha256": output_hash,
    }
    write_json(sidecar, payload, overwrite=False)
    return payload


__all__ = [
    "OUTCOME_ORDER",
    "QualitativeEvidenceError",
    "generate_augmentation_audit_grid",
    "generate_gradcam_outcome_grid",
    "select_outcome_examples",
    "validate_prediction_frame",
]
