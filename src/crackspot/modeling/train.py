"""Reproducible validation-only training runner for experiments E1-E4."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crackspot import __version__
from crackspot.config import load_config
from crackspot.constants import LABEL_MAPPING, PREPROCESSING_NAME
from crackspot.data import (
    audit_split,
    build_tf_dataset,
    compute_balanced_class_weights,
    load_manifest_table,
    manifest_sha256,
    verify_official_dataset_preconditions,
)
from crackspot.modeling.metrics import compute_binary_metrics
from crackspot.modeling.model import (
    build_from_config,
    compile_binary_model,
    configure_fine_tuning,
)
from crackspot.reporting.export import write_json
from crackspot.reporting.plots import plot_confusion_matrix, plot_training_history
from crackspot.utils.environment import verify_official_git_state, write_environment
from crackspot.utils.hashing import sha256_file, sha256_json
from crackspot.utils.reproducibility import set_global_determinism

TRAINING_CONTRACT_FILENAME = "training_contract.json"
RUN_COMPLETION_FILENAME = "training_complete.json"
FINAL_EVIDENCE_FILES = (
    RUN_COMPLETION_FILENAME,
    "run_summary.json",
)
FINALIZATION_ARTIFACTS = (
    "model.keras",
    "metrics_validation.json",
    "classification_report_validation.json",
    "predictions_validation.csv",
    "errors_validation.csv",
    "confusion_matrix_validation.png",
    "confusion_matrix_validation_normalized.png",
    "trainability_final.json",
    "training_duration.json",
    "model.metadata.json",
    "run_summary.json",
    RUN_COMPLETION_FILENAME,
)


class TrainingProtocolError(RuntimeError):
    """Raised when a run would violate the locked experimental protocol."""


def _tensorflow():
    try:
        import tensorflow as tf
    except (ImportError, RuntimeError, ValueError) as exc:
        raise RuntimeError("TensorFlow không sẵn sàng trong môi trường hiện tại") from exc
    return tf


def _new_run_id(experiment: str, config_hash: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment}-{timestamp}-{config_hash[:8]}"


def _load_manifest(path: Path) -> pd.DataFrame:
    return load_manifest_table(path)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingProtocolError(f"Thiếu {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise TrainingProtocolError(f"{description} không hợp lệ: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingProtocolError(f"{description} phải là JSON object: {path}")
    return payload


def _validate_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value or Path(value).name != value or value in {".", ".."}:
        raise TrainingProtocolError("run_id phải là một tên thư mục đơn, không chứa đường dẫn")
    return value


def _smoke_preconditions() -> dict[str, Any]:
    return {
        "required": False,
        "status": "NOT_VALID_FOR_REPORT",
        "git": None,
        "official_dataset": None,
    }


def _official_preconditions(
    *,
    git: dict[str, Any],
    official_dataset: dict[str, Any],
) -> dict[str, Any]:
    return {
        "required": True,
        "status": "OFFICIAL_EXPERIMENT_PRECONDITIONS_VERIFIED",
        "git": git,
        "official_dataset": official_dataset,
    }


def _precondition_evidence(preconditions: dict[str, Any]) -> dict[str, Any]:
    git = preconditions.get("git")
    dataset = preconditions.get("official_dataset")
    integrity = dataset.get("dataset_integrity") if isinstance(dataset, dict) else None
    return {
        "official_preconditions": preconditions,
        "git_commit": git.get("head_commit") if isinstance(git, dict) else None,
        "git_tracked_status_porcelain": (
            git.get("tracked_status_porcelain") if isinstance(git, dict) else None
        ),
        "dataset_integrity_fingerprint_sha256": (
            integrity.get("content_fingerprint_sha256") if isinstance(integrity, dict) else None
        ),
    }


def _prepare_run_directory(
    output_root: str | Path,
    run_id: str | None,
    *,
    experiment: str,
    config_hash: str,
    resume: bool,
) -> tuple[Path, str]:
    if resume and run_id is None:
        raise TrainingProtocolError("--resume bắt buộc đi cùng --run-id")
    resolved_run_id = _validate_run_id(run_id or _new_run_id(str(experiment), config_hash))
    run_dir = Path(output_root) / resolved_run_id
    if resume:
        if not run_dir.is_dir():
            raise TrainingProtocolError(
                f"--resume chỉ dùng cho run directory đã tồn tại: {run_dir}"
            )
        completed = [name for name in FINAL_EVIDENCE_FILES if (run_dir / name).exists()]
        if completed:
            raise TrainingProtocolError(
                "Run đã hoàn tất; từ chối resume và không ghi đè evidence: " + ", ".join(completed)
            )
    else:
        if run_dir.exists():
            raise FileExistsError(f"Không ghi đè run directory: {run_dir}")
        run_dir.mkdir(parents=True)
    return run_dir, resolved_run_id


def _write_initial_contract(
    run_dir: Path,
    *,
    values: dict[str, Any],
    config_hash: str,
    manifest_hash: str,
    manifest_path: Path,
    run_id: str,
    official_preconditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = run_dir / "config_snapshot.json"
    write_json(snapshot, values, overwrite=False)
    preconditions = official_preconditions or _smoke_preconditions()
    contract = {
        "schema_version": 2,
        "run_id": run_id,
        "config_sha256": config_hash,
        "config_snapshot_file_sha256": sha256_file(snapshot),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        **_precondition_evidence(preconditions),
        "resume_policy": {
            "same_run_id_only": True,
            "config_and_manifest_must_match": True,
            "completed_phases_are_immutable": True,
        },
    }
    write_json(run_dir / TRAINING_CONTRACT_FILENAME, contract, overwrite=False)
    return contract


def _verify_resume_contract(
    run_dir: Path,
    *,
    values: dict[str, Any],
    config_hash: str,
    manifest_hash: str,
    run_id: str,
    official_preconditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _load_json_object(run_dir / TRAINING_CONTRACT_FILENAME, "training resume contract")
    snapshot_path = run_dir / "config_snapshot.json"
    snapshot = _load_json_object(snapshot_path, "config snapshot")
    if contract.get("run_id") != run_id:
        raise TrainingProtocolError("run_id không khớp training contract")
    if snapshot != values:
        raise TrainingProtocolError(
            "Giá trị config hiện tại không khớp config_snapshot.json; từ chối resume"
        )
    snapshot_semantic_hash = sha256_json(snapshot)
    if snapshot_semantic_hash != config_hash or contract.get("config_sha256") != config_hash:
        raise TrainingProtocolError("Config hash không khớp training contract; từ chối resume")
    snapshot_file_hash = sha256_file(snapshot_path)
    if contract.get("config_snapshot_file_sha256") != snapshot_file_hash:
        raise TrainingProtocolError("File hash config_snapshot.json đã thay đổi; từ chối resume")
    if contract.get("manifest_sha256") != manifest_hash:
        raise TrainingProtocolError(
            "Canonical manifest hash không khớp training contract; từ chối resume"
        )
    if official_preconditions is not None:
        expected_evidence = _precondition_evidence(official_preconditions)
        for field, expected in expected_evidence.items():
            if contract.get(field) != expected:
                raise TrainingProtocolError(
                    f"Official precondition {field} không khớp training contract; từ chối resume"
                )
    return contract


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise TrainingProtocolError(f"{field} phải là SHA-256 64 ký tự hex")
    return text


def _resolve_project_artifact(project_root: Path, value: Any, field: str) -> Path:
    text = str(value or "").strip().replace("\\", "/")
    portable = Path(text)
    if not text or portable.is_absolute() or ".." in portable.parts:
        raise TrainingProtocolError(f"model_selection.{field} không phải path portable an toàn")
    target = (project_root / portable).resolve()
    try:
        target.relative_to(project_root)
    except ValueError as exc:  # pragma: no cover - protected by path checks.
        raise TrainingProtocolError(f"model_selection.{field} thoát khỏi project root") from exc
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def _verified_selection_candidate(
    candidate: Any,
    *,
    expected_experiment: str,
    expected_manifest_hash: str,
    project_root: Path,
) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise TrainingProtocolError("Mỗi model_selection candidate phải là JSON object")
    experiment = str(candidate.get("experiment", "")).strip().upper()
    if experiment != expected_experiment:
        raise TrainingProtocolError(
            f"model_selection phải chứa đúng candidate {expected_experiment}, nhận {experiment!r}"
        )
    run_id = str(candidate.get("run_id", "")).strip()
    if not run_id:
        raise TrainingProtocolError(f"Candidate {experiment} thiếu run_id")
    try:
        best_val_loss = float(candidate["best_val_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingProtocolError(f"Candidate {experiment} thiếu best_val_loss hợp lệ") from exc
    if not math.isfinite(best_val_loss) or best_val_loss < 0:
        raise TrainingProtocolError(f"Candidate {experiment} có best_val_loss không hợp lệ")

    manifest_hash = _require_sha256(
        candidate.get("manifest_sha256"), f"candidate {experiment} manifest_sha256"
    )
    if manifest_hash != expected_manifest_hash:
        raise TrainingProtocolError(
            f"Candidate {experiment} không dùng cùng manifest với E4 hiện tại"
        )
    config_hash = _require_sha256(
        candidate.get("config_sha256"), f"candidate {experiment} config_sha256"
    )
    checkpoint_hash = _require_sha256(
        candidate.get("checkpoint_sha256"), f"candidate {experiment} checkpoint_sha256"
    )
    config_path = _resolve_project_artifact(project_root, candidate.get("config"), "config")
    checkpoint_path = _resolve_project_artifact(
        project_root, candidate.get("checkpoint"), "checkpoint"
    )
    if config_path.parent != checkpoint_path.parent:
        raise TrainingProtocolError(f"Candidate {experiment} config/checkpoint khác run directory")
    run_dir = config_path.parent
    if run_dir.name != run_id:
        raise TrainingProtocolError(f"Candidate {experiment} run_id không khớp artifact path")

    config_payload = _load_json_object(config_path, f"candidate {experiment} config snapshot")
    if sha256_json(config_payload) != config_hash:
        raise TrainingProtocolError(f"Candidate {experiment} config semantic hash không khớp")
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise TrainingProtocolError(f"Candidate {experiment} checkpoint hash không khớp")
    config_experiment = config_payload.get("experiment")
    if (
        not isinstance(config_experiment, dict)
        or str(config_experiment.get("id", "")).strip().upper() != experiment
    ):
        raise TrainingProtocolError(f"Candidate {experiment} config snapshot sai experiment")

    summary_path = run_dir / "run_summary.json"
    completion_path = run_dir / RUN_COMPLETION_FILENAME
    recorded_completion_path = _resolve_project_artifact(
        project_root,
        candidate.get("training_complete"),
        "training_complete",
    )
    if recorded_completion_path != completion_path:
        raise TrainingProtocolError(
            f"Candidate {experiment} training_complete không cùng run directory"
        )
    recorded_completion_hash = _require_sha256(
        candidate.get("training_complete_sha256"),
        f"candidate {experiment} training_complete_sha256",
    )
    if sha256_file(completion_path) != recorded_completion_hash:
        raise TrainingProtocolError(
            f"Candidate {experiment} training_complete hash không khớp model_selection"
        )
    summary = _load_json_object(summary_path, f"candidate {experiment} run summary")
    completion = _load_json_object(completion_path, f"candidate {experiment} training completion")
    if (
        summary.get("valid_for_report") is not True
        or summary.get("status") != "VALIDATION_COMPLETE_TEST_LOCKED"
        or completion.get("status") != "VALIDATION_COMPLETE_TEST_LOCKED"
        or completion.get("immutable") is not True
    ):
        raise TrainingProtocolError(f"Candidate {experiment} không report-valid/hoàn tất")
    expected_pairs = {
        "run_id": run_id,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "model_sha256": checkpoint_hash,
    }
    for field, expected in expected_pairs.items():
        if str(summary.get(field, "")).strip().lower() != expected.lower():
            raise TrainingProtocolError(f"Candidate {experiment} run_summary.{field} không khớp")
        if str(completion.get(field, "")).strip().lower() != expected.lower():
            raise TrainingProtocolError(
                f"Candidate {experiment} training_complete.{field} không khớp"
            )
    try:
        summary_val_loss = float(summary["best_val_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingProtocolError(
            f"Candidate {experiment} run_summary thiếu best_val_loss"
        ) from exc
    if not math.isclose(summary_val_loss, best_val_loss, rel_tol=0.0, abs_tol=1e-15):
        raise TrainingProtocolError(f"Candidate {experiment} best_val_loss không khớp summary")

    artifact_hashes = completion.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise TrainingProtocolError(f"Candidate {experiment} completion thiếu artifact hashes")
    required_artifacts = {
        "config_snapshot.json": config_path,
        "model.keras": checkpoint_path,
        "run_summary.json": summary_path,
    }
    for filename, path in required_artifacts.items():
        expected_file_hash = _require_sha256(
            artifact_hashes.get(filename),
            f"candidate {experiment} training_complete.artifact_sha256.{filename}",
        )
        if sha256_file(path) != expected_file_hash:
            raise TrainingProtocolError(
                f"Candidate {experiment} artifact {filename} không khớp training_complete"
            )
    return {
        "experiment": experiment,
        "run_id": run_id,
        "best_val_loss": best_val_loss,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "config_sha256": config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
    }


def _resolve_e4_selection(
    values: dict[str, Any],
    selection_path: Path | None,
    *,
    expected_manifest_hash: str | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    fine_tune = values.get("training", {}).get("fine_tune", {})
    if not fine_tune.get("inherit_from_model_selection", False):
        return values
    if selection_path is None or not selection_path.is_file():
        raise TrainingProtocolError(
            "E4 cần --model-selection từ E2/E3 validation; không được tự chọn"
        )
    if expected_manifest_hash is None or project_root is None:
        raise TrainingProtocolError("E4 selection guard cần manifest hash và project root")
    expected_hash = _require_sha256(expected_manifest_hash, "E4 manifest_sha256")
    root = Path(project_root).resolve()
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise TrainingProtocolError(f"model_selection.json không hợp lệ: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingProtocolError("model_selection.json phải là JSON object")
    required_contract = {
        "schema_version": 1,
        "selected_by": "validation",
        "selection_split": "validation",
        "metric": "val_loss",
        "mode": "min",
        "tie_break": "experiment_id_ascending",
        "valid_for_report": True,
        "status": "MODEL_SELECTED",
    }
    for field, expected in required_contract.items():
        if payload.get(field) != expected:
            raise TrainingProtocolError(
                f"model_selection.{field} phải là {expected!r}, nhận {payload.get(field)!r}"
            )
    if (
        _require_sha256(payload.get("manifest_sha256"), "model_selection manifest_sha256")
        != expected_hash
    ):
        raise TrainingProtocolError("model_selection không dùng cùng manifest với E4 hiện tại")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise TrainingProtocolError("model_selection phải chứa đúng hai candidates E2/E3")
    candidates_by_experiment: dict[str, Any] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TrainingProtocolError("model_selection candidate phải là object")
        experiment_id = str(candidate.get("experiment", "")).strip().upper()
        if experiment_id in candidates_by_experiment:
            raise TrainingProtocolError(f"model_selection lặp candidate {experiment_id}")
        candidates_by_experiment[experiment_id] = candidate
    if set(candidates_by_experiment) != {"E2", "E3"}:
        raise TrainingProtocolError("model_selection candidates phải chính xác là E2 và E3")
    verified = [
        _verified_selection_candidate(
            candidates_by_experiment[experiment],
            expected_experiment=experiment,
            expected_manifest_hash=expected_hash,
            project_root=root,
        )
        for experiment in ("E2", "E3")
    ]
    winner_evidence = min(verified, key=lambda item: (item["best_val_loss"], item["experiment"]))
    winner_fields = {
        "winner_experiment": winner_evidence["experiment"],
        "winner_run_id": winner_evidence["run_id"],
        "winner_config_sha256": winner_evidence["config_sha256"],
        "winner_checkpoint_sha256": winner_evidence["checkpoint_sha256"],
    }
    for field, expected in winner_fields.items():
        if str(payload.get(field, "")).strip() != str(expected):
            raise TrainingProtocolError(f"model_selection.{field} không khớp winner tính lại")
    try:
        recorded_winner_loss = float(payload["winner_best_val_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingProtocolError("model_selection thiếu winner_best_val_loss") from exc
    if not math.isclose(
        recorded_winner_loss,
        float(winner_evidence["best_val_loss"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise TrainingProtocolError("model_selection winner_best_val_loss không khớp")
    expected_config_portable = str(winner_evidence["config_path"].relative_to(root)).replace(
        "\\", "/"
    )
    expected_checkpoint_portable = str(
        winner_evidence["checkpoint_path"].relative_to(root)
    ).replace("\\", "/")
    if str(payload.get("winner_config", "")).replace("\\", "/") != expected_config_portable:
        raise TrainingProtocolError("model_selection winner_config không khớp candidate thắng")
    if str(payload.get("winner_checkpoint", "")).replace("\\", "/") != expected_checkpoint_portable:
        raise TrainingProtocolError("model_selection winner_checkpoint không khớp candidate thắng")

    winner = load_config(winner_evidence["config_path"]).values
    winner_fine_tune = winner.get("training", {}).get("fine_tune", {})
    boundary = winner_fine_tune.get("unfreeze_from")
    learning_rate = winner_fine_tune.get("learning_rate")
    if not boundary or learning_rate is None:
        raise TrainingProtocolError("Cấu hình thắng thiếu fine-tune boundary/learning rate")
    resolved = copy.deepcopy(values)
    resolved["training"]["fine_tune"].update(
        {
            "unfreeze_from": boundary,
            "learning_rate": float(learning_rate),
            "inherit_from_model_selection": False,
            "inherited_from": winner_evidence["experiment"],
            "selection_sha256": sha256_file(selection_path),
            "selection_manifest_sha256": expected_hash,
            "selection_winner_run_id": winner_evidence["run_id"],
            "selection_winner_checkpoint_sha256": winner_evidence["checkpoint_sha256"],
        }
    )
    return resolved


def _phase_log_path(run_dir: Path, phase: str) -> Path:
    return run_dir / f"{phase}_keras_log.csv"


def _phase_backup_dir(run_dir: Path, phase: str) -> Path:
    return run_dir / ".training_backup" / phase


def _resume_callback_state(history: pd.DataFrame) -> dict[str, int | float]:
    if history.empty:
        return {"best": float("inf"), "best_epoch": 0, "early_wait": 0, "reduce_wait": 0}
    losses = history["val_loss"].to_numpy(dtype=np.float64)
    best_position = int(np.argmin(losses))
    early_wait = len(history) - best_position - 1
    reduce_wait = early_wait
    if "learning_rate" in history.columns:
        rates = pd.to_numeric(history["learning_rate"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(rates).all() and len(rates) > 1:
            changes = np.flatnonzero(~np.isclose(rates[1:], rates[:-1], rtol=1e-12, atol=0.0))
            if len(changes):
                reduce_wait = len(rates) - int(changes[-1] + 1) - 1
    return {
        "best": float(losses[best_position]),
        "best_epoch": int(history.iloc[best_position]["epoch"]),
        "early_wait": int(early_wait),
        "reduce_wait": int(reduce_wait),
    }


def _callbacks(
    tf: Any,
    run_dir: Path,
    phase: str,
    config: dict[str, Any],
    *,
    resume: bool = False,
    prior_best: float | None = None,
) -> tuple[list[Any], Path]:
    callback_config = config.get("training", {}).get("callbacks", {})
    checkpoint = run_dir / f"{phase}_best.keras"
    early = callback_config.get("early_stopping", {})
    reduce = callback_config.get("reduce_lr_on_plateau", {})
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    if prior_best is not None:
        checkpoint_callback.best = float(prior_best)
    callbacks: list[Any] = [
        checkpoint_callback,
        tf.keras.callbacks.CSVLogger(
            _phase_log_path(run_dir, phase),
            append=bool(resume or _phase_log_path(run_dir, phase).exists()),
        ),
    ]
    early_callback = None
    if early.get("enabled", True):
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(early.get("patience", 5)),
            restore_best_weights=bool(early.get("restore_best_weights", True)),
            verbose=1,
        )
        callbacks.append(early_callback)
    reduce_callback = None
    if reduce.get("enabled", True):
        reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=float(reduce.get("factor", 0.2)),
            patience=int(reduce.get("patience", 2)),
            min_lr=float(reduce.get("min_learning_rate", 1e-7)),
            verbose=1,
        )
        callbacks.append(reduce_callback)

    if resume and prior_best is not None:
        history = _read_phase_log(run_dir, phase)
        backup_dir = _phase_backup_dir(run_dir, phase)
        if not checkpoint.is_file() or not backup_dir.is_dir():
            raise TrainingProtocolError(
                f"Phase {phase} có history nhưng thiếu best checkpoint/BackupAndRestore state"
            )
        state = _resume_callback_state(history)
        best_weights = tf.keras.models.load_model(checkpoint, compile=False).get_weights()

        class _ResumeCallbackState(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs: Any = None) -> None:
                del logs
                if early_callback is not None:
                    early_callback.best = state["best"]
                    early_callback.best_epoch = state["best_epoch"]
                    early_callback.wait = state["early_wait"]
                    early_callback.best_weights = best_weights
                if reduce_callback is not None:
                    reduce_callback.best = state["best"]
                    reduce_callback.wait = state["reduce_wait"]
                    reduce_callback.cooldown_counter = 0

        # EarlyStopping/ReduceLROnPlateau reset themselves on_train_begin, so
        # restore their journal-derived state afterwards and before epoch 1.
        callbacks.append(_ResumeCallbackState())
    callbacks.append(
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=_phase_backup_dir(run_dir, phase),
            save_freq="epoch",
            double_checkpoint=True,
            delete_checkpoint=False,
        )
    )
    return callbacks, checkpoint


def _read_phase_log(run_dir: Path, phase: str, *, required: bool = True) -> pd.DataFrame:
    path = _phase_log_path(run_dir, phase)
    if not path.is_file():
        if required:
            raise TrainingProtocolError(f"Thiếu CSV history của phase {phase}: {path}")
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise TrainingProtocolError(f"CSV history phase {phase} không hợp lệ: {exc}") from exc
    if frame.empty:
        if required:
            raise TrainingProtocolError(f"CSV history phase {phase} rỗng")
        return frame
    if "epoch" not in frame.columns or "val_loss" not in frame.columns:
        raise TrainingProtocolError(f"CSV history phase {phase} thiếu epoch/val_loss")
    try:
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="raise").astype(int)
        frame["val_loss"] = pd.to_numeric(frame["val_loss"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise TrainingProtocolError(f"CSV history phase {phase} có giá trị không hợp lệ") from exc
    if (frame["epoch"] < 0).any() or not np.isfinite(frame["val_loss"]).all():
        raise TrainingProtocolError(f"CSV history phase {phase} có epoch/val_loss không hợp lệ")
    duplicate_epochs = frame.loc[frame["epoch"].duplicated(keep=False)]
    for epoch, rows in duplicate_epochs.groupby("epoch", sort=False):
        values = rows.drop(columns=["epoch"]).reset_index(drop=True)
        if any(values[column].nunique(dropna=False) > 1 for column in values.columns):
            raise TrainingProtocolError(
                f"CSV history phase {phase} có duplicate epoch {epoch} xung đột"
            )
    return (
        frame.drop_duplicates(subset=["epoch"], keep="last")
        .sort_values("epoch", kind="stable")
        .reset_index(drop=True)
    )


def _phase_completion_path(run_dir: Path, phase: str) -> Path:
    return run_dir / f"{phase}_complete.json"


def _load_phase_completion(run_dir: Path, phase: str) -> tuple[dict[str, Any], Path] | None:
    marker = _phase_completion_path(run_dir, phase)
    if not marker.exists():
        return None
    payload = _load_json_object(marker, f"phase completion marker {phase}")
    if payload.get("phase") != phase:
        raise TrainingProtocolError(f"Phase marker không khớp phase {phase}")
    checkpoint_name = payload.get("checkpoint")
    if checkpoint_name != f"{phase}_best.keras":
        raise TrainingProtocolError(f"Phase marker {phase} tham chiếu checkpoint không hợp lệ")
    checkpoint = run_dir / checkpoint_name
    if not checkpoint.is_file():
        raise TrainingProtocolError(f"Checkpoint phase {phase} bị thiếu: {checkpoint}")
    if payload.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise TrainingProtocolError(f"Checkpoint hash phase {phase} không khớp marker")
    log_path = _phase_log_path(run_dir, phase)
    if not log_path.is_file() or payload.get("history_log_sha256") != sha256_file(log_path):
        raise TrainingProtocolError(f"CSV history hash phase {phase} không khớp marker")
    history = _read_phase_log(run_dir, phase)
    observed_best = float(history["val_loss"].min())
    try:
        marker_best = float(payload["best_val_loss"])
        completed_epochs = int(payload["completed_epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingProtocolError(f"Phase marker {phase} thiếu metric/epoch hợp lệ") from exc
    if not np.isclose(marker_best, observed_best, rtol=1e-12, atol=1e-12):
        raise TrainingProtocolError(f"best_val_loss phase {phase} không khớp CSV history")
    if completed_epochs != len(history):
        raise TrainingProtocolError(f"completed_epochs phase {phase} không khớp CSV history")
    return payload, checkpoint


def _write_phase_completion(
    run_dir: Path,
    phase: str,
    checkpoint: Path,
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    history = _read_phase_log(run_dir, phase)
    log_path = _phase_log_path(run_dir, phase)
    if not checkpoint.is_file():
        raise TrainingProtocolError(f"Phase {phase} hoàn tất nhưng thiếu best checkpoint")
    payload = {
        "schema_version": 1,
        "phase": phase,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "history_log": log_path.name,
        "history_log_sha256": sha256_file(log_path),
        "completed_epochs": len(history),
        "best_val_loss": float(history["val_loss"].min()),
        "completed_fit_seconds": float(duration_seconds),
    }
    write_json(_phase_completion_path(run_dir, phase), payload, overwrite=False)
    return payload


def _cleanup_phase_backup(run_dir: Path, phase: str) -> None:
    backup_root = (run_dir / ".training_backup").resolve()
    target = _phase_backup_dir(run_dir, phase).resolve()
    try:
        target.relative_to(backup_root)
    except ValueError as exc:  # pragma: no cover - internal path invariant.
        raise TrainingProtocolError("Backup phase nằm ngoài run directory") from exc
    if target.is_dir():
        shutil.rmtree(target)


def _combined_history(run_dir: Path, phases: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for phase in phases:
        phase_frame = _read_phase_log(run_dir, phase).copy()
        phase_frame.insert(0, "phase_epoch", phase_frame.pop("epoch") + 1)
        phase_frame.insert(0, "phase", phase)
        frames.append(phase_frame)
    if not frames:
        raise TrainingProtocolError("Không có phase history để tổng hợp")
    history = pd.concat(frames, ignore_index=True, sort=False)
    history.insert(1, "global_epoch", np.arange(1, len(history) + 1))
    return history


def _write_frame_once_or_verify(path: Path, frame: pd.DataFrame) -> None:
    content = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise TrainingProtocolError(f"Từ chối ghi đè evidence CSV đã tồn tại: {path}")
        return
    path.write_text(content, encoding="utf-8")


def _write_json_once_or_verify(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _load_json_object(path, path.name)
        if existing != payload:
            raise TrainingProtocolError(f"Từ chối ghi đè evidence JSON đã tồn tại: {path}")
        return
    write_json(path, payload, overwrite=False)


def _predict(model: Any, dataset: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float64).reshape(-1)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise RuntimeError("Model trả về xác suất không hợp lệ")
    return probabilities


def run_training(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path = "artifacts/runs",
    run_id: str | None = None,
    model_selection: str | Path | None = None,
    weights_none: bool = False,
    head_epochs: int | None = None,
    fine_tune_epochs: int | None = None,
    smoke: bool = False,
    resume: bool = False,
) -> Path:
    """Train E1-E4 without reading the locked test labels for model selection."""

    loaded = load_config(config_path)
    values = copy.deepcopy(loaded.values)
    experiment = values["experiment"].get("slug") or values["experiment"].get("id")
    if str(values["experiment"].get("id", "")).upper() == "E5" or not values.get(
        "training", {}
    ).get("head", {}).get("enabled", True):
        raise TrainingProtocolError("E5 không huấn luyện model mới")
    if weights_none and not smoke:
        raise TrainingProtocolError("--weights-none chỉ được dùng cho smoke NOT_VALID_FOR_REPORT")
    if weights_none:
        values.setdefault("model", {})["weights"] = None
    if head_epochs is not None:
        if head_epochs <= 0:
            raise ValueError("head_epochs phải dương")
        values["training"]["head"]["max_epochs"] = int(head_epochs)
    if fine_tune_epochs is not None:
        if fine_tune_epochs < 0:
            raise ValueError("fine_tune_epochs không được âm")
        values["training"]["fine_tune"]["max_epochs"] = int(fine_tune_epochs)
    values.setdefault("run", {})["smoke"] = bool(smoke)
    if not smoke and str(values.get("model", {}).get("weights", "")).strip().lower() != "imagenet":
        raise TrainingProtocolError(
            "Full experiment bắt buộc khởi tạo MobileNetV2 weights=imagenet"
        )

    manifest_file = Path(manifest_path).resolve()
    manifest = _load_manifest(manifest_file)
    audit = audit_split(manifest, enforce_official_balance=not smoke)
    if not audit.get("valid"):
        raise TrainingProtocolError(f"Split audit không hợp lệ: {audit.get('errors')}")
    manifest_hash = manifest_sha256(manifest)
    if smoke:
        preconditions = _smoke_preconditions()
        project_root = Path.cwd().resolve()
    else:
        git_state = verify_official_git_state(loaded.path)
        project_root = Path(git_state.repository_root)
        official_dataset = verify_official_dataset_preconditions(
            manifest,
            manifest_file,
            dataset_root,
        )
        preconditions = _official_preconditions(
            git=git_state.to_dict(),
            official_dataset=official_dataset.to_dict(),
        )

    values = _resolve_e4_selection(
        values,
        Path(model_selection).resolve() if model_selection else None,
        expected_manifest_hash=manifest_hash,
        project_root=project_root,
    )
    config_hash = sha256_json(values)
    experiment = values["experiment"].get("slug") or values["experiment"].get("id")
    run_dir, resolved_run_id = _prepare_run_directory(
        output_root,
        run_id,
        experiment=str(experiment),
        config_hash=config_hash,
        resume=resume,
    )
    if resume:
        _verify_resume_contract(
            run_dir,
            values=values,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            run_id=resolved_run_id,
            official_preconditions=preconditions,
        )
    else:
        _write_initial_contract(
            run_dir,
            values=values,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            manifest_path=manifest_file,
            run_id=resolved_run_id,
            official_preconditions=preconditions,
        )

    precondition_path = run_dir / "official_preconditions.json"
    if resume and not precondition_path.is_file():
        raise TrainingProtocolError("Run resume thiếu official_preconditions.json")
    _write_json_once_or_verify(precondition_path, preconditions)
    seed = int(values.get("experiment", {}).get("seed", values.get("seed", 42)))
    determinism = set_global_determinism(seed, enable_ops=True)
    if resume:
        if not (run_dir / "determinism.json").is_file():
            raise TrainingProtocolError("Run resume thiếu determinism.json")
    else:
        environment = write_environment(run_dir / "environment.json", project_path=project_root)
        pip_freeze = environment.get("pip_freeze")
        if isinstance(pip_freeze, str):
            (run_dir / "pip_freeze.txt").write_text(pip_freeze + "\n", encoding="utf-8")
        write_json(run_dir / "determinism.json", determinism, overwrite=False)
    _write_json_once_or_verify(run_dir / "split_audit.json", audit)

    train_frame = manifest.loc[manifest["split"].astype(str).str.lower() == "train"].copy()
    val_frame = manifest.loc[
        manifest["split"].astype(str).str.lower().isin({"val", "validation"})
    ].copy()
    if train_frame.empty or val_frame.empty:
        raise TrainingProtocolError("Manifest phải có train và validation")

    pipeline = values.get("pipeline", {})
    batch_size = int(pipeline.get("batch_size", 32))
    image_size = tuple(pipeline.get("image_size", (224, 224)))
    augmentation_config = pipeline.get("augmentation", {})
    augmentation = bool(augmentation_config.get("enabled", False))
    train_dataset = build_tf_dataset(
        train_frame,
        batch_size=batch_size,
        image_size=image_size,
        training=True,
        seed=seed,
        augment=augmentation,
        dataset_root=dataset_root,
        rotation_degrees=float(augmentation_config.get("max_rotation_degrees", 15.0)),
        brightness_delta=float(augmentation_config.get("brightness_delta", 0.15)),
        contrast_delta=float(augmentation_config.get("contrast_delta", 0.15)),
    )
    val_dataset = build_tf_dataset(
        val_frame,
        batch_size=batch_size,
        image_size=image_size,
        training=False,
        seed=seed,
        augment=False,
        dataset_root=dataset_root,
    )
    class_weights = compute_balanced_class_weights(train_frame)
    _write_json_once_or_verify(
        run_dir / "class_weights.json",
        {str(key): value for key, value in class_weights.items()},
    )

    tf = _tensorflow()
    phase_candidates: list[tuple[str, float, Path]] = []
    phase_durations: dict[str, float] = {}
    completed_phases: list[str] = []
    training_started = time.perf_counter()

    head = values["training"]["head"]
    head_completion = _load_phase_completion(run_dir, "head")
    if head_completion is None:
        model = build_from_config(values)
        head_trainability = configure_fine_tuning(model, None)
        _write_json_once_or_verify(run_dir / "trainability_head.json", head_trainability)
        compile_binary_model(model, float(head.get("learning_rate", 1e-3)))
        prior_head_log = _read_phase_log(run_dir, "head", required=False)
        prior_head_best = (
            float(prior_head_log["val_loss"].min()) if not prior_head_log.empty else None
        )
        head_callbacks, head_checkpoint = _callbacks(
            tf,
            run_dir,
            "head",
            values,
            resume=resume,
            prior_best=prior_head_best,
        )
        phase_started = time.perf_counter()
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=int(head.get("max_epochs", 20)),
            callbacks=head_callbacks,
            class_weight=class_weights,
            shuffle=False,
            verbose=2,
        )
        head_marker = _write_phase_completion(
            run_dir,
            "head",
            head_checkpoint,
            duration_seconds=time.perf_counter() - phase_started,
        )
        _cleanup_phase_backup(run_dir, "head")
    else:
        head_marker, head_checkpoint = head_completion
    phase_durations["head_seconds"] = float(head_marker["completed_fit_seconds"])
    phase_candidates.append(("head", float(head_marker["best_val_loss"]), head_checkpoint))
    completed_phases.append("head")

    fine_tune = values["training"].get("fine_tune", {})
    if fine_tune.get("enabled", False) and int(fine_tune.get("max_epochs", 0)) > 0:
        boundary = fine_tune.get("unfreeze_from")
        if not boundary:
            raise TrainingProtocolError("Fine-tune run chưa resolve unfreeze_from")
        fine_tune_completion = _load_phase_completion(run_dir, "fine_tune")
        if fine_tune_completion is None:
            # The phase handoff always uses the immutable best head checkpoint,
            # making fresh and resumed runs enter fine-tuning from the same state.
            model = tf.keras.models.load_model(head_checkpoint, compile=False)
            summary = configure_fine_tuning(model, str(boundary))
            _write_json_once_or_verify(run_dir / "trainability.json", summary)
            compile_binary_model(model, float(fine_tune["learning_rate"]))
            prior_ft_log = _read_phase_log(run_dir, "fine_tune", required=False)
            prior_ft_best = (
                float(prior_ft_log["val_loss"].min()) if not prior_ft_log.empty else None
            )
            ft_callbacks, ft_checkpoint = _callbacks(
                tf,
                run_dir,
                "fine_tune",
                values,
                resume=resume,
                prior_best=prior_ft_best,
            )
            phase_started = time.perf_counter()
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=int(fine_tune.get("max_epochs", 30)),
                callbacks=ft_callbacks,
                class_weight=class_weights,
                shuffle=False,
                verbose=2,
            )
            ft_marker = _write_phase_completion(
                run_dir,
                "fine_tune",
                ft_checkpoint,
                duration_seconds=time.perf_counter() - phase_started,
            )
            _cleanup_phase_backup(run_dir, "fine_tune")
        else:
            ft_marker, ft_checkpoint = fine_tune_completion
        phase_durations["fine_tune_seconds"] = float(ft_marker["completed_fit_seconds"])
        phase_candidates.append(("fine_tune", float(ft_marker["best_val_loss"]), ft_checkpoint))
        completed_phases.append("fine_tune")

    history_frame = _combined_history(run_dir, completed_phases)
    _write_frame_once_or_verify(run_dir / "history.csv", history_frame)
    training_curves_path = run_dir / "training_curves.png"
    if not training_curves_path.exists():
        plot_training_history(
            {
                "loss": history_frame["loss"].tolist(),
                "val_loss": history_frame["val_loss"].tolist(),
                "accuracy": history_frame.get("accuracy", pd.Series(dtype=float)).tolist(),
                "val_accuracy": history_frame.get("val_accuracy", pd.Series(dtype=float)).tolist(),
            },
            training_curves_path,
        )

    selected_phase, best_val_loss, selected_checkpoint = min(
        phase_candidates, key=lambda item: (item[1], item[0])
    )
    finalization_conflicts = [
        run_dir / name for name in FINALIZATION_ARTIFACTS if (run_dir / name).exists()
    ]
    if finalization_conflicts:
        raise TrainingProtocolError(
            "Run có final evidence dở dang; từ chối ghi đè, cần audit thủ công: "
            + ", ".join(path.name for path in finalization_conflicts)
        )
    final_model_path = run_dir / "model.keras"
    with selected_checkpoint.open("rb") as source, final_model_path.open("xb") as target:
        shutil.copyfileobj(source, target)
    selected_model = tf.keras.models.load_model(final_model_path, compile=False)
    probabilities = _predict(selected_model, val_dataset)
    if len(probabilities) != len(val_frame):
        raise RuntimeError("Số prediction validation không khớp manifest")
    threshold = float(values.get("evaluation", {}).get("test_threshold", 0.5))
    metrics = compute_binary_metrics(val_frame["label"], probabilities, threshold)
    write_json(run_dir / "metrics_validation.json", metrics, overwrite=False)
    write_json(
        run_dir / "classification_report_validation.json",
        metrics["classification_report"],
        overwrite=False,
    )
    prediction_frame = val_frame.loc[
        :, ["relative_path", "label", "surface", "source_group", "sha256"]
    ].copy()
    prediction_frame = prediction_frame.rename(columns={"label": "y_true"})
    prediction_frame["split"] = "validation"
    prediction_frame["probability_crack"] = probabilities
    prediction_frame["y_pred"] = (probabilities >= threshold).astype(int)
    prediction_frame["outcome"] = np.select(
        [
            (prediction_frame["y_true"] == 1) & (prediction_frame["y_pred"] == 1),
            (prediction_frame["y_true"] == 0) & (prediction_frame["y_pred"] == 0),
            (prediction_frame["y_true"] == 0) & (prediction_frame["y_pred"] == 1),
        ],
        ["TP", "TN", "FP"],
        default="FN",
    )
    prediction_frame.to_csv(
        run_dir / "predictions_validation.csv",
        index=False,
        lineterminator="\n",
        mode="x",
    )
    prediction_frame.loc[prediction_frame["outcome"].isin({"FP", "FN"})].to_csv(
        run_dir / "errors_validation.csv",
        index=False,
        lineterminator="\n",
        mode="x",
    )
    plot_confusion_matrix(metrics["confusion_matrix"], run_dir / "confusion_matrix_validation.png")
    plot_confusion_matrix(
        metrics["confusion_matrix_normalized"],
        run_dir / "confusion_matrix_validation_normalized.png",
        normalized=True,
    )

    model_hash = sha256_file(final_model_path)
    current_session_seconds = time.perf_counter() - training_started
    training_duration = float(sum(phase_durations.values()))
    final_trainability = {
        "selected_checkpoint_phase": selected_phase,
        "trainable_parameters_at_end": int(
            sum(int(np.prod(tuple(value.shape))) for value in selected_model.trainable_weights)
        ),
        "total_parameters": int(selected_model.count_params()),
    }
    write_json(run_dir / "trainability_final.json", final_trainability, overwrite=False)
    duration_payload = {
        **phase_durations,
        "known_completed_fit_seconds": training_duration,
        "current_session_training_and_validation_artifact_seconds": float(current_session_seconds),
        "resumed": bool(resume),
    }
    write_json(run_dir / "training_duration.json", duration_payload, overwrite=False)
    metadata = {
        "run_id": resolved_run_id,
        "model_version": __version__,
        "threshold": threshold,
        "input_size": list(image_size),
        "preprocessing": PREPROCESSING_NAME,
        "label_mapping": LABEL_MAPPING,
        "gradcam_layer": values.get("model", {}).get("gradcam_layer", "out_relu"),
        "tensorflow_version": tf.__version__,
        "model_sha256": model_hash,
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "smoke_test": bool(smoke),
        "resumed": bool(resume),
        "status": "NOT_VALID_FOR_REPORT" if smoke else "VALIDATION_COMPLETE_TEST_LOCKED",
        **_precondition_evidence(preconditions),
    }
    write_json(run_dir / "model.metadata.json", metadata, overwrite=False)
    summary = {
        "run_id": resolved_run_id,
        "experiment": experiment,
        "selected_phase": selected_phase,
        "best_val_loss": float(best_val_loss),
        "validation_accuracy": metrics["accuracy"],
        "validation_f1_crack": metrics["crack"]["f1"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "model_sha256": model_hash,
        "model_path": str(final_model_path.resolve()),
        "trainable_parameters": final_trainability["trainable_parameters_at_end"],
        "total_parameters": final_trainability["total_parameters"],
        "training_duration_seconds": float(training_duration),
        "resumed": bool(resume),
        "valid_for_report": not smoke,
        "status": "NOT_VALID_FOR_REPORT" if smoke else "VALIDATION_COMPLETE_TEST_LOCKED",
        **_precondition_evidence(preconditions),
    }
    write_json(run_dir / "run_summary.json", summary, overwrite=False)
    completion_artifacts = [
        "config_snapshot.json",
        TRAINING_CONTRACT_FILENAME,
        "official_preconditions.json",
        *[f"{phase}_complete.json" for phase in completed_phases],
        "history.csv",
        "model.keras",
        "metrics_validation.json",
        "predictions_validation.csv",
        "model.metadata.json",
        "run_summary.json",
    ]
    completion = {
        "schema_version": 2,
        "status": summary["status"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "run_id": resolved_run_id,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "model_sha256": model_hash,
        "completed_phases": completed_phases,
        "artifact_sha256": {name: sha256_file(run_dir / name) for name in completion_artifacts},
        "immutable": True,
        **_precondition_evidence(preconditions),
    }
    # This marker is deliberately the final write. A run with this marker or
    # run_summary.json can never be resumed or overwritten.
    write_json(run_dir / RUN_COMPLETION_FILENAME, completion, overwrite=False)
    return run_dir


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="Train CrackSpot E1-E4 using validation only")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--run-id")
    parser.add_argument("--model-selection")
    parser.add_argument("--weights-none", action="store_true")
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--fine-tune-epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục đúng --run-id đã tồn tại sau khi xác minh config/manifest hash",
    )
    args = parser.parse_args()
    path = run_training(
        config_path=args.config,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        run_id=args.run_id,
        model_selection=args.model_selection,
        weights_none=args.weights_none,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(path)


if __name__ == "__main__":
    main()
