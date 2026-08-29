from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import crackspot.modeling.train as training
from crackspot.utils.hashing import sha256_json


def test_resume_requires_explicit_existing_run_id(tmp_path: Path) -> None:
    with pytest.raises(training.TrainingProtocolError, match="--run-id"):
        training._prepare_run_directory(
            tmp_path,
            None,
            experiment="e1",
            config_hash="a" * 64,
            resume=True,
        )

    with pytest.raises(training.TrainingProtocolError, match="đã tồn tại"):
        training._prepare_run_directory(
            tmp_path,
            "missing-run",
            experiment="e1",
            config_hash="a" * 64,
            resume=True,
        )


def test_resume_refuses_completed_run_without_writing(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    summary = run_dir / "run_summary.json"
    summary.write_text('{"status":"VALIDATION_COMPLETE_TEST_LOCKED"}', encoding="utf-8")
    before = summary.read_bytes()

    with pytest.raises(training.TrainingProtocolError, match="đã hoàn tất"):
        training._prepare_run_directory(
            tmp_path,
            "completed-run",
            experiment="e1",
            config_hash="a" * 64,
            resume=True,
        )

    assert summary.read_bytes() == before
    assert {path.name for path in run_dir.iterdir()} == {"run_summary.json"}


def test_resume_contract_requires_exact_config_values_file_hash_and_manifest(
    tmp_path: Path,
) -> None:
    values = {"experiment": {"id": "E1"}, "run": {"smoke": False}}
    config_hash = sha256_json(values)
    manifest_hash = "b" * 64
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("relative_path\nimage.jpg\n", encoding="utf-8")
    training._write_initial_contract(
        tmp_path,
        values=values,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        manifest_path=manifest_path,
        run_id="run-1",
    )

    verified = training._verify_resume_contract(
        tmp_path,
        values=values,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        run_id="run-1",
    )
    assert verified["manifest_sha256"] == manifest_hash

    changed_values = {**values, "run": {"smoke": True}}
    with pytest.raises(training.TrainingProtocolError, match="Giá trị config"):
        training._verify_resume_contract(
            tmp_path,
            values=changed_values,
            config_hash=sha256_json(changed_values),
            manifest_hash=manifest_hash,
            run_id="run-1",
        )
    with pytest.raises(training.TrainingProtocolError, match="Canonical manifest hash"):
        training._verify_resume_contract(
            tmp_path,
            values=values,
            config_hash=config_hash,
            manifest_hash="c" * 64,
            run_id="run-1",
        )

    snapshot = tmp_path / "config_snapshot.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    snapshot.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(training.TrainingProtocolError, match="File hash"):
        training._verify_resume_contract(
            tmp_path,
            values=values,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            run_id="run-1",
        )


def _write_phase_log(path: Path, rows: list[tuple[int, float, float]]) -> None:
    pd.DataFrame(rows, columns=["epoch", "loss", "val_loss"]).to_csv(
        path, index=False, lineterminator="\n"
    )


def test_phase_marker_locks_checkpoint_and_history_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "head_best.keras"
    checkpoint.write_bytes(b"checkpoint-v1")
    _write_phase_log(
        tmp_path / "head_keras_log.csv",
        [(0, 0.8, 0.7), (1, 0.6, 0.5)],
    )

    marker = training._write_phase_completion(
        tmp_path,
        "head",
        checkpoint,
        duration_seconds=1.25,
    )
    loaded = training._load_phase_completion(tmp_path, "head")
    assert loaded is not None
    assert loaded[0] == marker
    assert loaded[1] == checkpoint
    assert marker["completed_epochs"] == 2
    assert marker["best_val_loss"] == pytest.approx(0.5)

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(training.TrainingProtocolError, match="Checkpoint hash"):
        training._load_phase_completion(tmp_path, "head")


def test_combined_history_deduplicates_resume_log_and_keeps_global_epochs(
    tmp_path: Path,
) -> None:
    _write_phase_log(
        tmp_path / "head_keras_log.csv",
        [(0, 0.9, 0.8), (1, 0.65, 0.55), (1, 0.65, 0.55)],
    )
    _write_phase_log(
        tmp_path / "fine_tune_keras_log.csv",
        [(0, 0.5, 0.45), (1, 0.4, 0.35)],
    )

    history = training._combined_history(tmp_path, ["head", "fine_tune"])

    assert history["phase"].tolist() == ["head", "head", "fine_tune", "fine_tune"]
    assert history["phase_epoch"].tolist() == [1, 2, 1, 2]
    assert history["global_epoch"].tolist() == [1, 2, 3, 4]
    assert history.loc[1, "val_loss"] == pytest.approx(0.55)


def test_conflicting_duplicate_epoch_is_rejected(tmp_path: Path) -> None:
    _write_phase_log(
        tmp_path / "head_keras_log.csv",
        [(0, 0.9, 0.8), (1, 0.7, 0.6), (1, 0.65, 0.55)],
    )

    with pytest.raises(training.TrainingProtocolError, match="duplicate epoch 1 xung đột"):
        training._read_phase_log(tmp_path, "head")


class _FakeCallback:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeModelCheckpoint(_FakeCallback):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.best = float("inf")


def test_callbacks_use_append_log_and_phase_scoped_backup(tmp_path: Path) -> None:
    callbacks = SimpleNamespace(
        Callback=_FakeCallback,
        ModelCheckpoint=_FakeModelCheckpoint,
        CSVLogger=_FakeCallback,
        EarlyStopping=_FakeCallback,
        ReduceLROnPlateau=_FakeCallback,
        BackupAndRestore=_FakeCallback,
    )
    fake_tf = SimpleNamespace(keras=SimpleNamespace(callbacks=callbacks))
    config = {
        "training": {
            "callbacks": {
                "early_stopping": {"enabled": False},
                "reduce_lr_on_plateau": {"enabled": False},
            }
        }
    }

    result, checkpoint = training._callbacks(
        fake_tf,
        tmp_path,
        "head",
        config,
        resume=True,
    )

    assert checkpoint == tmp_path / "head_best.keras"
    assert result[1].kwargs["append"] is True
    backup = result[-1]
    assert backup.kwargs["backup_dir"] == tmp_path / ".training_backup" / "head"
    assert backup.kwargs["save_freq"] == "epoch"
    assert backup.kwargs["double_checkpoint"] is True
    assert backup.kwargs["delete_checkpoint"] is False


def test_resume_callback_state_uses_real_epoch_journal() -> None:
    history = pd.DataFrame(
        {
            "epoch": [2, 3, 4, 5],
            "val_loss": [0.5, 0.4, 0.42, 0.43],
            "learning_rate": [1e-3, 1e-3, 2e-4, 2e-4],
        }
    )

    state = training._resume_callback_state(history)

    assert state["best"] == pytest.approx(0.4)
    assert state["best_epoch"] == 3
    assert state["early_wait"] == 2
    assert state["reduce_wait"] == 1
