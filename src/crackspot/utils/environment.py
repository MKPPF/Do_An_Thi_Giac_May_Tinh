"""Capture software, hardware, and Git state for each experiment."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class GitStateError(RuntimeError):
    """Raised when an official experiment has no clean, committed code state."""


@dataclass(frozen=True, slots=True)
class OfficialGitState:
    """Git provenance required before a report-valid training run."""

    repository_root: str
    head_commit: str
    branch: str | None
    tracked_status_porcelain: str
    tracked_worktree_clean: bool
    untracked_files_ignored_for_gate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(command: list[str], *, cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=cwd,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _run_git_required(arguments: list[str], *, cwd: Path, description: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise GitStateError("Git is required for an official experiment") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {details}" if details else ""
        raise GitStateError(f"cannot {description}{suffix}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitStateError(f"cannot {description}: {exc}") from exc
    return result.stdout.strip()


def verify_official_git_state(project_path: str | Path) -> OfficialGitState:
    """Require a valid HEAD commit and no staged/unstaged tracked changes.

    Untracked files do not alter already committed experiment code, so they are
    deliberately omitted from this gate.  Generated data/artifacts remain
    governed by their own hashes and immutable bundle contracts.
    """

    candidate = Path(project_path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise GitStateError(f"project path does not exist: {candidate}")
    repository_text = _run_git_required(
        ["rev-parse", "--show-toplevel"],
        cwd=candidate,
        description="locate the Git repository",
    )
    repository = Path(repository_text).resolve()
    head = _run_git_required(
        ["rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repository,
        description="resolve a valid Git HEAD commit",
    ).lower()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head):
        raise GitStateError(f"Git HEAD is not a valid commit object ID: {head!r}")
    status = _run_git_required(
        ["status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repository,
        description="inspect tracked Git status",
    )
    if status:
        raise GitStateError(
            "official experiment requires a clean tracked worktree; staged/unstaged "
            f"changes:\n{status}"
        )
    branch = _run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repository)
    return OfficialGitState(
        repository_root=str(repository),
        head_commit=head,
        branch=branch or None,
        tracked_status_porcelain=status,
        tracked_worktree_clean=True,
        untracked_files_ignored_for_gate=True,
    )


def capture_environment(project_path: str | Path | None = None) -> dict[str, Any]:
    git_cwd = Path(project_path).expanduser().resolve() if project_path is not None else None
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=git_cwd),
        "git_status": _run(["git", "status", "--short"], cwd=git_cwd),
        "git_tracked_status": _run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=git_cwd
        ),
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        payload["memory_total_bytes"] = int(memory.total)
        payload["memory_available_bytes_at_capture"] = int(memory.available)
        payload["physical_cpu_count"] = psutil.cpu_count(logical=False)
    except (ImportError, OSError, RuntimeError) as exc:
        payload["psutil_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import tensorflow as tf

        payload["tensorflow"] = tf.__version__
        devices = tf.config.list_physical_devices()
        payload["tensorflow_devices"] = [
            {"name": device.name, "device_type": device.device_type} for device in devices
        ]
        payload["tensorflow_build_info"] = dict(tf.sysconfig.get_build_info())
        gpu_details: list[dict[str, Any]] = []
        for device in tf.config.list_physical_devices("GPU"):
            try:
                details = dict(tf.config.experimental.get_device_details(device))
            except (RuntimeError, ValueError):
                details = {}
            gpu_details.append({"name": device.name, **details})
        payload["gpu_details"] = gpu_details
    except (ImportError, RuntimeError, ValueError) as exc:
        payload["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def write_environment(
    path: str | Path, *, project_path: str | Path | None = None
) -> dict[str, Any]:
    payload = capture_environment(project_path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "GitStateError",
    "OfficialGitState",
    "capture_environment",
    "verify_official_git_state",
    "write_environment",
]
