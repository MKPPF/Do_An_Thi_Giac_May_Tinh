from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from crackspot.utils import environment


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=stdout, stderr="")


def test_official_git_state_requires_commit_and_ignores_untracked_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "a" * 40
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        if command[1:] == ["rev-parse", "--show-toplevel"]:
            return _completed(str(tmp_path))
        if command[1:] == ["rev-parse", "--verify", "HEAD^{commit}"]:
            return _completed(head)
        if command[1:] == ["status", "--porcelain=v1", "--untracked-files=no"]:
            return _completed("")
        if command[1:] == ["symbolic-ref", "--quiet", "--short", "HEAD"]:
            return _completed("main")
        raise AssertionError(command)

    monkeypatch.setattr(environment.subprocess, "run", fake_run)

    result = environment.verify_official_git_state(tmp_path)

    assert result.head_commit == head
    assert result.branch == "main"
    assert result.tracked_worktree_clean is True
    assert result.tracked_status_porcelain == ""
    assert result.untracked_files_ignored_for_gate is True
    assert ("git", "status", "--porcelain=v1", "--untracked-files=no") in calls
    assert all("--untracked-files=all" not in call for call in calls)


def test_official_git_state_rejects_tracked_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        if command[1:] == ["rev-parse", "--show-toplevel"]:
            return _completed(str(tmp_path))
        if command[1:] == ["rev-parse", "--verify", "HEAD^{commit}"]:
            return _completed("b" * 40)
        if command[1:] == ["status", "--porcelain=v1", "--untracked-files=no"]:
            return _completed(" M src/crackspot/modeling/train.py")
        raise AssertionError(command)

    monkeypatch.setattr(environment.subprocess, "run", fake_run)

    with pytest.raises(environment.GitStateError, match="clean tracked worktree"):
        environment.verify_official_git_state(tmp_path)


def test_official_git_state_rejects_missing_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        if command[1:] == ["rev-parse", "--show-toplevel"]:
            return _completed(str(tmp_path))
        if command[1:] == ["rev-parse", "--verify", "HEAD^{commit}"]:
            raise subprocess.CalledProcessError(128, command, stderr="ambiguous argument HEAD")
        raise AssertionError(command)

    monkeypatch.setattr(environment.subprocess, "run", fake_run)

    with pytest.raises(environment.GitStateError, match="valid Git HEAD"):
        environment.verify_official_git_state(tmp_path)
