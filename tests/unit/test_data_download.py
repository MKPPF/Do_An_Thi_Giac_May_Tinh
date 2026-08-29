from __future__ import annotations

import importlib.util
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any

import pytest


def _download_module():
    script = Path(__file__).parents[2] / "scripts" / "download_sdnet2018.py"
    specification = importlib.util.spec_from_file_location("crackspot_download_sdnet2018", script)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_safe_extract_accepts_normal_archive_without_network(tmp_path: Path) -> None:
    module = _download_module()
    archive_path = tmp_path / "small.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SDNET2018/D/CD/patch.jpg", b"fixture")
        archive.writestr("SDNET2018/ReadMe.txt", b"source grouping notes")

    destination = tmp_path / "raw"
    module.safe_extract_zip(archive_path, destination)

    assert (destination / "SDNET2018/D/CD/patch.jpg").read_bytes() == b"fixture"
    assert (destination / "SDNET2018/ReadMe.txt").is_file()


@pytest.mark.parametrize(
    "member_name",
    ["../escape.txt", "/absolute.txt", "C:/windows-path.txt", "dir/../../escape.txt"],
)
def test_safe_extract_rejects_path_traversal(tmp_path: Path, member_name: str) -> None:
    module = _download_module()
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, b"must not escape")

    destination = tmp_path / "raw"
    with pytest.raises(module.UnsafeArchiveError):
        module.safe_extract_zip(archive_path, destination)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_refuses_to_overwrite_existing_dataset(tmp_path: Path) -> None:
    module = _download_module()
    archive_path = tmp_path / "small.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("SDNET2018/D/CD/patch.jpg", b"new")
    existing = tmp_path / "raw/SDNET2018"
    existing.mkdir(parents=True)
    marker = existing / "marker.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.safe_extract_zip(archive_path, tmp_path / "raw")

    assert marker.read_text(encoding="utf-8") == "preserve me"


def _dataset_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("D/CD", "D/UD", "P/CP", "P/UP", "W/CW", "W/UW"):
            archive.writestr(f"{folder}/source-1.jpg", folder.encode())


def test_safe_extract_sdnet_native_bundle(tmp_path: Path) -> None:
    module = _download_module()
    inner = tmp_path / "inner.zip"
    _dataset_zip(inner)
    bundle = tmp_path / "official-native.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("DATA_Maguire/ReadMe_SDNET2018 .txt", b"official notes")
        archive.write(inner, "DATA_Maguire/SDNET2018.zip")

    extracted = module.safe_extract_sdnet_archive(bundle, tmp_path / "raw")

    assert extracted == (tmp_path / "raw/SDNET2018").resolve()
    assert (extracted / "D/CD/source-1.jpg").read_bytes() == b"D/CD"
    assert (extracted / "ReadMe_SDNET2018 .txt").read_bytes() == b"official notes"
    assert not (extracted / "SDNET2018.zip").exists()


def test_safe_extract_sdnet_direct_archive_has_stable_root(tmp_path: Path) -> None:
    module = _download_module()
    archive = tmp_path / "direct.zip"
    _dataset_zip(archive)

    extracted = module.safe_extract_sdnet_archive(archive, tmp_path / "raw")

    assert extracted == (tmp_path / "raw/SDNET2018").resolve()
    assert (extracted / "W/UW/source-1.jpg").read_bytes() == b"W/UW"


class _FakeResponse:
    def __init__(self, data: bytes, *, status: int, content_type: str) -> None:
        self._data = data
        self._offset = 0
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(data))

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _FakeOpener:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def open(self, request: Any, timeout: int) -> _FakeResponse:
        assert timeout == 60
        self.requests.append(request)
        if len(self.requests) == 1:
            return _FakeResponse(b"landing", status=200, content_type="text/html")
        return _FakeResponse(b"tail", status=206, content_type="application/zip")


def test_download_bootstraps_cookie_session_and_resumes_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _download_module()
    opener = _FakeOpener()
    monkeypatch.setattr(module, "build_opener", lambda *_: opener)
    destination = tmp_path / "SDNET2018.zip"
    destination.with_name("SDNET2018.zip.part").write_bytes(b"head")

    result = module.download_with_resume("https://official.example/native", destination)

    assert result.read_bytes() == b"headtail"
    assert len(opener.requests) == 2
    assert opener.requests[0].full_url == module.OFFICIAL_PAGE
    assert opener.requests[1].full_url == "https://official.example/native"
    assert opener.requests[1].get_header("Range") == "bytes=4-"
    assert opener.requests[1].get_header("Referer") == module.OFFICIAL_PAGE
