#!/usr/bin/env python3
"""Download, verify and safely extract the official SDNET2018 archive."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from contextlib import suppress
from hashlib import md5
from http.cookiejar import CookieJar
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

OFFICIAL_PAGE = "https://digitalcommons.usu.edu/all_datasets/48/"
DEFAULT_URL = (
    "https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent"
)
OFFICIAL_MD5 = "677411e784f194422c90f52d9ed0d7c6"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36 "
    "CrackSpot/1.0"
)
CHUNK_SIZE = 1024 * 1024
MAX_MEMBERS = 100_000
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3
EXPECTED_DATASET_FOLDERS = ("D/CD", "D/UD", "P/CP", "P/UP", "W/CW", "W/UW")


class DownloadError(RuntimeError):
    """Network or integrity failure while obtaining the archive."""


class UnsafeArchiveError(RuntimeError):
    """Archive content violates extraction safety constraints."""


def md5_file(path: Path) -> str:
    """Compute the publisher-provided MD5 integrity checksum."""

    digest = md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path) -> Path:
    """Stream to ``.part`` after bootstrapping the official repository session.

    DigitalCommons currently protects ``viewcontent`` with Cloudflare.  A
    direct request receives HTTP 403, while a normal visit to the official
    landing page first supplies the short-lived cookie needed by both the
    native download URL and subsequent HTTP Range requests.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "application/zip,application/octet-stream;q=0.9,*/*;q=0.1",
        "Accept-Encoding": "identity",
        "Referer": OFFICIAL_PAGE,
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"

    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    try:
        landing_request = Request(
            OFFICIAL_PAGE,
            headers={
                "User-Agent": BROWSER_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
            },
        )
        with opener.open(landing_request, timeout=60) as landing_response:
            # Reading one byte ensures response headers/cookies are processed
            # without retaining the landing page in memory.
            landing_response.read(1)
        response = opener.open(Request(url, headers=headers), timeout=60)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DownloadError(
            f"cannot download SDNET2018 from {url}: {exc}. "
            f"Open the official page manually if its download URL changed: {OFFICIAL_PAGE}"
        ) from exc
    with response:
        status = getattr(response, "status", response.getcode())
        content_type = response.headers.get_content_type()
        if content_type == "text/html":
            raise DownloadError(
                "server returned HTML instead of the ZIP archive; use --url with "
                f"the current link from {OFFICIAL_PAGE}"
            )
        if offset and status != 206:
            # The server ignored Range. Restart in the same .part file so a stale
            # prefix can never be concatenated to a full response.
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        expected_remaining_text = response.headers.get("Content-Length")
        expected_remaining = int(expected_remaining_text) if expected_remaining_text else None
        received = 0
        with partial.open(mode) as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
        if expected_remaining is not None and received != expected_remaining:
            raise DownloadError(
                f"incomplete response: received {received} of {expected_remaining} expected bytes"
            )
    os.replace(partial, destination)
    return destination


def _configure_utf8_stdio() -> None:
    """Keep Vietnamese diagnostics printable on legacy Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # Captured/test streams do not always support reconfiguration.
        with suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="replace")


def _safe_member_path(member_name: str) -> PurePosixPath:
    normalised = member_name.replace("\\", "/")
    path = PurePosixPath(normalised)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise UnsafeArchiveError(f"unsafe ZIP member path: {member_name!r}")
    return path


def _validate_archive(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    members = archive.infolist()
    if len(members) > MAX_MEMBERS:
        raise UnsafeArchiveError(f"archive contains {len(members)} entries; limit is {MAX_MEMBERS}")
    total_size = sum(member.file_size for member in members)
    if total_size > MAX_UNCOMPRESSED_BYTES:
        raise UnsafeArchiveError(
            f"archive expands to {total_size} bytes; safety limit is {MAX_UNCOMPRESSED_BYTES} bytes"
        )
    validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for member in members:
        path = _safe_member_path(member.filename)
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise UnsafeArchiveError(f"symbolic link is not accepted: {member.filename}")
        if member.compress_size > 0 and member.file_size / member.compress_size > 2_000:
            raise UnsafeArchiveError(f"suspicious compression ratio for {member.filename}")
        validated.append((member, path))
    return validated


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract through a staging directory, rejecting traversal and links."""

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = _validate_archive(archive)
        with tempfile.TemporaryDirectory(
            prefix="sdnet2018-extract-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary).resolve()
            for info, relative in members:
                target = (staging / Path(*relative.parts)).resolve()
                try:
                    target.relative_to(staging)
                except ValueError as exc:  # defence in depth after path validation.
                    raise UnsafeArchiveError(
                        f"archive member escapes destination: {info.filename}"
                    ) from exc
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)

            top_level = list(staging.iterdir())
            conflicts = [
                destination / item.name for item in top_level if (destination / item.name).exists()
            ]
            if conflicts:
                raise FileExistsError(
                    "refusing to overwrite extracted data: "
                    + ", ".join(str(path) for path in conflicts)
                )
            for item in top_level:
                shutil.move(str(item), destination / item.name)


def _copy_zip_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path) -> Path:
    """Copy one already-validated regular member with bounded memory."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=CHUNK_SIZE)
    if destination.stat().st_size != member.file_size:
        raise DownloadError(
            f"incomplete nested ZIP member: received {destination.stat().st_size} "
            f"of {member.file_size} bytes"
        )
    return destination


def safe_extract_sdnet_archive(archive_path: Path, extract_root: Path) -> Path:
    """Extract the publisher's native bundle into ``<extract_root>/SDNET2018``.

    The checksum-bearing DigitalCommons native download is a ZIP bundle that
    contains the README and a nested ``SDNET2018.zip``.  This function also
    accepts the historical direct dataset ZIP, but always produces one stable
    dataset root and stages everything before the final move.
    """

    target = extract_root.resolve() / "SDNET2018"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite extracted data: {target}")
    extract_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="sdnet2018-bundle-", dir=extract_root.parent
    ) as temporary:
        temporary_root = Path(temporary).resolve()
        staged_dataset = temporary_root / "SDNET2018"
        with zipfile.ZipFile(archive_path) as outer:
            validated = _validate_archive(outer)
            nested_members = [
                (info, path)
                for info, path in validated
                if not info.is_dir() and path.name.lower() == "sdnet2018.zip"
            ]
            if nested_members:
                if len(nested_members) != 1:
                    raise UnsafeArchiveError(
                        f"expected one nested SDNET2018.zip, found {len(nested_members)}"
                    )
                nested_info, _ = nested_members[0]
                nested_path = _copy_zip_member(
                    outer, nested_info, temporary_root / "nested-SDNET2018.zip"
                )
                safe_extract_zip(nested_path, staged_dataset)
                readme_members = [
                    (info, path)
                    for info, path in validated
                    if not info.is_dir() and path.suffix.lower() == ".txt"
                ]
                for info, path in readme_members:
                    _copy_zip_member(outer, info, staged_dataset / path.name)
            else:
                safe_extract_zip(archive_path, staged_dataset)

        missing = [
            folder for folder in EXPECTED_DATASET_FOLDERS if not (staged_dataset / folder).is_dir()
        ]
        if missing:
            raise UnsafeArchiveError(
                "archive does not have the expected SDNET2018 class folders: " + ", ".join(missing)
            )
        extract_root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite extracted data: {target}")
        shutil.move(str(staged_dataset), target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official SDNET2018 ZIP, verify MD5 and extract safely."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Official ZIP URL")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/SDNET2018.zip"),
        help="Local archive path",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory receiving archive top-level entries",
    )
    parser.add_argument("--expected-md5", default=OFFICIAL_MD5)
    parser.add_argument(
        "--skip-extract", action="store_true", help="Verify download but do not extract"
    )
    return parser.parse_args()


def main() -> int:
    _configure_utf8_stdio()
    args = parse_args()
    archive = args.archive.resolve()
    try:
        if archive.exists():
            actual = md5_file(archive)
            if actual.lower() != args.expected_md5.lower():
                raise DownloadError(
                    f"existing archive MD5 is {actual}, expected {args.expected_md5}; "
                    "move the bad file aside before retrying"
                )
            print(f"Archive already present and verified: {archive}")
        else:
            print(f"Downloading from {args.url}")
            download_with_resume(args.url, archive)
            actual = md5_file(archive)
            if actual.lower() != args.expected_md5.lower():
                raise DownloadError(
                    f"MD5 mismatch: received {actual}, expected {args.expected_md5}. "
                    f"Archive retained for inspection at {archive}"
                )
            print(f"MD5 verified: {actual}")
        if not args.skip_extract:
            extracted = safe_extract_sdnet_archive(archive, args.extract_dir.resolve())
            print(f"Extracted safely into: {extracted}")
    except (DownloadError, UnsafeArchiveError, FileExistsError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
