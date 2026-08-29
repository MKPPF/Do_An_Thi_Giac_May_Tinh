from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from crackspot.data import (
    DatasetStructureError,
    GroupResolutionError,
    build_manifest,
    parse_sdnet_path,
    write_group_map_template,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("D/CD/crack_1.jpg", ("D", 1)),
        ("D/UD/no_crack_1.jpg", ("D", 0)),
        ("P/CP/crack_2.jpg", ("P", 1)),
        ("P\\UP\\no_crack_2.jpg", ("P", 0)),
        ("SDNET2018/W/CW/crack_3.jpg", ("W", 1)),
        ("W/UW/no_crack_3.jpg", ("W", 0)),
    ],
)
def test_parse_sdnet_path_uses_folder_as_single_source_of_label(
    path: str, expected: tuple[str, int]
) -> None:
    assert parse_sdnet_path(path) == expected


@pytest.mark.parametrize(
    "path",
    ["image.jpg", "D/XD/image.jpg", "D/CD/UP/image.jpg", "W/CP/image.jpg"],
)
def test_parse_sdnet_path_rejects_ambiguous_or_inconsistent_structure(path: str) -> None:
    with pytest.raises(DatasetStructureError):
        parse_sdnet_path(path)


def _save_image(root: Path, relative_path: str, mode: str = "RGB") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    color: object = 80 if mode == "L" else (30, 60, 90)
    Image.new(mode, (8, 6), color=color).save(path)
    return path


def test_build_manifest_decodes_content_and_applies_verified_group_map(
    tmp_path: Path,
) -> None:
    _save_image(tmp_path, "D/CD/c1.jpg")
    _save_image(tmp_path, "D/UD/u1.png", mode="L")
    provisional = build_manifest(tmp_path)
    group_map = provisional.loc[:, ["relative_path", "sha256"]].copy()
    group_map["source_group"] = ["deck-original-01", "deck-original-02"]
    group_map["verified"] = True

    frame = build_manifest(tmp_path, group_map=group_map)

    assert frame["label"].tolist() == [1, 0]
    assert frame["class_name"].tolist() == ["Crack", "Non-crack"]
    assert frame["surface"].tolist() == ["D", "D"]
    assert frame["source_group_verified"].all()
    assert set(frame["source_mode"]) == {"RGB", "L"}
    assert set(frame["image_format"]) == {"JPEG", "PNG"}
    assert frame["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert frame.attrs["dataset_root"] == str(tmp_path.resolve())


def test_build_manifest_keeps_invalid_files_visible(tmp_path: Path) -> None:
    _save_image(tmp_path, "P/CP/good.jpg")
    bad = tmp_path / "P/UP/bad.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"broken")

    frame = build_manifest(tmp_path)
    bad_row = frame.loc[frame["relative_path"].eq("P/UP/bad.jpg")].iloc[0]

    assert bad_row["audit_status"] == "invalid"
    assert "CorruptImageError" in bad_row["audit_error"]
    assert len(frame) == 2


def test_verified_regex_requires_complete_match(tmp_path: Path) -> None:
    _save_image(tmp_path, "W/CW/original01_patch001.jpg")
    _save_image(tmp_path, "W/UW/unmatched.jpg")

    with pytest.raises(GroupResolutionError, match="did not resolve"):
        build_manifest(
            tmp_path,
            group_regex=r"(original\d+)_patch",
            group_rule_verified=True,
        )


def test_regex_groups_are_explicitly_unverified_by_default(tmp_path: Path) -> None:
    _save_image(tmp_path, "W/CW/original01_patch001.jpg")

    frame = build_manifest(tmp_path, group_regex=r"(original\d+)_patch")

    assert frame.loc[0, "source_group"] == "original01"
    assert not bool(frame.loc[0, "source_group_verified"])
    assert frame.loc[0, "group_resolution_method"] == "unverified_regex"


def test_group_map_must_cover_exact_audited_dataset(tmp_path: Path) -> None:
    _save_image(tmp_path, "D/CD/a.jpg")
    incomplete = pd.DataFrame(columns=["relative_path", "source_group"])

    with pytest.raises(GroupResolutionError, match="coverage"):
        build_manifest(tmp_path, group_map=incomplete, group_rule_verified=True)


def test_group_map_sha_prevents_stale_review_from_being_reused(tmp_path: Path) -> None:
    _save_image(tmp_path, "D/CD/a.jpg")
    group_map = pd.DataFrame(
        {
            "relative_path": ["D/CD/a.jpg"],
            "source_group": ["original-1"],
            "sha256": ["0" * 64],
            "verified": [True],
        }
    )

    with pytest.raises(GroupResolutionError, match="SHA-256 mismatch"):
        build_manifest(tmp_path, group_map=group_map)


def test_write_group_map_template_never_claims_verification(tmp_path: Path) -> None:
    _save_image(tmp_path, "D/CD/a.jpg")
    frame = build_manifest(tmp_path)
    output = tmp_path / "review" / "group_map.csv"

    write_group_map_template(frame, output)
    result = pd.read_csv(output, keep_default_na=False)

    assert result["source_group"].eq("").all()
    assert result["verified"].eq(False).all()
