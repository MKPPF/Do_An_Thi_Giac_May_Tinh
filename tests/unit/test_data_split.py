from __future__ import annotations

from hashlib import sha256

import pandas as pd
import pandas.testing as pdt
import pytest

from crackspot.data import (
    MAX_IMAGE_FRACTION_DEVIATION,
    MAX_SOURCE_GROUP_FRACTION_DEVIATION,
    MAX_STRATUM_FRACTION_DEVIATION,
    SplitValidationError,
    audit_split,
    create_group_splits,
    manifest_sha256,
)


def _manifest(groups_per_stratum: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for surface in ("D", "P", "W"):
        for label in (0, 1):
            for group_index in range(groups_per_stratum):
                payload = f"{surface}-{label}-{group_index}".encode()
                rows.append(
                    {
                        "relative_path": f"{surface}/{'C' if label else 'U'}{surface}/{index}.jpg",
                        "label": label,
                        "surface": surface,
                        "source_group": f"{surface}-{label}-original-{group_index}",
                        "source_group_verified": True,
                        "sha256": sha256(payload).hexdigest(),
                        "audit_status": "ok",
                        "split": "",
                    }
                )
                index += 1
    return pd.DataFrame(rows)


def test_group_split_is_deterministic_leakage_free_and_near_requested_ratios() -> None:
    frame = _manifest()

    first = create_group_splits(frame, seed=42, restarts=12)
    second = create_group_splits(frame.sample(frac=1, random_state=7), seed=42, restarts=12)
    first_sorted = first.sort_values("relative_path").reset_index(drop=True)
    second_sorted = second.sort_values("relative_path").reset_index(drop=True)

    pdt.assert_series_equal(first_sorted["split"], second_sorted["split"])
    report = audit_split(first)
    assert report["valid"] is True
    assert report["errors"] == []
    assert abs(report["counts"]["train"]["fraction"] - 0.70) <= 0.05
    assert abs(report["counts"]["validation"]["fraction"] - 0.15) <= 0.05
    assert abs(report["counts"]["test"]["fraction"] - 0.15) <= 0.05
    for category in ("path", "source_group", "sha256"):
        assert all(not values for values in report["overlaps"][category].values())


def test_all_patches_from_same_source_group_stay_together() -> None:
    frame = _manifest()
    duplicate_row = frame.iloc[0].copy()
    duplicate_row["relative_path"] = "D/UD/second_patch_same_original.jpg"
    duplicate_row["sha256"] = sha256(b"different patch").hexdigest()
    frame = pd.concat([frame, duplicate_row.to_frame().T], ignore_index=True)

    result = create_group_splits(frame, seed=42, restarts=8)

    group_rows = result.loc[result["source_group"].eq(frame.iloc[0]["source_group"])]
    assert group_rows["split"].nunique() == 1


def test_exact_duplicates_link_distinct_source_groups_into_same_split() -> None:
    frame = _manifest()
    first_index = frame.index[(frame["surface"] == "D") & (frame["label"] == 0)][0]
    second_index = frame.index[(frame["surface"] == "P") & (frame["label"] == 0)][0]
    frame.loc[second_index, "sha256"] = frame.loc[first_index, "sha256"]

    result = create_group_splits(frame, seed=42, restarts=8)

    assert result.loc[first_index, "split"] == result.loc[second_index, "split"]
    assert audit_split(result)["valid"]


def test_conflicting_labels_for_identical_bytes_fail_fast() -> None:
    frame = _manifest()
    zero_index = frame.index[frame["label"].eq(0)][0]
    one_index = frame.index[frame["label"].eq(1)][0]
    frame.loc[one_index, "sha256"] = frame.loc[zero_index, "sha256"]

    with pytest.raises(SplitValidationError, match="conflicting labels"):
        create_group_splits(frame)


@pytest.mark.parametrize("column", ["source_group", "sha256"])
def test_missing_group_or_hash_fails_fast(column: str) -> None:
    frame = _manifest()
    frame.loc[0, column] = ""

    with pytest.raises(SplitValidationError):
        create_group_splits(frame)


def test_unverified_group_rule_is_not_report_valid() -> None:
    frame = _manifest()
    frame.loc[0, "source_group_verified"] = False

    with pytest.raises(SplitValidationError, match="NOT_VALID_FOR_REPORT"):
        create_group_splits(frame)


def test_split_fails_when_stratum_cannot_cover_all_splits() -> None:
    frame = _manifest(groups_per_stratum=3)
    target = frame.index[(frame["surface"] == "D") & (frame["label"] == 1)]
    frame.loc[target, "source_group"] = "one-deck-crack-original"

    with pytest.raises(SplitValidationError, match="cannot represent"):
        create_group_splits(frame)


def test_audit_detects_source_group_and_hash_overlap() -> None:
    frame = create_group_splits(_manifest(), restarts=8)
    train_index = frame.index[frame["split"].eq("train")][0]
    test_index = frame.index[frame["split"].eq("test")][0]
    frame.loc[test_index, "source_group"] = frame.loc[train_index, "source_group"]
    frame.loc[test_index, "sha256"] = frame.loc[train_index, "sha256"]

    report = audit_split(frame)

    assert report["valid"] is False
    assert report["overlaps"]["source_group"]["train__test"]
    assert report["overlaps"]["sha256"]["train__test"]


def test_manifest_hash_is_independent_of_row_order_but_not_assignment() -> None:
    frame = create_group_splits(_manifest(), restarts=8)
    shuffled = frame.sample(frac=1, random_state=1)
    changed = frame.copy()
    changed.loc[0, "split"] = "test" if changed.loc[0, "split"] != "test" else "train"

    assert manifest_sha256(frame) == manifest_sha256(shuffled)
    assert manifest_sha256(frame) != manifest_sha256(changed)


@pytest.mark.parametrize("invalid_label", [0.5, True])
def test_split_rejects_fractional_and_boolean_labels(invalid_label: object) -> None:
    frame = _manifest()
    frame["label"] = frame["label"].astype(object)
    frame.loc[0, "label"] = invalid_label

    with pytest.raises(SplitValidationError, match="labels must be"):
        create_group_splits(frame)


def test_split_requires_exact_official_surface_vocabulary() -> None:
    frame = _manifest()
    frame.loc[0, "surface"] = "A"

    with pytest.raises(SplitValidationError, match="exactly D, P and W"):
        create_group_splits(frame)


def test_split_rejects_malformed_sha256() -> None:
    frame = _manifest()
    frame.loc[0, "sha256"] = "not-a-sha256"

    with pytest.raises(SplitValidationError, match="malformed SHA-256"):
        create_group_splits(frame)


def test_official_seed_and_ratios_are_fixed() -> None:
    frame = _manifest()

    with pytest.raises(SplitValidationError, match="seed is fixed"):
        create_group_splits(frame, seed=41)
    with pytest.raises(SplitValidationError, match="ratios are fixed"):
        create_group_splits(frame, ratios=(0.60, 0.20, 0.20))


def test_audit_requires_every_surface_label_stratum_in_every_split() -> None:
    frame = create_group_splits(_manifest(), restarts=8)
    target = frame["split"].eq("test") & frame["surface"].eq("D") & frame["label"].eq(0)
    frame.loc[target, "split"] = "train"

    report = audit_split(frame)

    assert report["valid"] is False
    assert any(
        "test is missing required stratum surface=D, label=0" in error for error in report["errors"]
    )


def test_audit_reports_official_targets_tolerances_and_deviations() -> None:
    report = audit_split(create_group_splits(_manifest(), restarts=8))

    assert report["protocol"]["target_ratios"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    assert report["protocol"]["tolerances"] == {
        "units": "absolute_fraction_percentage_points",
        "image_fraction": MAX_IMAGE_FRACTION_DEVIATION,
        "source_group_fraction": MAX_SOURCE_GROUP_FRACTION_DEVIATION,
        "surface_label_fraction": MAX_STRATUM_FRACTION_DEVIATION,
    }
    assert report["balance"]["within_official_tolerances"] is True
    for split_name in ("train", "validation", "test"):
        counts = report["counts"][split_name]
        assert "target_fraction" in counts
        assert "fraction_deviation" in counts
        assert "source_group_fraction_deviation" in counts
        assert set(counts["surface_label_fraction_deviation"]) == {
            "D|0",
            "D|1",
            "P|0",
            "P|1",
            "W|0",
            "W|1",
        }
