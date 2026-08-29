from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crackspot.data import load_manifest_table, manifest_sha256
from crackspot.modeling import evaluate, train
from crackspot.reporting import aggregate, qualitative


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relative_path": "D/CD/001-1.jpg",
                "label": 1,
                "class_name": "Crack",
                "surface": "D",
                "source_group": "001",
                "source_group_verified": True,
                "group_resolution_method": "verified_regex",
                "sha256": "0" * 63 + "1",
                "perceptual_hash": "0000000000000001",
                "width": 256,
                "height": 256,
                "source_mode": "RGB",
                "image_format": "JPEG",
                "audit_status": "ok",
                "audit_error": "",
                "split": "train",
            }
        ]
    )


def test_csv_manifest_loader_preserves_text_ids_and_canonical_hash(tmp_path: Path) -> None:
    expected = _manifest()
    path = tmp_path / "manifest.csv"
    expected.to_csv(path, index=False, lineterminator="\n")

    loaded = load_manifest_table(path)

    assert loaded.loc[0, "source_group"] == "001"
    assert loaded.loc[0, "sha256"] == "0" * 63 + "1"
    assert loaded.loc[0, "perceptual_hash"] == "0000000000000001"
    assert loaded.loc[0, "relative_path"] == "D/CD/001-1.jpg"
    assert loaded.loc[0, "label"] == 1
    assert bool(loaded.loc[0, "source_group_verified"]) is True
    assert manifest_sha256(loaded) == manifest_sha256(expected)


@pytest.mark.parametrize(
    "consumer",
    [
        train._load_manifest,
        evaluate._load_manifest,
        aggregate._load_manifest,
        qualitative._read_table,
    ],
)
def test_every_manifest_consumer_uses_lossless_csv_loader(tmp_path: Path, consumer) -> None:
    expected = _manifest()
    path = tmp_path / "manifest.csv"
    expected.to_csv(path, index=False, lineterminator="\n")

    loaded = consumer(path)

    assert loaded.loc[0, "source_group"] == "001"
    assert manifest_sha256(loaded) == manifest_sha256(expected)


def test_parquet_loader_preserves_stored_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.pq"
    path.touch()
    expected = _manifest()
    calls: list[Path] = []

    def fake_read_parquet(source: Path) -> pd.DataFrame:
        calls.append(source)
        return expected

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    loaded = load_manifest_table(path)

    assert loaded is expected
    assert calls == [path]


def test_manifest_loader_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV or Parquet"):
        load_manifest_table(path)
