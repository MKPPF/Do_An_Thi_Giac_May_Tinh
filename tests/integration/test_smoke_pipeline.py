from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from crackspot.data import audit_split


def _load_smoke_module() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "smoke_pipeline.py"
    spec = importlib.util.spec_from_file_location("crackspot_smoke_pipeline_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.slow
def test_complete_synthetic_smoke_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the process-wide one-shot registry isolated from the developer's real
    # artifact registry while preserving the exact same marker implementation.
    import crackspot.modeling.evaluate as final_evaluation

    marker_registry = tmp_path / "final_test_registry"
    monkeypatch.setattr(final_evaluation, "FINAL_EVALUATION_REGISTRY", marker_registry)
    smoke = _load_smoke_module()
    output_root = tmp_path / "artifacts" / "smoke"

    run_dir = smoke.run_smoke_pipeline(
        output_root=output_root,
        run_id="smoke-integration",
    )

    assert run_dir == output_root.resolve() / "smoke-integration"
    assert not (output_root / ".internal-smoke-work").exists()

    manifest = pd.read_csv(run_dir / "manifest.csv", keep_default_na=False)
    split_report = audit_split(manifest)
    assert split_report["valid"] is True
    assert len(manifest) == 60
    assert manifest["source_group"].nunique() == 30
    assert set(manifest["surface"]) == {"D", "P", "W"}
    assert set(manifest["label"]) == {0, 1}
    assert manifest["source_group_verified"].all()

    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    assert config["model"]["weights"] is None
    assert config["training"]["head"]["max_epochs"] == 1
    assert config["run"]["smoke"] is True

    selection = json.loads((run_dir / "selection_complete.json").read_text(encoding="utf-8"))
    threshold = json.loads((run_dir / "threshold_validation.json").read_text(encoding="utf-8"))
    assert selection["selected_by"] == "validation"
    assert selection["threshold"] == pytest.approx(threshold["threshold"])

    final_metrics = json.loads(
        (run_dir / "final_test" / "metrics_test.json").read_text(encoding="utf-8")
    )
    final_complete = json.loads(
        (run_dir / "final_test" / "evaluation_complete.json").read_text(encoding="utf-8")
    )
    assert final_metrics["status"] == "NOT_VALID_FOR_REPORT"
    assert final_metrics["valid_for_report"] is False
    assert final_complete["status"] == "NOT_VALID_FOR_REPORT"
    assert len(list(marker_registry.glob("*.json"))) == 1

    metadata = json.loads((run_dir / "selected_model.metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "smoke_pipeline_summary.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "NOT_VALID_FOR_REPORT"
    assert metadata["valid_for_report"] is False
    assert metadata["threshold"] == pytest.approx(selection["threshold"])
    assert summary["status"] == "NOT_VALID_FOR_REPORT"
    assert summary["valid_for_report"] is False
    assert summary["synthetic_data"] is True
    assert summary["network_used"] is False
    assert summary["imagenet_weights_used"] is False
    assert summary["final_evaluation_status"] == "NOT_VALID_FOR_REPORT"
    assert summary["cli_service_probability_delta"] <= 1e-6
    assert (
        summary["cli_prediction"]["predicted_class"]
        == summary["service_prediction"]["predicted_class"]
    )
    assert len(summary["gradcam_heatmap_shape"]) == 2

    with Image.open(run_dir / "gradcam_smoke.png") as overlay:
        overlay.verify()
