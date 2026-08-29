"""Run CrackSpot's complete protocol on a tiny, generated dataset.

This smoke run is deliberately self-contained: it never downloads SDNET2018 or
ImageNet weights, and every resulting artifact is marked NOT_VALID_FOR_REPORT.
It exercises the same manifest, group split, training, threshold-selection,
final-test, inference, CLI, and Grad-CAM paths used by a real experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

try:
    from crackspot import __version__
    from crackspot.constants import LABEL_MAPPING, PREPROCESSING_NAME
    from crackspot.data import audit_split, build_manifest, create_group_splits
    from crackspot.inference import InferenceService
    from crackspot.modeling.evaluate import run_final_evaluation
    from crackspot.modeling.selection import lock_run_selection
    from crackspot.modeling.threshold import tune_from_predictions
    from crackspot.modeling.train import run_training
    from crackspot.reporting.export import write_json
    from crackspot.utils.hashing import sha256_file
except ModuleNotFoundError:  # Support direct execution before an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot import __version__
    from crackspot.constants import LABEL_MAPPING, PREPROCESSING_NAME
    from crackspot.data import audit_split, build_manifest, create_group_splits
    from crackspot.inference import InferenceService
    from crackspot.modeling.evaluate import run_final_evaluation
    from crackspot.modeling.selection import lock_run_selection
    from crackspot.modeling.threshold import tune_from_predictions
    from crackspot.modeling.train import run_training
    from crackspot.reporting.export import write_json
    from crackspot.utils.hashing import sha256_file


STATUS = "NOT_VALID_FOR_REPORT"
SURFACE_FOLDERS = {
    ("D", 1): "D/CD",
    ("D", 0): "D/UD",
    ("P", 1): "P/CP",
    ("P", 0): "P/UP",
    ("W", 1): "W/CW",
    ("W", 0): "W/UW",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"smoke-{timestamp}"


def _synthetic_pixels(
    *, surface_index: int, label: int, source_index: int, patch_index: int
) -> np.ndarray:
    """Create a deterministic RGB texture with a conspicuous crack-like band."""

    seed = 10_000 * surface_index + 1_000 * label + 100 * source_index + patch_index
    rng = np.random.default_rng(seed)
    height = width = 64
    base = (190, 215, 235)[surface_index]
    noise = rng.integers(-10, 11, size=(height, width, 1), dtype=np.int16)
    tint = np.asarray(
        (
            (base, base - 8, base - 15),
            (base - 5, base, base - 10),
            (base - 12, base - 5, base),
        )[surface_index],
        dtype=np.int16,
    )
    pixels = np.clip(tint[None, None, :] + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    if label == 1:
        offset = source_index * 2 + patch_index
        points = [
            (5 + offset % 7, 0),
            (24 + offset % 5, 17),
            (17 + offset % 9, 35),
            (45 - offset % 6, 63),
        ]
        draw.line(points, fill=(8, 8, 8), width=7, joint="curve")
        draw.line([(point[0] + 5, point[1]) for point in points], fill=(45, 45, 45), width=2)
    else:
        # A faint, label-preserving surface mark also makes every encoded file unique.
        row = 4 + source_index * 7 + patch_index
        draw.line((2, row, 61, row), fill=(base - 18, base - 18, base - 18), width=1)
    return np.asarray(image, dtype=np.uint8)


def _create_synthetic_dataset(dataset_root: Path) -> pd.DataFrame:
    """Write two patches for five verified source groups in every stratum."""

    rows: list[dict[str, Any]] = []
    for surface_index, surface in enumerate(("D", "P", "W")):
        for label in (0, 1):
            folder = dataset_root / SURFACE_FOLDERS[(surface, label)]
            folder.mkdir(parents=True, exist_ok=True)
            for source_index in range(5):
                source_group = f"synthetic-{surface}-{label}-{source_index:02d}"
                for patch_index in range(2):
                    filename = f"{surface.lower()}_{label}_{source_index:02d}_{patch_index:02d}.png"
                    path = folder / filename
                    pixels = _synthetic_pixels(
                        surface_index=surface_index,
                        label=label,
                        source_index=source_index,
                        patch_index=patch_index,
                    )
                    Image.fromarray(pixels).save(path, format="PNG")
                    rows.append(
                        {
                            "relative_path": path.relative_to(dataset_root).as_posix(),
                            "source_group": source_group,
                            "verified": True,
                            "review_notes": "Generated source identity; synthetic smoke only",
                        }
                    )
    return pd.DataFrame(rows)


def _write_smoke_config(path: Path) -> Path:
    base_config = (_project_root() / "configs" / "experiments" / "e1_baseline.yaml").resolve()
    payload = {
        "extends": str(base_config),
        "experiment": {
            "id": "SMOKE",
            "name": "smoke_end_to_end",
            "slug": "smoke_end_to_end",
            "description": "Synthetic protocol smoke; never valid evidence for the report.",
        },
        "pipeline": {"batch_size": 32, "augmentation": {"enabled": False}},
        "model": {"weights": None, "dropout": 0.0},
        "training": {
            "head": {"enabled": True, "max_epochs": 1, "learning_rate": 0.003},
            "fine_tune": {"enabled": False, "max_epochs": 0, "unfreeze_from": None},
            "callbacks": {
                "early_stopping": {"enabled": False},
                "reduce_lr_on_plateau": {"enabled": False},
            },
        },
        "evaluation": {"threshold": 0.5, "test_threshold": 0.5},
        "run": {"smoke": True, "status": STATUS, "valid_for_report": False},
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _run_cli(*, image_path: Path, model_path: Path, metadata_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(_project_root() / "scripts" / "predict.py"),
        str(image_path),
        "--model",
        str(model_path),
        "--metadata",
        str(metadata_path),
        "--json",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "TF_CPP_MIN_LOG_LEVEL": environment.get("TF_CPP_MIN_LOG_LEVEL", "2"),
        }
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=environment,
        timeout=180,
    )
    stdout = completed.stdout.decode("utf-8", errors="strict")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("predict.py returned no JSON output")
    return json.loads(lines[-1])


def _safe_remove_internal_workdir(work_dir: Path, staging_root: Path) -> None:
    resolved_work = work_dir.resolve()
    resolved_staging = staging_root.resolve()
    if resolved_work.parent != resolved_staging or not resolved_work.name.startswith(
        "crackspot-smoke-"
    ):
        raise RuntimeError(f"Refusing to remove a non-internal smoke path: {resolved_work}")
    shutil.rmtree(resolved_work)


def run_smoke_pipeline(
    *, output_root: str | Path = "artifacts/smoke", run_id: str | None = None
) -> Path:
    """Execute the complete, non-reportable smoke protocol and return its run directory."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or _new_run_id()
    final_run_dir = output / resolved_run_id
    if final_run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite smoke run: {final_run_dir}")

    staging_root = output / ".internal-smoke-work"
    staging_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="crackspot-smoke-", dir=staging_root))
    try:
        dataset_root = work_dir / "dataset"
        group_map = _create_synthetic_dataset(dataset_root)
        group_map.to_csv(work_dir / "group_map.csv", index=False, lineterminator="\n")
        manifest = build_manifest(
            dataset_root,
            group_map=group_map,
            group_rule_verified=True,
            raise_on_image_error=True,
        )
        split_manifest = create_group_splits(manifest, seed=42, restarts=16)
        split_audit = audit_split(split_manifest)
        if not split_audit["valid"]:
            raise RuntimeError(f"Synthetic split audit failed: {split_audit['errors']}")
        manifest_path = work_dir / "manifest.csv"
        split_manifest.to_csv(manifest_path, index=False, lineterminator="\n")
        config_path = _write_smoke_config(work_dir / "smoke_config.yaml")

        run_dir = run_training(
            config_path=config_path,
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            output_root=output,
            run_id=resolved_run_id,
            weights_none=True,
            head_epochs=1,
            fine_tune_epochs=0,
            smoke=True,
        )
        if run_dir.resolve() != final_run_dir.resolve():
            raise RuntimeError("Training returned an unexpected smoke run directory")

        # Preserve synthetic provenance beneath the immutable run after training created it.
        shutil.copytree(dataset_root, run_dir / "dataset")
        shutil.copy2(manifest_path, run_dir / "manifest.csv")
        shutil.copy2(work_dir / "group_map.csv", run_dir / "group_map.csv")
        shutil.copy2(config_path, run_dir / "smoke_config.yaml")
        write_json(run_dir / "synthetic_split_audit.json", split_audit, overwrite=False)

        threshold_result = tune_from_predictions(
            run_dir / "predictions_validation.csv",
            run_dir / "threshold_validation.json",
            allow_smoke=True,
        )
        selection_path = run_dir / "selection_complete.json"
        lock_run_selection(
            run_dir=run_dir,
            output=selection_path,
            experiment="smoke_end_to_end",
            threshold_result=run_dir / "threshold_validation.json",
            allow_smoke=True,
        )

        evaluation_dir = run_final_evaluation(
            selection_path=selection_path,
            manifest_path=run_dir / "manifest.csv",
            dataset_root=run_dir / "dataset",
            output_dir=run_dir / "final_test",
            smoke=True,
        )

        metadata_payload = json.loads((run_dir / "model.metadata.json").read_text(encoding="utf-8"))
        metadata_payload.update(
            {
                "threshold": threshold_result.threshold,
                "status": STATUS,
                "valid_for_report": False,
                "smoke_test": True,
            }
        )
        selected_metadata = run_dir / "selected_model.metadata.json"
        write_json(selected_metadata, metadata_payload, overwrite=False)

        probe_row = (
            split_manifest.loc[split_manifest["split"] == "test"]
            .sort_values("relative_path")
            .iloc[0]
        )
        probe_path = run_dir / "dataset" / str(probe_row["relative_path"])
        service = InferenceService.from_files(
            run_dir / "model.keras", selected_metadata, verify_hash=True
        )
        service_result = service.predict_image(probe_path, include_gradcam=True)
        if service_result.overlay is None or service_result.heatmap is None:
            raise RuntimeError("Grad-CAM smoke did not produce an overlay and heatmap")
        overlay_path = run_dir / "gradcam_smoke.png"
        service_result.overlay.save(overlay_path, format="PNG")
        cli_result = _run_cli(
            image_path=probe_path,
            model_path=run_dir / "model.keras",
            metadata_path=selected_metadata,
        )
        probability_delta = abs(
            float(cli_result["crack_probability"]) - service_result.crack_probability
        )
        if probability_delta > 1e-6:
            raise RuntimeError(
                f"CLI and InferenceService probabilities differ by {probability_delta}"
            )
        if int(cli_result["predicted_class"]) != service_result.predicted_class:
            raise RuntimeError("CLI and InferenceService predicted different classes")

        summary = {
            "run_id": resolved_run_id,
            "status": STATUS,
            "valid_for_report": False,
            "synthetic_data": True,
            "network_used": False,
            "imagenet_weights_used": False,
            "model_version": __version__,
            "preprocessing": PREPROCESSING_NAME,
            "label_mapping": LABEL_MAPPING,
            "rows": len(split_manifest),
            "source_groups": int(split_manifest["source_group"].nunique()),
            "split_audit_valid": bool(split_audit["valid"]),
            "threshold_selection": asdict(threshold_result),
            "selection_contract": str(selection_path.resolve()),
            "final_evaluation_dir": str(evaluation_dir.output_dir.resolve()),
            "final_evaluation_status": evaluation_dir.status,
            "model_sha256": sha256_file(run_dir / "model.keras"),
            "probe_relative_path": str(probe_row["relative_path"]),
            "service_prediction": service_result.to_dict(),
            "cli_prediction": cli_result,
            "cli_service_probability_delta": probability_delta,
            "gradcam_heatmap_shape": list(service_result.heatmap.shape),
            "gradcam_overlay": str(overlay_path.resolve()),
        }
        write_json(run_dir / "smoke_pipeline_summary.json", summary, overwrite=False)
        return run_dir
    finally:
        _safe_remove_internal_workdir(work_dir, staging_root)
        with suppress(OSError):
            staging_root.rmdir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full CrackSpot protocol on generated, non-reportable data."
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--run-id", help="Optional immutable run directory name")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    run_dir = run_smoke_pipeline(output_root=args.output_root, run_id=args.run_id)
    print(
        json.dumps(
            {"run_dir": str(run_dir.resolve()), "status": STATUS},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
