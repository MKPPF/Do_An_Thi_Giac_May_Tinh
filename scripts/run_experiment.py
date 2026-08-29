"""Run one immutable E1-E4 training experiment from a YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crackspot.config import load_config
from crackspot.modeling.train import run_training


def _default_paths(config_path: str | Path) -> tuple[Path, Path, Path]:
    loaded = load_config(config_path)
    data = loaded.values.get("data", {})
    project = loaded.values.get("project", {})
    manifest_dir = Path(data.get("manifest_dir", "data/manifests"))
    return (
        manifest_dir / "split_v1" / "manifest.csv",
        Path(data.get("raw_dir", "data/raw/SDNET2018")),
        Path(project.get("artifact_root", "artifacts")) / "runs",
    )


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(
        description="Train CrackSpot E1-E4 without opening the locked test split"
    )
    parser.add_argument("--config", required=True, help="Experiment YAML (E1-E4)")
    parser.add_argument("--manifest", help="Locked split CSV/Parquet")
    parser.add_argument("--dataset-root", help="SDNET2018 directory")
    parser.add_argument("--output-root", help="Immutable run root")
    parser.add_argument("--run-id", help="Optional unique run identifier")
    parser.add_argument("--model-selection", help="E2/E3 validation decision required by E4")
    parser.add_argument(
        "--weights-none", action="store_true", help="Smoke only: no ImageNet download"
    )
    parser.add_argument("--head-epochs", type=int, help="Validation-approved override")
    parser.add_argument("--fine-tune-epochs", type=int, help="Validation-approved override")
    parser.add_argument("--smoke", action="store_true", help="Mark NOT_VALID_FOR_REPORT")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --run-id after strict config/manifest verification",
    )
    args = parser.parse_args()

    default_manifest, default_dataset, default_output = _default_paths(args.config)
    run_dir = run_training(
        config_path=args.config,
        manifest_path=args.manifest or default_manifest,
        dataset_root=args.dataset_root or default_dataset,
        output_root=args.output_root or default_output,
        run_id=args.run_id,
        model_selection=args.model_selection,
        weights_none=args.weights_none,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
