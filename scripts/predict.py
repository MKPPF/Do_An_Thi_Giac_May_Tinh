"""Predict one image with the same service used by Streamlit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from crackspot.inference import ImageValidationError, InferenceError, InferenceService
    from crackspot.modeling.gradcam import GradCAMError
except ModuleNotFoundError:  # Support `python scripts/predict.py` before editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crackspot.inference import ImageValidationError, InferenceError, InferenceService
    from crackspot.modeling.gradcam import GradCAMError


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    default_model = os.environ.get(
        "CRACKSPOT_MODEL_PATH", str(project_root / "models" / "crackspot.keras")
    )
    parser = argparse.ArgumentParser(
        description="Phân loại Crack/Non-crack bằng checkpoint CrackSpot đã khóa."
    )
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh JPEG hoặc PNG")
    parser.add_argument("--model", type=Path, default=Path(default_model), help="Full model .keras")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(os.environ["CRACKSPOT_METADATA_PATH"])
        if os.environ.get("CRACKSPOT_METADATA_PATH")
        else None,
        help="Metadata JSON; nếu bỏ trống sẽ dò sidecar cạnh model",
    )
    parser.add_argument(
        "--overlay-output",
        type=Path,
        help="Tùy chọn: tạo Grad-CAM và lưu overlay PNG tại đây",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Bỏ kiểm tra model_sha256 (chỉ dùng khi debug)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Cho phép ghi đè tệp overlay đã tồn tại",
    )
    parser.add_argument("--json", action="store_true", help="Chỉ in kết quả JSON")
    return parser


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run(args: argparse.Namespace) -> dict[str, object]:
    service = InferenceService.from_files(
        args.model,
        args.metadata,
        verify_hash=not args.skip_hash_check,
    )
    include_gradcam = args.overlay_output is not None
    if args.overlay_output is not None and args.overlay_output.exists() and not args.overwrite:
        raise InferenceError(
            f"Tệp overlay đã tồn tại: {args.overlay_output}. Dùng --overwrite nếu muốn ghi đè."
        )
    result = service.predict_image(args.image, include_gradcam=include_gradcam)
    if args.overlay_output is not None:
        if result.overlay is None:
            raise InferenceError("Grad-CAM không trả về overlay.")
        output = args.overlay_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.overlay.save(output, format="PNG")
    payload = result.to_dict()
    if args.overlay_output is not None:
        payload["overlay_path"] = str(args.overlay_output.expanduser().resolve())
    return payload


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run(args)
    except (ImageValidationError, InferenceError, GradCAMError, OSError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Kết quả: {payload['display_label_vi']}")
        print(f"P(Crack): {float(payload['crack_probability']):.6f}")
        print(f"Ngưỡng: {float(payload['threshold']):.6f}")
        print(f"Thời gian suy luận: {float(payload['latency_ms']):.2f} ms")
        print(f"Model: {payload['model_version']} | Run: {payload['run_id']}")
        if "overlay_path" in payload:
            print(f"Grad-CAM overlay: {payload['overlay_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
