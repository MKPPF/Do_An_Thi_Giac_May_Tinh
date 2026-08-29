"""Measured, reproducible latency benchmarks for CrackSpot inference."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np

from crackspot.utils.environment import capture_environment


class BenchmarkError(ValueError):
    """Raised when a latency benchmark cannot produce trustworthy evidence."""


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{name} phải là số nguyên")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "không âm" if allow_zero else "dương"
        raise BenchmarkError(f"{name} phải {qualifier}")
    return value


def benchmark_callable(
    operation: Callable[[], Any],
    *,
    warmup_runs: int = 10,
    measured_runs: int = 100,
    target_seconds: float = 5.0,
    environment: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Benchmark one operation, excluding warm-up calls from all statistics.

    ``operation`` must include the complete scope the caller wants to claim.  The
    command-line wrapper uses decode + preprocessing + model inference.  A clock
    can be injected so the measurement and summary logic are unit-testable.
    """

    warmup_count = _positive_integer(warmup_runs, "warmup_runs", allow_zero=True)
    measured_count = _positive_integer(measured_runs, "measured_runs")
    if not callable(operation):
        raise BenchmarkError("operation phải callable")
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise BenchmarkError("target_seconds phải hữu hạn và dương")

    for _ in range(warmup_count):
        operation()

    timings_ms: list[float] = []
    for _ in range(measured_count):
        started = float(clock())
        operation()
        elapsed_ms = (float(clock()) - started) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise BenchmarkError("Clock trả về latency không hợp lệ")
        timings_ms.append(elapsed_ms)

    values = np.asarray(timings_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    median_ms = float(np.median(values))
    p50_ms = float(np.percentile(values, 50))
    p95_ms = float(np.percentile(values, 95))
    target_ms = float(target_seconds * 1000.0)
    captured_environment = dict(environment) if environment is not None else capture_environment()
    return {
        "schema_version": 1,
        "kind": "inference_latency_benchmark",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "warmup_runs": warmup_count,
        "measured_runs": measured_count,
        "latency_unit": "ms_per_image",
        "latency_ms": {
            "raw": timings_ms,
            "mean": mean_ms,
            "median": median_ms,
            "p50": p50_ms,
            "p95": p95_ms,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        },
        "target": {
            "seconds_per_image": float(target_seconds),
            "milliseconds_per_image": target_ms,
            "statistic": "p95",
            "observed_ms": p95_ms,
            "meets_target": bool(p95_ms <= target_ms),
            "all_measured_runs_meet_target": bool(np.max(values) <= target_ms),
        },
        "environment": captured_environment,
    }


def benchmark_inference_service(
    service: Any,
    image: Any,
    *,
    warmup_runs: int = 10,
    measured_runs: int = 100,
    target_seconds: float = 5.0,
    include_gradcam: bool = False,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Benchmark the shared inference service and attach model provenance."""

    if not hasattr(service, "predict_image") or not hasattr(service, "metadata"):
        raise BenchmarkError("service không tuân theo InferenceService contract")

    result = benchmark_callable(
        lambda: service.predict_image(image, include_gradcam=include_gradcam),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        target_seconds=target_seconds,
        environment=environment,
    )
    metadata = service.metadata
    smoke = bool(getattr(metadata, "extra", {}).get("smoke_test", False))
    result.update(
        {
            "run_id": metadata.run_id,
            "model_version": metadata.display_version,
            "checkpoint_sha256": metadata.model_sha256,
            "manifest_sha256": metadata.manifest_sha256,
            "threshold": float(metadata.threshold),
            "include_gradcam": bool(include_gradcam),
            "timing_scope": (
                "decode_preprocess_inference_gradcam"
                if include_gradcam
                else "decode_preprocess_inference"
            ),
            "valid_for_report": not smoke,
            "status": "NOT_VALID_FOR_REPORT" if smoke else "MEASURED",
        }
    )
    return result


__all__ = [
    "BenchmarkError",
    "benchmark_callable",
    "benchmark_inference_service",
]
