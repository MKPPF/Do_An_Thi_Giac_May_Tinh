from __future__ import annotations

from types import SimpleNamespace

import pytest

from crackspot.reporting.benchmark import (
    BenchmarkError,
    benchmark_callable,
    benchmark_inference_service,
)


def test_benchmark_separates_warmup_and_computes_all_statistics() -> None:
    calls: list[int] = []
    clock_values = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
    result = benchmark_callable(
        lambda: calls.append(1),
        warmup_runs=2,
        measured_runs=3,
        target_seconds=0.5,
        environment={"device": "test-cpu"},
        clock=lambda: next(clock_values),
    )

    assert len(calls) == 5
    assert result["latency_ms"]["raw"] == pytest.approx([100.0, 200.0, 300.0])
    assert result["latency_ms"]["mean"] == pytest.approx(200.0)
    assert result["latency_ms"]["median"] == pytest.approx(200.0)
    assert result["latency_ms"]["p50"] == pytest.approx(200.0)
    assert result["latency_ms"]["p95"] == pytest.approx(290.0)
    assert result["target"]["meets_target"] is True
    assert result["environment"] == {"device": "test-cpu"}


def test_benchmark_service_marks_smoke_metadata_not_valid_for_report() -> None:
    class FakeService:
        metadata = SimpleNamespace(
            run_id="smoke-run",
            display_version="0.1.0",
            model_sha256="a" * 64,
            manifest_sha256="b" * 64,
            threshold=0.5,
            extra={"smoke_test": True},
        )

        def predict_image(self, image: object, *, include_gradcam: bool) -> None:
            assert image == "image"
            assert include_gradcam is False

    result = benchmark_inference_service(
        FakeService(),
        "image",
        warmup_runs=0,
        measured_runs=1,
        environment={"device": "fake"},
    )
    assert result["valid_for_report"] is False
    assert result["status"] == "NOT_VALID_FOR_REPORT"
    assert result["timing_scope"] == "decode_preprocess_inference"


@pytest.mark.parametrize(
    ("warmup", "measured", "target"),
    [(-1, 1, 5.0), (0, 0, 5.0), (0, 1, 0.0)],
)
def test_benchmark_rejects_invalid_counts_and_target(
    warmup: int, measured: int, target: float
) -> None:
    with pytest.raises(BenchmarkError):
        benchmark_callable(
            lambda: None,
            warmup_runs=warmup,
            measured_runs=measured,
            target_seconds=target,
            environment={},
        )
