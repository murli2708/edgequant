"""Measure size, latency (p50/p95), and accuracy for each precision variant."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .types import Precision, RunResult


def measure_latency(run_once, *, warmup: int = 5, iters: int = 50) -> tuple[float, float]:
    """Return (p50_ms, p95_ms). `run_once` is a zero-arg callable doing one inference."""
    for _ in range(warmup):
        run_once()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def benchmark(model_path: str | Path, eval_set, precision: Precision) -> RunResult:
    """Load the variant, run the eval set, and collect all KPIs.

    TODO (Phase 2): load the artifact for `precision`, build a `run_once` closure,
    and compute accuracy on `eval_set`.
    """
    raise NotImplementedError("TODO: wire up a backend, then return a RunResult")
