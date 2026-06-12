"""Shared data types for the harness."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class RunResult:
    precision: Precision
    size_mb: float
    latency_p50_ms: float
    latency_p95_ms: float
    accuracy: float

    def as_row(self) -> dict:
        return {
            "precision": self.precision.value,
            "size_mb": round(self.size_mb, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "accuracy": round(self.accuracy, 4),
        }
