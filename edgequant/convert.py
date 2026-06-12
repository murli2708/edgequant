"""Convert an FP32 model into lower-precision variants.

This is a backend-agnostic skeleton. Implement one path at a time (ONNX Runtime
dynamic/static quantization is the gentlest place to start).
"""
from __future__ import annotations

from pathlib import Path

from .types import Precision


def convert(model_path: str | Path, precision: Precision, out_dir: str | Path) -> Path:
    """Produce a quantized artifact and return its path.

    TODO (Phase 2):
      - FP16:  onnxruntime float16 conversion, or torch .half().
      - INT8:  dynamic quantization first; then static (needs a calibration set).
      - INT4:  weight-only group quantization (llama.cpp / bitsandbytes style).
      - Keep sensitive layers (first/last, embeddings) in higher precision.
    """
    raise NotImplementedError(f"TODO: implement {precision.value} conversion")
