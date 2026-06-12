# EdgeQuant — a quantization & benchmarking harness

> Phase 2 starter. *How much can I shrink a model before it stops being useful — and how do I prove it?*

## Problem

A model that runs in a notebook is not a model that runs on a phone. EdgeQuant takes an
FP32 model and produces a **decision table**: for each precision (FP16 / INT8 / INT4) it
reports size, latency, and accuracy so you can pick the smallest variant that still clears
your quality bar.

## Constraints

- Must run fully offline on a laptop CPU (no cloud quantization services).
- Every number in the report must be reproducible from a single command.
- The report is the product — a Markdown/HTML table a stakeholder can read.

## Architecture

```
   model.fp32  ──▶  convert.py  ──▶  {fp16, int8, int4} artifacts
                                          │
   eval set    ──▶  benchmark.py  ◀───────┘   (size, latency p50/p95, accuracy)
                         │
                         ▼
                    report.py  ──▶  report.md / report.html  (decision table)
```

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m edgequant.benchmark --model path/to/model.onnx --eval path/to/eval
python -m edgequant.report --runs runs/ --out report.md
```

## KPI table (the bar this repo holds itself to)

| Metric                  | Target                                              |
| ----------------------- | --------------------------------------------------- |
| Size reduction (INT8)   | ~4x vs. FP32                                         |
| Accuracy drop (INT8)    | < 1% absolute on the eval set                       |
| Latency improvement     | Reported per backend; measured p50 **and** p95      |
| Reproducibility         | One command regenerates every row                   |

## Failure modes & what I'd change next

- **INT4 falls off a cliff** on some layers — keep sensitive layers (first/last, embeddings)
  in higher precision (mixed precision) instead of quantizing uniformly.
- **Latency is bimodal** — always report p95, not just the mean; a good p50 hides jank.
- **Calibration set too small** → biased INT8 ranges. Use a representative sample.

## License

MIT.
