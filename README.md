# EdgeQuant — a quantization & benchmarking harness

![Phase](https://img.shields.io/badge/Phase%202-The%20Diet-7C3AED)
![Status](https://img.shields.io/badge/status-scaffold%20%C2%B7%20building%20in%20public-orange)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Backends](https://img.shields.io/badge/backends-ONNX%20%C2%B7%20CoreML%20%C2%B7%20LiteRT%20%C2%B7%20llama.cpp-005CED)
![License](https://img.shields.io/badge/license-MIT-green)

> **Phase 2 · The Diet.** *How much can I shrink a model before it stops being useful — and how do I prove it?*

> **Status — honest:** scaffold + engineering plan. The KPI table is the *bar this harness
> holds itself to*, not a results claim — numbers get filled in as real models run through it.

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

## Key design decisions & tradeoffs

- **The report is the product.** EdgeQuant doesn't just quantize — it emits a *decision table* (size × latency × accuracy, per backend, per device class) so the output is a stakeholder-readable artifact, not a throwaway script.
- **Per-channel + entropy calibration over naive per-tensor.** Per-channel weight quantization is what keeps INT8 accuracy loss under 1% on convs/linears; the extra scale vector is cheap insurance.
- **Always report p95, never just the mean.** A good p50 hides jank; tail latency is what users actually feel. Every latency row carries p50 **and** p95.
- **One command regenerates everything.** Reproducibility is a first-class KPI — a benchmark you can't re-run is an anecdote, not evidence.

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

## Where this fits

Part of the **[Edge AI Architect roadmap](https://github.com/murli2708/edge-ai-roadmap)** — a 6-month, 6-project build-in-public series.

**Phase 2 of 6** · ← prev **[microtorch](https://github.com/murli2708/microtorch)** · next → **[ZeroCopyVision](https://github.com/murli2708/zerocopyvision)** (Phase 3 · real-time camera sight)

## License

MIT © Murli — see [LICENSE](LICENSE).
