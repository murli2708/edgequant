"""Render a list of RunResults as a Markdown (or HTML) decision table.

The report is the product — keep it readable enough for a non-ML stakeholder.
"""
from __future__ import annotations

from typing import Iterable

from .types import RunResult

_HEADERS = ["precision", "size_mb", "latency_p50_ms", "latency_p95_ms", "accuracy"]


def to_markdown(results: Iterable[RunResult]) -> str:
    rows = [r.as_row() for r in results]
    lines = ["| " + " | ".join(_HEADERS) + " |",
             "| " + " | ".join("---" for _ in _HEADERS) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in _HEADERS) + " |")
    return "\n".join(lines) + "\n"


def to_html(results: Iterable[RunResult]) -> str:
    rows = [r.as_row() for r in results]
    head = "".join(f"<th>{h}</th>" for h in _HEADERS)
    body = "".join(
        "<tr>" + "".join(f"<td>{row[h]}</td>" for h in _HEADERS) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
