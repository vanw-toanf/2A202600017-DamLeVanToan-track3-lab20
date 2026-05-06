"""Tracing hooks — JSON file trace (no external provider needed).

Each span is written to reports/trace_<session_id>.jsonl.
Students can replace or augment with LangSmith/Langfuse provider spans.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# One session ID per process run — all spans share the same trace file.
_SESSION_ID = uuid.uuid4().hex[:8]
_REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports"


def _trace_file() -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return _REPORTS_DIR / f"trace_{_SESSION_ID}.jsonl"


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    Writes a JSON line to ``reports/trace_<session>.jsonl`` on exit.
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "session_id": _SESSION_ID,
        "name": name,
        "attributes": attributes or {},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": None,
    }
    try:
        yield span
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        _append_span(span)
        logger.debug("trace_span | %s | %.3fs", name, span["duration_seconds"])


def _append_span(span: dict[str, Any]) -> None:
    try:
        with _trace_file().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(span) + "\n")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not write trace span: %s", exc)


def get_trace_file_path() -> str:
    """Return the path of the current session trace file."""
    return str(_trace_file())
