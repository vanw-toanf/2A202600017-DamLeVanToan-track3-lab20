"""Report generator — writes reports/benchmark_report.md."""

from __future__ import annotations


from datetime import datetime, timezone
from pathlib import Path

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import get_trace_file_path


def generate_report(
    query: str,
    baseline_state: ResearchState,
    baseline_metrics: BenchmarkMetrics,
    multi_state: ResearchState,
    multi_metrics: BenchmarkMetrics,
) -> str:
    """Build markdown report and write to reports/benchmark_report.md.

    Returns the path of the generated file.
    """
    reports_dir = Path(__file__).resolve().parents[3] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "benchmark_report.md"

    def _fmt(v: float | None, fmt: str = ".4f") -> str:
        return f"{v:{fmt}}" if v is not None else "N/A"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    trace_path = get_trace_file_path()

    md = f"""# Benchmark Report — Multi-Agent Research System

**Generated:** {now}
**Query:** `{query}`
**Model:** gpt-4o-mini
**Search provider:** Wikipedia API (free, no key)

---

## Results Summary

| Metric | Single-Agent Baseline | Multi-Agent System |
|---|---|---|
| Latency (s) | {_fmt(baseline_metrics.latency_seconds, '.2f')} | {_fmt(multi_metrics.latency_seconds, '.2f')} |
| Estimated cost (USD) | ${_fmt(baseline_metrics.estimated_cost_usd, '.5f')} | ${_fmt(multi_metrics.estimated_cost_usd, '.5f')} |
| Quality score (0-10) | {_fmt(baseline_metrics.quality_score, '.1f')} | {_fmt(multi_metrics.quality_score, '.1f')} |
| Notes | {baseline_metrics.notes} | {multi_metrics.notes} |

---

## Single-Agent Baseline Answer

{baseline_state.final_answer or "_No answer generated._"}

---

## Multi-Agent Answer

{multi_state.final_answer or "_No answer generated._"}

---

## Agent Trace (Multi-Agent)

| Step | Agent | Tokens In | Tokens Out | Cost (USD) |
|---|---|---|---|---|
"""

    for ar in multi_state.agent_results:
        tin = ar.metadata.get("input_tokens", "—")
        tout = ar.metadata.get("output_tokens", "—")
        cost = ar.metadata.get("cost_usd")
        cost_str = f"${cost:.5f}" if cost else "—"
        md += f"| {ar.agent} | {ar.agent} | {tin} | {tout} | {cost_str} |\n"

    md += f"""
---

## Failure Mode Analysis

### Observed failure modes
- **Wikipedia search miss**: if the query is too specific or niche, Wikipedia opensearch returns 0 results.
  *Fix*: fall back to model's internal knowledge + flag in research notes.
- **Max-iterations cap**: supervisor enforces `MAX_ITERATIONS` to prevent infinite loops.
  *Fix*: increase limit or add retry logic in individual agents.
- **LLM citation hallucination**: writer may cite sources not in the provided list.
  *Fix*: post-process final_answer to validate citations against `state.sources`.

### When to use multi-agent
- Complex, multi-step research tasks that benefit from specialised roles.
- When traceability / auditability of each step is important.

### When NOT to use multi-agent
- Simple factual queries answerable in one LLM call (adds latency & cost for no gain).
- Tight latency budgets where the orchestration overhead matters.

---

## Trace File

Full JSON-lines trace: `{trace_path}`
"""

    report_path.write_text(md, encoding="utf-8")
    return str(report_path)


def render_markdown_report(metrics_list: list[BenchmarkMetrics]) -> str:  # noqa: E501
    """Render a simple markdown string from a list of BenchmarkMetrics.

    Thin helper used by tests and ad-hoc display without needing full state objects.
    """
    lines = ["# Benchmark Report\n"]
    lines.append("| Run | Latency (s) | Cost (USD) | Quality |")
    lines.append("|---|---|---|---|")
    for m in metrics_list:
        lines.append(
            f"| {m.run_name} "
            f"| {m.latency_seconds:.2f} "
            f"| {f'${m.estimated_cost_usd:.5f}' if m.estimated_cost_usd is not None else 'N/A'} "
            f"| {f'{m.quality_score:.1f}' if m.quality_score is not None else 'N/A'} |"
        )
    return "\n".join(lines)
