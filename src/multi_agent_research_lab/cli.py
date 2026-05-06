"""Command-line entrypoint for the lab starter."""

from __future__ import annotations

import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import generate_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import get_trace_file_path
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

_BENCHMARK_QUERY = "What is GraphRAG and how does it work?"

_BASELINE_SYSTEM = (
    "You are a helpful research assistant. Answer the user's question thoroughly "
    "in ~500 words using your knowledge. Cite relevant concepts clearly."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


# ── baseline ───────────────────────────────────────────────────────────────────

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one LLM call, no search)."""
    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    t0 = time.perf_counter()
    llm = LLMClient()
    response = llm.complete(_BASELINE_SYSTEM, query)
    latency = time.perf_counter() - t0

    state.final_answer = response.content
    state.agent_results = []

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(
        f"[dim]Latency: {latency:.2f}s | "
        f"Tokens in: {response.input_tokens} | "
        f"Tokens out: {response.output_tokens} | "
        f"Cost: ${response.cost_usd:.5f}[/dim]"
    )


# ── multi-agent ────────────────────────────────────────────────────────────────

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(f"[dim]Iterations: {result.iteration} | Sources: {len(result.sources)}[/dim]")
    console.print(f"[dim]Trace file: {get_trace_file_path()}[/dim]")


# ── benchmark ──────────────────────────────────────────────────────────────────

@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = _BENCHMARK_QUERY,
) -> None:
    """Run both baseline and multi-agent, compare metrics, write benchmark_report.md."""
    _init()
    console.print(f"\n[bold]Benchmark query:[/bold] {query}\n")

    # --- single-agent baseline runner ---
    def _baseline_runner(q: str) -> ResearchState:
        req = ResearchQuery(query=q)
        st = ResearchState(request=req)
        llm = LLMClient()
        resp = llm.complete(_BASELINE_SYSTEM, q)
        st.final_answer = resp.content
        return st

    # --- multi-agent runner ---
    def _multi_runner(q: str) -> ResearchState:
        wf = MultiAgentWorkflow()
        return wf.run(ResearchState(request=ResearchQuery(query=q)))

    console.print("[yellow]Running single-agent baseline...[/yellow]")
    baseline_state, baseline_metrics = run_benchmark(
        "single-agent", query, _baseline_runner, score_quality=True
    )

    console.print("[yellow]Running multi-agent workflow...[/yellow]")
    multi_state, multi_metrics = run_benchmark(
        "multi-agent", query, _multi_runner, score_quality=True
    )

    # Print comparison table
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Single-Agent", justify="right")
    table.add_column("Multi-Agent", justify="right")

    def _fmt(v: object) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    table.add_row("Latency (s)", _fmt(baseline_metrics.latency_seconds), _fmt(multi_metrics.latency_seconds))
    table.add_row("Est. cost (USD)", f"${baseline_metrics.estimated_cost_usd:.5f}", f"${multi_metrics.estimated_cost_usd:.5f}")
    table.add_row("Quality score", _fmt(baseline_metrics.quality_score), _fmt(multi_metrics.quality_score))
    table.add_row("Notes", baseline_metrics.notes or "", multi_metrics.notes or "")
    console.print(table)

    # Generate markdown report
    report_path = generate_report(
        query=query,
        baseline_state=baseline_state,
        baseline_metrics=baseline_metrics,
        multi_state=multi_state,
        multi_metrics=multi_metrics,
    )
    console.print(f"\n[green]Report written to:[/green] {report_path}")
    console.print(f"[green]Trace file:[/green] {get_trace_file_path()}")


if __name__ == "__main__":
    app()
