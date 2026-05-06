"""Benchmark skeleton — measures latency, cost, quality for single vs multi-agent."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _total_cost(state: ResearchState) -> float:
    """Sum cost_usd across all agent results."""
    total = 0.0
    for ar in state.agent_results:
        total += ar.metadata.get("cost_usd") or 0.0
    return total


def _citation_coverage(state: ResearchState) -> float:
    """Fraction of available sources cited in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0
    cited = sum(1 for s in state.sources if s.title in (state.final_answer or ""))
    return round(cited / len(state.sources), 2)


def _llm_quality_score(query: str, answer: str) -> float:
    """Use gpt-4o-mini as judge — returns 0.0-10.0."""
    from multi_agent_research_lab.services.llm_client import LLMClient

    system = (
        "You are a strict quality judge. Rate the following answer to the given query "
        "on a scale of 0 to 10 (10 = perfect). Respond with ONLY a single number, e.g. 7.5."
    )
    user = f"Query: {query}\n\nAnswer:\n{answer}"
    try:
        client = LLMClient()
        response = client.complete(system, user)
        score = float(response.content.strip().split()[0])
        return max(0.0, min(10.0, score))
    except Exception as exc:
        logger.warning("Quality scoring failed: %s", exc)
        return 0.0


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    *,
    score_quality: bool = True,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, citation coverage, and quality score."""
    started = perf_counter()
    state = runner(query)
    latency = round(perf_counter() - started, 3)

    estimated_cost = _total_cost(state)
    citation_cov = _citation_coverage(state)

    quality: float | None = None
    if score_quality and state.final_answer:
        quality = _llm_quality_score(query, state.final_answer)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost,
        quality_score=quality,
        notes=(
            f"citation_coverage={citation_cov:.0%} | "
            f"sources={len(state.sources)} | "
            f"iterations={state.iteration}"
        ),
    )
    logger.info(
        "Benchmark [%s] | latency=%.2fs cost=$%.5f quality=%s citation=%s",
        run_name, latency, estimated_cost, quality, citation_cov,
    )
    return state, metrics
