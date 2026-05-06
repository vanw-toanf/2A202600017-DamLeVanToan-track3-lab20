"""Researcher agent — searches Wikipedia and summarises findings."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research assistant. Given a set of Wikipedia excerpts, write concise \
research notes (300-500 words) that:
- Summarise the key facts relevant to the user's query.
- Cite each source by its title in square brackets, e.g. [GraphRAG].
- Stay factual; do not add opinions.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""
        with trace_span("researcher", {"query": state.request.query}) as span:
            # 1. Search
            sources = self._search.search(
                state.request.query, max_results=state.request.max_sources
            )
            state.sources = sources

            if not sources:
                logger.warning("ResearcherAgent: no sources found, using fallback note.")
                state.research_notes = (
                    "No Wikipedia sources found for this query. "
                    "Answer will rely on model's internal knowledge."
                )
                state.agent_results.append(
                    AgentResult(agent=AgentName.RESEARCHER, content=state.research_notes)
                )
                span["attributes"]["sources_found"] = 0
                return state

            # 2. Build context for LLM
            context = "\n\n".join(
                f"[{s.title}] ({s.url})\n{s.snippet}" for s in sources
            )
            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Wikipedia excerpts:\n{context}"
            )

            # 3. LLM summarise
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content

            # 4. Record
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "sources_count": len(sources),
                    },
                )
            )
            span["attributes"].update({
                "sources_found": len(sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            })

        state.add_trace_event("researcher_done", {"notes_len": len(state.research_notes or "")})
        return state
