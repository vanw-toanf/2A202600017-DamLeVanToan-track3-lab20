"""Analyst agent — turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an analytical assistant. Given research notes, produce structured analysis \
(200-350 words) that:
- Extracts 3-5 key claims or findings as a numbered list.
- Identifies any conflicting viewpoints or gaps in evidence.
- Flags claims that are weakly supported (mark with ⚠).
- Does NOT repeat full source text — only distil insights.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``."""
        with trace_span("analyst", {"notes_len": len(state.research_notes or "")}) as span:
            notes = state.research_notes or "(no research notes available)"

            user_prompt = (
                f"Original query: {state.request.query}\n\n"
                f"Research notes:\n{notes}"
            )
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            span["attributes"].update({
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            })

        state.add_trace_event("analyst_done", {"analysis_len": len(state.analysis_notes or "")})
        return state
