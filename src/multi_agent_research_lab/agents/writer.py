"""Writer agent — synthesises a final answer with citations."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical writer. Using the provided research notes and analytical insights, \
write a clear, well-structured answer (~500 words) for the target audience: {audience}.

Requirements:
- Start with a one-sentence direct answer.
- Use markdown headings (##) to organise sections.
- Cite sources inline using [Title] notation.
- End with a "## References" section listing all cited titles and URLs.
- Do not fabricate facts beyond the provided notes.
"""


class WriterAgent(BaseAgent):
    """Synthesises a clear response with citations."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer``."""
        with trace_span("writer") as span:
            research = state.research_notes or "(no research notes)"
            analysis = state.analysis_notes or "(no analysis notes)"

            sources_block = "\n".join(
                f"- [{s.title}]({s.url})" for s in state.sources
            )

            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{research}\n\n"
                f"Analysis:\n{analysis}\n\n"
                f"Available sources:\n{sources_block}"
            )

            system = _SYSTEM_PROMPT.format(audience=state.request.audience)
            response = self._llm.complete(system, user_prompt)
            state.final_answer = response.content

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
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

        state.add_trace_event("writer_done", {"answer_len": len(state.final_answer or "")})
        return state
