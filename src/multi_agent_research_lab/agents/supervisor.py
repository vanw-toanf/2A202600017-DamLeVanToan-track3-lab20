"""Supervisor / router — decides which worker runs next."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# Routing order — each step runs exactly once in sequence.
_ROUTE_SEQUENCE = ["researcher", "analyst", "writer"]


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update ``state.route_history`` with the next route.

        Routing policy:
        - researcher  → if research_notes is missing
        - analyst     → if analysis_notes is missing
        - writer      → if final_answer is missing
        - done        → if all fields are filled OR max_iterations exceeded
        - done        → fallback on unexpected state
        """
        settings = get_settings()
        with trace_span("supervisor", {"iteration": state.iteration}) as span:
            if state.iteration >= settings.max_iterations:
                logger.warning(
                    "SupervisorAgent: max_iterations=%d reached, forcing 'done'.",
                    settings.max_iterations,
                )
                route = "done"
            elif state.research_notes is None:
                route = "researcher"
            elif state.analysis_notes is None:
                route = "analyst"
            elif state.final_answer is None:
                route = "writer"
            else:
                route = "done"

            state.record_route(route)
            span["attributes"]["route"] = route
            logger.info("SupervisorAgent: iteration=%d → route=%s", state.iteration, route)

        state.add_trace_event("supervisor_route", {"route": route, "iteration": state.iteration})
        return state
