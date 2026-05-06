"""LangGraph workflow — wires Supervisor → Researcher → Analyst → Writer."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# ── node wrappers ──────────────────────────────────────────────────────────────
# LangGraph passes state as dict; we convert to/from ResearchState for type safety.


def _as_dict(state: ResearchState) -> dict:  # type: ignore[type-arg]
    return state.model_dump()


def _from_dict(data: dict) -> ResearchState:  # type: ignore[type-arg]
    return ResearchState.model_validate(data)


_supervisor = SupervisorAgent()
_researcher = ResearcherAgent()
_analyst = AnalystAgent()
_writer = WriterAgent()


def _supervisor_node(data: dict) -> dict:  # type: ignore[type-arg]
    state = _from_dict(data)
    state = _supervisor.run(state)
    return _as_dict(state)


def _researcher_node(data: dict) -> dict:  # type: ignore[type-arg]
    state = _from_dict(data)
    state = _researcher.run(state)
    return _as_dict(state)


def _analyst_node(data: dict) -> dict:  # type: ignore[type-arg]
    state = _from_dict(data)
    state = _analyst.run(state)
    return _as_dict(state)


def _writer_node(data: dict) -> dict:  # type: ignore[type-arg]
    state = _from_dict(data)
    state = _writer.run(state)
    return _as_dict(state)


def _route_decision(data: dict) -> str:  # type: ignore[type-arg]
    """Read the last route set by Supervisor and return the next node name."""
    route_history: list[str] = data.get("route_history", [])
    if not route_history:
        return "researcher"
    last = route_history[-1]
    if last == "done":
        return END
    return last  # "researcher" | "analyst" | "writer"


# ── graph builder ──────────────────────────────────────────────────────────────

class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph."""

    def build(self) -> object:
        """Create and compile the LangGraph StateGraph."""
        # Use plain dict as state type (LangGraph-compatible)
        graph = StateGraph(dict)

        # Nodes
        graph.add_node("supervisor", _supervisor_node)
        graph.add_node("researcher", _researcher_node)
        graph.add_node("analyst", _analyst_node)
        graph.add_node("writer", _writer_node)

        # Entry point
        graph.set_entry_point("supervisor")

        # After supervisor: conditional routing
        graph.add_conditional_edges(
            "supervisor",
            _route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )

        # After each worker → back to supervisor
        for worker in ("researcher", "analyst", "writer"):
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile graph, invoke with initial state, return final ResearchState."""
        compiled = self.build()
        initial = _as_dict(state)
        logger.info("MultiAgentWorkflow.run | query=%r", state.request.query)
        result_dict = compiled.invoke(initial)
        return _from_dict(result_dict)
