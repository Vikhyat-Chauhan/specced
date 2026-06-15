"""LangGraph state machine: plan → retrieve → act → evaluate + self-refine.

    graph = build_graph()
    result = graph.invoke(initial_state(case, max_refines=3))

Conditional edge out of evaluate: if done → END, else → act (refine loop).
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import plan, retrieve, act, evaluate


def _router(state: AgentState) -> str:
    return END if state["done"] else "act"


def build_graph() -> "CompiledGraph":
    g = StateGraph(AgentState)

    g.add_node("plan", plan.run)
    g.add_node("retrieve", retrieve.run)
    g.add_node("act", act.run)
    g.add_node("evaluate", evaluate.run)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "act")
    g.add_edge("act", "evaluate")
    g.add_conditional_edges("evaluate", _router, {"act": "act", END: END})

    return g.compile()
