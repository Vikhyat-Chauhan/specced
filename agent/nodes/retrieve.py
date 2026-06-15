"""Retrieve node — fetch standard terminology codes relevant to this note.

Calls rag.retrieve.retrieve_hints() when the RAG index is available (US-5).
Falls back gracefully to empty hints so the agent works without the index.
"""

from __future__ import annotations

from agent.state import AgentState


def run(state: AgentState) -> AgentState:
    hints = ""
    try:
        from rag.retrieve import retrieve_hints
        hints = retrieve_hints(state["case"].note, state["case"].target_resources)
        if hints:
            print(f"[retrieve] {len(hints.splitlines())} code hints")
    except Exception:
        pass
    return {**state, "retrieved_hints": hints}
