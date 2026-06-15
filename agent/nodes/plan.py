"""Plan node — validates the case and logs the extraction outline.

No LLM call: planning is implicit in the act prompt. This node is a natural
extension point if we later want to break the note into sections or dynamically
determine target resource types before retrieval.
"""

from __future__ import annotations

from agent.state import AgentState


def run(state: AgentState) -> AgentState:
    case = state["case"]
    print(
        f"[plan] case={case.id} | note={len(case.note)} chars "
        f"| targets={case.target_resources} | deidentify={case.deidentify}"
    )
    return state
