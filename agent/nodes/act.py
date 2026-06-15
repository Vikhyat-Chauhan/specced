"""Act node — run the fine-tuned model to extract FHIR + PHI spans.

On refine iterations (refine_count > 0) the previous eval's error_context is
appended to the user message so the model sees what it got wrong and can fix it.

The HFBackend is kept as a module-level singleton loaded once per graph run
(via load_backend / unload_backend called from agent.run) to avoid reloading
4-bit weights on every self-refine iteration.
"""

from __future__ import annotations

import os
from typing import Optional, Any

from agent.state import AgentState

_DEFAULT_ADAPTER = os.environ.get("SPECCED_ADAPTER", "train/checkpoints/adapter")

# Singleton — loaded once by agent.run, reused across refine iterations.
_backend: Any = None


def load_backend(adapter_path: Optional[str] = None) -> Any:
    from serve.client import HFBackend
    global _backend
    path = adapter_path if adapter_path and os.path.exists(adapter_path) else None
    _backend = HFBackend(path)
    _backend.load()
    return _backend


def unload_backend() -> None:
    global _backend
    if _backend is not None:
        _backend.unload()
        _backend = None


def _build_prompt(
    note: str,
    target_resources: list[str],
    hints: str,
    error_ctx: str,
) -> str:
    from train.prompt import SYSTEM_PROMPT
    types = ", ".join(target_resources) if target_resources else (
        "Condition, MedicationStatement, Observation, AllergyIntolerance"
    )
    user_msg = f"Clinical note:\n{note}\n\nTarget resource types: {types}"
    if hints:
        user_msg += f"\n\nRelevant standard codes — use these where they match:\n{hints}"
    if error_ctx:
        user_msg += (
            f"\n\nYour previous attempt had these issues — please fix them:\n{error_ctx}"
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run(state: AgentState) -> AgentState:
    if _backend is None:
        load_backend(_DEFAULT_ADAPTER)

    case = state["case"]
    iteration = state["refine_count"]
    prompt = _build_prompt(
        case.note,
        case.target_resources,
        state["retrieved_hints"],
        state["error_context"] if iteration > 0 else "",
    )

    label = "refine" if iteration > 0 else "extract"
    print(f"[act/{label}] iteration={iteration+1}")

    result = _backend.extract(
        case.note,
        case.target_resources,
        prompt_override=prompt,
    )
    print(f"[act] {result.tokens_used} tokens | {result.latency_ms:.0f}ms | "
          f"{len(result.prediction.resources)} resources | {len(result.prediction.phi_spans)} PHI spans")

    return {**state, "prediction": result.prediction}
