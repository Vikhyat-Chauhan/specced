"""Retrieve standard terminology codes relevant to a clinical note.

    from rag.retrieve import retrieve_hints
    hints = retrieve_hints(note, ["Condition", "MedicationStatement"])
    # → "- lisinopril → RxNorm 29046\n- hypertension → ICD-10 I10\n..."

The index is loaded lazily on first call and cached for the process lifetime.
Build the index first: `python -m rag.index`

retrieve_hints() returns an empty string if the index doesn't exist, so the
agent degrades gracefully without RAG.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Any

_INDEX_PATH = "rag/index.lancedb"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_TOP_K = 8

_db: Any = None
_table: Any = None
_embedder: Any = None


def _load_index() -> bool:
    global _db, _table, _embedder
    if _table is not None:
        return True
    if not Path(_INDEX_PATH).exists():
        return False
    try:
        import lancedb
        from sentence_transformers import SentenceTransformer
        _db = lancedb.connect(_INDEX_PATH)
        _table = _db.open_table("concepts")
        _embedder = SentenceTransformer(_EMBED_MODEL)
        return True
    except Exception:
        return False


def _extract_query_terms(note: str) -> str:
    """Use the note text directly as the query; limit length for efficiency."""
    return note[:512]


def retrieve_hints(note: str, target_resources: Optional[list[str]] = None) -> str:
    """Return a formatted string of top code hints for the given note.

    Returns empty string if the index is unavailable.
    """
    if not _load_index():
        return ""

    query_text = _extract_query_terms(note)
    query_vec = _embedder.encode([query_text], convert_to_numpy=True)[0]

    results = _table.search(query_vec).limit(_TOP_K).to_list()

    # Filter by target resource types if specified
    if target_resources:
        results = [r for r in results if r.get("resource_type") in target_resources]

    if not results:
        return ""

    lines = []
    seen: set[str] = set()
    for r in results:
        if not r.get("code"):
            continue
        key = f"{r['concept']}:{r['code']}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- {r['concept']} → {r['system_label']} {r['code']} "
            f"(system: {r['code_system']}, for {r['resource_type']})"
        )

    return "\n".join(lines)


def search(query: str, top_k: int = _TOP_K) -> list[dict]:
    """Low-level search — returns raw result dicts. Useful for debugging."""
    if not _load_index():
        return []
    vec = _embedder.encode([query], convert_to_numpy=True)[0]
    return _table.search(vec).limit(top_k).to_list()
