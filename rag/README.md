# rag/ — terminology retrieval

⬜ Planned — **US-5** in `STORIES.md`.

Ground extracted concepts in standard medical terminologies during the agent's `retrieve` step, so codes are correct (not hallucinated).

## Contents
- `terminologies/` — loaders + embeddings for SNOMED CT, RxNorm, ICD-10-CM, LOINC. ⚠️ **No licensed terminology files committed** — load from the user's local licensed copy at runtime.
- `principles/` — extraction principle notes **authored in our own words** (what to extract, how to map concepts to resources, de-id guidance).

## Planned layout
- `index.py` — embed (bge-small) + build the vector store (LanceDB/Chroma) over concept names/synonyms.
- `retrieve.py` — given a note/concept, return top-k candidate codes (e.g. "lisinopril" → RxNorm `29046`).

## Notes
- Retrieval is keyed by extracted concept text; the agent binds the best code into the FHIR resource.
- Evaluate impact via an ablation: agent with vs without terminology RAG on the held-out set (coding accuracy).
