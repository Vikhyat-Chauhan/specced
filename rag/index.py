"""Build and persist a LanceDB vector index over clinical terminology concepts.

    python -m rag.index [--out rag/index.lancedb] [--model all-MiniLM-L6-v2]

Default data source: the curated KB in data/knowledge.py (ships with the repo,
real codes, no licensing issues). Users can extend the index with their own
SNOMED/RxNorm/ICD-10/LOINC dumps via loaders in rag/terminologies/ — see the
README for the expected schema (name, system, code, resource_type, synonyms).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

_DEFAULT_INDEX = "rag/index.lancedb"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _kb_records() -> list[dict[str, Any]]:
    """Extract all concept records from the curated knowledge base."""
    from data.knowledge import (
        MEDICATIONS, CONDITIONS, LABS, ALLERGENS, RXNORM, ICD10, SNOMED, LOINC,
    )

    records: list[dict[str, Any]] = []

    for m in MEDICATIONS:
        text = " ".join([m.display] + list(m.synonyms))
        records.append({
            "concept": m.display,
            "synonyms": ", ".join(m.synonyms),
            "text": text,
            "code_system": RXNORM,
            "system_label": "RxNorm",
            "code": m.rxnorm,
            "resource_type": "MedicationStatement",
        })

    for c in CONDITIONS:
        text = " ".join([c.display] + list(c.synonyms))
        records.append({
            "concept": c.display,
            "synonyms": ", ".join(c.synonyms),
            "text": text,
            "code_system": ICD10,
            "system_label": "ICD-10",
            "code": c.icd10,
            "resource_type": "Condition",
        })
        records.append({
            "concept": c.display,
            "synonyms": ", ".join(c.synonyms),
            "text": text,
            "code_system": SNOMED,
            "system_label": "SNOMED",
            "code": c.snomed,
            "resource_type": "Condition",
        })

    for lab in LABS:
        text = " ".join([lab.display] + list(lab.synonyms))
        records.append({
            "concept": lab.display,
            "synonyms": ", ".join(lab.synonyms),
            "text": text,
            "code_system": LOINC,
            "system_label": "LOINC",
            "code": lab.loinc,
            "resource_type": "Observation",
        })

    for allergen in ALLERGENS:
        text = " ".join([allergen.substance] + list(allergen.synonyms))
        records.append({
            "concept": allergen.substance,
            "synonyms": ", ".join(allergen.synonyms),
            "text": text,
            "code_system": SNOMED,
            "system_label": "SNOMED",
            "code": "",  # allergens in KB don't carry SNOMED codes yet
            "resource_type": "AllergyIntolerance",
        })

    return records


def build(out_path: str = _DEFAULT_INDEX, model_name: str = _DEFAULT_MODEL) -> str:
    import lancedb
    from sentence_transformers import SentenceTransformer
    import pyarrow as pa

    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    print("Extracting KB records...")
    records = _kb_records()

    # Embed the text field (concept + synonyms)
    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} concepts...")
    vectors = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Add vectors to records
    for rec, vec in zip(records, vectors):
        rec["vector"] = vec.tolist()

    print(f"Writing index to {out_path} ...")
    db = lancedb.connect(out_path)
    if "concepts" in db.list_tables():
        db.drop_table("concepts")
    db.create_table("concepts", data=records)
    print(f"Index ready: {len(records)} entries in '{out_path}/concepts'")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag.index")
    ap.add_argument("--out", default=_DEFAULT_INDEX)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    args = ap.parse_args(argv)
    build(args.out, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
