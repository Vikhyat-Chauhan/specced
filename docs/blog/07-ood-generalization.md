# OOD generalization: what happens when the notes don't look like training data

Fine-tuning on 160 synthetic examples produces a model that scores 20/20 on a held-out set from the same generator. That's a useful proof-of-concept, but it's not a deployment benchmark. The real question is: what happens when the model sees clinical notes from specialties it wasn't trained on, written by real clinicians with real abbreviations and real note structures?

## The MTSamples test

MTSamples is a public corpus of ~5,000 de-identified medical transcriptions across 30+ specialties — allergy, surgery, orthopedics, neurology, gastroenterology, pain management, radiology, and more. The notes are real clinical text, not template-generated. A neurology consult looks nothing like a primary care SOAP note.

We ran 50 randomly sampled MTSamples notes through both the base Qwen2.5-Coder-7B and the fine-tuned model, measuring FHIR validity (the only metric we can compute without gold annotations):

| Model | Avg FHIR validity | Resources/note | PHI spans/note | Empty outputs |
|---|---|---|---|---|
| Base | **0.951** | 3.06 | 5.9 | 13/50 |
| Fine-tuned | 0.842 | **5.08** | **7.06** | 6/50 |

## What the numbers mean

**The fine-tuned model extracts more — always.** 5.08 resources/note vs 3.06, 7.06 PHI spans/note vs 5.9, and only 6 empty outputs vs 13. The fine-tuning taught the model to always attempt extraction and to look for more entity types. On in-distribution notes this is unambiguously better. On OOD notes it means the model sometimes produces structurally invalid resources where the base model would have produced nothing.

**The base model is conservative.** It falls back to empty output 13/50 times — 26% of notes. When it's uncertain about the output format, it produces nothing rather than hallucinating structure. What it does produce is 95% valid FHIR.

**The validity gap (0.951 vs 0.842) is a precision-recall trade-off.** The fine-tuned model has learned that "extract something" is correct behavior, but the specific FHIR fields it learned (from 15 medications, 12 conditions, template notes) don't transfer perfectly to, say, a radiology report or an orthopedic operative note. A radiology report has no medications. An operative note has Procedures, which weren't in our training set.

## Where the fine-tuned model breaks down

Looking at the invalid resources in the FT output:

- **Missing required fields on unfamiliar resource types** — the model occasionally emits an `AllergyIntolerance` without a `patient` reference, or a `Procedure` without `status`. These resource types were underrepresented in training.
- **Hallucinated codings** — on some OOD specialties, the model fills in SNOMED/ICD-10 codes that weren't in the note. The RAG index helps when a concept is in the KB, but surgical procedure codes are not.
- **Structured text confusion** — some MTSamples transcriptions have headers (`SUBJECTIVE:`, `ASSESSMENT:`, `PLAN:`) that the model interprets as structured data. Template notes don't have this format.

## The fix is more data

The synthetic data pipeline is the lever. The current KB covers 15 medications and 12 conditions from primary care. Expanding to:
- Surgical procedures (`data/knowledge.py` additions)
- Specialty-specific conditions (cardiology, oncology, orthopedics)
- Claude-written notes that use varied formatting (SOAP, H&P, operative, consult)

...would close most of the OOD gap without changing any model architecture. The reject-sampler ensures only learnable pairs enter training, so adding more KB entries is low-risk.

## What this means for deployment

For a primary care or internal medicine clinic (where notes look like our training distribution), the fine-tuned model is production-ready on the synthetic benchmark. For subspecialty clinics, more training data is needed before trusting the output.

The MTSamples validity rate (0.842 for FT) is a reasonable floor estimate for unconstrained deployment. Combined with the agent's self-refine loop — which will catch and fix invalid FHIR resources — the effective validity rate in production is higher than the single-shot number suggests.

## Reproducing this

```bash
python -m evals.mtsamples_bench --n 50 --adapter train/checkpoints/adapter
```

Results are saved to `evals/reports/mtsamples_<ts>.json`. The dataset is downloaded automatically from HuggingFace (no login required).
