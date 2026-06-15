# Local vs Claude: cost, latency, privacy

The default question in AI engineering is "which hosted API should I use?" For clinical NLP, the question is different: **can you use a hosted API at all?**

## The privacy constraint

Clinical notes contain PHI. Sending them to a hosted API — even one with a BAA — introduces risk:
- Network transmission is an attack surface
- The provider's logging and retention policies become your liability
- Regulatory frameworks (HIPAA, GDPR, NHS DSP Toolkit) may require data to stay on-premises

Specced's design answer: the extraction model runs locally. PHI never leaves the machine that holds the clinical data. Claude (Anthropic's API) is used only for *data generation* — synthetic notes with Faker-generated PHI, not real patient data — and is optional (the offline path runs without any API key).

## The three roles of Claude in Specced

1. **Note-writer** (`data/teacher.py:note_via_claude`) — Sonnet 4.6 writes realistic clinical prose from structured facts. Used once during data generation, not at inference time.

2. **Teacher / reject-sampler** (`data/teacher.py:extract_via_claude`) — Sonnet 4.6 extracts FHIR from generated notes to filter out unleachable pairs. Also used during data generation only.

3. **Clinical judge** (`evals/judges/clinical.py`) — Opus 4.8 reviews extractions for clinical plausibility (hallucinated meds, wrong doses, etc.). Optional, used for benchmarking.

None of these touch real patient data.

## Performance comparison

On the 20 held-out synthetic cases:

| System | Passed | F1 | De-id | Latency | Cost/note |
|---|---|---|---|---|---|
| Base Qwen2.5-7B | 0/20 | 0.416 | 0.850 | 29.8 s | $0 |
| Fine-tuned (local) | 20/20 | 0.992 | 1.000 | 22.7 s | $0 |
| FT + Agent (local) | 20/20 | 1.000 | 1.000 | 23.2 s | $0 |

For reference: Claude Sonnet 4.6 at ~$3/M input + $15/M output tokens, with a ~1500-token prompt and ~1500-token output, costs about **$0.027/note**. At 1000 notes/day that's $27/day or ~$800/month — before volume discounts.

The fine-tuned local model: **$0/note** (electricity included in operating cost), **22.7 s/note** on a consumer GPU.

## Latency

22–30 seconds per note is too slow for real-time scribing (a clinician finishing a note and waiting half a minute is untenable). It's acceptable for:
- Batch processing (overnight de-id + coding of the day's notes)
- Async workflows where the note is submitted and the result retrieved later
- Revenue-cycle automation where latency is measured in hours, not seconds

For lower-latency requirements:
- **vLLM** for batched inference (throughput, not latency) — included in the project plan
- **GGUF Q4_K_M on Ollama** — slightly faster than direct HF inference for single-note requests
- **Smaller model** (3B) — loses some F1 but fits in fewer GB and runs faster

## The real trade-off

The genuine trade-off isn't cost or latency — it's **capability on out-of-distribution inputs**.

A fine-tuned 7B model trained on 160 synthetic examples works well on notes that look like the training distribution (the template format, the KB's 15 medications and 12 conditions). A real hospital note from a subspecialty clinic — with 40 conditions, unusual abbreviations, and medication regimens not in the KB — will likely degrade the fine-tuned model more than it degrades Claude.

Claude has seen orders of magnitude more clinical text in pre-training. The local model's advantage is privacy and zero marginal cost; Claude's advantage is robustness to novel inputs.

The right production design: **local model as the primary extractor, Claude as the fallback for low-confidence cases** (e.g., when the local model's eval score is below a threshold). This gives the privacy and cost benefits of local inference while using Claude as a safety net — and Claude still never sees the PHI (the fallback can route to a de-identified version of the note, or the local model can be used for de-id first).

## What the synthetic benchmark doesn't measure

The held-out set is drawn from the same generator as the training set. This means the results (20/20 passing, F1=1.0 for FT+Agent) are optimistic for production:
- The vocabulary of medications and conditions is fixed to the 15+12 in `data/knowledge.py`
- Note length and style are controlled (template or Claude-written from the same structured facts)
- PHI is synthetic (Faker)

## OOD test: MTSamples

To measure real-world generalization, we ran both models on 50 randomly sampled transcriptions from [MTSamples](https://huggingface.co/datasets/NickyNicky/medical_mtsamples) — 4,999 real de-identified clinical notes across 30+ specialties (allergy, surgery, orthopedics, neurology, gastroenterology, etc.). No gold annotations, so only FHIR validity can be measured.

| Model | Avg FHIR validity | Resources/note | PHI spans/note | Empty outputs |
|---|---|---|---|---|
| Base | **0.951** | 3.06 | 5.9 | 13/50 |
| Fine-tuned | 0.842 | **5.08** | **7.06** | 6/50 |

**What this shows:**

The fine-tuned model extracts 66% more resources and 20% more PHI spans on real notes — it's more aggressive and rarely gives up (6 empty vs 13). But its OOD validity drops to 0.842 vs the base model's 0.951. The extra resources include some that are structurally invalid FHIR — fields the model learned in training don't map cleanly to all specialty-specific note styles.

The base model, by contrast, is conservative: when it's unsure, it produces nothing (13/50 empty) rather than hallucinating structure. What it does produce is 95% valid.

**The OOD trade-off in one line:** fine-tuning improves recall on in-distribution data dramatically (0 → 20/20 passing) but trades some OOD precision for that gain. The fix is more training data with broader specialty coverage — `make data` with a larger `--n` and ideally Claude-written notes from varied specialties.

Real-world evaluation against i2b2 2014 (de-id) or n2c2 2018 (medication extraction) would give a gold-annotated OOD picture. Those datasets require a DUA but are free for researchers.
