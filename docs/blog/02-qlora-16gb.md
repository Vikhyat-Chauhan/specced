# QLoRA on 16 GB — fitting a 7B model on a consumer GPU

The standard complaint about fine-tuning LLMs is that you need a rack of A100s. QLoRA makes that untrue. Here's what it actually looks like to fine-tune Qwen2.5-Coder-7B on a single NVIDIA RTX 5060 Ti (16 GB VRAM).

## Why Qwen2.5-Coder-7B

Three reasons:
1. **Structured JSON output.** Code models are trained on structured text. FHIR extraction is a JSON generation task. Qwen-Coder consistently outperforms general-purpose models of the same size on structured output tasks.
2. **16 GB fit.** In NF4 4-bit quantization, the 7B model loads in ~4.5 GB. That leaves ~10 GB for activations, gradients, and the LoRA parameters during training — enough headroom for batch size 2 with gradient accumulation.
3. **Strong base.** A capable base model needs fewer examples to learn a new output format. With reject-sampled synthetic data, we don't have millions of examples — we have 160. The base model's prior does most of the heavy lifting.

## The QLoRA setup

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,   # NF4 quantization
    dtype=None,          # auto (bfloat16 on Ampere+)
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                # LoRA rank — 40M trainable params out of 7.6B (0.53%)
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)
```

**Only 0.53% of parameters are trained.** The rest are frozen in 4-bit and never moved to float32. This is why it fits.

## Training details

- **Dataset:** 160 training examples + 20 validation (offline synthetic, template notes)
- **Epochs:** 3 (30 gradient steps with effective batch 16)
- **Learning rate:** 2e-4 with linear warmup (10 steps) and linear decay
- **Loss masking:** response-only via Unsloth's `train_on_responses_only` — the system prompt and user message don't contribute to loss
- **Time:** 17 minutes on the RTX 5060 Ti
- **Peak VRAM:** ~9.9 GB during training

## Loss curve

| Epoch | Train loss |
|---|---|
| 0.5 | 0.1611 |
| 1.0 | 0.1001 |
| 1.5 | 0.0551 |
| 2.0 | 0.0331 |
| 2.5 | 0.0207 |
| 3.0 | 0.0122 |

**Final val loss: 0.0107.** Clean convergence, no plateau. The model is clearly learning the output format — the loss drop from epoch 1 to 2 is almost as steep as epoch 0 to 1, which suggests we're not in an overfit regime at epoch 3 (a smaller dataset would typically plateau earlier).

## What the training actually teaches

The base model knows about FHIR conceptually. What it doesn't know is:
- Our exact ChatML prompt format
- That the output must be a single JSON object (not wrapped in markdown, not partial)
- The specific field structure we expect (phi_spans with type/start/end, resources with required fields)
- That de-id is required alongside extraction

Three epochs on 160 examples is enough to internalize this format. The base model's resource F1 on held-out cases was 0.416 even with correct generation — it was producing valid-ish output but mismatching resources. Post fine-tuning: 0.992.

## The response-only masking trick

Without masking, the model's loss includes tokens from the system prompt and the user message — text that's identical across all training examples. Training on that text wastes capacity and can cause the model to "memorize the prompt" rather than learning the extraction task.

Unsloth's `train_on_responses_only` masks everything before the `<|im_start|>assistant` token. The model only sees gradients from its own output: the JSON it needs to learn to generate.

## Flash Attention note

The RTX 5060 Ti uses the Blackwell architecture. At the time of this writing, Flash Attention 2 doesn't support Blackwell, so Unsloth falls back to Xformers. No performance difference observed in practice — the attention computation is not the bottleneck at batch size 2 with 160 examples.

## Exporting for serving

After training, the LoRA adapter (40M params, ~160 MB) is merged back into the base model and quantized to GGUF for Ollama:

```bash
python -m train.export_gguf --adapter train/checkpoints/adapter --out serve/ --quant q4_k_m
```

Unsloth handles the llama.cpp quantization internally. The result is a single `.gguf` file that Ollama can serve locally with `ollama create specced-qwen7b -f serve/Modelfile`.
