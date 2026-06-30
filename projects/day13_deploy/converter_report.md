# AetherConvert — universal model → VQ-compressed converter

A general tool (`aether_convert.py`) that compresses ANY Hugging Face causal-LM with
our P-A recipe (vector quantization, d=4 K=256, ~2 bits/weight) and loads it back as
a normal model. Architecture-agnostic (GPT-2, Llama, Qwen, Phi, ...).

## Verified end-to-end on GPT-2
- `convert models/gpt2 experiments/gpt2_aether`: quantized 48 matrices, kept 100
  raw → **652 MB → 179 MB (3.6× smaller)**.
- `load(...)`: reconstructs a normal `AutoModelForCausalLM`, **124,439,808 params**,
  generates real text.
- sample (post-hoc, no heal): *"The future of AI is a good for the future of the
  future…"* — coherent words, repetitive (the honest 2-bit-without-healing quality).

## Key design decisions (learned the hard way, measured)
- **Skip embeddings / lm_head from VQ.** Quantizing the token embedding at 2-bit
  produced pure garbage ("jongjong…") — the embedding IS token identity. Keeping it
  raw → coherent output. (This is why the ratio is 3.6× not 16× on GPT-2: its
  embedding is 38.6M of 124M params; on bigger models the embedding is a smaller
  fraction, so the ratio climbs.)
- **Dedup tied weights** (GPT-2 ties wte↔lm_head): store shared storage once, use a
  manifest "ref" — avoids a 154 MB duplicate and the safetensors shared-memory error.
- Quantize only 2D matrices with both dims ≥ 256 (so codebook overhead amortises).

## Honest limits
- Post-hoc only (no healing): text is degraded at ~2 bits. Healing (the GPT-2 path,
  Day-6 P3 → ppl 94.7) recovers quality but is model-specific; folding general
  healing into the converter is the next step.
- DISK shrinks for real; RAM/speed do NOT (dequantised to fp32; no low-bit kernel —
  Phase B). "Use like a normal model" = functionally yes, with disk savings.
- GPT-2 is a weak/old base, so even healed output is modest. A newer, larger model
  is a better target (more redundancy → better compression + less quality loss) —
  and the converter already supports any such model.

## Usage
```
python projects/day13_deploy/aether_convert.py convert <model_dir> <out_dir>
python projects/day13_deploy/aether_convert.py load   <out_dir>     # sanity
python projects/day13_deploy/aether_convert.py chat   <out_dir>     # interactive
```
