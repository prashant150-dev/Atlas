# Day 22 — Streaming converter: model → AetherCore architecture AS IT ARRIVES

## The idea (user's)
As a model downloads, convert each chunk to our 2-bit architecture on the fly and
discard the FP, so the full FP model is NEVER held in RAM or on disk. This is how SOTA
low-bit quantization (GPTQ/AQLM) already works — quantize layer-by-layer with bounded
memory.

## How it works
safetensors LAZY loading: `safe_open(file)` + `get_tensor(key)` brings ONE tensor into
RAM at a time. For each tensor: load → mixed-precision VQ (Part-1) → write the compressed
arrays → free the FP tensor → next. Peak RAM = the single largest tensor, not the model.

## Result (on GPT-2, models/gpt2 → experiments/gpt2_streamed)

| metric | value |
|---|---|
| full FP model on disk | 498 MB |
| **peak FP held in RAM at once** | **154 MB** (never the whole model) |
| compressed output | 111 MB (4.5× smaller) |
| tensors | 48 quantized (2-bit mixed) + 100 kept raw (embeds/norms/bias) |
| round-trip max \|err\| | 0.264 (post-hoc, no healing) |
| reload verified | ✅ manifest + per-tensor npz reconstructs correctly |

## Honest reading of the weak-looking numbers (GPT-2 is a bad showcase)
- **Peak RAM 154 MB ≈ 3× reduction only** — because GPT-2's `wte` embedding is 154 MB,
  a HUGE fraction of a 124M model, and we keep embeddings raw. On a large model the
  embedding is a tiny fraction and a single transformer-layer weight is the peak →
  peak-resident ≈ one layer (hundreds of MB) vs an 800 GB full model = **>1000× less**.
- **Compression only 4.5×** — same reason: embeddings dominate GPT-2 and are kept fp16
  (2-bit on embeddings = garbage, the "jongjong" lesson). On an FFN/attention-dominated
  large model the quantized weights are the bulk → **~7-16× compression**.
- **Round-trip err 0.264** is post-hoc 2-bit with NO healing — the streaming path does
  fast per-tensor VQ. Behavioural healing (the thing that got GPT-2 to 1.42× FP) is the
  heavier offline step; a per-layer local-heal hook is the honest TODO.

## Large-model projection (same loop, unchanged)
| model | full FP | peak resident (≈1 layer) | compressed @2-bit |
|---|---|---|---|
| 7B | ~14 GB | ~0.4 GB | ~1.9 GB |
| 70B | ~140 GB | ~1.6 GB | ~18 GB |
| 400B | ~800 GB | ~few GB | ~100 GB |

The loop is identical; only the per-tensor size changes. The full FP never needs to be
resident — which is the whole point.

## What this DOES and does NOT solve
- ✅ Solves: never holding the full FP model (the 800 GB problem). Bounded RAM. Reloadable.
- ❌ Does NOT solve: the compressed output still needs disk (400B@2bit ≈ 100 GB > this
  PC's 50 GB → external drive, or cap at ~180B); the 800 GB FP still has to be DOWNLOADED
  (streamed through, discarded); best-quality healing is offline/heavier.

## Files
- `stream_convert.py` — the streaming tensor-by-tensor converter (safetensors lazy load)
- `stream_results.json` — measured peak-RAM / compression / round-trip
- output: `experiments/gpt2_streamed/` (manifest.json + per-tensor npz)
