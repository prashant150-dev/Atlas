# ATLAS

**Run big AI models on a weak PC** — an efficiency-first architecture and research project.

ATLAS (*Active-sparse, Tiered-memory, Low-bit, Augmented-with-tools, Self-verifying*) is an
attempt to make large language models small and fast enough to run on a "potato PC"
(an old 4-core CPU, 8 GB RAM, no GPU) **without dropping below the base model's quality** on
verifiable tasks. It is built from the ground up with an honesty rule: every claim is measured
against a real baseline, and the project never claims a result it cannot reproduce.

> Built solo, on the exact potato PC it targets (Intel i5-4590T, 8 GB RAM, no GPU).
> CPU-only. No CUDA assumed anywhere.

---

## The idea in one picture

```
  prompt
    │
    ▼
 ┌────────┐   math ─────────────▶  exact calculator tool        (100% on verifiable)
 │ ROUTER │   code ─────────────▶  generate + RUN + self-fix
 │        │   fact ─────────────▶  retrieve from knowledge base  (grounded)
 │        │   open/creative ────▶  the base model (low-bit)
 └────────┘
    │
    ▼
  VERIFY  ──▶ confident answer  /  honest "I don't know"
```

Three substrates instead of one giant dense model:

- **A small reasoner** that stays in RAM.
- **Retrieval memory** on disk (cheap, huge, only the relevant bit is read).
- **Sparse experts** — only a small fraction of the weights are active per token.

On top of that: **low-bit weights**, a **LUT kernel** that computes on packed weights
(no float multiplies), **tools** for exact answers, and a **verify** step that prefers an
honest "I don't know" over a confident wrong answer.

---

## What is actually proven (CPU, small scale)

These are measured, reproducible results in this repo — not promises:

| Claim | Result | Where |
|---|---|---|
| Native-sparse training survives extreme sparsity | 0.62M char-LM, 95% sparse: **native 0.970** acc vs dense 0.977 vs **post-hoc 0.184** (collapses) | `projects/v2_design/T11_training/` |
| Exact-weight compression has a hard floor | Rate-distortion floor from real GPT-2 weights (~2.04 bits/weight) — no method may cross it | `src/compression/rate_distortion_limit.py` |
| "Preserve behaviour, not weights" reopens the frontier | Healing/QAT recovers naive ternary from ~3% → ~28% top-1 in 30 steps | `src/compression/healing_qat.py` |
| Error-compensated post-hoc quant beats naive | GPTQ-style: **8.4× better** output error at 2-bit, no training | `projects/v2_design/T1_size/gptq_no_train.py` |
| The tiers run together as one engine | Router + tools + retrieval + verify in a single pipeline | `projects/v2_design/integration/` |

**Honest status:** this is a **research project + working prototype**, roughly **~25% of the way
to a product**. The orchestration layer (router → tools → retrieve → verify) runs end-to-end on
CPU today. The remaining work is large-model validation (needs a GPU), integrated live inference,
and hardening. See `projects/v2_design/PRODUCTION_READINESS.md` for a brutally honest breakdown.

---

## Quick start

```bash
# one question (verifiable tasks answered exactly, instantly)
python atlas.py ask "What is 47389 * 8291?"

# convert an fp16/fp32 Hugging Face model -> ATLAS low-bit format (streaming, bounded RAM)
python atlas.py convert models/qwen2.5-1.5b

# interactive chat over a real base model
python atlas.py chat --model qwen
```

The compression and GPT-2 paths:

```bash
python main.py --mode demo --prompt "solve x^2 - 4 = 0"
python main.py --mode gpt2-compress --gpt2-compression int8 --compressed-gpt2-path experiments/gpt2_packed
python main.py --mode gpt2-compare  --compressed-gpt2-path experiments/gpt2_packed
```

> Model weights (`models/`) and experiment outputs (`experiments/`) are **not** in git — they are
> large and regenerable. Download a base model into `models/` to use the real-model paths.

---

## Testing

There is no pytest suite — **every module has a `__main__` self-test that is its contract**:

```bash
python -m src.compression.engine
python -m src.math_engine.symbolic
python -m src.router.beast_router
python projects/v2_design/integration/atlas_engine.py
```

---

## Repository layout

```
atlas.py                       # unified CLI: convert / ask / chat
main.py                        # compression + GPT-2 demos and benchmarks
src/                           # the engine: compression, router, math/code engines, memory, quality
projects/                      # the research track (the heart of the project)
  ROADMAP.md                   # source of truth for direction
  day1_compression_limit/      # D1: the proven rate-distortion floor
  day2_healing_qat/            # D2: healing recovers ternary collapse
  day3_aethernet/              # D3: native ternary+sparse vs post-hoc
  v2_design/                   # the ATLAS tiers (T1–T21), integration engine, honest status docs
```

---

## Honesty guardrails (the project's rules)

- **Measure, don't assume.** A `__main__` self-test is the contract.
- **Always keep a naive baseline.** A result is a "win" only if it beats the baseline at equal size.
- **Count all the overhead** (scales, correction tables, seeds) in bits/weight.
- **Don't re-attack the proven-impossible floor.** Compression loses information; it cannot add
  intelligence. The wins come from changing the *goal* (preserve behaviour, route, retrieve,
  verify), not from magic.

---

## Status & roadmap

- `projects/v2_design/PRODUCTION_READINESS.md` — honest ~20–25% assessment.
- `projects/v2_design/COMPLETION_ROADMAP.md` — Research → Production → UI.
- `projects/ROADMAP.md` — the research direction (source of truth).

## License

MIT — see [LICENSE](LICENSE).
