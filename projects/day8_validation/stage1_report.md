# Day 8 Stage 1 — does VQ+healing survive REAL English? (multi-seed ablation)

Directly answers reviewer critiques #3 (single seed), #4 (no std), #6/#20 (PPL on
synthetic ≠ language), #19 (no ablation): we test on **real English prose** with a
**real LM (GPT-2)**, across **seeds**, with a **full ablation** at matched ~2
bits/weight.

Eval = held-out English passage (offline; standard benchmarks need a download this
box lacks). Healing = 40 distillation steps on a disjoint English corpus.
**3 of 5 planned seeds completed** (the run was killed on session exit); 3 seeds
already give an unambiguous answer.

## Results (perplexity on real English, mean ± std over seeds 0,1,2)

| method | bits/weight | ppl (mean ± std) | runs |
|---|---|---|---|
| FP teacher | 32 | **77.8** | — |
| scalar ternary post-hoc | 2.04 | 7,329 (det) | — |
| VQ post-hoc | 2.02 | **651.6 ± 113.1** | 725, 492, 738 |
| scalar ternary + heal | 2.04 | **1,322.8 ± 470.4** | 872, 1125, 1972 |
| **VQ + heal** | 2.02 | **303.7 ± 33.8** | 319, 335, 257 |

## Verdict — the win survives real English AND is statistically clean

- **VQ + heal (303.7 ± 33.8) beats ternary + heal (1322.8 ± 470.4) by ~4.4×** at
  equal bits, on real English. Crucially the distributions **do not overlap**:
  VQ+heal's worst run (335) is far below ternary+heal's best run (872). So the gain
  is **not a lucky seed** — it holds every seed.
- **VQ + heal is also far more STABLE**: std 33.8 (~11% of mean) vs ternary+heal's
  470 (~36%). The shared-codebook lever is both better and steadier.
- **VQ post-hoc (651.6) beats ternary post-hoc (7,329) by ~11×** — the
  cross-weight-structure head-start reproduces on language.

## What this does and does NOT establish (honest)
- **Establishes:** the Day-6 VQ-vs-scalar size win transfers to **real English**
  with multi-seed statistics and a clean ablation — it was not a synthetic-task or
  single-seed artifact.
- **Still open** (from the critique): standard benchmarks (WikiText/TinyStories/
  downstream tasks — need a download), MoE-on-language (here MoE was validated on
  synthetic capacity tasks only), scaling laws to large models, training/healing
  cost economics, wall-clock speed, comparison vs AWQ/GPTQ/AQLM/QuIP#. Perplexity
  is a proxy, not reasoning.
- 3 seeds (not 5), GPT-2-small, 40 heal steps. The direction and significance are
  clear; tighter numbers want the full 5 seeds + more steps.

## Bottom line
On **real English, across seeds, with an ablation**, vector-quantization + healing
beats the scalar-ternary baseline ~4.4× at equal bits with **non-overlapping**
distributions — the size lever is real on language, not a synthetic mirage. The
broader 400B-capability claims remain unproven and explicitly out of scope here.
