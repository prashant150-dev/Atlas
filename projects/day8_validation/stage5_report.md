# Day 8 Stage 5 — remaining critique points, tested where possible

## Probe A — MoE routing health (#2, #8)
Trained an 8-expert top-2 MoE on char_lm; measured router behaviour on eval tokens.

| metric | value |
|---|---|
| per-expert load | 0.06–0.20 (ideal 0.125) |
| **dead experts** | **0 / 8** |
| routing entropy (norm, 1=balanced) | **0.971** |
| load CV | 0.37 |

**No routing collapse at this scale** — every expert is used, near-balanced. Honest:
expert-imbalance / dead-expert problems are real at *large* scale and we cannot
reproduce them on a tiny model; this only shows routing is healthy here.

## Probe B — shared codebook across layers (#9)
Per-matrix codebooks vs ONE codebook shared across three differently-distributed
GPT-2 matrices (NMSE):

| | attn.c_proj | mlp.c_fc | mlp.c_proj | penalty |
|---|---|---|---|---|
| per-matrix | 0.069 | 0.109 | 0.121 | — |
| shared-across-layers | 0.112 | 0.115 | 0.131 | **1.20×** |

**Confirmed limitation:** sharing one codebook across layers with different
distributions costs ~20% reconstruction error (worst on the most distinct matrix).
Day-7's shared codebook was *within a layer's experts* (similar distribution),
where this penalty is small; **cross-layer sharing should be per-layer**, as the
data shows.

## Probe C — memory storage realism (#15)
Templated facts vs rich facts (relations + context + metadata), gzip bits/fact:

| | bits/fact | 1B facts |
|---|---|---|
| templated | 23 | ~2.9 GB |
| **rich (relations+context)** | **76 (3.2×)** | **~9.5 GB** |

**Confirmed and corrected:** real knowledge is ~3× more expensive than the R2
templated estimate — but still **disk-friendly and linear** (1B rich facts ≈ 9.5 GB,
fits the 56 GB disk). The "knowledge is cheap to store" conclusion survives; the
exact number was optimistic.

## Probe D — D1 floor varies by weight type (#10)
Gaussian-entropy proxy (relative bits) per weight type:

| attn.c_attn | mlp.c_fc | wte_embed | ln_1 |
|---|---|---|---|
| −0.28 | −0.78 | −0.75 | −2.55 |

**The floor is heterogeneous:** attention weights carry more spread/entropy than
MLP, and LayerNorm is extremely low-entropy (highly compressible). D1's single
~2.04 bits is an *aggregate*; per-type floors differ. We still cannot test other
architectures (Llama/Qwen) offline — that generalisation remains open.

## Points we honestly did NOT close
- **#5 standard benchmarks** (WikiText/GSM8K/MMLU/HumanEval): need a download this
  offline box lacks. Real-English held-out perplexity (Stage 1) is our offline
  substitute, not a substitute for task benchmarks.
- **#11 multi-hop / compositional retrieval**: R3 was single-hop; multi-hop not built.
- **#12 catastrophic forgetting** (train→compress→heal→continue): not tested.
- **#16 official AWQ/GPTQ/AQLM/QuIP# numbers**: reimplemented the backbone (Stage 2),
  did not run the official libraries.
- **#2 full MoE-on-language**: routing is healthy (Probe A) but a full MoE *language*
  model was not trained; MoE validated on synthetic capacity tasks.

## #17 — softening the "hardware-only gap" claim (important)
The Master Report said remaining gaps are "hardware/scale, not unknowns." **That was
too strong.** Honest restatement: the size lever is well-evidenced at small scale,
but **unknown algorithmic barriers at large scale remain possible** — large-MoE
routing instability, cross-layer codebook degradation (Probe B shows it is real and
modest at small scale), the sub-1-bit cliff (Day-6 P4), and architecture-dependent
floors (Probe D). These are not proven blockers, but they are **not proven absent
either**. The correct claim is: *promising, well-measured small-scale evidence; the
jump to 400B remains an open empirical question, not a foregone conclusion.*
