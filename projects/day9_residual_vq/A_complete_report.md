# P-A (Size) → ~complete: the locked recipe + full frontier

This consolidates Day-6/8/9 into the finished P-A picture: the compression recipe,
the full bits↔quality frontier, and the honest open edge.

## A1 — residual / additive VQ (Day-9), reconstruction NMSE on GPT-2 mlp.c_fc

| method | bits/weight | NMSE |
|---|---|---|
| scalar ternary | 2.04 | 0.224 |
| **single VQ d4K256** | 2.01 | **0.109** |
| residual d4 [16,16] | 2.00 | 0.143 |
| residual d8 [256,256] | 2.06 | 0.119 |
| **residual d8 [256,16]** | **1.53** | **0.211** |
| residual d4 [16,4] | 1.50 | 0.244 |
| single VQ d8K256 | 1.03 | 0.335 |
| **residual d4 [256,64,16]** | 4.52 | **0.012** |

Findings:
- **At 2 bits, single big-codebook VQ wins** (0.109) — naive residual does NOT beat
  it. (AQLM's additive gains need *joint* codebook optimisation + beam search; our
  sequential residual is the simple version, so it under-delivers at equal bits.)
- **Residual extends the frontier where single VQ couldn't:** 1.53 bits → 0.211,
  matching scalar-ternary's 2.04-bit quality at **25% fewer bits** (plain VQ failed
  sub-2-bit, Day-6 P4).
- **Stacking codebooks gives a clean near-lossless point:** 4.52 bits → 0.012 NMSE.

## The full P-A frontier (reconstruction NMSE, lower=better)
```
NMSE
0.34 | VQ 1.0b ●
0.24 | ternary 2.0b ●        residual 1.5b ●
0.21 |                       residual 1.53b ●
0.18 | VQ 1.94b ●
0.14 | residual 2.0b ●
0.11 | VQ 2.0b ★ (sweet spot)
0.03 | VQ 3.2b ●
0.01 | residual 4.5b ●
     +----------------------------------------------- bits/weight
```

## The LOCKED recipe (P-A standard)
- **Default (2-bit sweet spot):** single VQ, group d=4, K=256, **+ healing**
  (codebook fine-tune). Measured: whole-model GPT-2 held-out English ppl
  **304 ± 34** vs ternary+heal 1323 ± 470 (Day-8 S1); ~2× FP.
- **Sub-2-bit operating point:** residual VQ d8 [256,16] (~1.5 b/w) when storage is
  king and some quality can be traded.
- **Near-lossless point:** residual VQ d4 [256,64,16] (~4.5 b/w).
- **For MoE experts:** shared codebook per layer (Day-7) so overhead amortises.
- All bits/weight figures include index + codebook + scale overhead.

## P-A status: ~90% (this-PC-complete)
Done: VQ>scalar (recon+behaviour), real-language + seeds + SOTA crossover + scaling
trend + cost/latency honesty + residual-VQ frontier + locked recipe.

Remaining ~10% (effort/hardware-gated, not unknowns):
- **AQLM-style JOINT additive-codebook optimisation** (beam search + jointly trained
  codebooks) — the lever to actually beat single-VQ at 2-bit and push usable
  sub-1-bit. Real, known, not yet built here.
- Confirmation on a larger model (GPT-2-medium+) — needs the model/hardware.

**Verdict:** P-A is, on this machine, a finished and validated low-bit compression
recipe with a mapped frontier and an honest open edge (joint additive opt). Ready
to hand its small low-bit weights to **P-B (the kernel)** next.
