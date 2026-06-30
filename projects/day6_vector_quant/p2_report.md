# Day 6 P2 — whole-model VQ: the reconstruction win transfers to behaviour

P1 showed VQ halves reconstruction error vs scalar ternary at equal bits/weight on
one matrix. P2 applies VQ to EVERY GPT-2 block linear, rebuilds a real model, and
measures end-to-end perplexity + top-1 vs the FP teacher — against the scalar-ternary
baseline at honest equal bits/weight (eval: a 256-token English passage).

## Results

| method | bits/weight | perplexity | top-1 vs teacher |
|---|---|---|---|
| FP teacher | 32 | **8.19** | 1.000 |
| scalar ternary (baseline) | 2.03 | **49,357** | 0.023 |
| **VQ d=4, K=256** | 2.02 | **1,880** | 0.078 |

## Verdict — the win is real end-to-end

- **At equal ~2 bits/weight, VQ gives 26× lower perplexity than scalar ternary**
  (1,880 vs 49,357). The reconstruction advantage (P1: 2× lower NMSE) translates
  into a large *behavioural* advantage. The cross-weight-structure lever works on
  real model behaviour, not just on paper.
- **But post-hoc VQ alone is still far from FP** (1,880 vs teacher 8.19, top-1 8%).
  At ~2 bits, *no* post-hoc method is FP-usable — consistent with D1/P1.1. VQ moves
  the floor down a lot, but healing (D2) is still needed to approach FP.

## Honest limitations
- **Large codebooks (K=4096) are impractical on this CPU** — the k-means
  assignment over 48 matrices did not finish in a reasonable time. The practical
  VQ regime here is small groups + small codebooks (K≤256). This is itself a real
  finding for a CPU lab.
- One eval passage, one seed; perplexity only (no downstream task accuracy yet).
- VQ here is plain k-means (no learnable transform / outlier handling that
  AQLM/BTC-LLM add — those would push further).

## Frontier so far (bits/weight ↔ behaviour, post-hoc, no healing)
```
ppl (log)
 49357 | ternary  ● (2.03 b/w)
  1880 | VQ d4K256 ● (2.02 b/w)   <- 26x better at equal bits
     8 | FP teacher ●────────────────────── (32 b/w)
```

## Next (P3)
1. **VQ + healing:** distill the FP teacher into the VQ student (STE on the
   codebook/assignments or shadow weights) — does healed-VQ beat healed-ternary at
   equal bits, approaching FP? This is the real frontier-mover.
2. Fill the sub-2-bit curve with CPU-cheap points (small K, varied group size).
