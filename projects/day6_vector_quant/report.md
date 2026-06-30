# Day 6 (P-A) — Vector quantization beats scalar at equal bits/weight

**Claim under test:** Day-1's ~2.04 bits/weight floor is the *per-weight marginal*
entropy (independent coding). Coding GROUPS of weights against a shared codebook
(vector quantization) should beat it by exploiting cross-weight correlation
(joint entropy < sum of marginals). Tested on a real GPT-2 weight matrix
(`transformer.h.0.mlp.c_fc.weight`, 768×3072 = 2.36M weights). Metric:
reconstruction NMSE vs honest bits/weight (codebook overhead included).

## Results

| method | bits/weight | NMSE |
|---|---|---|
| scalar int2 | 2.00 | 0.998 |
| **scalar ternary (D1/D2 baseline)** | 2.04 | **0.224** |
| scalar int4 | 4.00 | 0.838 |
| **VQ d=4, K=256** | 2.01 | **0.109** |
| VQ d=8, K=4096 | 1.94 | 0.177 |
| VQ d=2, K=16 | 2.00 | 0.128 |
| VQ d=4, K=4096 | 3.22 | 0.033 |
| VQ d=4, K=16 | 1.00 | 0.368 |
| VQ d=8, K=256 | 1.03 | 0.335 |

## Verdict — a real, measured win

- **At ~2 bits/weight, VQ halves the error of the strong scalar baseline:**
  VQ d4_K256 (2.01 b/w) NMSE **0.109** vs ternary (2.04 b/w) **0.224** — ~2× lower
  error at equal size.
- **VQ d8_K4096 gives lower error AND fewer bits** (0.177 @ 1.94 b/w) than ternary
  (0.224 @ 2.04). A strict win on both axes.
- **At 1 bit/weight VQ is still usable** (0.37) where scalar int1 is dead (1.0).

This is the genuinely-new P-A lever working: **cross-weight structure is real and
exploitable**, so the per-weight floor is not the joint floor. D1 is not
contradicted — its 2.04 bits was the *i.i.d. per-weight* bound; real weights are
more compressible *jointly*.

## Honest caveats
- This is **reconstruction NMSE**, not yet downstream behaviour. Lower NMSE is
  necessary, not sufficient — the next test is whether it preserves perplexity /
  top-1 (and how much healing helps).
- `scalar_int` here lacks per-channel scaling, so it is a weak baseline; the fair
  comparison is against **ternary** (per-column scaled), which VQ still beats ~2×.
- One matrix, one seed. Needs the full-model + behaviour test.
- Codebook overhead is included in bits/weight (negligible at K≤256, ~0.4 b/w at
  K=4096 — already counted).

## Next (Day-6 P2)
1. **Whole-model VQ + behaviour:** apply VQ to all GPT-2 block linears, measure
   perplexity / top-1 vs the ternary baseline at equal bits/weight.
2. **VQ + healing (D2):** does distillation on top of VQ recover behaviour even
   further, beating healed-ternary at equal size?
3. Sweep group size d and K to trace the VQ bits/weight ↔ quality frontier — our
   new, lower frontier curve.
