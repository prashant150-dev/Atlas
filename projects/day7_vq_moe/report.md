# Day 7 — VQ + healing + MoE sparsity together (size AND effective-speed)

Combines the two proven levers in one model on the char_lm task:
- **MoE sparsity** — 8 experts, top-2 active → big total capacity, small active cost.
- **VQ + healing** (Day-6) — expert weights at ~2 bits against a per-layer SHARED
  codebook (so the codebook overhead amortises across all experts).

We report accuracy + honest FFN bit budgets (stored total, and bits touched per
token). chance = 0.168.

## Results

| variant | accuracy | FFN stored bits | FFN active bits / token |
|---|---|---|---|
| DenseFP-small (H=128) | 0.548 | 524,288 | 524,288 |
| DenseFP-big (H=1024) | 0.547 | 4,194,304 | 4,194,304 |
| MoE-FP (8×128, top-2) | 0.549 | 4,210,688 | 1,064,960 |
| **VQ-MoE + heal** | **0.557** | **344,064** | **81,920** |

## Verdict — the two levers multiply

Versus the **DenseFP-big** model (same accuracy, 0.547 vs 0.557):
- **Stored: 12.2× smaller** (4.19M → 344k bits) — from VQ's 2-bit shared-codebook
  experts.
- **Active / token: 51× smaller** (4.19M → 82k bits) — from MoE sparsity (only 2/8
  experts) *and* VQ (2-bit), composed.

So VQ-MoE delivers the **big model's accuracy at ~the small model's stored size
and a fraction of any variant's active compute** — exactly the dream's
"huge total capacity, tiny stored + active cost" shape, measured.

## Honest caveats
- **char_lm is capacity-saturated**: DenseFP-small (0.548) ≈ DenseFP-big (0.547),
  so extra capacity buys no accuracy here. This experiment therefore proves the
  **iso-accuracy bit-efficiency** composition (same quality, far fewer stored +
  active bits), NOT a capacity→quality gain. A harder, capacity-hungry task would
  be needed to show MoE's quality upside.
- Shared-codebook VQ is what makes VQ viable on small experts (per-expert codebooks
  would dominate). Confirmed it works post-hoc (acc 0.549) and heals to 0.557.
- Accounting covers the FFN (the varied part); attention/embed/head are identical
  across variants. Small model, single seed.

## Significance for the dream
This is the first model in the project where **both** dream levers act together
and are measured: stored size (VQ) **and** effective active cost (MoE) drop by
~12× / ~51× at iso-accuracy. The architecture pattern — sparse experts +
shared-codebook VQ + healing — is the concrete "huge-but-cheap" design.

## Next
- A capacity-hungry task to show MoE's quality upside (so total capacity actually
  helps, not just costs less).
- Scale experts / add more layers; combine with the reasoner+memory stack (Day-5).
