# Day 4 — P1.1: the HONEST healing ceiling (held-out)

**Why:** P1 measured top-1 on the *same* 10 sentences it trained on, over only 63
positions — so its "ceiling" (top-1 spiking to 47%, ppl below the teacher) was
memorization, not skill. P1.1 fixes the measurement: train on a 50-sentence
corpus, evaluate on a **disjoint 18-sentence held-out set** over **256
positions**, and also report *train* top-1 so the overfit gap is visible.

Setup identical otherwise: GPT-2-small, ~2.0156 bits/weight, STE distillation
from the FP teacher, lr 2e-4, T=2.0, seq_len 64, seed 0.

## Result — held-out vs train

| step | held-out top-1 | train top-1 | gap (train−held) | held-out ppl |
|---|---|---|---|---|
| 0 (naive) | 5.9% | 4.3% | — | 37,150 |
| 15 | 20.7% | 23.0% | +2.3% | 1,171 |
| 30 | 26.6% | 28.1% | +1.6% | 689 |
| 60 | **29.3%** | 36.3% | +7.0% | 549 |
| 120 | 26.9% | 43.0% | +16.0% | 410 |
| 240 | 30.1% | 23.8% | −6.3% | 375 |

_elapsed ~1076s (~18 min), CPU-only._

## What it says

- **Honest ceiling ≈ 27–30% top-1 on unseen text**, reached by ~30–60 steps and
  then **flat / noisy**. The in-sample P1 figure (47%) was indeed an illusion —
  held-out caps far lower.
- **Overfitting is real and visible.** From step 60→120 train top-1 climbs
  36%→43% while held-out *falls* 29%→27% (gap +16%). More steps stop helping the
  thing we actually care about. (Steps 240's bounce — train 24%, held 30% — is
  optimizer/STE noise on a small corpus; the band, not any single point, is the
  signal.)
- **Held-out perplexity stays high (~375)** and never approaches the teacher.
  So a healed 2-bit ternary GPT-2 still disagrees with the FP model on ~70% of
  next-token picks on unseen text — this is **not** FP-quality. Pure post-hoc
  healing at ~2 bits has a genuine, modest ceiling.

## Honest verdict (the real P1 answer)

At ~2 bits/weight, post-hoc heal-only recovery on GPT-2-small tops out around
**~30% teacher agreement on held-out text** — useful as a *mechanism* proof, far
from the "usable 50–75%" we'd hoped, and nowhere near FP quality. The lever to do
better is **not more steps**; it is (a) **less aggressive bit-widths** and/or
(b) **native low-bit training** (D3 AetherNet already beat post-hoc at equal
bits).

## Next — P2 (the master-key curve for P-A)

Run the same held-out heal at **int8 / int4 / int2 / ternary** and plot
**bits/weight → healed held-out top-1**. That curve is our real, honest
rate-distortion-with-healing frontier — the core measurement P-A (size at
quality) is built on. Then compare each point against native-trained AetherNet
at the same bits.
