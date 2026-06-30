# Day 5 — R6: native low-bit reasoner vs the post-hoc wall

R5 showed a *post-hoc* ternary GPT-2 reasoner collapses to 0.000 on the read-and-
answer task. R6 asks D3's question directly: does a model trained **natively** in
ternary+sparse space survive where post-hoc dies? We compare DenseFP /
PostHocTernary / AetherNet (native ternary + sparse MoE, same config) across
tasks of increasing computational demand. (Model is a single transformer block,
so true multi-hop induction is out of reach; we use tasks it can learn.)

## Results across tasks

| task | type | DenseFP | PostHocTernary | AetherNet (native) | chance |
|---|---|---|---|---|---|
| copy_m6 (D3) | attention copy | 1.00 | 0.948 | 1.00 | 0.083 |
| **R6 indexed retrieval** | attention routing | 1.00 | 0.934 | **0.435** | 0.083 |
| **char_lm (D3)** | computation-heavy | 0.918 | **0.470** | **0.838** | 0.168 |

AetherNet stored size ≈ **2× smaller** than DenseFP (native ternary + sparse);
PostHoc ≈ 5–6× smaller but with the accuracy shown.

## What this honestly says

- **The "wall" (post-hoc collapse) is task-dependent.** On attention-routing
  tasks (copy, indexed retrieval) ternary perturbation barely hurts — post-hoc
  stays ~0.93+. The wall appears on **computation-heavy** tasks where the FFN
  must encode rich structure: char_lm post-hoc falls 0.918 → **0.470**.
- **Native low-bit breaks the wall exactly where the wall exists.** On char_lm,
  AetherNet recovers **0.838** (vs post-hoc 0.470) — most of FP's 0.918 — at ~2×
  smaller stored bits. This is the Lever-1 proof: *where post-hoc dies, native
  survives.*
- **But native is NOT a universal win.** On the easy indexed-retrieval task it
  *underperformed* both (0.435) — the ternary + sparse-MoE model is harder to
  optimise and likely undertrained on a task that doesn't need that capacity.
  Honest: native pays off on hard, computation-heavy capability, not on
  everything.

## Connecting to the real reasoner (R5's 0.000)

The GPT-2 reasoner's job — read language and produce the right token — is
**computation-heavy**, the same regime where post-hoc collapsed (char_lm 0.470,
GPT-2 0.000). So the R6 verdict transfers: a usable low-bit reasoner must be
**natively trained (or strongly healed), not post-hoc quantized.** Lever 1's
*direction* is confirmed; the magnitude at GPT-2 scale remains to be earned.

## Honest limits
- Tiny single-block models; char_lm/ copy are proxies, not language.
- AetherNet retrieval result may improve with more steps; we report the
  as-measured number, not a tuned-to-win one.
- Native at GPT-2 scale (train a real ternary reasoner from scratch) is beyond
  this CPU; that is the scale gap.

## Next
- **R7:** a real packed-ternary matmul **kernel** — the RAM+speed lever R5 showed
  is missing (compression buys disk only without it).
- Longer-horizon: a small natively-ternary reasoner trained on real text, dropped
  into the R4 pipeline in place of GPT-2.
