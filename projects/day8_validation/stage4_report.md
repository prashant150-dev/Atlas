# Day 8 Stage 4 — healing cost (#7) and wall-clock latency (#13/#14)

Two concrete measured questions from the critique.

## Part 1 — wall-clock latency: FP vs VQ-reconstructed GPT-2 (CPU)

| model | forward (64 tok) | generate |
|---|---|---|
| FP | 175.9 ms | 14.8 tok/s |
| VQ (dequantised) | 241.4 ms | 16.6 tok/s |

**Honest result: VQ gives NO speed win on this CPU** — the two are within
measurement noise (VQ forward is even a touch slower; generate a touch faster).
This is exactly what R5/R7 predicted: `load_packed`/`_apply_recon` reconstruct the
weights to fp32, and there is no low-bit CPU kernel, so the matmul runs in fp32 at
fp32 speed. **The compression win is DISK (and, with a SIMD kernel, RAM), not
latency.** We do not claim a speedup we cannot measure.

## Part 2 — healing economics (the one-time cost, #7)

| quantity | measured |
|---|---|
| heal step time (teacher fwd + student fwd+bwd, GPT-2-small) | 3,368 ms/step |
| total heal (40 steps) | 134.7 s |
| = inference-forwards of compute (one-time) | **766** |
| heal tokens processed | 2,560 |
| fraction of pretraining (~9e9 tokens, ESTIMATE) | **2.8e-7** |

**Honest result: healing is a tiny ONE-TIME fine-tune, not a re-training.** It costs
~766 forward-passes of compute (≈135 s for GPT-2-small) and processes ~2.8e-7 of
pretraining's token budget. The reviewer's worry ("maybe 100× training cost") is
the opposite of what we measure — healing is **negligible** vs pretraining and is
amortised over all future inference.

**The real catch (stated honestly):** heal cost scales ~linearly with model params
(each step is a forward+backward over the whole model), so a 400B heal, while still
a tiny fraction of a 400B *pretraining*, requires hardware that can backprop
through 400B — the standing hardware gap, not a cost-economics blocker.

## Verdict
- **#13/#14 (speed):** measured, honest — **no wall-clock win on CPU** without a
  low-bit kernel; the win is storage. We don't overclaim 40–50 tok/s.
- **#7 (heal cost):** measured — healing is a **negligible one-time** cost vs
  pretraining (2.8e-7), not a training-economics problem; the only barrier at 400B
  is hardware to run the backward pass.

## Honest caveats
- GPT-2-small, CPU, single machine; generate timing is noisy (greedy, batch 1).
- Pretraining token count is an ESTIMATE (WebText scale).
- The "win is disk" conclusion is contingent on no kernel; R7 shows the kernel
  math (RAM 15.7×, 0 weight-mults) that a SIMD/C implementation would convert to
  real speed/energy — not built here.
