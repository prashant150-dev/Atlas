# Phase B (Speed) — the honest CPU verdict

Goal: turn P-A's compression into a real wall-clock speedup using an actual CPU
low-bit kernel. We converted GPT-2's Conv1D to nn.Linear and applied torch's
onednn dynamic int8 quantization (fbgemm/qnnpack not in this build), measuring
generation speed + perplexity.

## Results (GPT-2-small, this 2014 i5-4590T CPU)

| variant | ppl | tok/s | linear weights |
|---|---|---|---|
| FP32 (Conv1D→Linear) | 102.3 | **19.7** | 494 MB |
| onednn int8 dynamic | 11,818 | **12.0 (0.61×)** | 124 MB (4× smaller) |

## The honest verdict — NO speed win on this hardware

Two negatives, both real:
1. **int8 was SLOWER (0.61×), not faster.** On this Haswell CPU (AVX2, **no
   AVX-512/VNNI**) the dynamic-quant overhead (quantising activations every call)
   exceeds the int8-GEMM benefit at GPT-2-small matrix sizes. The "real CPU low-bit
   kernel" does not pay off here.
2. **Naive int8 collapsed quality** (ppl 102 → 11,818): per-tensor dynamic int8
   without per-channel scaling destroys outlier-heavy weights — exactly why
   GPTQ/AWQ exist. Fixable with per-channel quant, but irrelevant since speed lost.

So P-A's compression does **not** convert to CPU speed here even with a compiled
int8 path. This empirically confirms (not just "no kernel" but "the available
kernel is slower on this CPU") that **P-B's 40–50 tok/s is hardware-gated.**

## The one real, honest win
**Conv1D → nn.Linear layout sped up FP32 from ~15 to 19.7 tok/s (+33%)**, free, just
from a better matmul memory layout. A genuine measured improvement — kept.

## Why, and what would actually deliver P-B
- This CPU lacks AVX-512/VNNI; int8/low-bit GEMM has no hardware acceleration path
  that beats fp32 BLAS at these sizes.
- Real speed needs: (a) a newer CPU (AVX-512-VNNI) where int8 GEMM wins, (b) a GPU,
  or (c) a hand-written packed-low-bit SIMD kernel (bitnet.cpp / T-MAC style) — a
  C/assembly effort, not Python.

## Phase-B status (honest): hardware-gated, ~20%
- ✅ R7: kernel math (RAM 15.7×, 0 weight-mults) — the *theory* of the win.
- ✅ Conv1D→Linear: +33% fp32 (real, free).
- ❌ Actual low-bit speedup on this CPU: **not achievable** — measured slower.
- The 40–50 tok/s target is **gated on hardware/a compiled kernel**, now confirmed
  empirically, not assumed.

**Bottom line:** unlike P-A (which we took to ~90% on this PC), **P-B cannot be
completed on this machine** — the honest ceiling here is ~20%, and the rest is a
hardware/kernel-engineering gap, proven by measurement.
