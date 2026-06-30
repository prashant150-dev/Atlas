# Day 5 — R7: packed-ternary matmul kernel (Lever 2)

R5 showed that loading a packed-ternary model and dequantizing to fp32 buys DISK
only — no RAM or speed win, because the weights inflate to fp32 in memory and the
matmul runs in fp32. R7 writes the missing piece: a kernel that computes
`y = x @ W` directly on the packed 2-bit weights, and proves what such a kernel
unlocks. (Honest scope: this is a reference kernel in Python/NumPy; the *speed*
win needs a SIMD/C kernel and is NOT claimable against BLAS here — but RAM,
op-count, and correctness are.)

## Measured (GPT-2-ish linear: K=768, N=768, batch 64)

| claim | fp32 | packed-ternary kernel | result |
|---|---|---|---|
| **RAM (resident weight)** | 2.36 MB | **0.151 MB** | **15.7× smaller** (2.05 bits/wt) |
| peak transient unpack | (full matrix) | **49 KB** (one block) | fp32 matrix never materialised |
| **weight multiplies** | 37,748,736 | **0** | only N=49,152 scale mults → **768× fewer mults** |
| **correctness** | — | max\|err\| 4.6e-5 | bit-exact vs dequant matmul |
| wall-clock | 0.8 ms (BLAS) | 53 ms (Python) | Python loses — see note |

Plus a literal add-only micro-kernel (K=32,N=8) reading 2-bit codes by bit-shift:
**0 weight multiplies, 154 adds, 8 scale mults, max\|err\| 1.4e-6.**

## What this proves (Lever 2)

1. **The RAM floor is real and reachable.** The weight resides at **2 bits/weight**
   (15.7× under fp32); the kernel streams it in small blocks so the fp32 matrix is
   *never allocated* — peak extra RAM is one 49 KB block. This is exactly the
   "fit a huge model in tiny RAM" lever, demonstrated on a real layer.
2. **Multiplies are eliminated.** A ±1/0 weight turns every multiply into an
   add / skip / subtract; only N per-output scales remain. **768× fewer
   multiplies** for this layer. On hardware this is where the energy/throughput
   win comes from.
3. **It is exact.** The packed kernel reproduces the dequantized matmul to float
   precision — the reorganisation is mathematically sound, not an approximation
   on top of the ternary error.

## The honest gap

NumPy/Python cannot turn the op-count + RAM win into wall-clock speed against a
hyper-optimised BLAS (0.8 ms vs 53 ms). The **speed** payoff requires a SIMD/C
kernel (popcount/add over packed lanes, bitnet.cpp-style). R7 proves the *math*
that such a kernel exploits; building that kernel is a C-engineering task beyond
this pure-Python CPU lab.

## Lever 2 verdict

The RAM floor and the multiply-elimination are **proven**. The remaining work is
pure systems engineering (a SIMD kernel) — not a research unknown and not blocked
by any law. The blueprint is sound.
