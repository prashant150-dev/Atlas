# Phase 1 — C++/AVX2 ternary kernel (the speed enabler)

Goal (math): on this PC, decode is bandwidth-bound. A 2-bit packed weight is read
16x faster than fp32, so a kernel that computes directly on packed weights should
break the "compression = disk only" wall (Phase B) and approach the
`tok/s = 0.5 * 18 GB/s / (active_params * 0.25 B)` ceiling → ~40 tok/s at ~900M
active params.

## Step 0 — install a C compiler (you do this once)
MinGW-w64 gcc (free, ~100 MB). Easiest: **winlibs.com** — download the UCRT zip,
unzip to e.g. `C:\mingw64`, add `C:\mingw64\bin` to PATH (or run from that shell).
Verify in this session:
```
!gcc --version
```

## Step 1 — build the kernel
```
!bash projects/day14_kernel/build.sh
```
(produces `libternary.dll`)

## Step 2 — benchmark vs fp32
```
!python projects/day14_kernel/bench_kernel.py
```
Reports: correctness (must match dequantised fp32), weight-RAM reduction (~16x),
and wall-clock speedup of the kernel vs numpy fp32 on a real FFN-sized mat-vec.

## What success looks like
- correctness max|err| ~ 0 (exact ternary arithmetic)
- packed weight ~16x smaller in RAM
- kernel faster than numpy fp32 (the real win this CPU couldn't show before)

If the kernel is NOT faster, that is an honest result too — it tells us the
2-bit-unpack overhead still loses to BLAS on Haswell, and the next lever is a
LUT/T-MAC-style kernel or AVX2 intrinsics. We measure, we don't assume.

## Files
- `ternary_kernel.c` — the kernel (matvec + batched matmul)
- `build.sh` — gcc -O3 -mavx2 -mfma build
- `bench_kernel.py` — correctness + timing vs fp32
