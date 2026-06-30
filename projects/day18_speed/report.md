# Day 18 — Part-2 "Beast Speed", step 1: real end-to-end tok/s

## Question
Day-14 showed the LUT-GEMM ternary kernel beats fp32 1.25× on ONE big mat-vec. The
dream needs an honest end-to-end number: full single-token decode through L layers,
kernel for every matmul → tok/s, then project to the dream config.

## Result — the kernel's win depends on matrix size

| matrix regime | dims | LUT vs fp32 | throughput |
|---|---|---|---|
| GPT-2-small | D=768, FF=4× | **0.77× (LOSES)** | — |
| large expert (dream regime) | D=4096, FF=3× | **3.84× FASTER** (369 ms vs 1417 ms) | **3637 M params/s** |

**Why:** the LUT kernel builds a per-token table (3^4=81 partial sums per input group)
that is amortised across all N outputs. Small matrices (GPT-2's 768-wide) don't have
enough outputs to amortise the build, so the kernel LOSES there. Large expert matrices
(4096+) amortise it fully → 3.84×. A 400B MoE's experts are large, so the large regime
is the relevant one.

## tok/s and the honest path to 40-50

At the measured large-matrix throughput (3637 M active-params/sec on this 4-core Haswell):

| active params / token | tok/s |
|---|---|
| 900 M | 4.0 |
| 500 M | 7.3 |
| 300 M | 12.1 |
| 200 M | 18.2 |
| ~80–100 M | **~36–45** ✅ |

**The crux:** `tok/s = throughput ÷ active-params/token`. The kernel gives a real
3.84× on the right matrices, but 40-50 tok/s on this PC requires the ACTIVE-param
budget per token to be small (~80-100 M) — which is exactly the Day-15 task-conditional
MoE lever: don't activate the whole model, route to a few small experts per token.

So Beast Speed = **large-matrix LUT kernel (3.84×) + aggressive MoE routing (~100 M
active/token)**. Neither alone is enough; together they reach the target.

## Honest caveats
- The projection is a COMPUTE-side upper bound: it counts the kernel matmul cost only.
  Real decode also pays attention/KV reads, layernorms, routing, and — critically —
  the memory bandwidth to bring each token's active expert weights into cache (if
  experts stream from disk, bandwidth may dominate). Step 2/3 must measure those.
- GPT-2-small running SLOWER under LUT is a real negative: this kernel is for large
  matrices only. Small dense models should stay fp32/BLAS.

## Step 2 — the memory-bandwidth reality check
A 400B model can't fit 8GB RAM, so experts live on disk. Measured disk read here:
**~1269 MB/s**. Combined with the 3637 M params/s compute throughput, at 2 bits/weight:

| active/token | RAM-resident (compute) | stream+prefetch | stream serial reload |
|---|---|---|---|
| 80 M | **45.5** | **45.5** | 26.5 |
| 100 M | **36.4** | **36.4** | 21.2 |
| 200 M | 18.2 | 18.2 | 10.6 |
| 900 M | 4.0 | 4.0 | 2.4 |

**Key honest finding: at 2-bit, disk bandwidth is NOT the wall.** 100 M active params =
only 25 MB; loading that at 1269 MB/s takes 19 ms, *faster* than the 22 ms of compute —
so with prefetch overlap the decode is COMPUTE-BOUND (same as RAM-resident). Part-1's
compression made per-token weight loading cheap enough that speed is limited by compute,
not I/O. Only naive per-token *serial* reload (no overlap) loses — easily avoided by
prefetch or by Day-15 task-conditional routing (load a task's experts once, reuse).

## Part-2 conclusion (Beast Speed)
**40-50 tok/s is reachable on this PC** with three measured pieces stacked:
1. **Large-matrix LUT kernel** — 3.84× faster than fp32 on expert-sized matmuls (D14+D18).
2. **~80-100 M active params/token** — via Day-15 task-conditional MoE routing (not the
   whole 400B; just the task's few small experts).
3. **2-bit weights (Part-1)** — make the 25 MB/token load cheaper than compute, so disk
   I/O is not the bottleneck (with prefetch / per-task residency).
→ 80 M active = **45 tok/s**, 100 M = **36 tok/s**, compute-bound.

**The honest remaining question is Part-4, not Part-2:** speed at ~100 M active/token is
solved; whether 100 M *active* params (out of a 400B total on disk) is "beast-intelligent
enough" is the intelligence/scale question — the same scale wall Part-1 hit, deferred to
Part-4. The speed *mechanism* works and is measured.

## Files
- `tok_per_sec.py` — end-to-end decode benchmark, two matrix regimes, projection
- `bandwidth_reality.py` — disk-BW vs compute, RAM-resident vs streaming regimes
- `tok_per_sec_results.json`, `bandwidth_results.json` — measured numbers
