# AetherCore — Master Report (Day-1 → Day-7)

> Rule of the project: **claim nahi, evidence.** Every number below is measured on
> an Intel i5-4590T (4c/4t @2.0GHz), 8 GB RAM, no GPU, CPU-only. Where a figure is
> a projection (not run), it is labelled ESTIMATE.

---

## 1. What we set out to do

Dream: a very large, very smart model on this 8 GB CPU — small footprint, minimal
quality loss. Decomposed into 4 measurable targets:
**P-A Size**, **P-B Speed**, **P-C Context**, **P-D Intelligence**.

This report covers the architecture and proofs built so far, with the math.

---

## 2. The architecture we built (two connected stacks)

```
  ┌──────────────────────── THE COMPRESSION STACK (P-A: Size) ─────────────────────────┐
  │  weights ── Vector Quantization (shared codebook) ── + Healing (codebook fine-tune) │
  │            └ groups of d weights → 1 index into K-entry codebook  (~2 bits/weight)  │
  │  experts ── MoE sparsity (N experts, top-k active)  → big TOTAL, small ACTIVE       │
  │  kernel  ── packed-ternary matmul (add-only, 2-bit resident)  → RAM floor proof     │
  └────────────────────────────────────────────────────────────────────────────────────┘
                                   ▼ supplies a small, cheap "brain"
  ┌──────────────────────── THE ARCHITECTURE STACK (P-C/P-D) ──────────────────────────┐
  │  [ small reasoner ] ──hybrid retrieval (lexical + learned)──> [ external memory ]    │
  │   in RAM, fast            free + alias-robust                  on disk, ~55 b/fact   │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

The two were merged on Day-7: the reasoner/experts are stored with VQ+MoE; the
knowledge lives in external memory. Each block below is a measured result.

---

## 3. Proven results, with the calculations

### D1 — the compression floor (the law we must respect)
- GPT-2 2D weights behave ~Gaussian; rate-distortion gives **NMSE = 2^(-2R)**.
- Their measured entropy ≈ **2.04 bits/weight** (independent per-weight coding).
- **Proof a monolithic 400B/1T can't fit 8 GB:** to fit 1T params in 8 GB,
  `8·8e9 bits / 1e12 = 0.064 bits/weight` needed — **32× below the 2.04 floor** →
  impossible. (56 GB disk → 0.45 bits/wt, still below the ternary regime.) ESTIMATE
  from the floor; the floor itself is measured.

### D2 — healing (change the goal: behaviour, not weights)
- Naive ternary GPT-2: top-1 **3%**, ppl **27,325**.
- After 30 distillation steps: top-1 **28%**, ppl **401**, KL 6.0→2.05.
- Run time: **268.6 s**.

### D3 — native low-bit beats post-hoc (co-design)
- char_lm: DenseFP **0.918**, PostHocTernary **0.470** (collapse), AetherNet
  (native ternary+MoE) **0.838**, at ~**2×** smaller stored. Run ~**483 s**.

### D4 — honest healing ceiling
- P1 (in-sample): top-1 plateaus **35–47%** by ~60 steps (memorisation). Run **2032 s**.
- P1.1 (held-out): honest ceiling **27–30%** top-1 at ~2 bits, then overfits. Run **1076 s**.
- Lesson: 2-bit heal-only is not FP-quality; need a better lever → led to VQ.

### D5 — reasoner + external memory (R1–R7)
- **R1:** closed-book (weights only) **0.00** vs open-book (retrieved) **1.00**,
  flat as KB grows 20→500 → capability scales with memory, not parameters.
- **R2:** content store **~55 bits/fact** (gzip), linear → **1B facts ≈ 7 GB**
  (calc: 1e9 × 55 / 8 / 1e9 = 6.9 GB). Dense GPT-2 embedding index was a poor
  retriever (0.10 vs lexical 1.00).
- **R3:** lexical retrieval 1.00 when the entity is named, **0.00** on aliases; a
  small learned 768→128 projection recovers **0.825 retrieval / 0.883 answers**.
- **R4 (end-to-end):** full pipeline on 8 GB CPU — **0.912** answer accuracy,
  **15.4 tok/s**, memory 55 b/fact.
- **R5 (merge):** int8 reasoner **128 MB @ 0.912** (free win); post-hoc ternary
  36 MB but **0.000** (collapse). RAM/speed identical across all → on CPU the
  compression win is **disk only** without a low-bit kernel.
- **R6:** native low-bit > post-hoc where the wall exists (char_lm 0.838 vs 0.470);
  not universal.
- **R7:** packed-ternary matmul kernel — RAM **15.7×** smaller (2.05 bits/wt,
  fp32 never materialised), **weight-multiplies 37.7M → 0 (768× fewer)**, bit-exact.
  Honest: wall-clock speed needs a SIMD/C kernel (NumPy 53 ms vs BLAS 0.8 ms).

### D6 — Vector Quantization: the genuinely-new size lever
**The key calculation (VQ bits/weight):**
```
bits/weight = (log2(K) · n_weights/d  +  K·d·32) / n_weights
            =  log2(K)/d              +  (K·d·32)/n_weights
                 ^ index cost            ^ shared codebook overhead (amortised)
for d=4, K=256, big matrix:  8/4 + tiny  ≈ 2.02 bits/weight
```
- **P1 (reconstruction):** at ~2 b/w, VQ NMSE **0.109** vs scalar ternary **0.224**
  — ~2× lower error at equal size. Why it's legal: D1's 2.04 floor is the
  *per-weight marginal* entropy; coding groups jointly uses **joint entropy < sum
  of marginals**. Run ~**100 s**.
- **P2 (behaviour):** whole-model VQ ppl **1,880** vs ternary **49,357** at equal
  ~2 b/w (**26×** better); FP teacher 8.19.
- **P3 (CROWN):** VQ+healing **ppl 94.7** vs ternary+healing **663.3** (~**7×**),
  within ~2× of FP (48.4), held-out. VQ *post-hoc* (458) already beats ternary
  *fully healed* (663).
- **P4 (honest cliff):** sub-1-bit plain VQ collapses; best (0.57 b/w) heals only
  to ppl ~4,000 — unusable. Rotation "transform" proven useless for VQ (NMSE
  0.45314 == 0.45314). Sweet spot = ~2-bit VQ+heal.

### D7 — VQ + healing + MoE sparsity together
**The key calculation (combined budget):**
```
stored(VQ-MoE) = log2(K)/d · (per-expert weights · N_expert) + 2·K·d·32 + router
active(VQ-MoE) = log2(K)/d · (per-expert weights · top_k)   + router
   sparsity factor  = top_k / N_expert       (e.g. 2/8 = 4×)
   VQ factor        = 16 / 2  = 8×           (fp16 → 2-bit)
   combined active  ≈ 4× · 8× = 32×   (measured 21–51× vs dense-big)
```
- **P1 (char_lm, capacity-saturated):** vs DenseFP-big at iso-accuracy — stored
  **12.2×** smaller, active/token **51×** smaller.
- **P2 (capacity-hungry, 4,800 mappings):** capacity now matters —

  | variant | accuracy | stored | active/token |
  |---|---|---|---|
  | DenseFP-small (H16) | 0.791 | 65.5 kb | 65.5 kb |
  | DenseFP-big (H128) | 0.950 | 524.3 kb | 524.3 kb |
  | MoE-FP (8×16, top2) | 0.951 | 540.7 kb | 147.5 kb |
  | **VQ-MoE + heal** | **0.961** | **114.7 kb** | **24.6 kb** |

  VQ-MoE matches the big model's accuracy at **4.6× less stored** and **21× less
  active**, and beats the small model on **both** accuracy and active cost. The
  dream shape — big capability, small stored, sub-small active — measured.

---

## 4. The architecture's headline equations (proofs in one place)

1. **Per-weight floor (D1):** `min bits/weight ≈ H(W) ≈ 2.04` for independent coding.
2. **VQ beats it (D6):** `bits/weight = log2(K)/d + K·d·32/n` ; joint coding →
   below the marginal floor (measured 2× lower NMSE, 7× lower healed ppl).
3. **MoE effective ratio (D7):** `active/total = top_k/N_expert`.
4. **Combined (D7):** `active_saving ≈ (N_expert/top_k) · (16/bits_vq)` → measured 21–51×.
5. **Memory scaling (R2):** `disk = facts · 55 / 8` bytes → 1B facts ≈ 7 GB.
6. **Kernel RAM floor (R7):** resident `= bits_vq/16 ×` the fp16 size → 15.7× smaller,
   0 weight-multiplies.

---

## 5. Time spent (measured CPU compute, from run logs)

These are pure experiment run-times (the CPU actually crunching), not the
back-and-forth. Cumulative-checkpoint runs use their final time.

| experiment | measured run time |
|---|---|
| D2 healing | 269 s |
| D3 native vs post-hoc (2 tasks) | ~483 s |
| D4 P1 healing ceiling sweep | 2,032 s |
| D4 P1.1 held-out ceiling | 1,076 s |
| D5 R1–R7 reasoner+memory (sum) | ~1,500 s |
| D6 P1–P4 vector quantization (sum) | ~2,000 s |
| D7 P1+P2 VQ-MoE (incl retunes) | ~900 s |
| **Total measured compute** | **≈ 8,260 s ≈ 2.3 hours** |

(D1 rate-distortion + R8/R2b calculators were near-instant; excluded.)

**Honest framing of "time":** ~**2.3 hours of actual CPU compute** produced the
whole evidence base. The calendar labels (Day-1…Day-7) are milestone names, not
seven days — the bulk was built in focused sessions. No GPU, no cloud, one 2014
desktop.

---

## 6. Where each dream target stands (honest)

| target | status | best measured result |
|---|---|---|
| **P-A Size** | 🟢 **strong win** | VQ+heal ~2 b/w beats ternary 7×; VQ-MoE big-acc at 4.6× stored / 21× active |
| **P-B Speed** | 🟡 path proven, not hit | 15.4 tok/s actual; kernel math (R7) proves RAM/0-mult, needs SIMD/C for speed |
| **P-C Context** | 🟡 reframed | retrieval-as-context works (R1–R4); true 15M attention impossible |
| **P-D Intelligence** | 🟡 partial | native low-bit + MoE give capacity-per-bit; real "beast" needs scale (hardware gap) |

---

## 7. Honest limitations (what this is NOT)
- Mostly GPT-2-small / tiny synthetic models; single seeds; perplexity/accuracy
  proxies. The *method* is proven small; LLM-scale proof needs bigger hardware.
- Sub-1-bit, true 40–50 tok/s, and a genuinely "beast" reasoner are **not** achieved
  here — they are bounded by hardware/scale, not by a missing idea.
- Speed wins from low-bit need a real SIMD/C kernel (not built).

## 8. The one-line truth
On a 2014 8 GB CPU, in ~2.3 hours of compute, we built and **measured** a new
size/efficiency architecture — **vector-quantization + healing + shared-codebook
MoE sparsity**, feeding a **small-reasoner + external-memory** stack — that beats
strong baselines at equal size and reproduces the dream's "big-but-cheap" shape at
small scale.

**Honest scope of the gap (revised after Day-8 validation):** the remaining gaps
are *largely* hardware/scale, but **not purely** — unknown algorithmic barriers at
large scale remain possible (large-MoE routing instability, cross-layer codebook
degradation [measured ~1.2× penalty, Stage 5], the sub-1-bit cliff [Day-6 P4],
architecture-dependent floors [Stage 5]). The defensible claim is: *promising,
well-measured small-scale evidence; the jump to 400B is an open empirical question,
not a foregone conclusion.*

## 9. Day-8 external-critique validation (what survived scrutiny)
A reviewer raised 20 weaknesses. Measured responses (`projects/day8_validation/`):
- **Real English + 5-seed-style ablation (#3,4,6,19,20):** VQ+heal 304±34 vs
  ternary+heal 1323±470 — non-overlapping, not a lucky seed.
- **SOTA crossover (#16):** scalar (GPTQ/AWQ-style) wins at 4-bit; VQ wins at 3-bit
  and dominates at 2-bit (~14×). We are a *low-bit* method, honestly positioned.
- **Scaling (#1,18):** VQ advantage constant ~1.9× over a 256× weight range
  (controlled) and 1.6–3.4× across real GPT-2 (65× range) — does not decay.
- **Cost/speed (#7,13,14):** no CPU speed win without a kernel (honest); healing is
  a negligible one-time cost (~2.8e-7 of pretraining).
- **MoE routing / shared codebook / memory realism / floor-by-type (#2,8,9,10,15):**
  routing healthy (0 dead experts); cross-layer codebook ~1.2× penalty (use
  per-layer); rich facts 3.2× the templated estimate but still ~9.5 GB/1B.
- **Still open:** standard task benchmarks (#5), multi-hop (#11), forgetting (#12),
  full MoE-on-language (#2), other architectures (#10), official-library numbers (#16).
