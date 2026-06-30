# 🏛️ ATLAS — Master Progress (the one place to look). Updated 2026-06-28.

**ATLAS** (Active-sparse, Tiered-memory, Low-bit, Augmented-with-tools, Self-verifying) =
the 11-lever AetherCore-V2 architecture. Carries a giant model on tiny hardware.
Full design: `COMPLETE_ARCHITECTURE.md`.

This file is the memory of the whole "ATLAS = 101× better AI" effort: the framework, every
task's status, the measured results, and the honest lessons. If you forget where things
are — start HERE.

## The mission
Build "AI V2": **101× better than current AI** — but honestly:
- 🟢 EFFICIENCY (same result at 101× less size/memory/energy/cost) = REAL, achievable.
- 🔴 ABSOLUTE intelligence (101× smarter than GPT-4) = physics-blocked (intelligence ≤
  compute ≤ hardware). Reframed: "match best quality at 101× less cost on 101× weaker HW".

Design docs: `01_how_current_ai_works.md`, `02_problem_tracker.md`, `03_target_101x.md`,
`04_attack_flow.md`. This is `00_PROGRESS.md` (status). Older base work: `projects/ROADMAP.md`,
`projects/DREAM_BLUEPRINT.md`.

## The 11 tasks + attack order
`T1 Size → T2 Memory → T3 Speed → T4 Context → ⑥ INTEGRATION → T6-T10 (make smart) ; T11 train = physics`

## SCOREBOARD (measured)

| Task | 101× goal | Status | Best measured result | Needs GPU/training? |
|---|---|---|---|---|
| **T1+T11** | 101× / native-train | ✅ METHOD PROVEN (small, no-GPU) | **native-sparse char-LM, CPU, no GPU: 95% sparse (~0.13 bits) NATIVE acc 0.970 (99% of dense 0.977) vs post-hoc 0.184 collapse = 5.3× better.** 0.15-bit @ near-dense quality PROVEN on real language. No-GPU post-hoc ceiling ~19× (GPTQ/SparseGPT). Fast-tech ~50× shifts CPU ceiling to ~100M; 1B+ still GPU | proven small no-GPU; big=GPU |
| **T2 Memory** | 101× less RAM | ✅ mechanism | paging: 1.6GB model in 67MB RAM (24×); projects 640× | NO |
| **T3 Speed** | 101× faster | ✅ mechanism | paged-LUT: 68MB RAM + 4.13× (vs fp32 paging); budget 106× vs naive fp32 | NO |
| **T4 Context** | 101× longer | ✅ proven | 15M tokens = 41MB, flat latency, multi-hop (Day-19) | NO |
| **T5 Energy** | 101× cheaper | ✅ mechanism | 126× at 1/8 sparsity (DRAM-bytes dominate → low-bit is top lever) + ~100× hw-cost tier | NO |
| **T6 Intelligence/compute** | same smarts cheaper | ✅ mechanism | test-time compute on Qwen-1.5B: 0.38→0.88 (direct→self-consistency), no training/GPU | NO (inference) |
| **T7 Reasoning depth** | 101× deeper | ✅ principle | flat p^N collapses (0.85^20=4%); per-step verify holds (0.99^20=82%) → decompose+verify-each | NO |
| **T8 Reliability** | 101× fewer hallucin. | ✅ mechanism | grounding + honest-IDK: hallucinations 3→0 (answer only what you can ground) | NO |
| **T9 Capability** | 101× more tasks | ✅ mechanism | router+tools: 6/6 across math/code/fact (each tool adds a capability) | NO |
| **T10 Latency** | 101× faster start | ✅ mechanism | speculative decoding ~2.4-3.4× (lossless) + small-active (90M→25ms vs 7B→1925ms) | NO |
| **T11 Training** | 101× cheaper | 🔴 physics | ~27× stacked (distill/data/low-bit/LoRA), NOT 101× — must do the compute to learn | YES |

## DETAILED RESULTS (so we don't forget HOW)

### T1 SIZE — `projects/v2_design/T1_size/` (see SPARSITY_MAP.md)
- 101× size = stack of ~5 levers (2-bit 8× × sparsity 4× × shared-codebook × embed × dedup ≈ 107×).
- 0.15-bit is impossible PER-WEIGHT (entropy ~2 bits, Day-1); reachable as AVERAGE via ~98% ZEROS.
- Sparsity ladder (98% sparse, 2% weights, toy task, dense=0.65):
  - post-hoc prune: 0.011 (DEAD) → never the path
  - random-native: 0.287 → native training works
  - smart (lottery) mask: 0.384
  - **PROPER RigL (gradient regrowth): 0.540 = 83% of dense** ← ceiling broke (was "capacity", was mask-quality)
- **HONEST WALL:** on REAL GPT-2 with tiny CPU healing (60 steps), RigL & post-hoc BOTH collapse
  (~50-60× FP). Native sparse needs REAL training compute → toy works, real LM needs a GPU.
- Earlier RED (don't re-try): plain VQ sub-1-bit collapses; rotation doesn't help VQ; residual VQ loses.

### T2 MEMORY — `projects/v2_design/T2_memory/` (see MEMORY_MAP.md)
- Lever: page layers from disk, one resident at a time → peak RAM = one layer.
- Measured: 1.6GB model runs in 67MB (24×). reduction ≈ n_layers × (16/bits) → 80-layer 2-bit = 640×.
- Gotcha: naive 2-bit paging is SLOWER (unpack-to-fp32 cost). Fix = compute on packed (T3) + prefetch.

### T3 SPEED — `projects/v2_design/T3_speed/`
- **KEY WIN (paged_lut.py): computing DIRECTLY on packed ternary (LUT kernel) = lowest RAM (68MB)
  AND fastest (4.13× vs fp32 paging, 35× vs unpack). Breaks the T2 memory↔speed tension.**
- 101× budget (vs naive fp32 dense): kernel 4.13× × sparsity 8× × bandwidth 2× × matrix 1.6× = 106×.
- Honest: baseline is naive fp32, NOT optimized llama.cpp; compute-side only.

### T4 CONTEXT — `projects/day19_context/` (done earlier)
- 15M-token context via retrieval (inverted index): recall 1.000, 41MB, flat ~0.15ms latency.
- Multi-hop: single-hop/naive RAG 0.000 vs iterative multi-hop 1.000 at 15M. (vs attention's 2.9TB.)

## THE BIG HONEST LESSONS (recurring)
1. **Efficiency (memory/speed/context) is winnable on this CPU without training.** T2/T3/T4 done.
2. **Anything needing TRAINING a real model needs a GPU** (T1 extreme sparsity). Toy works, real doesn't.
   Free options exist: Kaggle (30h/wk free GPU), Colab; or heavy CPU healing (slow); or small-LM-from-scratch.
3. **101× is EFFICIENCY, never absolute intelligence** (physics).
4. **Measure, don't assume** — we killed many plausible levers honestly (post-hoc sparsity, residual VQ,
   2-bit-no-heal Qwen gibberish, naive 2-bit paging). Negatives are results.

## WHAT'S NEXT (when resuming)
- **⑥ INTEGRATION** (no training): combine 4-bit weights + paging + LUT kernel + retrieval + self-review
  into ONE working engine on a real model. Highest value; the current gate.
- **T6 test-time compute** (no training): make a small-active model reason better by thinking longer.
- **T1 real-scale** (needs GPU): validate native-sparse RigL on a real LM via free Kaggle/Colab GPU.

## Real models on disk
- `models/gpt2` (124M, trainable on CPU) · `models/qwen2.5-1.5b` (1.5B, inference-only on CPU)
- 2-bit conversions work (streaming converter, `projects/day22_streaming/` + `day23_realrun/`),
  but 2-bit-NO-HEAL breaks quality (Qwen gibberish) — 4-bit is the no-training usable point.
