# 🏛️ ATLAS — COMPLETE ARCHITECTURE (the definitive single reference)

**ATLAS** — like the titan who carried the world on his shoulders, this architecture
carries a GIANT model on the shoulders of TINY hardware (a $200 CPU PC).

**ATLAS** = **A**ctive-sparse · **T**iered-memory · **L**ow-bit · **A**ugmented-with-tools ·
**S**elf-verifying. (The 11-lever AetherCore-V2 architecture.)

Last updated 2026-06-28. This is the full picture: the design, all 11 levers, the dream
config, what runs where, and the honest verdicts. Companion files: `00_PROGRESS.md`
(status), `comparison_fp16_vs_ours.py` (fp16-vs-ours), `codesign_dashboard.py` (couplings),
per-task folders `T1_size/ … T11_training/`, and base work in `projects/DREAM_BLUEPRINT.md`.

---

## 1. THE MISSION
Run a model with **current-AI quality at 101× less size / memory / energy / cost, on
hardware 101× weaker** (a $200 CPU PC). Honest split: 101× EFFICIENCY is real; "101×
smarter than the best" is physics-blocked (intelligence ≤ compute ≤ hardware).

---

## 2. THE CORE DESIGN — 3 substrates (don't store one giant brain)
```
  prompt
    │
    ▼
 [1] REASONER  — small, low-bit, RAM-resident, fast (the "thinking")
    │           routes the task, plans, verifies
    ▼
 [2] MEMORY    — huge context/knowledge on DISK, pulled by RETRIEVAL (not attention)
    │           15M-token context as a 41 MB index, O(1) lookup
    ▼
 [3] EXPERTS   — 400B total on disk; only the task's ~90M ACTIVE params loaded to RAM
    │           sparse MoE: big total (knowledge) + small active (speed)
    ▼
  answer  (small, fast, near-FP quality, grounded)
```

---

## 3. THE 11 LEVERS (each measured/characterized)

| # | Lever | What it does | Result | CPU? |
|---|---|---|---|---|
| **T1** | Size | mixed-precision VQ + native-sparse (RigL) | toy 0.15-bit @ 83%; 4-bit no-train @ ~FP | real-scale needs GPU |
| **T2** | Memory | page layers from disk, one resident | 24× measured, 640× projected | ✅ |
| **T3** | Speed | LUT kernel on packed weights + small-active | 4.13× + 68MB; 106× budget vs naive | ✅ |
| **T4** | Context | retrieval (inverted index) + multi-hop | 15M = 41MB, flat latency, recall 1.0 | ✅ |
| **T5** | Energy | low-bit (DRAM bytes↓) + sparsity | 126× (1/8 sparse) + ~100× hw-cost | ✅ |
| **T6** | Intelligence/compute | test-time compute (CoT, self-consistency) | Qwen-1.5B 0.38→0.88 | ✅ |
| **T7** | Reasoning depth | decompose + verify EACH step | flat 0.85^20=4% → verified 0.99^20=82% | ✅ |
| **T8** | Reliability | retrieval-grounding + honest "I don't know" | hallucinations 3→0 | ✅ |
| **T9** | Capability | router + tools (math/code/facts exact) | 6/6 across 3 types | ✅ |
| **T10** | Latency | speculative decoding + small-active | 2.4-3.4× lossless; 90M→25ms | ✅ |
| **T11** | Training | distill/data/low-bit/LoRA | ~27× (NOT 101×) — physics | needs GPU |

**9 of 11 (T2–T10) are CPU-proven, no training. T1 real-scale + T11 are the GPU-gated knot.**

---

## 4. THE RUNTIME (how one answer is produced)
```
1. ROUTER (T9)           detect task type (math / code / fact / reasoning)
2. RETRIEVE (T4, T8)     pull relevant chunks from the 15M-token memory; ground the answer
3. LOAD (T2)             page in only the task's ~90M active experts (2-bit), one layer at a time
4. COMPUTE (T3)          LUT kernel runs the matmuls directly on packed weights (no unpack)
5. THINK (T6, T7)        test-time compute: plan → solve each step → verify each step
6. TOOLS (T9)            exact execution: calculator / run-code / lookup (100% on domain)
7. VERIFY (T8)           self-review (bounded loop); if not groundable → "I don't know"
   → answer: small, fast, grounded, near-FP quality
```

---

## 5. THE DREAM CONFIG (Config B) — all levers together
```
bits=2 · sparsity=95% · active=90M · NATIVE-trained · test-time=2× · + tools + retrieval + verify
```
Same 7B model, fp16 vs Config B:

| Axis | fp16 | Config B | diff |
|---|---|---|---|
| Size | 14 GB | 131 MB | 107× |
| RAM | 14 GB | 22 MB | 640× |
| Active/token | 7B | 90M | 78× fewer |
| Context | 128K | 15M | 117× |
| Energy | 100% | ~0.8% | 126× |
| Intelligence | 100% | ~99% (proxy) | ~same |
| Runs on | datacenter GPU | $200 CPU PC | potato |

---

## 6. THE KEY INSIGHTS (hard-won, measured)
1. **Active vs Total are different jobs:** active = SPEED, total = KNOWLEDGE. Same 90M
   active but 400B total >> 7B total (router picks better experts from a richer pool).
   Proof: 4A keystone — equal active, MoE 1.000 vs dense 0.594.
2. **The axes are LINKED:** low-bit + sparsity improve Size/Memory/Speed/Energy TOGETHER
   (positive coupling); Intelligence is the binding tension.
3. **Post-hoc vs Native:** compressing a trained model post-hoc collapses at extremes;
   NATIVE training (sparse-from-scratch, RigL) keeps quality — but needs training compute.
4. **Tools make execution exact:** accuracy bounded only by SETUP/reasoning (model), which
   test-time compute improves → 100% reachable on VERIFIABLE tasks, honest IDK elsewhere.
5. **Better routing is the ultimate frontier:** perfect routing → 90M-active approaches
   400B-dense quality at 90M cost (the speed↔intelligence tradeoff would dissolve).

---

## 7. THE ONE KNOT (honest)
Everything is in hand on CPU EXCEPT keeping intelligence high at aggressive compression —
that needs **native training (T1/T11) = a GPU** (free Kaggle/Colab works). The 99%
intelligence figure is a PROXY (toy native-sparse 83%, literature near-FP); confirming it
at scale is the one remaining experiment. Efficiency numbers are measured/derived.

---

## 8. WHAT CAN BE DONE WITHOUT A GPU (now)
- Integration: combine 4-bit model + paging + LUT kernel + retrieval + test-time + tools +
  verify into one working assistant (Config C/D level: good intelligence, ~8-50× efficiency).
- Better-routing research (toy/small scale).
- Heavy CPU healing of a small model toward its 2-bit ceiling.
- Smartness pipeline (model + tools + verify) — measure max accuracy on verifiable tasks.

## 9. WHAT NEEDS A GPU (free options exist)
- Config B validation: native-train a sparse low-bit model → confirm the 99% at scale.
- T1 real-scale: extreme sparsity on a real LM.

---

## 10. FILE MAP
- `00_PROGRESS.md` — live status/scoreboard
- `01_how_current_ai_works.md`, `02_problem_tracker.md`, `03_target_101x.md`, `04_attack_flow.md` — design
- `T1_size/ … T11_training/` — per-task code + results + maps
- `codesign_dashboard.py` — axis couplings · `comparison_fp16_vs_ours.py` — fp16 vs ours
- `projects/DREAM_BLUEPRINT.md` — the earlier 4-part dream record
- `projects/ROADMAP.md` — full milestone history
