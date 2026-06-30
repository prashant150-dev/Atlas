# ATLAS — Flow & Honest Drawbacks

## PART A — THE FLOW

### A1. Build flow (how a model becomes ATLAS) — one-time
```
1. DOWNLOAD     existing pretrained model (e.g. Llama 7B)              [internet]
2. QUANTIZE     weights -> 2-bit VQ (+ GPTQ error-compensation)        [CPU]
3. SPARSE-ADAPT prune to ~95-98% + RigL native fine-tune to recover    [GPU, one-time] ⚠️
                (distill from the original model; surviving weights
                 learn to do the removed weights' job)
4. PACKAGE      paging layout (T2) + LUT indices (T3) + retrieval       [CPU]
                index (T4) + tool wiring (T9)
5. READY        ~0.15-bit ATLAS model on disk
```
Only step 3 needs a GPU; everything else is CPU.

### A2. Runtime flow (how a query is answered) — every prompt
```
prompt
  │
  ▼
[ROUTER T9]      detect task type (math / code / fact / reason / chat)
  │
  ▼
[RETRIEVE T4,T8] pull relevant chunks from 15M-token memory; ground the answer
  │
  ▼
[LOAD T2]        page in only this task's ~90M active experts (2-bit), layer by layer
  │
  ▼
[COMPUTE T3]     LUT kernel runs matmuls directly on packed weights (no multiplies)
  │
  ▼
[THINK T6,T7]    test-time compute: plan -> solve each step -> verify each step
  │
  ▼
[TOOLS T9]       exact execution: calculator / run-code / lookup (100% on verifiable)
  │
  ▼
[VERIFY T8]      self-review (bounded); if not groundable -> honest "I don't know"
  │
  ▼
answer  (small, fast, grounded; near-FP on verifiable, model-bound on creative)
```

---

## PART B — HONEST DRAWBACKS (the real weaknesses)

### Hard limits (can't fully fix)
1. **One-time GPU step is unavoidable** — 0.15-bit at quality needs native sparse training
   (post-hoc collapses, measured). For big models that means a GPU (free Kaggle or ~$100-300).
2. **Intelligence is bounded by ACTIVE params** on non-verifiable tasks — ~90M-active reasons
   like a small model for creative/open work. Tools rescue only VERIFIABLE tasks.
3. **Speed ↔ intelligence tradeoff is real** — test-time compute makes hard answers slower;
   you can't max both.
4. **Autoregressive + test-time = slow long outputs** — a big chess-game-with-UI generation
   is many sequential steps; wall-clock is long even if per-token is cheap.

### Validation gaps (proven small, NOT at scale)
5. **Everything is proven at TINY scale** — char-LM, GPT-2, Qwen-1.5B. The 99% intelligence at
   0.15-bit is a PROXY/projection, NOT measured on a real big model. It could degrade at scale.
6. **0.15-bit @ full quality unproven on a real LLM** — our 99% is char-level (inherently easy);
   no one in the field has usable models at 0.15 bits/weight. This is the riskiest claim.
7. **Routing quality is the hidden bottleneck** — "90M-active = big-total smart" ONLY holds if
   the router picks the RIGHT experts. Bad routing = bad answers. Routing at scale is hard and
   we have NOT validated it on a real big MoE.

### Engineering / practical
8. **Per-token expert streaming risk** — if active experts change every token and must stream
   from disk, speed collapses. The dream assumes a stable per-task working set (unproven at scale).
9. **Speed wins are vs NAIVE fp32, not llama.cpp** — against already-optimized int4 inference the
   advantage shrinks a lot. Honest baseline matters.
10. **CPU sparse compute is regime-dependent** — sparse-skip only helps big matrices at 98% sparse;
    small/medium see overhead. Not a universal speedup.
11. **Complexity = more failure modes** — 11 levers + router + tools + retrieval + verify is a big
    system; harder to build, debug, and keep correct than one model.
12. **Retrieval ≠ true attention** — 15M "context" is fetched chunks, not full attention; subtle
    cross-context reasoning that true attention catches can be missed (multi-hop helps, not equal).
13. **Tool/verify help only where verifiable** — creative, aesthetic, open-ended quality is capped
    at the base model's ceiling; no external tool lifts it.
14. **Calibration/data dependence** — GPTQ/SparseGPT need good calibration activations; sparse-adapt
    needs fine-tuning data. Garbage calibration → worse results (we hit this: random calib failed).
15. **Text-only** — no images / audio / video.

---

## PART C — Honest one-line verdict
> **ATLAS is a real, measured efficiency architecture (100×+ on size/memory/speed/energy/context,
> CPU-proven) whose intelligence MATCHES current AI on VERIFIABLE tasks (tools+verify) but is
> bounded on CREATIVE tasks, requires a ONE-TIME GPU for big-model 0.15-bit, and is VALIDATED ONLY
> AT SMALL SCALE — the scale-up (routing quality, 0.15-bit on a real LLM) is the main open risk.**
