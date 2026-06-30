# AI V2 — Foundation Study #1: How current AI works & where it hurts

Before redesigning AI, understand EXACTLY how today's LLMs work and which limits are
real. This doc maps the full pipeline and every major problem, tagging each as
🟢 DESIGN (changeable — V2 opportunity) or 🔴 PHYSICS (fundamental — cannot remove).

---

## PART A — How a modern LLM actually works (the pipeline)

```
text -> [1 tokenize] -> [2 embed] -> [3 transformer layers xN] -> [4 unembed] -> next token
                                            ^                                         |
                                            └──────── repeat per token (autoregressive)
```

**1. Tokenization** — text is chopped into "tokens" (sub-words), each a number.
   "unhappy" -> ["un","happy"] -> [403, 7211]. ~50k-150k vocabulary.

**2. Embedding** — each token id -> a vector (e.g. 4096 numbers). A big lookup table.

**3. Transformer layers (the core, repeated 30-100×)** — each layer has TWO parts:
   - **Attention**: every token looks at every other token to decide what's relevant
     ("the cat ... it" — "it" attends to "cat"). Cost = O(n²) in sequence length n.
     Uses a KV-cache during generation (stores past keys/values).
   - **FFN / MLP**: a big 2-layer network — this is where most PARAMETERS and most
     "knowledge/computation" live. ~2/3 of all weights.
   Stack N of these; each refines the representation.

**4. Unembed (LM head)** — final vector -> a score for every vocabulary token ->
   softmax -> probability -> pick the next token.

**5. Autoregressive generation** — produce ONE token, append it, run the WHOLE network
   again for the next token. A 100-word answer = ~130 full forward passes. Sequential.

**6. Training (done once, offline, hugely expensive)**
   - **Pretraining**: predict the next token over trillions of words. Months, $10-100M.
   - **Fine-tuning + RLHF**: make it follow instructions / be helpful & safe.

---

## PART B — The major problems (where it hurts), tagged

### 1. 🟡 SIZE — billions of weights
A 70B model in fp16 = 140 GB; 400B = 800 GB; GPT-4-class ~ TBs. Won't fit consumer RAM.
- **Why**: capability scales with parameters (more params = more knowledge/skill).
- **Movable?** PARTLY. Low-bit (our 2-bit) + sparsity shrink storage a lot. But you
  can't compress *information* below its entropy (🔴 floor ~1-2 bits/useful-weight).

### 2. 🔴+🟡 COMPUTE — mountains of multiply-adds
Each token = (params × 2) FLOPs roughly. 70B model = ~140 GFLOPs PER TOKEN. Needs GPUs.
- **Movable?** The ALGORITHM is partly movable (sparsity → fewer active; better kernels →
  T-MAC/LUT; lower precision). But "intelligence needs computation" is 🔴 PHYSICS — you
  cannot get N units of reasoning for free. Compute-per-second = hardware.

### 3. 🟡 MEMORY BANDWIDTH — the real decode bottleneck
Generating one token reads ALL the active weights from memory. Decode is BANDWIDTH-bound,
not compute-bound: the GPU/CPU waits for weights to arrive. This is why low-bit helps
speed (fewer bytes to move), and why our 2-bit unlocked it.
- **Movable?** YES — smaller weights (low-bit), keep hot weights in fast memory, sparsity.

### 4. 🟡 ATTENTION is O(n²) — long context explodes
Doubling the context 4×s the attention cost; the KV-cache grows linearly and eats RAM
(15M tokens ≈ TBs of KV — impossible). This caps context length.
- **Movable?** YES, actively being replaced: FlashAttention (faster exact), Mamba/SSM &
  RWKV (linear O(n), no quadratic attention), and RETRIEVAL (our approach — don't attend
  to 15M, fetch the few relevant chunks). Big V2 frontier.

### 5. 🟡 DENSE activation — every weight used for every token (wasteful)
A dense model runs ALL its parameters for every single token, even trivial ones.
- **Movable?** YES — **MoE/sparsity**: route each token to a few expert sub-networks.
  Active params << total. (We proved sparse-active ≈ big-total capacity.) Huge lever.

### 6. 🔴 AUTOREGRESSIVE — one token at a time, sequential
You can't compute token #50 before #49; generation can't be parallelized across the
sequence. This caps latency fundamentally for a given model.
- **Movable?** PARTLY — speculative decoding (draft + verify), diffusion-LM, multi-token
  prediction help, but the left-to-right dependency is largely 🔴 inherent to the task.

### 7. 🔴 TRAINING COST — months, millions of dollars, megawatts
Pretraining a frontier model needs thousands of GPUs for months.
- **Movable?** Efficiency (better data, curricula, distillation) helps, but the raw
  compute to *learn* trillions of tokens is largely 🔴 hardware/energy bound.

### 8. 🟡 ENERGY — datacenters draw megawatts
Inference at scale + training = enormous energy (and $$$).
- **Movable?** YES via efficiency (low-bit, sparsity, better kernels) — but bounded below
  by 🔴 Landauer-ish thermodynamic limits on computation.

### 9. 🟡 QUALITY problems — hallucination, forgetting, no grounding
Models confidently make things up, forget old context, aren't grounded in truth.
- **Movable?** YES — retrieval/grounding (our memory substrate), verification/self-review
  (our bounded loop), better training. Active V2 territory.

---

## PART C — The one law to never forget

> **Intelligence-per-second is bounded by Computation-per-second, which is bounded by
> Hardware.** No architecture removes this — it is 🔴 physics. Everything we CAN do is make
> each unit of computation cheaper/smarter (efficiency), not make computation free.

So a realistic "AI V2" = **10-100× more EFFICIENT** (real, world-changing) — NOT
"unlimited intelligence on a potato" (impossible). The movable 🟡 limits above are where
a determined builder wins; the 🔴 ones are where years get wasted.

---

## PART D — The V2 opportunity map (what to attack)

| Current AI uses | V2 could use | Status |
|---|---|---|
| Dense Transformer | Sparse MoE + retrieval | partly built here |
| Quadratic attention | Linear (SSM/Mamba/RWKV) or retrieval | open frontier |
| FP16 weights | Low-bit (2-bit) + healing | built here |
| Plain matmul | LUT/T-MAC kernels (no multiplies) | built here |
| One-shot answer | Test-time compute + self-review | started here |
| Ungrounded | Retrieval-grounded memory | built here |
| Reload weights/token | Task-conditional residency | designed here |

**We are already building AetherCore = a V2 attempt on the 🟡 limits.** Next docs:
#2 quantify each limit with real numbers; #3 design V2's architecture against them.
