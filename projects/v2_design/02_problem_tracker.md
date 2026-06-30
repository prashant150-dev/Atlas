# AI V2 — Problem Tracker: attack each limit, one by one (measured)

Rule (same as the whole project): for each problem — (1) understand how current AI does it
& why, (2) study how SOTA already attacks it, (3) design OUR angle, (4) prove it small with
a measured experiment + honest verdict, (5) only claim a win vs a real baseline at equal cost.

Status legend: ✅ done · 🟡 in progress · 🔲 not started · 🔴 physics (don't fight)

---

## The attack list (priority = leverage × movability × buildable-here)

| # | Problem | Current AI | Our angle / SOTA | Status |
|---|---|---|---|---|
| P1 | **Size** (FP weights huge) | fp16, dense | mixed-precision VQ 2-bit + healing | ✅ done (1.42× FP) |
| P2 | **Dense waste** (all params/token) | dense FFN | sparse MoE, task-routing | ✅ done (sparse=big-total) |
| P3 | **Compute/bandwidth** (matmul, weight reads) | dense matmul | LUT/T-MAC kernel (no multiplies) | ✅ done (3.84×) |
| P4 | **Context / O(n²) attention** | quadratic attn + KV cache | retrieval (sidestep) | ✅ sidestepped; 🔲 core attn untouched |
| P5 | **Quality under low-bit** | — | healing + self-review | 🟡 healing not in converter yet |
| P6 | **Hallucination/grounding** | ungrounded | retrieval + verify loop | 🟡 started (self-review) |
| P7 | **Tokenizer** (brittle, huge vocab embed) | BPE 50-150k vocab | byte/patch-level, tokenizer-free | 🔲 NOT started |
| P8 | **Attention core itself** (still O(n²)) | softmax attention | linear attn: SSM/Mamba/RWKV | 🔲 NOT started |
| P9 | **Autoregressive** (1 token/step, sequential) | left-to-right | speculative / multi-token / diffusion | 🔲 NOT started |
| P10 | **Reasoning depth at low active** | big active | test-time compute (o1-style) | 🔲 NOT started |
| P11 | **Integration** (all pieces one engine) | — | unified AetherCore runtime | 🔲 the big one |
| PX | **Training cost / energy floor** | huge | efficiency only | 🔴 mostly physics |

---

## What's already won (don't re-fight)
P1, P2, P3 are measured wins; P4 sidestepped via retrieval; P5/P6 started. These are the
🟡 "movable" limits we've already cracked at small scale.

## The genuinely-new frontiers to attack (the "change the Transformer" fun)
- **P8 Attention core** — replace O(n²) softmax attention with LINEAR attention (SSM /
  Mamba / RWKV style). The most iconic "change the Transformer" move. Testable small:
  measure quality vs speed of linear-attn vs softmax-attn on a toy seq task.
- **P7 Tokenizer-free** — drop the 50-150k-vocab BPE + its giant embedding table; work on
  raw bytes/patches. Removes a brittle, memory-heavy component.
- **P9 Autoregressive** — speculative decoding (small draft model proposes, big verifies)
  or multi-token prediction to break the 1-token-per-step wall.
- **P10 Test-time compute** — let a small-active model THINK longer to reason like a bigger
  one (the realistic path to better reasoning without a GPU).

## Honest priority recommendation
1. **Finish the won ones into a working whole** first (P5 healing + P11 integration) — make
   a real model actually run WELL on our stack. Highest dream-leverage.
2. Then open ONE new frontier — **P8 (linear attention)** is the most iconic and is a clean
   measurable probe (does linear-attn keep quality while killing O(n²)?).
3. Then P10 (test-time compute), P7 (tokenizer-free), P9 (autoregressive).

> We attack ONE at a time, prove it, commit it, move on — exactly how we got here.

## Next doc: pick #1 from "new frontiers", run a small measured probe, record verdict.
