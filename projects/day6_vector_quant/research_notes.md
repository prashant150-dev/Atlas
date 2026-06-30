# P-A (Size) — deep research notes (2026-06-21)

**Integrity note:** the deep-research workflow's *search + fetch* succeeded (3 primary
sources, 14 claims extracted) but the *adversarial verification* step failed on an
API rate-limit (402), so every claim shows a bogus "0-0 / refuted". These are NOT
refuted — they are **unverified-by-workflow**. Below they are recorded with my own
confidence based on prior knowledge of the literature. Treat extreme numbers as
"claimed, verify before trusting".

## Sources
- ParetoQ — arxiv 2502.02631 (Meta; unified low-bit QAT). *Confidence: high — well-known.*
- BitNet b1.58-2B-4T — huggingface.co/microsoft/bitnet-b1.58-2B-4T. *Confidence: high.*
- BTC-LLM — arxiv 2506.12040 (sub-1-bit via binary codebook). *Confidence: medium — recent, extreme claims, verify.*

## Key claimed findings

### Native low-bit / sub-1.58-bit (ParetoQ, BitNet)
- ParetoQ: ternary/2-bit/3-bit comparable in size-accuracy; generally beat 4-bit & binary.
- **Learning transition between 2 and 3 bits**: >=3-bit stays near the pretrained
  distribution; <=2-bit the internal representations change drastically (regime change).
- ParetoQ ternary 600M **beats prior SOTA ternary 3B** using 1/5 the params (native QAT).
- BitNet b1.58-2B-4T: native ternary, 4T tokens, avg **54.19** vs FP Qwen2.5-1.5B **55.23**
  (near-FP at similar size). Non-embedding memory **0.4 GB** vs 1.4–4.8 GB FP (3.5–12×).
- **CONFIRMS OUR R5/R7:** "efficiency/memory/speed gains require a dedicated C++ kernel
  (bitnet.cpp); under standard transformers there is NO efficiency benefit." CPU decode
  29 ms / 0.028 J with the kernel.

### Cross-weight structure / sub-1-bit (BTC-LLM)
- **Sub-1-bit weight quant: 1.11 → 0.7 bits/weight** via a **Binary Codebook** (cluster
  recurring weight *vectors* into compact indices) + a learnable transform that reduces
  outliers and promotes shared sign patterns.
- LLaMA-2-13B at **0.8 bits/weight → only 3.1% accuracy drop**, 1.6× speedup over FP16.
  *(extreme; verify — but the mechanism is sound.)*

## The information-theoretic resolution (question 5)
Our Day-1 floor (~2.04 bits/weight) is the **per-weight MARGINAL entropy** — the cost of
coding each weight *independently*. You CAN go below it without breaking any law by coding
**groups of weights jointly**, because the **joint entropy of correlated weights is lower
than the sum of their marginals**. That is exactly what **vector / codebook quantization**
does (AQLM, QuIP#, BTC-LLM). So "sub-(per-weight)-entropy via structure" is a **real, open
frontier**, not a confusion — D1's floor was never the joint-entropy floor.

## The 2 most promising open levers for us
1. **Vector / codebook quantization (cross-weight structure).** Code small *groups* of
   weights against a shared learned codebook (k-means). Reaches ~2 bits and below where
   scalar quantization (our D1/D2 ternary) cannot, because it exploits cross-weight
   correlation. **CPU-reproducible at GPT-2 scale** (k-means on weight vectors is cheap).
2. **Native low-bit QAT (already our R6/D3 direction).** Confirmed SOTA: native ternary ≈
   FP at equal size; but needs scale + a C++ kernel for the actual speed/RAM win.

## Decision
Build lever #1 — **vector quantization vs scalar at EQUAL bits/weight** on real GPT-2
weights. Win condition (ROADMAP discipline): VQ must beat scalar ternary/int at equal
bits/weight in reconstruction error and (with healing) behaviour. This is our genuinely-new
P-A experiment, and the honest route below the per-weight floor.
