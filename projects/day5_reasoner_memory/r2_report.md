# Day 5 — R2: compressing the external memory

**Goal:** how few bits per fact before the reasoner starts missing answers?
Two parts came out of it: the retrieval *index*, and the stored *content*.

## R2a — quantizing a dense (GPT-2) embedding index: it doesn't help, because the index is weak

We made the memory a vector index (GPT-2 mean-pooled sentence embeddings) and
quantized it fp32 → int8 → int4 → ternary, vs the lexical TF-IDF baseline.

| index | bits/fact | retrieval@1 | open-book acc |
|---|---|---|---|
| **TF-IDF (lexical)** | (sparse, ~free) | **1.00** | — |
| GPT-2 emb fp32 | 24,576 | 0.10 | 0.15 |
| GPT-2 emb int8 | 6,176 (4.0×) | 0.00 | 0.05 |
| GPT-2 emb int4 | 3,104 (7.9×) | 0.02 | 0.03 |
| GPT-2 emb ternary | 1,568 (15.7×) | 0.00 | 0.02 |

**Finding:** the dense GPT-2 embedding is a *poor* index here (10% vs lexical
100%) — GPT-2 was never trained to make sentence embeddings, and mean-pooling
washes out the distinguishing nonsense keys. Quantizing an already-broken index
just finishes it off. **Lesson:** don't compress a bad index; the *good* index
(lexical) is already tiny and perfect. A proper semantic index needs a real
trained embedder — a separate future piece, not a quantization problem.

## R2b — the content storage floor: knowledge is cheap and scales linearly

The real memory cost is the *content* (the fact text the reasoner reads). Lossless
gzip floor (≈ entropy):

| facts | raw bits/fact | gzip bits/fact | × |
|---|---|---|---|
| 500 | 456 | 71.5 | 6.4× |
| 5,000 | 457 | 58.4 | 7.8× |
| 50,000 | 457 | 56.6 | 8.1× |

bits/fact **flat ~57** as the store grows → linear scaling. Projection (lossless):

| knowledge base | lossless size |
|---|---|
| 1,000,000 facts | **7.1 MB** |
| 100,000,000 facts | **708 MB** |
| 1,000,000,000 facts | **~7.1 GB** |

**1 billion facts ≈ 7 GB — fits on this PC's 56 GB free disk.** Storing the
"knowledge" is *not* the bottleneck; it is disk-cheap and scales linearly, far
cheaper than parameters.

## Honest verdict & redirect

- **Storage is basically solved/cheap.** The dream's "1T knowledge" lives happily
  on a small disk (these templated facts are low-entropy, so 57 b/fact is
  optimistic for real text — but the *linear, disk-friendly* scaling is the real
  point).
- **The hard, open problem is semantic RETRIEVAL at scale**, not storage. Lexical
  retrieval is free and perfect when keys are distinct; it will fail on
  paraphrase/ambiguity, and dense GPT-2 embeddings are too weak. So the research
  lever moves to: **a small, accurate, compressible embedder + retrieval that
  survives ambiguity at scale.**

## Next
- **R3:** stress retrieval — paraphrased / ambiguous queries and near-duplicate
  keys where lexical breaks; measure where it falls and what a better index buys.
- Parallel: shrink the *reasoner* (smaller model still answers from context?).
