# Day 5 — R1 keystone result: external memory = scalable capability

**Claim tested:** for a fixed model (same RAM), a reasoner that can *retrieve*
from external memory answers questions a parametric-only model of the same size
cannot — and that capability **scales with the memory, not the parameters**.

**Setup:** local GPT-2-small as the reasoner (unchanged in both arms). KB of K
*invented* facts (nonsense subjects so pretraining can't know them; real
attribute values it can copy). Offline TF-IDF retriever. Closed-book = continue
the cloze stem from weights alone; open-book = retrieved fact in context, then
the same stem. Held-out questions per size; seed 0.

## Result

| KB size | closed-book acc (weights only) | open-book acc (reasoner+memory) | retrieval@1 |
|---|---|---|---|
| 20 | **0.00** | **1.00** | 1.00 |
| 100 | **0.00** | **1.00** | 1.00 |
| 500 | **0.00** | **1.00** | 1.00 |

_~50s per size, CPU-only._

## What it proves

- **The asymmetry is total here.** Same GPT-2, same RAM: with external memory it
  answers everything (1.00), without it answers nothing (0.00). Knowledge placed
  in retrievable memory is capability the parameters never held.
- **Capability tracks memory, not parameters.** Open-book stays 1.00 as the KB
  grows 20 → 500 at *constant* model size. This is the core of the
  reasoner+memory design: grow the store, not the weights — the "1T" can live in
  external memory.
- Retrieval@1 stayed perfect even at 500 facts, so the reasoner was never
  starved of the right fact.

## Honest caveats (what this does NOT yet show)

- **Easy regime.** Invented keys are unique, so TF-IDF retrieval is trivial; and
  the answer is a direct copy from context, trivial for any LM. R1 proves the
  *principle*, not a hard setting.
- **Closed-book 0% is by construction** (invented facts). The fair reading: to
  put these into parametric memory you'd pay fine-tuning + capacity; external
  memory holds unbounded facts at ~no model cost. That asymmetry is the point.
- Memory here is raw text, uncompressed, small. The dream needs a **huge,
  compressed** store and **harder** retrieval.

## Next — R2

Compress the external memory with our D1/D2/D3 work and measure **bits per fact
vs retrieval/answer accuracy** — how small can the "knowledge" get before the
reasoner starts missing? Then R3: shrink the reasoner; harder, ambiguous queries
at larger scale.
