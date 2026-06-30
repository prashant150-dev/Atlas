# Day 5 — R4: the full pipeline, flown once and measured

First end-to-end flight of the reasoner + hybrid-retriever + external-memory
architecture on the real machine (i5-4590T, 8 GB, CPU-only).

## The system

```
question --> HYBRID retriever --------------> retrieved fact --> [ GPT-2 reasoner ] --> answer
             |  lexical (free) if it scores high (entity is named)
             |  learned head (768->128) if lexical score < 0.10 (alias/paraphrase)
             v
        external memory: 240 facts, gzip 1,656 bytes (55.2 bits/fact)
```

Mixed workload: 480 queries = 240 that **name** the entity + 240 that use an
**alias** the fact never stores.

## Retrieval (mixed workload)

| retriever | accuracy | note |
|---|---|---|
| lexical only | 0.502 | perfect on named, 0 on alias |
| learned only | 0.850 | handles both, needs embed+train |
| **hybrid** | **0.848** | 200/480 routed to the learned head |

Hybrid matches learned-only accuracy while sending **58% of queries through the
free lexical path** — the learned head is only invoked for the hard minority.
(The small gap from ideal is alias words that coincidentally collide with a fact
token, e.g. "salt".)

## End-to-end + the system bill

| metric | value |
|---|---|
| **end-to-end answer accuracy (hybrid)** | **0.912** |
| reasoner speed | **15.4 tok/sec** (GPT-2, greedy, CPU) |
| reasoner | 124.4 M params · 497.8 MB fp32 |
| learned retrieval head | 98 K params · 393 KB |
| external memory | 240 facts · 1,656 bytes · **55.2 bits/fact** |

(answer accuracy > retrieval accuracy because a wrong fact sometimes carries the
same value by chance — ~8–10 values per attribute.)

## What this proves — and the honest gaps

**Proven:** the whole architecture runs on this 8 GB CPU box and answers a mixed
named/alias QA workload at **91%**, where the "extra intelligence" (the
knowledge) costs **55 bits/fact + a 393 KB head** instead of billions of added
parameters. Project the memory: **1 billion facts ≈ 7 GB** — the "1T-equivalent"
knowledge fits on disk while the reasoner stays small in RAM.

**Honest gaps vs the dream:**
- **Speed 15.4 tok/sec** is below the 40–50 target. The reasoner is an
  unoptimized fp32 124M GPT-2. This is exactly where the *other* half of the
  project plugs in: **quantize/heal the reasoner to ternary (D2/D3)** → ~16×
  smaller, faster, lower RAM.
- **Reasoner still 498 MB fp32.** Ternary reasoner ≈ 30 MB.
- **91%, not 100%** — retrieval still misses on alias/token collisions.
- Synthetic, low-entropy facts; real knowledge is higher-entropy and harder to
  retrieve.

## This is the bridge between the two halves of AetherCore

Compression track (D1–D3: ternary + healing + native low-bit) now has a clear
job inside the architecture: **shrink and speed up the reasoner**, while the
reasoner+memory design supplies capability + context from a cheap external store.

## Next (R5 / Phase-merge)
- **Quantize the reasoner** (apply healing_qat / native ternary) and re-measure
  speed, RAM, and end-to-end accuracy — the two tracks combined.
- Harden retrieval (collision-aware routing) and scale the memory toward
  millions of facts on disk.
