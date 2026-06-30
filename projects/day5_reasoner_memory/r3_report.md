# Day 5 — R3: the retrieval boss fight

Goal: find where retrieval breaks, and what a learned (semantic) index buys over
cheap lexical matching. Three retrievers: lexical TF-IDF, raw GPT-2 embeddings,
and a small **learned projection** trained contrastively (InfoNCE) on
(query, fact) pairs and tested on a held-out query template it never saw.

## Round 1 (R3) — paraphrase, keys still present: lexical does NOT break

240 facts, colliding key pools, queries paraphrased but still naming the entity.

| retriever | canonical | paraphrase |
|---|---|---|
| lexical TF-IDF | 1.00 | **1.00** |
| GPT-2 raw embeddings | 0.12 | 0.03 |
| learned projection | 1.00 | 0.98 |

**Surprise:** lexical stayed perfect even under heavy paraphrase. The rare key
tokens (`zor`, `mintar`) appear in both query and fact, and their high IDF anchors
the match regardless of the surrounding words. The learned head recovered raw
embeddings from 3% to 98%, but couldn't *beat* an unbroken baseline.

## Round 2 (R3b) — alias reference, keys absent: lexical collapses, learned wins

Now the query names each entity by an **alias** (`duskmane`, `the salt city`)
that is **never stored in the fact** -> zero lexical overlap with the right fact.
A retriever must have *learned* the alias<->entity association.

| retriever | canonical | alias query | open-book answer (alias) |
|---|---|---|---|
| lexical TF-IDF | 1.00 | **0.00** | 0.083 |
| GPT-2 raw embeddings | — | 0.008 (chance) | — |
| **learned projection** | — | **0.825** | **0.883** |

**The boss appears and is beaten.** Lexical retrieval goes to **0%** when the
surface tokens disappear — string matching has nothing to grab. A small learned
projection (a 768->128 linear head on frozen GPT-2 embeddings, trained on alias
paraphrases, tested on a held-out template) reaches **82.5% retrieval / 88%
answers** — it does something lexical fundamentally cannot: associate different
surface forms with the same entity.

## Honest verdict

- **Lexical retrieval is shockingly strong and nearly free whenever the query
  shares the fact's rare tokens.** For a huge fraction of "look up the entity I
  named" queries, you do NOT need a heavy semantic index — a key insight for a
  tiny-footprint system.
- **Where surface tokens vanish (aliases, synonyms, descriptions), lexical dies
  and learned retrieval is essential** — and a *small, cheap, learnable* head on
  top of a frozen LM recovers most of it. This is the "learned retrieval" pillar
  (ROADMAP R4), shown small and measured.
- Raw GPT-2 embeddings are unusable for retrieval (chance-level); the value is in
  the *trained* projection, not the base embeddings.

## Design implication for the architecture

Use a **hybrid retriever**: cheap lexical first (free, perfect when the entity is
named), with a small learned semantic head as the fallback for alias/paraphrase
queries. Both are tiny next to the memory store (R2: ~57 bits/fact) and the
reasoner.

## Next
- **R3.1:** scale the learned retriever (bigger KB, ambiguous many-to-one) and
  **compress the learned head** (quantize it) — bits vs retrieval.
- **R4:** shrink the reasoner; end-to-end honest report (RAM, disk-memory, tok/s,
  accuracy).
