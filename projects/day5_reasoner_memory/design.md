# Day 5 — Reasoner + External Memory: design & keystone (R1)

## The architecture (one picture)

Today's model glues everything into weights. Our new architecture splits capability
into three substrates so the dream's parts stop fighting physics:

```
   [ small reasoner ]      full precision, in RAM, fast      -> "thinking"
          |  query
          v  retrieve
   [ external memory ]     huge, on disk, compressed         -> "knowledge" (the '1T')
          ^
          |  (later) route
   [ sparse experts ]      only the active few in RAM        -> "skills"
```

Why this beats "compress a 1T monolith into 8GB" (which D1 proved impossible):
we never store 1T weights. Knowledge lives in retrievable memory; only a small
reasoner must fit in RAM. Different information budget, no law broken.

## R1 — the keystone we must prove first

**Claim:** for a *fixed model/RAM budget*, a reasoner that can **retrieve** from
external memory has capability that a same-size **parametric-only** model cannot
hold — and that capability **scales with the memory, not the parameters**.

If this is false at small scale, the whole tower falls, so we test it before
building anything else.

### Experiment

- **Knowledge base (KB):** K *invented* facts so GPT-2's pretraining cannot have
  memorized them. Each fact: an invented `creature` + invented `place` + a real
  `attribute`/`value` (e.g. colour=cerulean) the model can read and copy.
  Example fact: "The zorblax of Mintar is cerulean in colour."
- **Reasoner:** the local GPT-2-small (unchanged, same RAM for both arms).
- **Baseline (closed-book):** prompt `Q: What colour is the zorblax of Mintar? A:`
  — the model must answer from weights alone. It never learned this fact -> should
  fail.
- **System (open-book):** prepend the retrieved fact as context, then the same Q
  -> the model can read and answer.
- **Retriever:** offline lexical TF-IDF cosine over fact strings vs the question
  (no extra model, no network).
- **Metric:** answer accuracy (gold value appears in the short greedy decode),
  plus retrieval@1 (did we fetch the right fact).
- **Scale sweep:** K in {20, 100, 500}. Expect closed-book ~0 at every K, and
  open-book high and roughly **flat** as K grows — capability tracking memory size
  at constant parameters.

### Honest expectations / failure modes

- Open-book acc is capped by retrieval@1 (if the retriever misses, the reasoner
  can't answer). We report both so the bottleneck is visible.
- This proves the *principle* (external memory = free, scalable capability), not
  yet a compressed or huge store — that's R2+.
