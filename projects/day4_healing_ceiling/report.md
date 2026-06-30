# Day 4 (P1) — Healing ceiling sweep on ternary GPT-2

**Question (ROADMAP P1):** D2 healed naive ternary GPT-2 from ~3% to ~28% top-1
in 30 steps. *Where does it stop?* Keep healing (more steps) and watch the
top-1 / perplexity curve for an asymptote.

**Setup:** GPT-2-small, block linears ternarized (2.0156 bits/weight, 48 layers,
~84.9M ternary params), trainable FP shadow + STE distillation from the frozen
FP teacher. One process, continuous training, evaluated at checkpoints. Same
config as D2 (lr 2e-4, T=2.0, ce_weight 0.1, seq_len 64, threshold 0.7, seed 0).

## Result — the curve

| step | top-1 vs teacher | perplexity | KL | distill loss |
|---|---|---|---|---|
| 0 (naive) | 3.1% | 19,449 | 6.80 | — |
| 15 | 25.0% | 141.5 | 2.83 | 6.07 |
| 30 | 34.4% | 43.7 | 2.54 | 2.93 |
| 60 | 35.9% | 35.5 | 3.56 | 2.18 |
| 120 | **46.9%** | 11.2 | 2.00 | 0.87 |
| 240 | 37.5% | 9.48 | 2.17 | 0.51 |
| 480 | 34.4% | 8.37 | 2.00 | 0.27 |

_elapsed: ~2030s (~34 min) on i5-4590T, CPU-only._

## Interpretation

- **The mechanism is strong and fast.** Almost all recovery happens in the
  first ~30 steps: 3% -> 34% top-1, perplexity 19,449 -> 44 (teacher 33.6).
  This re-confirms the D2 lever cleanly.
- **Top-1 does NOT keep climbing with steps — it plateaus and oscillates**
  in the ~34-47% band (120 up to 46.9%, then back to 34-37%). More steps past
  ~60 do not buy more top-1 agreement.
- **But the distillation loss and perplexity keep dropping monotonically**
  (loss 6.07 -> 0.27; ppl down to 8.37, *below* the teacher's 33.6). The student
  is now **memorizing** the 10-sentence set, not getting closer to the teacher's
  behaviour. Classic overfitting on a tiny, in-sample corpus.
- **The measurement is the bottleneck, not the method.** Two flaws cap what this
  number can mean:
  1. **Eval overlaps training** — the 64-token eval window is built from the
     same 10 sentences we heal on, so ppl < teacher is memorization, not skill.
  2. **63 eval positions, single seed** — each token is ~1.6%, so the top-1
     band (34-47%) is mostly noise.

## Honest verdict

P1's literal question ("how high does top-1 go with more steps?") has a clear
answer: **more steps alone hit diminishing returns; top-1 plateaus ~35-45% on
this tiny in-sample setup.** The healing *mechanism* saturates quickly; the lever
to push the real ceiling higher is **better data and a clean eval**, not more
training steps.

## Next (P1.1 — make the ceiling real)

1. **Held-out eval:** heal on corpus A, measure top-1/ppl on unseen corpus B.
2. **Bigger healing set:** hundreds-thousands of windows, not 10 sentences.
3. **More eval tokens + multiple seeds:** kill the ±noise so the asymptote is
   a trustworthy number.
4. Then **P2:** repeat the held-out heal at int4 / int2 / ternary to draw the
   real "bits/weight -> healed top-1" frontier (the master-key curve for P-A).
