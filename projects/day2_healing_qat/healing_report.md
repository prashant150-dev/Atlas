# Day 2 — Healing / QAT on ternary GPT-2

**Question:** post-hoc ternary GPT-2 collapses (Day 1). If we stop trying to
match the FP weights and instead *preserve behaviour* (distil the FP teacher
into the ternary student, training shadow weights through a straight-through
estimator), how much of the collapse is recovered?

## Result

| metric | teacher (FP) | naive ternary | healed ternary |
| --- | --- | --- | --- |
| perplexity | 42.8 | 27325.8 | 401.3 |
| top-1 vs teacher | 100% | 3.0% | 28.4% |
| KL vs teacher | 0.000 | 6.003 | 2.050 |

- Recovery (top-1): 26% of the gap closed
- Distillation loss: 477.555 -> 26.562
- Layers wrapped: 48 | ternary params: 84,934,656 | ~2.02 bits/weight
- Elapsed: 268.6s

## Generation samples

- **teacher:** 'The most important idea in science is that the universe is a collection of particles. The universe is a collection of particles.\n\nThe universe is a collection'
- **naive ternary:** 'The most important idea in science is details details details details details details details details details details details details\n cpl cpl cpl cpl cpl c'
- **healed ternary:** 'The most important idea in science is not a good and is a good and it is a good and it is a good and it is a good and it'

## Interpretation

- The naive row reproduces the Day-1 post-hoc collapse: ternarizing the FP
  weights directly destroys the next-token distribution.
- Healing does **not** break information theory. It changes the target: the
  irreducible information of the *behaviour* is lower than that of the exact
  FP weights, so distillation can recover quality that no post-hoc correction
  of the weights could reach. This is the BitNet-style co-design lever, shown
  small and measured on CPU.
- Honest ceiling: this is GPT-2-small with a tiny healing set and few steps;
  it demonstrates the *mechanism*, not a production-grade ternary model. A
  longer heal on more data closes more of the gap, asymptoting at the
  behaviour's own irreducible content — not at zero.

