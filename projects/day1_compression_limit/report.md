# Day 1 — Maximum Compression Limit (GPT-2 simulation)

**Question:** A model ko size mein maximum kitna compress kar sakte hain (50x/100x/200x/400x), aur har step pe kitni intelligence/accuracy girti hai?

**Method:** Two zones. (1) Per-weight quantization (int8/int4/ternary/int2/binary) covers the viable ~2x-16x range. (2) Low-rank SVD truncation (rank tuned per matrix) is the only knob that reaches the requested 50x-400x. Quality is measured functionally on real GPT-2 (perplexity + next-token agreement + generation). Sizes extrapolated to 400000000000 parameters.

## Baseline (original GPT-2)

- Parameters: 163,037,184
- Perplexity on eval passage: **32.62** (lower = smarter)
- FP16 size at 400000000000 params: **745 GB**

## Results (sorted by realized compression)

| Method | Realized | bits/w | 400B size | Perplexity | PPL vs orig | Top-1 agree | Intelligence |
|:-------|---------:|-------:|----------:|-----------:|------------:|------------:|-------------:|
| original FP16 | 1.0x | 16.0 | 745 GB | 32.6 | 1.0x | 100.0% | 100.0% |
| int8 (2x) | 2.0x | 8.01 | 372.92 GB | 32 | 1.0x | 96.1% | 97.3% |
| int4 (4x) | 4.0x | 4.01 | 186.65 GB | 55 | 1.7x | 61.2% | 72.6% |
| ternary 1.58b | 8.0x | 2.01 | 93.52 GB | 16,091 | 493.4x | 0.0% | 28.9% |
| int2 (8x) | 8.0x | 2.01 | 93.52 GB | 21,708 | 665.6x | 3.9% | 30.5% |
| binary 1b | 15.9x | 1.01 | 46.96 GB | 68,762 | 2,108.3x | 0.0% | 28.8% |
| low-rank 50x | 50.2x | 0.32 | 14.83 GB | 8,914 | 273.3x | 9.3% | 34.1% |
| low-rank 100x | 97.5x | 0.16 | 7.65 GB | 6,593 | 202.1x | 9.3% | 35.0% |
| low-rank 200x | 194.9x | 0.08 | 3.82 GB | 3,431 | 105.2x | 9.3% | 35.0% |
| low-rank 400x | 365.6x | 0.04 | 2.04 GB | 44,092 | 1,351.9x | 9.3% | 35.1% |

_Note: perplexity is capped at exp(60) for fully-broken models; treat huge values as "collapsed". Top-1 agreement = fraction of next-token predictions that match the original model; this is the cleanest intelligence-preservation signal._

## Generation samples (qualitative intelligence)

Prompt: `The most important idea in science is`

- **original** (ppl 32.6): see model output
- **int8 (2x)**: 'The most important idea in science is that the universe is a collection of particles. The universe is a collection of particles. The universe is a collection of particles.\n\nThe universe is'
- **int4 (4x)**: 'The most important idea in science is that it is not just about the data, it is about the data that is used to make that data.\n\nI think that is the most'
- **ternary 1.58b**: 'The most important idea in science is first first first first first first first first first first first first first first first first first first first first first first first first first first first first first first'
- **int2 (8x)**: 'The most important idea in science is,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
- **binary 1b**: 'The most important idea in science is quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality quality'
- **low-rank 50x**: 'The most important idea in science is the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the'
- **low-rank 100x**: 'The most important idea in science is the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the'
- **low-rank 200x**: 'The most important idea in science is the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the'
- **low-rank 400x**: 'The most important idea in science is the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the'

## Interpretation

- **Lossless-ish zone (~2x):** int8 per-output-channel quantization is essentially free (perplexity unchanged, >95% next-token agreement). This is the safe, production-grade win.
- **Usable zone (~4x):** int4 still works (perplexity roughly 1.5-2x, agreement ~60%) -- this is the realistic post-hoc compression floor for GPT-2-small without any retraining.
- **Cliff below 4-bit:** ternary / int2 / binary all collapse toward near-random (top-1 agreement crashes to ~0%). GPT-2-small (124M) has little redundancy, so sub-4-bit *post-hoc* quantization destroys it. This is exactly why **healing / QAT** is the next step.
- **Collapse zone (50x-400x):** only low-rank truncation even *reaches* these ratios, and GPT-2's weight matrices are high-rank, so the model is already fully collapsed. 267x-400x at high quality is **not** achievable post-hoc -- this matches information theory.
- **Caveat (important for the dream):** GPT-2-small has little redundancy; a true 400B model is far more over-parameterized and would tolerate lower bit-widths better. But the hard wall for *post-hoc* methods at usable quality (~4-bit / ~4x here) is fundamental without retraining.
- **The real lever** for large size wins at usable quality is **native low-bit training (BitNet-style) + MoE sparsity (active << total params)** and **healing finetunes**, not post-hoc stacking of ratios. That is the next experiment.
