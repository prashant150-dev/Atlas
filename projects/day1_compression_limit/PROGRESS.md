# PROGRESS — AetherCore Compression Research

## Day 1 (done)
- Established the core principle: compression only has meaning vs a fidelity (quality) axis.
- Built `src/compression/compression_limit.py`: low-rank SVD compressor tunable to any target ratio + functional quality eval (perplexity, top-1 agreement, logit cosine, generation).
- Ran 50x / 100x / 200x / 400x on GPT-2 and measured the actual intelligence drop at each.
- Saved `results.json` + `report.md` in this folder.

## Key finding
- Post-hoc size compression past ~16x requires discarding rank/information; intelligence drops sharply and measurably. 267x-400x at high quality is not achievable post-hoc (matches information theory).

## Next steps (Day 2+)
1. Add per-weight quantization rows (int8/int4/int2/ternary/binary) to the same curve for a full rate-distortion picture (1x..16x range).
2. Add a 4-bit group-wise quantizer with a HEALING finetune (QAT) on GPT-2 backbone; measure how much perplexity is recovered vs naive ternary.
3. Explore the real high-ratio lever: MoE sparsity (active vs total params) + native low-bit training.
4. Once a method beats the naive curve at equal size, scale the experiment to a 1B+ model.
