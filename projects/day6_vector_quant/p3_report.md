# Day 6 P3 — VQ + healing vs ternary + healing (the crown jewel)

The real win test: at EQUAL bits/weight, heal both a vector-quantized and a
scalar-ternary GPT-2 from the FP teacher and compare on a HELD-OUT passage.

- **ternary + healing** = D2 method: trainable FP shadow weights, STE.
- **VQ + healing** = fix the k-means assignments, make the CODEBOOK trainable,
  distil. Bits/weight unchanged (indices fixed); only the 256×4 codebook per
  matrix moves.

Train: 50-sentence corpus (`day4/corpus.py TRAIN_TEXT`). Eval: separate science
passage (held-out). 60 distillation steps, seq_len 64, seed 0.

## Results (held-out perplexity, lower = better)

| model | bits/weight | post-hoc ppl | healed ppl (60 steps) | healed top-1 |
|---|---|---|---|---|
| FP teacher | 32 | — | **48.4** | 1.00 |
| ternary + healing | 2.016 | 37,402 | **663.3** | 0.27 |
| **VQ + healing** | 2.019 | 458 | **94.7** | 0.45 |

## Verdict — VQ wins decisively at equal bits

- **VQ + healing reaches ppl 94.7 vs ternary + healing 663.3 — ~7× better at
  equal bits/weight**, and within ~2× of the FP teacher (48.4).
- **VQ *post-hoc* (458) already beats ternary *fully healed* (663).** The
  cross-weight-structure head-start is so large that un-trained VQ > trained
  scalar ternary.
- VQ needed **~3× fewer trainable params** to heal (39.5M vs 124.4M) — because
  only the codebooks + FP embeddings move, not 85M shadow body weights — yet won.

## The new frontier (≈2 bits/weight, held-out)
```
ppl   37402 ● ternary post-hoc
        663 ● ternary + heal
        458 ● VQ post-hoc
         95 ● VQ + heal        <- 7x better than ternary+heal, equal bits
         48 ● FP teacher
```

## Honest caveats
- Both arms also fine-tuned the FP embeddings / layernorms (non-wrapped, equal
  footing for both); the bits/weight figure covers the quantized body weights.
- Still ~2× the FP perplexity at 2 bits — a big step, not lossless. More steps /
  data / a learnable transform (AQLM/BTC-LLM-style) would close more.
- One held-out passage, one seed, perplexity only; 60 steps.
- Healing the codebook keeps bits/weight fixed (assignments frozen) — honest.

## Conclusion for P-A (Size)
**Vector quantization + healing is our best size lever**: at ~2 bits/weight it is
~7× better than the scalar-ternary baseline on held-out perplexity and approaches
FP, exploiting cross-weight structure that per-weight methods cannot. This beats
the D1/D2 frontier at equal size — the genuinely-new formula we set out to find.

## Next
- Sweep group size / codebook to trace the full VQ+heal bits/weight ↔ ppl curve.
- Add a learnable transform before VQ (the AQLM/BTC-LLM lever) toward sub-1-bit.
- Combine with MoE sparsity (P-B/effective-size) for the full stored+active story.
