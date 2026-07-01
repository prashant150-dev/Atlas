# ATLAS GPU validation (R1) — native sparse + 2-bit on a real model

This is the **first thing to run when a GPU is available**. It validates the riskiest
ATLAS claim on a real Hugging Face model: that native sparse + ternary training
**preserves quality** where naive post-hoc compression **collapses**, and it reports
the **honest achieved bits/weight** (all overhead counted).

## Why this matters — where 100× comes from (honest)

Compression = `16 / bits_per_weight` (vs fp16). Average bits/weight from sparsity + 2-bit:

| Sparsity | density | achieved bits/weight | compression vs fp16 |
|---|---|---|---|
| 90% | 10% | ~1.3 | ~12× |
| 95% | 5% | ~0.77 | ~21× |
| 98% | 2% | ~0.28 | ~57× |
| **99%** | **1%** | **~0.14** | **~114×** ✅ (the 100× target) |

**100× needs ~99% sparsity.** The open question this script answers: *does the model
stay good at that sparsity when trained natively?* Post-hoc definitely dies there —
native training is the bet.

## Run on Kaggle (free T4 GPU)

1. **New Notebook** → Settings → **Accelerator: GPU T4 x1**, **Internet: ON**.
2. Upload `atlas_gpu_validate.py` (Add Data → or paste into a cell).
3. In a cell:
   ```python
   !pip -q install "transformers>=4.44" datasets accelerate
   from atlas_gpu_validate import main
   for s in (0.90, 0.95, 0.98, 0.99):
       main(model_name="EleutherAI/pythia-1b", sparsity=s, steps=100)
   ```
4. Read the printed table for each sparsity. `gpu_validate_results.json` is written each run.

### Models to try (small → bigger as GPU allows)
- `EleutherAI/pythia-410m` (fast first check)
- `EleutherAI/pythia-1b` (the ~1B target)
- `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` (Linear-based, same wrapper)

## What to look for (thesis holds if)
- **HEALED perplexity << NAIVE perplexity** at high sparsity → native beats post-hoc.
- Healed perplexity stays close to the **teacher** (FP) baseline.
- The sparsity where healed quality finally breaks = the honest ceiling for this method.

## Local CPU smoke test (no GPU, no internet)
```bash
python projects/gpu_kaggle/atlas_gpu_validate.py --self-test
```
Proves the mechanism on a toy Linear stack (healed beats frozen post-hoc) and sanity-checks
the bits/weight accounting — so you know the code is correct before spending GPU time.

## Honest scope
This validates **T1 (size) + T11 (native sparse training)** — the compression + quality
half of the 100× stack. Speed/RAM at inference (T2 paging + T3 LUT kernel) is a separate
integration; this script measures storage compression and behaviour preservation, not tok/s.
