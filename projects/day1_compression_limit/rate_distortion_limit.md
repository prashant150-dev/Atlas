# Mathematical Compression Limit (rate-distortion floor)

**Question:** Kisi bhi architecture se, ek model ke weights ko fixed bit-budget mein store karne par, *koi bhi* compressor + correction + healing scheme minimum kitni distortion tak pahunch sakti hai? Yeh floor information theory deti hai -- isse neeche real world mein jaana impossible hai.

> **Key fact:** correction-table aur healing khud bits kharch karte hain jo isi budget mein aate hain. Isliye ye floor ko sirf *approach* kar sakte hain, *cross* nahi.

## Measured weight statistics (real GPT-2)

- Compressible params: 162,915,840 across 51 matrices
- Differential entropy (unit-variance): **2.040 bits**
- Gaussian entropy (unit-variance): 2.047 bits
- Non-Gaussianity gain: **0.007 bits/weight** (kurtosis 12.24; >3 = heavier tails => slightly more compressible than Gaussian)

## The rate-distortion floor (best case for ANY method)

`NMSE` = normalized weight MSE = D / sigma^2 (0 = perfect, 1 = signal fully lost). `RMS rel` = sqrt(NMSE) = best-possible relative RMS weight error.

| Ratio | bits/weight | 400B size | NMSE floor (Gauss) | NMSE floor (SLB) | best RMS rel err |
|------:|------------:|----------:|-------------------:|-----------------:|-----------------:|
| 2x | 8.000 | 372.53 GB | 0.0000 | 0.0000 | 0.39% |
| 4x | 4.000 | 186.26 GB | 0.0039 | 0.0039 | 6.25% |
| 8x | 2.000 | 93.13 GB | 0.0625 | 0.0619 | 25.00% |
| 16x | 1.000 | 46.57 GB | 0.2500 | 0.2476 | 50.00% |
| 32x | 0.500 | 23.28 GB | 0.5000 | 0.4952 | 70.71% |
| 50x | 0.320 | 14.90 GB | 0.6417 | 0.6356 | 80.11% |
| 100x | 0.160 | 7.45 GB | 0.8011 | 0.7934 | 89.50% |
| 200x | 0.080 | 3.73 GB | 0.8950 | 0.8864 | 94.61% |
| 400x | 0.040 | 1.86 GB | 0.9461 | 0.9370 | 97.27% |

## Inverse view: high fidelity FORCES low compression

To hold weight distortion at or below a target, this is the *maximum* compression physically permitted -- no correction/healing can exceed it.

| Target weight distortion | min bits/weight (Gauss) | max ratio (Gauss) | max ratio (SLB) |
|:-------------------------|------------------------:|------------------:|----------------:|
| 10% distortion (NMSE 1e-01) | 1.66 | 9.63x | 9.67x |
| 1% distortion (NMSE 1e-02) | 3.32 | 4.82x | 4.83x |
| 0.1% distortion (NMSE 1e-03) | 4.98 | 3.21x | 3.22x |
| 0.01% distortion (NMSE 1e-04) | 6.64 | 2.41x | 2.41x |
| 0.001% distortion (NMSE 1e-05) | 8.30 | 1.93x | 1.93x |

## Bottom line (the wall, measured)

- **At 100x** (=0.160 bits/weight): the floor is NMSE >= **0.80** (Gaussian) / **0.79** (SLB). The best-possible scheme still loses >= 80% of the weight signal. 100x at ~0% drop is **physically impossible** for post-hoc weight compression -- correction/healing included.
- High fidelity is expensive: every halving of distortion costs ~0.5 bit/weight, so '0.001%-style' targets pin you near full precision (~2-3x), not 100x.

## The only legitimate escape doors (genuine research frontier)

This floor assumes weights are an i.i.d. source we must reproduce. Two doors get past it WITHOUT breaking physics:
1. **Joint structure / learned reconstruction.** Weights are correlated, not i.i.d.; the *joint* entropy per weight is lower than this marginal bound. A generator/hypernetwork that rebuilds a layer from a tiny seed can beat per-weight coding -- bounded by joint entropy, still finite, but a real lever and largely unexplored.
2. **Change the function (co-design), not just its storage.** Native low-bit training (BitNet b1.58) reaches full quality at ~1.58 bits because the *trained function itself* lives in low-bit space -- there is no FP16 'original' to distort. Plus MoE sparsity makes *active* bits/token tiny even when *total* params are huge.

Neither escapes information theory; both move the goalposts honestly. That is where the god-level architecture has to live -- not in post-hoc squeezing, which this floor caps.
