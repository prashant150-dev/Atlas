"""T11 — how long does CPU training take on THIS PC? (honest FLOP estimate, before we
build faster-training tech).

Training compute ~= 6 * N_params * N_tokens FLOPs  (2 fwd + 4 bwd, standard).
This CPU (i5-4590T, 4 cores, AVX2): peak ~190 GFLOPS fp32, but real PyTorch CPU training
is memory/overhead bound -> effective ~15 GFLOPS (honest). Native-sparse adds ~1.7x
(mask updates + dense-grad passes).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "cpu_train_time_results.json"

EFF_FLOPS = 15e9            # honest effective training FLOPS on this CPU (PyTorch, fp32)
SPARSE_OVERHEAD = 1.7       # native-sparse RigL overhead


def train_hours(params, tokens):
    flops = 6 * params * tokens * SPARSE_OVERHEAD
    return flops / EFF_FLOPS / 3600


def main():
    print(f"this CPU: ~{EFF_FLOPS/1e9:.0f} GFLOPS effective (PyTorch training), "
          f"sparse overhead {SPARSE_OVERHEAD}x\n", flush=True)
    # token budgets: 'proof' (just show it learns) vs 'decent' (Chinchilla-ish 20x)
    print(f"{'model':>10} {'proof (5x tok)':>16} {'decent (20x tok)':>18}", flush=True)
    print("-" * 48, flush=True)
    rows = []
    for p in [1e6, 5e6, 10e6, 50e6, 100e6, 1e9]:
        proof = train_hours(p, p * 5)
        decent = train_hours(p, p * 20)
        rows.append({"params_M": p/1e6, "proof_hours": round(proof, 1),
                     "decent_hours": round(decent, 1)})
        def fmt(h):
            if h < 1: return f"{h*60:.0f} min"
            if h < 48: return f"{h:.1f} hr"
            return f"{h/24:.1f} days"
        print(f"{p/1e6:>8.0f}M {fmt(proof):>16} {fmt(decent):>18}", flush=True)

    print("\nHONEST READ:", flush=True)
    print("  - TINY (1-5M): hours -> method-proof FEASIBLE on CPU (free, local).", flush=True)
    print("  - 10-50M: ~1-7 days -> borderline (overnight runs).", flush=True)
    print("  - 100M+: weeks -> too slow on CPU.", flush=True)
    print("  - 1B+: months -> infeasible (this is why big models need a GPU).", flush=True)
    print("\n  => No-GPU method-proof: train a 1-5M native-sparse model -> hours. Good enough", flush=True)
    print("     to VALIDATE 0.15-bit native training works on real (tiny) language.", flush=True)
    print("\n  THEN: faster-training tech (next) can cut this — sparse-only compute, low-bit", flush=True)
    print("  training, LUT kernel for backprop, distillation (fewer tokens). Target: 5-20x faster.", flush=True)

    OUT.write_text(json.dumps({"eff_gflops": EFF_FLOPS/1e9, "sparse_overhead": SPARSE_OVERHEAD,
                   "rows": rows,
                   "note": "CPU training time = 6*N*tokens*overhead / eff_FLOPS. TINY models (1-5M) "
                           "= hours (method-proof feasible); 100M+ = weeks; 1B+ = months. Faster-"
                           "training tech (sparse-compute + low-bit + LUT-backprop + distill) is next."},
                   indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
