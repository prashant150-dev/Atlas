"""T11 TRAINING COST — the honest physics wall (the ONE axis where 101x is NOT reachable).

Inference efficiency (T2-T10) we can push 101x — same answer, less resource. But TRAINING
cost is different: to LEARN, the model must process the data and compute gradients. That
raw compute is largely fundamental (the recurring 660x-style wall). There ARE efficiency
levers, but they give ~2-10x, not 101x.

This lists the real training-cost levers honestly and the floor.
Run:  python projects/v2_design/T11_training/training_cost.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "training_results.json"

# realistic training-efficiency levers (multipliers vs naive from-scratch fp32 training)
LEVERS = [
    ("Distillation (learn from a teacher)", 3.0, "fewer tokens to reach quality"),
    ("Better data / curriculum", 2.0, "higher-quality tokens learn faster"),
    ("Low-bit / mixed-precision training", 1.8, "cheaper per gradient step"),
    ("PEFT / LoRA (train few params)", 2.5, "adapt, don't retrain everything"),
]


def main():
    print("TRAINING-COST efficiency levers (vs naive from-scratch fp32):\n", flush=True)
    cum = 1.0
    rows = []
    for name, mult, why in LEVERS:
        cum *= mult
        rows.append({"lever": name, "x": mult, "cumulative_x": round(cum, 1), "why": why})
        print(f"  {name:38s} {mult:>4.1f}x  (cum {cum:.0f}x)", flush=True)

    print(f"\n  stacked training-efficiency: ~{cum:.0f}x  (target 101x)", flush=True)
    print(f"  -> {'reaches 101x' if cum >= 101 else 'FALLS SHORT of 101x'}: training is "
          f"bounded by raw compute to learn the data.", flush=True)

    print("\nHONEST VERDICT — T11 is the physics axis:", flush=True)
    print("- Inference (T2-T10): 101x efficiency REACHABLE (same answer, less resource).", flush=True)
    print("- Training: efficiency levers give ~%dx, NOT 101x. To LEARN trillions of tokens you" % round(cum),
          flush=True)
    print("  must DO the compute — that's hardware/energy, the same wall as T1's real-scale.", flush=True)
    print("- This is why our T1 extreme-sparsity NEEDS a GPU: it requires (re)training.", flush=True)
    print("- The honest play: DON'T retrain from scratch. Use a pretrained model + cheap", flush=True)
    print("  adaptation (distill/LoRA) + free GPU (Kaggle/Colab) when training is unavoidable.", flush=True)

    payload = {"levers": rows, "stacked_x": round(cum, 1), "reaches_101x": cum >= 101,
               "note": "training cost is physics-bound: efficiency levers (distill/data/low-bit/"
                       "LoRA) give ~2-10x stacked, NOT 101x. To learn the data you must do the "
                       "compute. This is the same wall as T1 real-scale; mitigate with pretrained "
                       "+ cheap adaptation + free GPU, don't fight it."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
