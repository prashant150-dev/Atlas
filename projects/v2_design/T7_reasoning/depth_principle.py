"""T7 REASONING DEPTH — why deep chains need step-verification (the math), + the lever.

A hard problem = many steps. If each step is right with probability p, a FLAT N-step chain
is correct only with p^N (errors COMPOUND -> deep reasoning collapses). The fix: verify
(tool-check) each step, so each step is ~1.0, and the whole chain stays correct regardless
of depth. This is why "reasoning depth" needs decomposition + per-step verification, not
just one long think.

This shows the compounding math (using the per-step success measured in T6) and the lever.
Run:  python projects/v2_design/T7_reasoning/depth_principle.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "depth_results.json"

P_STEP_MODEL = 0.85       # model gets one step right ~85% (model-alone)
P_STEP_VERIFIED = 0.99    # with tool-check per step, a step is ~exact


def chain_accuracy(p_step, depth):
    return p_step ** depth


def main():
    depths = [1, 2, 3, 5, 8, 12, 20]
    print(f"per-step success: flat model {P_STEP_MODEL}, tool-verified {P_STEP_VERIFIED}\n", flush=True)
    print(f"{'depth':>6} {'FLAT (p^N)':>12} {'VERIFIED':>12}", flush=True)
    print("-" * 32, flush=True)
    rows = []
    for d in depths:
        flat = chain_accuracy(P_STEP_MODEL, d)
        ver = chain_accuracy(P_STEP_VERIFIED, d)
        rows.append({"depth": d, "flat": round(flat, 3), "verified": round(ver, 3)})
        print(f"{d:6d} {flat:11.1%} {ver:11.1%}", flush=True)

    print("\nHONEST READ:", flush=True)
    print(f"- FLAT reasoning DIES with depth: at 20 steps, {P_STEP_MODEL}^20 = "
          f"{chain_accuracy(P_STEP_MODEL,20):.1%} (errors compound).", flush=True)
    print(f"- STEP-VERIFIED stays high: {P_STEP_VERIFIED}^20 = "
          f"{chain_accuracy(P_STEP_VERIFIED,20):.1%}.", flush=True)
    print("- So DEPTH (hard, many-step problems) needs: decompose -> solve each step ->", flush=True)
    print("  TOOL-VERIFY each step -> combine. This is the T7 lever (builds on T6 + tools).", flush=True)
    print("- 'Reasoning depth 101x' = handle 101x-longer chains without collapse, via", flush=True)
    print("  per-step verification (the only way error doesn't explode with length).", flush=True)

    payload = {"p_step_model": P_STEP_MODEL, "p_step_verified": P_STEP_VERIFIED, "rows": rows,
               "note": "flat N-step reasoning = p^N (compounds, collapses with depth); "
                       "per-step tool-verification keeps deep chains correct. Depth needs "
                       "decompose + verify-each-step, not one long think."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
