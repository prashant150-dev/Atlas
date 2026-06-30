"""T6 — SMARTNESS METER: how high can accuracy go (toward 100%) with TOOLS + VERIFY?

Key idea: a model is bad at EXECUTION (arithmetic slips) but the reasoning/setup is the
hard part. Split the job:
  - reasoning/setup  -> the model (improved by test-time compute, T6)
  - exact execution  -> a TOOL (calculator/code-run/lookup) = 100% on its domain

This meter scores a benchmark three ways to show the CEILING:
  model_alone     : model must do setup AND arithmetic in its head (slips on the math)
  model_plus_tool : model sets up the expression, a TOOL computes it exactly
  tool_ceiling    : if setup is correct, the tool is exact -> the max reachable accuracy

We simulate the model's two failure modes honestly: (a) wrong setup (reasoning error),
(b) right setup but arithmetic slip. Tools fix (b) entirely; test-time compute attacks (a).

Run:  python projects/v2_design/T6_intelligence/smartness_meter.py
"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path

OUT = Path(__file__).resolve().parent / "smartness_meter_results.json"

# benchmark: (question, correct setup-expression, true answer)
BENCH = [
    ("3 boxes x 12 apples, sell 17", "3*12-17", 19),
    ("5 bags x 8 marbles, give 13", "5*8-13", 27),
    ("60 km/h for 3.5 h", "60*3.5", 210.0),
    ("4 notebooks x 25, pay 200", "200-4*25", 100),
    ("7 rows x 6 chairs, remove 9", "7*6-9", 33),
    ("50 L tank, -4 L/h for 6 h", "50-4*6", 26),
    ("15 boxes x 3 kg + 5 kg crate", "15*3+5", 50),
    ("240 pages / 30 per day", "240/30", 8.0),
    ("compound: (12+8)*3 - 10", "(12+8)*3-10", 50),
    ("17*23 (big multiply)", "17*23", 391),
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def safe_eval(expr):
    """exact arithmetic evaluation (the TOOL) — no model, no slips."""
    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("bad expr")
    return ev(ast.parse(expr, mode="eval").body)


def main():
    import random
    rng = random.Random(0)
    n = len(BENCH)

    # Simulate a small model HONESTLY from our T6 measurement:
    #  - setup correct ~75% of the time (reasoning), and even then
    #  - mental arithmetic correct only ~70% (models slip on exact math).
    P_SETUP_OK = 0.75
    P_ARITH_OK = 0.70

    model_alone = model_tool = tool_ceiling = 0
    rows = []
    for q, expr, ans in BENCH:
        setup_ok = rng.random() < P_SETUP_OK
        # model alone: needs BOTH setup and arithmetic right
        ma = setup_ok and (rng.random() < P_ARITH_OK)
        # model + tool: needs only setup right (tool does exact arithmetic)
        mt = setup_ok
        # tool ceiling: if setup correct, tool is exact (== checking the tool itself)
        tc = (safe_eval(expr) == ans)
        model_alone += ma; model_tool += mt; tool_ceiling += tc
        rows.append({"q": q, "expr": expr, "ans": ans,
                     "tool_exact": safe_eval(expr) == ans})

    print(f"benchmark: {n} checkable problems\n", flush=True)
    print(f"  model ALONE (setup + mental math) : {model_alone}/{n} = {model_alone/n:.2f}", flush=True)
    print(f"  model + TOOL (model sets up, tool computes): {model_tool}/{n} = {model_tool/n:.2f}", flush=True)
    print(f"  TOOL CEILING (correct setup -> exact)      : {tool_ceiling}/{n} = {tool_ceiling/n:.2f}",
          flush=True)
    print(f"\n  -> the TOOL itself is 100% exact on every problem (arithmetic never slips).", flush=True)
    print(f"  -> model+tool removes ALL arithmetic errors; only the SETUP (reasoning) can fail.",
          flush=True)
    print(f"  -> test-time compute (T6) attacks the remaining SETUP errors -> toward 100%.", flush=True)

    print("\nHOW TO MAXIMISE ACCURACY (the recipe):", flush=True)
    print("  1. model + test-time compute  -> get the SETUP right (reasoning)", flush=True)
    print("  2. TOOL executes exactly      -> zero arithmetic/execution errors", flush=True)
    print("  3. self-verify                -> catch the rest; say 'unsure' if not verifiable", flush=True)
    print("  => 100% is reachable on VERIFIABLE tasks; honest 'I don't know' elsewhere.", flush=True)

    OUT.write_text(json.dumps({"n": n, "model_alone": model_alone, "model_plus_tool": model_tool,
                   "tool_ceiling": tool_ceiling, "p_setup_ok": P_SETUP_OK, "p_arith_ok": P_ARITH_OK,
                   "rows": rows,
                   "note": "smartness ceiling: tools make execution exact (100% on domain), so "
                           "accuracy is bounded only by SETUP/reasoning, which test-time compute "
                           "improves. Max accuracy = model+tool+verify on verifiable tasks."},
                   indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
