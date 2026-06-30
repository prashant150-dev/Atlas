"""T6 INTELLIGENCE-per-compute — make a SMALL model reason like a bigger one by THINKING
longer (test-time compute), no training, no GPU. The realistic path to better reasoning.

Same small model (Qwen2.5-1.5B), three inference strategies on checkable reasoning Qs:
  1. DIRECT          : answer immediately (1 forward pass of generation)
  2. CHAIN-OF-THOUGHT: "think step by step", then state the answer
  3. SELF-CONSISTENCY: sample K chain-of-thoughts, majority-vote the final number

If accuracy rises DIRECT < CoT <= SelfConsistency, then spending more COMPUTE at inference
(not more params) buys reasoning — the o1-style lever, runnable on this CPU.

Run:  python projects/v2_design/T6_intelligence/test_time_compute.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

MODEL = "models/qwen2.5-1.5b"
K_SAMPLES = 3

# checkable multi-step word problems (answer = a number)
QS = [
    ("A shop has 3 boxes with 12 apples each. They sell 17 apples. How many apples remain?", 19),
    ("Tom has 5 bags, each with 8 marbles. He gives away 13 marbles. How many are left?", 27),
    ("A train travels 60 km in 1 hour. How far does it travel in 3.5 hours?", 210),
    ("Sara buys 4 notebooks at 25 rupees each and pays with a 200 rupee note. What change does she get?", 100),
    ("There are 7 rows of chairs with 6 chairs each. 9 chairs are removed. How many chairs remain?", 33),
    ("A tank holds 50 litres. It loses 4 litres per hour. How much is left after 6 hours?", 26),
    ("Each box weighs 3 kg. A truck carries 15 boxes and an empty crate of 5 kg. Total weight?", 50),
    ("A book has 240 pages. Raj reads 30 pages a day. How many days to finish?", 8),
]


def _last_number(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    v = float(nums[-1])
    return int(v) if v == int(v) else v


@torch.inference_mode()
def _gen(model, tok, prompt, max_new, sample=False):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if sample:
        kw.update(do_sample=True, temperature=0.8, top_p=0.95)
    else:
        kw.update(do_sample=False)
    out = model.generate(ids, **kw)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                                 torch_dtype=torch.bfloat16).eval()

    direct = cot = sc = 0
    rows = []
    for q, ans in QS:
        # 1. direct
        d = _gen(model, tok, q + " Answer with just the final number.", 24)
        d_ok = _last_number(d) == ans
        # 2. chain of thought
        c = _gen(model, tok, q + " Let's think step by step, then give the final number.", 200)
        c_ok = _last_number(c) == ans
        # 3. self-consistency: K sampled CoTs, majority vote
        votes = []
        for _ in range(K_SAMPLES):
            s = _gen(model, tok, q + " Let's think step by step, then give the final number.",
                     200, sample=True)
            v = _last_number(s)
            if v is not None:
                votes.append(v)
        sc_ans = Counter(votes).most_common(1)[0][0] if votes else None
        sc_ok = sc_ans == ans
        direct += d_ok; cot += c_ok; sc += sc_ok
        rows.append({"q": q[:40], "ans": ans, "direct_ok": d_ok, "cot_ok": c_ok, "sc_ok": sc_ok})
        print(f"  {'D' if d_ok else '.'}{'C' if c_ok else '.'}{'S' if sc_ok else '.'}  "
              f"ans={ans:<5} | {q[:50]}", flush=True)

    n = len(QS)
    print(f"\n  DIRECT          : {direct}/{n} = {direct/n:.2f}", flush=True)
    print(f"  CHAIN-OF-THOUGHT: {cot}/{n} = {cot/n:.2f}", flush=True)
    print(f"  SELF-CONSISTENCY: {sc}/{n} = {sc/n:.2f}  (K={K_SAMPLES} samples)", flush=True)
    gain = (sc - direct) / n
    verdict = ("test-time compute HELPS: more thinking -> more correct (same model)"
               if sc > direct else "no clear gain on this small set")
    print(f"\n  VERDICT: {verdict}  (+{gain:.2f} direct->self-consistency)", flush=True)

    OUT = Path(__file__).resolve().parent / "ttc_results.json"
    OUT.write_text(json.dumps({"model": MODEL, "n": n, "k_samples": K_SAMPLES,
                   "direct": direct, "cot": cot, "self_consistency": sc, "rows": rows,
                   "verdict": verdict,
                   "note": "test-time compute (CoT, self-consistency) on a fixed small model; "
                           "more inference-compute buys reasoning without more params/GPU."},
                   indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
