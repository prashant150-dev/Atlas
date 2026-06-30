"""Day-13 step 1: squeeze the real available CPU speed (honest, measured).

Phase B proved low-bit kernels don't help on this Haswell CPU. So here we sweep the
genuinely-available speed levers and report the best honest tok/s:
  - Conv1D vs converted nn.Linear (better matmul layout)
  - thread count {1,2,4}
  - fp32 vs bf16 weights
  - torch.compile (if it works on this CPU)

No magic — just the best achievable wall-clock on this machine, measured.

Run from repo root::  python projects/day13_deploy/speed_optimize.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
OUT = _HERE / "speed_results.json"
_MODEL = "models/gpt2"


def _to_linear(model):
    from transformers.pytorch_utils import Conv1D
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                w = child.weight
                lin = nn.Linear(w.shape[0], w.shape[1], bias=child.bias is not None)
                with torch.no_grad():
                    lin.weight.copy_(w.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, lin)
    return model


def _speed(model, tok, n_new=40, reps=4):
    inp = tok("The future of computing", return_tensors="pt")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        t = time.perf_counter(); total = 0
        for _ in range(reps):
            out = model.generate(**inp, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            total += out.shape[1] - inp.input_ids.shape[1]
        return total / (time.perf_counter() - t)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    rows = []

    def bench(label, build, threads):
        torch.set_num_threads(threads)
        m = build()
        s = _speed(m, tok)
        rows.append({"config": label, "threads": threads, "tok_per_s": round(s, 2)})
        print(f"  {label:34s} | threads {threads} | {s:5.1f} tok/s", flush=True)
        del m

    print("baseline + layout + threads:", flush=True)
    for th in (4, 2, 1):
        bench("Conv1D fp32 (baseline)", lambda: AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval(), th)
    for th in (4, 2, 1):
        bench("Linear fp32", lambda: _to_linear(AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()), th)

    print("bf16:", flush=True)
    bench("Linear bf16", lambda: _to_linear(AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()).to(torch.bfloat16), 4)

    print("torch.compile (may be slow to warm up / may fail):", flush=True)
    try:
        def build_compiled():
            m = _to_linear(AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval())
            m.forward = torch.compile(m.forward)
            return m
        bench("Linear fp32 + torch.compile", build_compiled, 4)
    except Exception as e:
        print(f"  torch.compile failed: {type(e).__name__}: {e}", flush=True)
        rows.append({"config": "torch.compile", "error": str(type(e).__name__)})

    best = max((r for r in rows if "tok_per_s" in r), key=lambda r: r["tok_per_s"])
    baseline = next(r for r in rows if r["config"].startswith("Conv1D") and r["threads"] == 4)
    print(f"\nBEST: {best['config']} @ {best['threads']}t = {best['tok_per_s']} tok/s "
          f"({best['tok_per_s']/baseline['tok_per_s']:.2f}x vs Conv1D-4t baseline {baseline['tok_per_s']})", flush=True)
    OUT.write_text(json.dumps({"results": rows, "best": best, "baseline_conv1d_4t": baseline},
                              indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
