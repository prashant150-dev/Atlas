"""Day-10 Phase B: a REAL low-bit CPU kernel speedup (not just op-count math).

R7 showed a pure-Python packed-ternary kernel can't beat BLAS fp32. The honest way
to turn compression into wall-clock speed on a CPU is a *compiled* low-bit kernel.
torch ships one: fbgemm int8 dynamic quantization for nn.Linear (x86/AVX2). We:

  1. convert GPT-2's Conv1D layers to equivalent nn.Linear (exact, lossless),
  2. measure fp32 generation speed + perplexity,
  3. dynamic-int8-quantize the Linear layers (fbgemm) and re-measure,
  4. report wall-clock speedup, model size, and quality.

This is the concrete P-B evidence: does an actual CPU low-bit kernel convert
compression into measured speed? (int8 is the accelerated path that EXISTS on CPU;
2-bit/ternary would need a bitnet.cpp-style kernel — noted honestly.)

Run from repo root::

    python projects/day10_speed/phase_b_kernel.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
OUT = _HERE / "phase_b_results.json"
LOG = _HERE / "phase_b_log.jsonl"
_MODEL = "models/gpt2"

_EVAL = ("The quiet persistence of small repairs outlasts every grand rebuilding, "
         "and the people who change the world are rarely the loudest in the room.")


def _convert_conv1d_to_linear(model):
    """Replace HF GPT-2 Conv1D with exact nn.Linear (weight = conv.weight.T)."""
    from transformers.pytorch_utils import Conv1D
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                w = child.weight  # [in, out]
                lin = nn.Linear(w.shape[0], w.shape[1], bias=child.bias is not None)
                with torch.no_grad():
                    lin.weight.copy_(w.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, lin)
                n += 1
    return n


def _ppl(model, tok):
    ids = tok(_EVAL, return_tensors="pt").input_ids
    with torch.inference_mode():
        lg = model(ids).logits
    sl = lg[:, :-1, :].reshape(-1, lg.size(-1)).float()
    return float(torch.exp(F.cross_entropy(sl, ids[:, 1:].reshape(-1))).item())


def _gen_speed(model, tok, n_new=40, reps=3):
    inp = tok("The future of computing", return_tensors="pt")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        t = time.perf_counter(); total = 0
        for _ in range(reps):
            out = model.generate(**inp, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            total += out.shape[1] - inp.input_ids.shape[1]
        dt = time.perf_counter() - t
    return total / dt


def _linear_bytes(model):
    b = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            b += m.weight.numel() * m.weight.element_size()
        elif hasattr(m, "weight") and hasattr(m, "_packed_params"):
            b += m.weight().numel()  # int8 -> 1 byte each
    return b


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.quantized.engine = "onednn"
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    fp = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    n = _convert_conv1d_to_linear(fp)
    print(f"converted {n} Conv1D -> Linear", flush=True)

    fp_ppl = _ppl(fp, tok)
    fp_spd = _gen_speed(fp, tok)
    fp_lin_mb = _linear_bytes(fp) / 1e6
    print(f"FP32  | ppl {fp_ppl:.2f} | {fp_spd:.1f} tok/s | linear weights {fp_lin_mb:.1f} MB", flush=True)
    _log({"variant": "fp32", "ppl": round(fp_ppl, 2), "tok_per_s": round(fp_spd, 2),
          "linear_MB": round(fp_lin_mb, 1)})

    q = torch.ao.quantization.quantize_dynamic(fp, {nn.Linear}, dtype=torch.qint8)
    q_ppl = _ppl(q, tok)
    q_spd = _gen_speed(q, tok)
    q_lin_mb = _linear_bytes(q) / 1e6
    print(f"int8  | ppl {q_ppl:.2f} | {q_spd:.1f} tok/s | linear weights {q_lin_mb:.1f} MB", flush=True)
    _log({"variant": "int8_fbgemm", "ppl": round(q_ppl, 2), "tok_per_s": round(q_spd, 2),
          "linear_MB": round(q_lin_mb, 1)})

    speedup = q_spd / fp_spd
    print(f"\nspeedup int8/fp32 = {speedup:.2f}x | size {fp_lin_mb/max(q_lin_mb,1e-9):.1f}x smaller | "
          f"ppl {fp_ppl:.1f}->{q_ppl:.1f}", flush=True)
    payload = {"engine": "onednn", "conv1d_converted": n,
               "fp32": {"ppl": fp_ppl, "tok_per_s": fp_spd, "linear_MB": fp_lin_mb},
               "int8": {"ppl": q_ppl, "tok_per_s": q_spd, "linear_MB": q_lin_mb},
               "speedup_x": round(speedup, 3),
               "note": "int8 fbgemm is the real CPU low-bit kernel; 2-bit/ternary would need a bitnet.cpp-style kernel"}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
