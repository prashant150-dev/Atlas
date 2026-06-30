"""Day-6 P2: whole-model vector quantization vs scalar ternary — REAL perplexity.

D6-P1 showed VQ halves reconstruction error vs scalar ternary at equal bits/weight
on one matrix. P2 tests whether that translates to end-to-end BEHAVIOUR: apply VQ
to every GPT-2 block linear, replace the weights in a real Transformers model, and
measure perplexity + top-1 agreement vs the FP teacher — against the scalar-ternary
baseline at honest equal bits/weight. Output traces the bits/weight ↔ perplexity
frontier.

Run from repo root::

    python projects/day6_vector_quant/p2_full_model.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
try:
    from projects.day6_vector_quant.vq_vs_scalar import scalar_ternary, vector_quant
except ModuleNotFoundError:
    sys.path.insert(0, str(_HERE))
    from vq_vs_scalar import scalar_ternary, vector_quant  # type: ignore

OUT = _HERE / "p2_results.json"
LOG = _HERE / "p2_log.jsonl"
_MODEL = "models/gpt2"
SEED = 0
SEQ_LEN = 256
_WRAP = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

_TEXT = (
    "The history of science is a history of patient observation slowly overturning "
    "comfortable belief. A theory survives only when it keeps making predictions that "
    "could have failed but did not. Language carries an idea from one mind to another, "
    "and mathematics sharpens that idea until its edges are exact. Rivers carve canyons "
    "not by force but by refusing to stop; knowledge grows the same quiet way, one "
    "careful question after another. The people who changed the world were rarely the "
    "loudest; they were the ones who looked again at what everyone assumed was settled, "
    "and found a small crack worth widening into a door."
)


def _eval_ids(tokenizer):
    ids = tokenizer(_TEXT, return_tensors="pt").input_ids[0]
    if ids.numel() < SEQ_LEN:
        ids = ids.repeat((SEQ_LEN // ids.numel()) + 1)
    return ids[:SEQ_LEN].unsqueeze(0)


@torch.inference_mode()
def _ppl(model, ids):
    logits = model(ids).logits
    sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
    lab = ids[:, 1:].reshape(-1)
    return float(torch.exp(F.cross_entropy(sl, lab)).item())


@torch.inference_mode()
def _top1(model, teacher_argmax, ids):
    logits = model(ids).logits[0].float()
    return float((logits.argmax(-1) == teacher_argmax).float().mean().item())


def _log(row):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n"); fh.flush()


def _quantize_model(model, method_fn):
    """Replace each wrapped weight with its quantized reconstruction in place.
    Returns honest model-wide bits/weight over the quantized matrices."""
    total_w = 0
    total_bits = 0.0
    for name, p in model.named_parameters():
        if name.endswith(_WRAP):
            W = p.detach().cpu().float().numpy()
            recon, bpw = method_fn(W)
            p.data.copy_(torch.from_numpy(recon.astype(np.float32)).reshape(p.shape))
            total_w += W.size
            total_bits += bpw * W.size
    return total_bits / max(total_w, 1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    ids = _eval_ids(tok)
    with torch.inference_mode():
        teacher_argmax = teacher(ids).logits[0].float().argmax(-1)
    teacher_ppl = _ppl(teacher, ids)
    print(f"FP teacher ppl {teacher_ppl:.2f}", flush=True)
    _log({"method": "FP_teacher", "bits_per_weight": 32.0, "ppl": round(teacher_ppl, 3), "top1": 1.0})

    methods = [
        ("ternary(baseline)", scalar_ternary),
        ("vq_d4_K256", lambda W: vector_quant(W, 4, 256, seed=SEED)),
        ("vq_d8_K4096", lambda W: vector_quant(W, 8, 4096, seed=SEED)),
        ("vq_d2_K256", lambda W: vector_quant(W, 2, 256, seed=SEED)),
    ]
    rows = []
    for label, fn in methods:
        t0 = time.perf_counter()
        m = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
        bpw = _quantize_model(m, fn)
        ppl = _ppl(m, ids)
        top1 = _top1(m, teacher_argmax, ids)
        row = {"method": label, "bits_per_weight": round(bpw, 4),
               "ppl": round(ppl, 3), "top1": round(top1, 4),
               "elapsed_sec": round(time.perf_counter() - t0, 1)}
        rows.append(row); _log(row)
        print(f"  {label:20s} | {bpw:6.3f} b/w | ppl {ppl:9.2f} | top1 {top1:.3f} | {row['elapsed_sec']}s", flush=True)
        del m

    payload = {"model": _MODEL, "seq_len": SEQ_LEN, "teacher_ppl": teacher_ppl,
               "wrapped_suffixes": list(_WRAP), "results": rows,
               "note": "whole-model VQ vs scalar ternary at honest equal bits/weight; lower ppl at equal/less bits = win"}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
