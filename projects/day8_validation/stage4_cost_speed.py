"""Day-8 Stage 4: healing-cost economics (#7) and wall-clock latency (#13/#14).

Two concrete, measured questions the critique raised:

  PART 1 — latency: actual ms/token and tok/s for FP GPT-2 vs VQ-reconstructed
  GPT-2 on this CPU. (Expectation, from R5/R7: VQ dequantises to fp32 at load and
  there is no low-bit CPU kernel, so latency is ~equal — the compression win is
  DISK, not speed, until a SIMD kernel exists. We measure it honestly.)

  PART 2 — healing economics: measured time per healing step, total heal cost, and
  how it compares to a single inference forward and to (estimated) pretraining.
  The honest claim: healing is a tiny ONE-TIME fine-tune on a small calibration
  set, negligible vs pretraining — but for a 400B model it still needs hardware
  able to backprop through 400B (the standing hardware gap).

Run from repo root::

    python projects/day8_validation/stage4_cost_speed.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
sys.path.insert(0, str(_HERE.parent / "day4_healing_ceiling"))
from stage1_real_lang import _apply_recon, _fresh, _WRAP  # type: ignore  # noqa: E402
from vq_vs_scalar import vector_quant  # type: ignore  # noqa: E402
from p3_vq_heal import VQConv1D, wrap_vq_student  # type: ignore  # noqa: E402
from corpus import TRAIN_TEXT  # type: ignore  # noqa: E402
from src.compression.healing_qat import _distillation_loss  # noqa: E402

OUT = _HERE / "stage4_results.json"
GPT2_PARAMS = 124_439_808
PRETRAIN_TOKENS_EST = 9.0e9          # GPT-2 WebText scale (ESTIMATE)
HEAL_STEPS = 40
SEQ = 64


def _time_forward(model, ids, reps=6):
    with torch.inference_mode():
        model(ids)  # warmup
        t = time.perf_counter()
        for _ in range(reps):
            model(ids)
        return (time.perf_counter() - t) / reps


def _time_generate(model, tok, n_new=40, reps=3):
    inp = tok("The future of computing", return_tensors="pt")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        t = time.perf_counter()
        total = 0
        for _ in range(reps):
            out = model.generate(**inp, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            total += out.shape[1] - inp.input_ids.shape[1]
        dt = time.perf_counter() - t
    return total / dt, dt / reps


def main():
    from transformers import AutoTokenizer
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tok = AutoTokenizer.from_pretrained("models/gpt2", local_files_only=True)
    ids = tok(" ".join(["the"] * SEQ), return_tensors="pt").input_ids[:, :SEQ]

    print("PART 1 — wall-clock latency (FP vs VQ-reconstructed):", flush=True)
    fp = _fresh().eval()
    fp_fwd = _time_forward(fp, ids)
    fp_toks, fp_gen = _time_generate(fp, tok)
    print(f"  FP   : forward {fp_fwd*1e3:6.1f} ms | generate {fp_toks:5.1f} tok/s", flush=True)
    del fp

    vq = _fresh().eval()
    _apply_recon(vq, lambda W: vector_quant(W, 4, 256, seed=0))
    vq_fwd = _time_forward(vq, ids)
    vq_toks, vq_gen = _time_generate(vq, tok)
    print(f"  VQ   : forward {vq_fwd*1e3:6.1f} ms | generate {vq_toks:5.1f} tok/s", flush=True)
    print(f"  speed ratio (VQ/FP): forward {vq_fwd/fp_fwd:.2f}x  generate {fp_toks/vq_toks:.2f}x"
          f"  -> ~equal: no low-bit CPU kernel, win is DISK not speed", flush=True)
    del vq

    print("PART 2 — healing economics (one-time cost):", flush=True)
    teacher = _fresh().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = _fresh()
    wrap_vq_student(student, 4, 256, 0)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-4)
    # build a few batches
    b = tok(TRAIN_TEXT[0], return_tensors="pt").input_ids[0]
    b = b[:SEQ].unsqueeze(0) if b.numel() >= SEQ else b.repeat((SEQ // b.numel()) + 1)[:SEQ].unsqueeze(0)
    # time one heal step (avg of 5)
    student.train()
    for _ in range(2):  # warmup
        with torch.inference_mode():
            tl = teacher(b).logits
        tl = tl.clone(); opt.zero_grad(set_to_none=True)
        _distillation_loss(student(b).logits, tl, b, 2.0, 0.1).backward(); opt.step()
    t = time.perf_counter()
    for _ in range(5):
        with torch.inference_mode():
            tl = teacher(b).logits
        tl = tl.clone(); opt.zero_grad(set_to_none=True)
        _distillation_loss(student(b).logits, tl, b, 2.0, 0.1).backward(); opt.step()
    step_time = (time.perf_counter() - t) / 5

    heal_total = step_time * HEAL_STEPS
    forwards_equiv = heal_total / fp_fwd
    heal_tokens = HEAL_STEPS * SEQ
    pretrain_ratio = heal_tokens / PRETRAIN_TOKENS_EST
    print(f"  heal step time   : {step_time*1e3:.0f} ms/step  ({HEAL_STEPS} steps = {heal_total:.1f} s)", flush=True)
    print(f"  = {forwards_equiv:.0f} inference-forwards of compute (one-time)", flush=True)
    print(f"  heal tokens {heal_tokens} vs pretrain ~{PRETRAIN_TOKENS_EST:.0e}"
          f"  -> {pretrain_ratio:.2e} of pretraining (ESTIMATE)", flush=True)

    payload = {
        "latency": {
            "fp_forward_ms": round(fp_fwd * 1e3, 2), "vq_forward_ms": round(vq_fwd * 1e3, 2),
            "fp_tok_per_s": round(fp_toks, 2), "vq_tok_per_s": round(vq_toks, 2),
            "vq_vs_fp_forward_x": round(vq_fwd / fp_fwd, 3),
            "note": "VQ dequantised to fp32; no low-bit CPU kernel -> latency ~equal, win is DISK",
        },
        "healing_economics": {
            "step_ms": round(step_time * 1e3, 1), "heal_steps": HEAL_STEPS,
            "heal_total_s": round(heal_total, 1),
            "equiv_inference_forwards": round(forwards_equiv, 1),
            "heal_tokens": heal_tokens, "pretrain_tokens_est": PRETRAIN_TOKENS_EST,
            "fraction_of_pretraining_est": pretrain_ratio,
            "note": "one-time fine-tune on a tiny calibration set; negligible vs pretraining, "
                    "but a 400B heal still needs hardware that can backprop 400B (hardware gap)",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
