"""Part-1 BEAST SIZE — how far can the PROVEN lever go? Healing-depth scaling.

Lever 1 (mixed-precision VQ, protect 5% @ int8) is GREEN at ppl 70.83 vs FP 48.41,
but healing was data-starved (3.8k-char corpus, 60 steps). Healing is DISTILLATION:
the FP teacher supplies the targets, so any input text works. We build a larger, more
diverse distillation corpus from the teacher's own sampled generations, then heal the
mixed-precision student deeper and watch the gap to FP: does it keep closing or plateau?

Honest question this answers: is 70.83 the lever's ceiling, or just under-healed?

Run from repo root::

    python projects/day17_beast_size/p3_heal_scale.py [steps]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
sys.path.insert(0, str(_HERE.parent / "day4_healing_ceiling"))
from corpus import TRAIN_TEXT  # type: ignore  # noqa: E402
from p3_vq_heal import SEED, SEQ_LEN, _eval_ids, _ppl, _top1  # noqa: E402
from p1_mixed_heal import wrap_mixed, _model_bpw, D  # noqa: E402

OUT = _HERE / "p3_scale_results.json"
_MODEL = "models/gpt2"
LR = 5e-4

# prompts to seed teacher generations -> a broad distillation corpus (input only;
# the teacher itself supplies the soft targets during healing).
_SEED_PROMPTS = [
    "The history of science", "In a distant country", "To compute the average of a list",
    "She opened the door and", "The economy depends on", "According to the theory of",
    "Once upon a time", "The recipe calls for", "Engineers designed the bridge",
    "The meaning of the word", "After the long winter", "Investors were worried that",
    "The function returns", "Children learn language by", "The ocean covers",
    "Music has the power to", "A balanced diet includes", "The ancient city was",
    "Climate models predict", "He picked up the phone and",
]


def _build_corpus(tok, teacher, n_per_prompt=2, gen_len=80):
    torch.manual_seed(SEED)
    texts = list(TRAIN_TEXT)
    for p in _SEED_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        for _ in range(n_per_prompt):
            with torch.inference_mode():
                out = teacher.generate(ids, do_sample=True, top_k=50, temperature=1.0,
                                       max_new_tokens=gen_len, pad_token_id=tok.eos_token_id)
            texts.append(tok.decode(out[0], skip_special_tokens=True))
    return texts


def _windows(tok, texts, seq_len):
    wins = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids[0]
        if ids.numel() < seq_len:
            ids = ids.repeat((seq_len // max(1, ids.numel())) + 1)
        for s in range(0, ids.numel() - seq_len + 1, seq_len):
            wins.append(ids[s:s + seq_len].unsqueeze(0))
    return wins


def _distill_loss(student_logits, teacher_logits, T=2.0):
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    eval_ids = _eval_ids(tok)
    with torch.inference_mode():
        t_argmax = teacher(eval_ids).logits[0].float().argmax(-1)
    teacher_ppl = _ppl(teacher, eval_ids)
    print(f"FP teacher ppl {teacher_ppl:.2f}", flush=True)

    print("building distillation corpus from teacher generations...", flush=True)
    texts = _build_corpus(tok, teacher)
    wins = _windows(tok, texts, SEQ_LEN)
    print(f"corpus: {len(texts)} passages -> {len(wins)} windows of {SEQ_LEN} tokens", flush=True)

    student = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    n = wrap_mixed(student, D, 256, 0.05, SEED)   # the PROVEN lever-1 arm
    bpw = _model_bpw(student)
    print(f"mixed-precision p5 student: {n} layers, {bpw:.3f} bits/weight", flush=True)

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rows = []
    ckpts = sorted(set([0, 30, 60, 120, 200, steps]))
    done = 0
    gen = torch.Generator().manual_seed(SEED)
    for ck in ckpts:
        student.train()
        for s in range(done, ck):
            b = wins[int(torch.randint(len(wins), (1,), generator=gen))]
            with torch.inference_mode():
                tl = teacher(b).logits.clone()
            opt.zero_grad(set_to_none=True)
            loss = _distill_loss(student(b).logits, tl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
        done = ck
        student.eval()
        ppl = _ppl(student, eval_ids); top1 = _top1(student, t_argmax, eval_ids)
        gap = ppl / teacher_ppl
        rows.append({"step": ck, "ppl": round(ppl, 2), "top1": round(top1, 4),
                     "gap_x": round(gap, 3)})
        print(f"  step {ck:4d} | ppl {ppl:8.2f} | top1 {top1:.3f} | {gap:.2f}x FP", flush=True)

    payload = {"model": _MODEL, "teacher_ppl": round(teacher_ppl, 2), "bpw": round(bpw, 3),
               "arm": "mixed_K256_p5", "n_windows": len(wins), "steps": steps, "rows": rows,
               "note": "deep healing of the proven mixed-precision lever on a teacher-generated "
                       "distillation corpus; does the FP gap keep closing?"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    best = min(rows, key=lambda r: r["ppl"])
    print(f"\nFP {teacher_ppl:.1f} | best mixed-p5 ppl {best['ppl']} @ step {best['step']} "
          f"({best['gap_x']}x FP) | bits/weight {bpw:.2f}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
