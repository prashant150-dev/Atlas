"""Day-4 (P1): healing CEILING sweep for ternary GPT-2.

D2 showed naive ternary GPT-2 (~3% top-1) heals to ~28% in 30 steps. P1 asks:
*where does it stop?* We load GPT-2 once, ternarize the block linears with
trainable FP shadows, and distil the FP teacher into the ternary student
**continuously**, pausing at checkpoint step-counts to measure top-1 agreement,
perplexity, and KL. The result is a curve top1(steps) whose asymptote is the
in-sample healing ceiling at ~2.02 bits/weight.

Honest caveats (carried into the report):
  * tiny healing set (10 sentences) and the eval window overlaps it, so this is
    an OPTIMISTIC, in-sample ceiling -- it measures the mechanism's reach, not
    generalization. Bigger/held-out data is the next refinement.
  * the student at step 0 (untrained shadow == FP weights) reproduces the D2
    "naive ternary" collapse, so we reuse it as the naive baseline.

Run from repo root::

    python projects/day4_healing_ceiling/heal_ceiling.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.compression.healing_qat import (  # noqa: E402
    HealingConfig,
    _agreement_and_kl,
    _build_eval_batch,
    _build_training_batches,
    _distillation_loss,
    _perplexity,
    compute_bits_per_weight,
    wrap_ternary_student,
)

CHECKPOINTS: tuple[int, ...] = (15, 30, 60, 120, 240, 480)
_HERE = Path(__file__).resolve().parent
OUT = _HERE / "ceiling_results.json"
LOG = _HERE / "ceiling_log.jsonl"
_MODEL = "models/gpt2"


def _log_row(row: dict) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = HealingConfig(steps=max(CHECKPOINTS))
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    LOG.write_text("", encoding="utf-8")  # reset log
    t_load = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True)
    layers_wrapped, ternary_params, scale_channels = wrap_ternary_student(
        student, threshold_factor=cfg.threshold_factor, train_shadow=True
    )
    bpw = compute_bits_per_weight(ternary_params, scale_channels)

    eval_ids = _build_eval_batch(tok, cfg.seq_len, cfg.seed)
    train_batches = _build_training_batches(tok, cfg.seq_len, max(CHECKPOINTS), cfg.seed)

    teacher_ppl = _perplexity(teacher, eval_ids)

    # step 0: untrained shadow == FP weights -> naive ternary collapse baseline
    student.eval()
    naive_top1, naive_kl = _agreement_and_kl(teacher, student, eval_ids)
    naive_ppl = _perplexity(student, eval_ids)
    print(
        f"loaded in {time.perf_counter() - t_load:.1f}s | bits/weight {bpw:.4f} | "
        f"teacher ppl {teacher_ppl:.1f}",
        flush=True,
    )
    row0 = {"step": 0, "top1": naive_top1, "ppl": naive_ppl, "kl": naive_kl, "note": "naive"}
    rows = [row0]
    _log_row(row0)
    print(f"step    0 | top1 {naive_top1:.3f} | ppl {naive_ppl:9.1f} | kl {naive_kl:.3f}", flush=True)

    shadow_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(shadow_params, lr=cfg.learning_rate)

    t0 = time.perf_counter()
    done = 0
    last_loss = float("nan")
    for ck in CHECKPOINTS:
        student.train()
        for step in range(done, ck):
            batch = train_batches[step]
            with torch.inference_mode():
                teacher_logits = teacher(batch).logits
            teacher_logits = teacher_logits.clone()
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(batch).logits
            loss = _distillation_loss(
                student_logits, teacher_logits, batch, cfg.kl_temperature, cfg.ce_weight
            )
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(shadow_params, cfg.grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
        done = ck

        student.eval()
        top1, kl = _agreement_and_kl(teacher, student, eval_ids)
        ppl = _perplexity(student, eval_ids)
        row = {
            "step": ck,
            "top1": top1,
            "ppl": ppl,
            "kl": kl,
            "loss": last_loss,
            "elapsed_sec": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        _log_row(row)
        print(
            f"step {ck:4d} | top1 {top1:.3f} | ppl {ppl:9.1f} | kl {kl:.3f} | "
            f"loss {last_loss:.3f} | {row['elapsed_sec']}s",
            flush=True,
        )

    payload = {
        "model": _MODEL,
        "bits_per_weight": bpw,
        "layers_wrapped": layers_wrapped,
        "ternary_params": ternary_params,
        "teacher_ppl": teacher_ppl,
        "naive": {"top1": naive_top1, "ppl": naive_ppl, "kl": naive_kl},
        "config": cfg.to_dict(),
        "checkpoints": rows,
        "caveats": [
            "tiny 10-sentence healing set; eval overlaps train -> optimistic in-sample ceiling",
            "single seed; GPT-2-small only",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
