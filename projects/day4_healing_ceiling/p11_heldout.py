"""Day-4 P1.1 — HONEST healing ceiling: held-out eval, bigger data, more tokens.

P1 (heal_ceiling.py) measured top-1 on the same text it trained on, over only 63
positions -> optimistic and noisy. P1.1 fixes both:
  * train only on `corpus.TRAIN_TEXT`, evaluate only on the disjoint
    `corpus.HELDOUT_TEXT` (real generalization),
  * evaluate over every held-out window (hundreds of positions, less noise),
  * also report the *train* top-1 so the train-vs-heldout GAP shows overfitting.

The held-out top-1 curve's peak is the honest ceiling; if it rises then falls
while train top-1 keeps rising, that gap is the overfitting we must beat.

Run from repo root::

    python projects/day4_healing_ceiling/p11_heldout.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.compression.healing_qat import (  # noqa: E402
    HealingConfig,
    _distillation_loss,
    compute_bits_per_weight,
    wrap_ternary_student,
)
try:
    from projects.day4_healing_ceiling.corpus import HELDOUT_TEXT, TRAIN_TEXT  # noqa: E402
except ModuleNotFoundError:  # run as a bare script: corpus.py sits next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus import HELDOUT_TEXT, TRAIN_TEXT  # type: ignore  # noqa: E402

CHECKPOINTS: tuple[int, ...] = (15, 30, 60, 120, 240)
SEQ_LEN = 64
_HERE = Path(__file__).resolve().parent
OUT = _HERE / "p11_results.json"
LOG = _HERE / "p11_log.jsonl"
_MODEL = "models/gpt2"


def _windows(tokenizer, texts: tuple[str, ...], seq_len: int) -> list[torch.Tensor]:
    """Tokenize joined text and cut into non-overlapping seq_len windows."""

    ids = tokenizer(" ".join(texts), return_tensors="pt").input_ids[0]
    n = (ids.numel() // seq_len) * seq_len
    if n < seq_len:  # pad up to one window if corpus is tiny
        reps = (seq_len // max(1, ids.numel())) + 1
        ids = ids.repeat(reps)
        n = seq_len
    chunks = ids[:n].reshape(-1, seq_len)
    return [chunks[i].unsqueeze(0) for i in range(chunks.size(0))]


def _eval_corpus(teacher, student, windows: list[torch.Tensor]) -> tuple[float, float, int]:
    """Top-1 agreement vs teacher and perplexity over ALL windows (pooled)."""

    matches = 0
    positions = 0
    ce_sum = 0.0
    ce_count = 0
    with torch.inference_mode():
        for w in windows:
            t_logits = teacher(w).logits[0].float()
            s_logits = student(w).logits[0].float()
            matches += int((t_logits.argmax(-1) == s_logits.argmax(-1)).sum().item())
            positions += t_logits.size(0)
            # perplexity of the student on the true next token
            shift_logits = s_logits[:-1]
            shift_labels = w[0, 1:]
            ce = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
            ce_sum += float(ce.item())
            ce_count += shift_labels.numel()
    top1 = matches / max(1, positions)
    ppl = float(torch.exp(torch.tensor(ce_sum / max(1, ce_count))).item())
    return top1, ppl, positions


def _log_row(row: dict) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = HealingConfig(steps=max(CHECKPOINTS))
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

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

    train_windows = _windows(tok, TRAIN_TEXT, SEQ_LEN)
    heldout_windows = _windows(tok, HELDOUT_TEXT, SEQ_LEN)
    # a small fixed train-eval subset to report the overfit gap cheaply
    train_eval = train_windows[: min(len(train_windows), len(heldout_windows))]

    print(
        f"loaded {time.perf_counter() - t_load:.1f}s | bits/weight {bpw:.4f} | "
        f"train windows {len(train_windows)} | held-out windows {len(heldout_windows)}",
        flush=True,
    )

    student.eval()
    h_top1, h_ppl, h_pos = _eval_corpus(teacher, student, heldout_windows)
    tr_top1, _, _ = _eval_corpus(teacher, student, train_eval)
    row0 = {"step": 0, "held_top1": h_top1, "held_ppl": h_ppl, "train_top1": tr_top1,
            "eval_positions": h_pos, "note": "naive"}
    _log_row(row0)
    rows = [row0]
    print(f"step    0 | held top1 {h_top1:.3f} | held ppl {h_ppl:9.1f} | train top1 {tr_top1:.3f} "
          f"| ({h_pos} eval positions)", flush=True)

    shadow_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(shadow_params, lr=cfg.learning_rate)
    t0 = time.perf_counter()
    done = 0
    last_loss = float("nan")
    for ck in CHECKPOINTS:
        student.train()
        for step in range(done, ck):
            batch = train_windows[step % len(train_windows)]
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
        h_top1, h_ppl, _ = _eval_corpus(teacher, student, heldout_windows)
        tr_top1, _, _ = _eval_corpus(teacher, student, train_eval)
        row = {"step": ck, "held_top1": h_top1, "held_ppl": h_ppl, "train_top1": tr_top1,
               "gap": round(tr_top1 - h_top1, 4), "loss": last_loss,
               "elapsed_sec": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        _log_row(row)
        print(f"step {ck:4d} | held top1 {h_top1:.3f} | held ppl {h_ppl:9.1f} | "
              f"train top1 {tr_top1:.3f} | gap {row['gap']:+.3f} | {row['elapsed_sec']}s", flush=True)

    best = max(rows, key=lambda r: r["held_top1"])
    payload = {
        "model": _MODEL, "bits_per_weight": bpw, "layers_wrapped": layers_wrapped,
        "ternary_params": ternary_params, "seq_len": SEQ_LEN,
        "train_windows": len(train_windows), "heldout_windows": len(heldout_windows),
        "eval_positions": h_pos, "config": cfg.to_dict(), "checkpoints": rows,
        "best_heldout": {"step": best["step"], "held_top1": best["held_top1"]},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nbest held-out top-1: {best['held_top1']:.3f} at step {best['step']}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
