"""Day-5 R5: The Grand Merge — compress the reasoner, re-fly the jet.

Brings the compression track (int8 / packed-ternary from gpt2_packed) into the
R4 reasoner+memory pipeline and re-measures the full bill for three reasoner
variants: fp32 (baseline), int8, ternary.

Retrieval is held FIXED (fp32 embeddings + the learned head + hybrid routing,
computed once) so the only variable is the reasoner used to *read the retrieved
fact and answer*. We report, per variant: on-disk size, runtime RAM, tok/sec,
and end-to-end answer accuracy.

Honest expectation (this is the point): on CPU there is no native low-bit kernel,
so load_packed_gpt2_model dequantizes back to fp32 -> the ternary/int8 win is
DISK, not RAM or speed. int8 keeps accuracy; naive ternary collapses it
(consistent with D1 / P1.1). Real RAM+speed wins need low-bit kernels (future).

Run from repo root::

    python projects/day5_reasoner_memory/r5_merge.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.compression.gpt2_packed import compress_gpt2_packed, load_packed_gpt2_model  # noqa: E402

_HERE = Path(__file__).resolve().parent
try:
    from projects.day5_reasoner_memory.r1_keystone import TfidfRetriever
    from projects.day5_reasoner_memory.r3_retrieval_stress import _embed, _train_projection
    from projects.day5_reasoner_memory.r3b_alias_break import _build
    from projects.day5_reasoner_memory.r4_end_to_end import _strip, _answer_timed
except ModuleNotFoundError:
    sys.path.insert(0, str(_HERE))
    from r1_keystone import TfidfRetriever  # type: ignore
    from r3_retrieval_stress import _embed, _train_projection  # type: ignore
    from r3b_alias_break import _build  # type: ignore
    from r4_end_to_end import _strip, _answer_timed  # type: ignore

OUT = _HERE / "r5_results.json"
LOG = _HERE / "r5_log.jsonl"
_MODEL = "models/gpt2"
_INT8_DIR = "experiments/gpt2_packed_int8"
_TERN_DIR = "experiments/gpt2_packed_ternary"
K = 240
N_OPEN = 80
LEX_THRESHOLD = 0.10
SEED = 0


def _log(row):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n"); fh.flush()


def _rss_mb():
    try:
        import psutil  # type: ignore
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return None


def _dir_mb(path):
    p = Path(path)
    if not p.exists():
        return None
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    # ---- compress the reasoner (disk) ----
    print("compressing reasoner -> int8 ...", flush=True)
    s8 = compress_gpt2_packed(_MODEL, _INT8_DIR, compression="int8")
    print(f"  int8: {s8.compressed_bytes/1e6:.1f} MB ({s8.compression_ratio:.1f}x), "
          f"rel_err {s8.average_relative_error:.4f}", flush=True)
    print("compressing reasoner -> ternary ...", flush=True)
    st = compress_gpt2_packed(_MODEL, _TERN_DIR, compression="ternary")
    print(f"  ternary: {st.compressed_bytes/1e6:.1f} MB ({st.compression_ratio:.1f}x), "
          f"rel_err {st.average_relative_error:.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    fp32 = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    facts = _build(random.Random(SEED))

    # ---- fixed retrieval (fp32 embeddings + learned head + hybrid) ----
    te = time.perf_counter()
    fact_vecs = _embed(fp32, tok, [f["fact"] for f in facts])
    canon_vecs = _embed(fp32, tok, [f["canon"] for f in facts])
    alias_vecs = _embed(fp32, tok, [f["eval_q"] for f in facts])
    train_texts, train_fidx = [], []
    for i, f in enumerate(facts):
        for q in f["train_qs"]:
            train_texts.append(q); train_fidx.append(i)
    train_vecs = _embed(fp32, tok, train_texts)
    W, _ = _train_projection(train_vecs, train_fidx, fact_vecs)
    for p in W.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        fact_proj = F.normalize(W(F.normalize(fact_vecs, dim=-1)), dim=-1)
    lex = TfidfRetriever([_strip(f["fact"]) for f in facts])
    print(f"fixed retrieval ready in {time.perf_counter()-te:.1f}s", flush=True)

    def learned_top1(qvec):
        with torch.no_grad():
            qp = F.normalize(W(F.normalize(qvec.unsqueeze(0), dim=-1)), dim=-1)
            return int((qp @ fact_proj.t()).argmax(dim=-1).item())

    workload = []
    for i, f in enumerate(facts):
        workload.append((i, f["canon"], canon_vecs[i]))
        workload.append((i, f["eval_q"], alias_vecs[i]))
    hyb_pred = []
    for true_i, qtext, qvec in workload:
        li, lscore = lex.top1_scored(_strip(qtext))
        hi = li if lscore >= LEX_THRESHOLD else learned_top1(qvec)
        hyb_pred.append((true_i, hi))
    sample = random.Random(SEED + 1).sample(range(len(workload)), N_OPEN)

    def evaluate(model, label, disk_mb):
        ans_ok, tot_tokens, tot_time = 0, 0, 0.0
        for s in sample:
            true_i, hi = hyb_pred[s]
            out, ntok, dt = _answer_timed(model, tok, f"{facts[hi]['fact']}\n{facts[true_i]['stem']}")
            ans_ok += int(facts[true_i]["value"] in out)
            tot_tokens += ntok; tot_time += dt
        params = sum(p.numel() for p in model.parameters())
        row = {
            "reasoner": label,
            "disk_MB": disk_mb,
            "ram_fp32_MB": round(params * 4 / 1e6, 1),
            "rss_MB": _rss_mb(),
            "tok_per_sec": round(tot_tokens / tot_time, 1) if tot_time else 0.0,
            "answer_acc": round(ans_ok / len(sample), 4),
        }
        _log(row)
        print(f"{label:8s} | disk {str(disk_mb):>6} MB | RAM {row['ram_fp32_MB']:.0f} MB | "
              f"{row['tok_per_sec']:.1f} tok/s | acc {row['answer_acc']:.3f}", flush=True)
        return row

    rows = []
    rows.append(evaluate(fp32, "fp32", _dir_mb(_MODEL)))
    int8_model = load_packed_gpt2_model(_INT8_DIR)
    rows.append(evaluate(int8_model, "int8", s8.compressed_bytes / 1e6))
    tern_model = load_packed_gpt2_model(_TERN_DIR)
    rows.append(evaluate(tern_model, "ternary", st.compressed_bytes / 1e6))

    payload = {
        "model": _MODEL, "kb_size": K, "n_open": N_OPEN,
        "compression": {
            "int8": {"disk_MB": round(s8.compressed_bytes/1e6, 1), "ratio": round(s8.compression_ratio, 2),
                     "rel_err": round(s8.average_relative_error, 4)},
            "ternary": {"disk_MB": round(st.compressed_bytes/1e6, 1), "ratio": round(st.compression_ratio, 2),
                        "rel_err": round(st.average_relative_error, 4),
                        "bits_per_weight": round(st.average_bits_per_packed_weight, 3)},
        },
        "variants": rows,
        "note": "retrieval fixed (fp32); load_packed dequantizes to fp32 so RAM/speed are reasoner-invariant on CPU; ternary win is DISK only",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
