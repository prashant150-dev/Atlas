"""Day-5 R2: how far can we COMPRESS the external memory before the reasoner
starts missing answers?

R1 proved external memory = scalable capability, but the store was raw and the
retriever was lexical+trivial. R2 makes the memory a *vector index* (GPT-2
sentence embeddings) and applies our D1/D2 quantization to it: store each fact
embedding at fp32 / int8 / int4 / ternary, then measure

    bits per fact  ->  retrieval@1  ->  open-book answer accuracy.

The query embedding is computed at runtime in full precision (realistic: only
the stored DB is compressed). This is the rate-distortion curve of *memory*: the
core measurement behind "1T-equivalent knowledge on a tiny disk".

Run from repo root::

    python projects/day5_reasoner_memory/r2_compress_memory.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
try:
    from projects.day5_reasoner_memory.r1_keystone import _answer, _make_kb, TfidfRetriever  # noqa: E402
except ModuleNotFoundError:
    sys.path.insert(0, str(_HERE))
    from r1_keystone import _answer, _make_kb, TfidfRetriever  # type: ignore  # noqa: E402

import random  # noqa: E402

OUT = _HERE / "r2_results.json"
LOG = _HERE / "r2_log.jsonl"
_MODEL = "models/gpt2"
KB_SIZE = 500
N_EVAL = 60
SEED = 0
_EPS = 1e-12


@torch.inference_mode()
def _embed(model, tokenizer, texts: list[str]) -> torch.Tensor:
    """Mean-pooled last-hidden-state sentence embedding for each text."""

    vecs = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt")
        hidden = model.transformer(ids.input_ids).last_hidden_state[0]  # [seq, 768]
        vecs.append(hidden.mean(dim=0))
    return torch.stack(vecs).float()  # [n, 768]


def _quantize(vecs: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-vector symmetric quantize+dequantize. bits in {8,4,2(ternary)}."""

    out = torch.empty_like(vecs)
    for i in range(vecs.size(0)):
        v = vecs[i]
        if bits == 2:  # ternary {-1,0,+1} * scale, threshold = 0.7*mean|v|
            thr = 0.7 * v.abs().mean().clamp_min(_EPS)
            sign = torch.zeros_like(v)
            sign[v > thr] = 1.0
            sign[v < -thr] = -1.0
            kept = sign != 0
            scale = (v.abs() * kept).sum() / kept.sum().clamp_min(1)
            out[i] = sign * scale
        else:
            levels = (1 << (bits - 1)) - 1  # int8->127, int4->7
            scale = v.abs().max().clamp_min(_EPS) / levels
            out[i] = torch.round(v / scale).clamp(-levels, levels) * scale
    return out


def _cosine_retrieve(query_vecs: torch.Tensor, db_vecs: torch.Tensor) -> torch.Tensor:
    """Return top-1 index in db for each query (cosine similarity)."""

    q = torch.nn.functional.normalize(query_vecs, dim=-1)
    d = torch.nn.functional.normalize(db_vecs, dim=-1)
    sims = q @ d.t()  # [n_query, n_db]
    return sims.argmax(dim=-1)


def _bits_per_fact(dim: int, bits: int) -> int:
    return dim * bits + 32  # codes + one fp32 scale per vector


def _log(row: dict) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    print(f"loaded gpt2 {time.perf_counter() - t0:.1f}s", flush=True)

    kb = _make_kb(KB_SIZE, random.Random(SEED))
    eval_idx = random.Random(SEED + 1).sample(range(KB_SIZE), N_EVAL)
    dim = 768

    te = time.perf_counter()
    fact_vecs = _embed(model, tokenizer, [f["fact"] for f in kb])
    query_vecs = _embed(model, tokenizer, [kb[i]["question"] for i in eval_idx])
    print(f"embedded {KB_SIZE} facts + {N_EVAL} queries in {time.perf_counter() - te:.1f}s", flush=True)

    gold_idx = torch.tensor(eval_idx)

    # lexical TF-IDF baseline (R1's retriever) for reference
    tfidf = TfidfRetriever([f["fact"] for f in kb])
    tfidf_hits = sum(1 for j, i in enumerate(eval_idx) if tfidf.top1(kb[i]["question"]) == i)

    def open_acc(retrieved: torch.Tensor) -> float:
        ok = 0
        for j, i in enumerate(eval_idx):
            ri = int(retrieved[j].item())
            opened = _answer(model, tokenizer, f"{kb[ri]['fact']}\n{kb[i]['stem']}")
            if kb[i]["value"] in opened:
                ok += 1
        return ok / len(eval_idx)

    methods = [("fp32", 32), ("int8", 8), ("int4", 4), ("ternary", 2)]
    rows = [{
        "method": "tfidf(lexical)", "bits_per_fact": None, "ratio_vs_fp32": None,
        "retrieval_acc": round(tfidf_hits / N_EVAL, 4), "open_acc": None,
    }]
    _log(rows[0])
    print(f"tfidf baseline retrieval {rows[0]['retrieval_acc']:.3f}", flush=True)

    for name, bits in methods:
        db = fact_vecs if bits == 32 else _quantize(fact_vecs, bits)
        retrieved = _cosine_retrieve(query_vecs, db)
        r_acc = float((retrieved == gold_idx).float().mean().item())
        ts = time.perf_counter()
        o_acc = open_acc(retrieved)
        bpf = dim * 32 if bits == 32 else _bits_per_fact(dim, bits)
        row = {
            "method": name, "bits_per_fact": bpf,
            "ratio_vs_fp32": round((dim * 32) / bpf, 2),
            "retrieval_acc": round(r_acc, 4), "open_acc": round(o_acc, 4),
            "elapsed_sec": round(time.perf_counter() - ts, 1),
        }
        rows.append(row)
        _log(row)
        print(f"{name:8s} | {bpf:6d} bits/fact ({row['ratio_vs_fp32']}x) | "
              f"retrieval {r_acc:.3f} | open {o_acc:.3f}", flush=True)

    payload = {"model": _MODEL, "kb_size": KB_SIZE, "n_eval": N_EVAL, "dim": dim,
               "seed": SEED, "vector_embedding": "gpt2 mean-pooled last hidden",
               "results": rows}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
