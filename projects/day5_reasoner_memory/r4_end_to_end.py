"""Day-5 R4: the full jet, flown once and measured honestly.

Integrates all three proven layers into one pipeline and reports end-to-end:

  reasoner (GPT-2)  <--  HYBRID retriever (lexical + learned fallback)  <--  memory

Workload is MIXED: half the queries name the entity (lexical's home turf), half
use aliases the fact never stores (only learned retrieval can resolve). We report
retrieval and answer accuracy for lexical-only / learned-only / hybrid, plus the
real system bill: reasoner size, learned-head size, compressed memory size, and
measured tokens/sec on this CPU.

Run from repo root::

    python projects/day5_reasoner_memory/r4_end_to_end.py
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = Path(__file__).resolve().parent
try:
    from projects.day5_reasoner_memory.r1_keystone import TfidfRetriever
    from projects.day5_reasoner_memory.r3_retrieval_stress import _embed, _train_projection
    from projects.day5_reasoner_memory.r3b_alias_break import _build
except ModuleNotFoundError:
    sys.path.insert(0, str(_HERE))
    from r1_keystone import TfidfRetriever  # type: ignore
    from r3_retrieval_stress import _embed, _train_projection  # type: ignore
    from r3b_alias_break import _build  # type: ignore

OUT = _HERE / "r4_results.json"
LOG = _HERE / "r4_log.jsonl"
_MODEL = "models/gpt2"
K = 240
N_OPEN = 80          # mixed open-book answer checks
LEX_THRESHOLD = 0.10  # below this lexical score -> fall back to learned
SEED = 0

# Generic words removed before lexical matching so only meaningful tokens (the
# rare entity keys / attributes) drive the score. Without this, alias queries
# score > 0 purely from shared function words and never route to the learned head.
_STOP = {"the", "of", "is", "are", "a", "an", "for", "its", "it", "in", "on", "to",
         "with", "what", "which", "does", "do", "known", "most", "noted", "people",
         "recall", "carry", "among", "wonders", "marks", "dwelling", "from", "certain",
         "one", "and", "that", "this", "by", "as", "at", "be", "famous"}


def _strip(text: str) -> str:
    return " ".join(w for w in text.lower().split() if w.strip(",.?!") not in _STOP)


def _log(row):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n"); fh.flush()


@torch.inference_mode()
def _answer_timed(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt")
    t = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    dt = time.perf_counter() - t
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).lower(), gen.numel(), dt


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    facts = _build(random.Random(SEED))
    print(f"built {K} facts", flush=True)

    # ---- embeddings + learned retrieval head (trained on alias paraphrases) ----
    te = time.perf_counter()
    fact_vecs = _embed(model, tok, [f["fact"] for f in facts])
    canon_vecs = _embed(model, tok, [f["canon"] for f in facts])      # named queries
    alias_vecs = _embed(model, tok, [f["eval_q"] for f in facts])     # alias queries
    train_texts, train_fidx = [], []
    for i, f in enumerate(facts):
        for q in f["train_qs"]:
            train_texts.append(q); train_fidx.append(i)
    train_vecs = _embed(model, tok, train_texts)
    W, _loss = _train_projection(train_vecs, train_fidx, fact_vecs)
    for p in W.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        fact_proj = F.normalize(W(F.normalize(fact_vecs, dim=-1)), dim=-1)
    print(f"embedded + trained head in {time.perf_counter()-te:.1f}s", flush=True)

    lex = TfidfRetriever([_strip(f["fact"]) for f in facts])

    def learned_top1(qvec):
        with torch.no_grad():
            qp = F.normalize(W(F.normalize(qvec.unsqueeze(0), dim=-1)), dim=-1)
            return int((qp @ fact_proj.t()).argmax(dim=-1).item())

    # ---- mixed workload: each fact contributes a NAMED and an ALIAS query ----
    workload = []  # (true_idx, query_text, query_vec, kind)
    for i, f in enumerate(facts):
        workload.append((i, f["canon"], canon_vecs[i], "named"))
        workload.append((i, f["eval_q"], alias_vecs[i], "alias"))

    lex_hits = lrn_hits = hyb_hits = 0
    routed_learned = 0
    hyb_pred = []
    for true_i, qtext, qvec, kind in workload:
        li, lscore = lex.top1_scored(_strip(qtext))
        ri = learned_top1(qvec)
        if lscore >= LEX_THRESHOLD:
            hi = li
        else:
            hi = ri
            routed_learned += 1
        lex_hits += int(li == true_i)
        lrn_hits += int(ri == true_i)
        hyb_hits += int(hi == true_i)
        hyb_pred.append((true_i, hi, kind))

    n = len(workload)
    retr = {"lexical": round(lex_hits / n, 4), "learned": round(lrn_hits / n, 4),
            "hybrid": round(hyb_hits / n, 4), "routed_to_learned": routed_learned,
            "workload": n}
    _log({"retrieval": retr})
    print(f"retrieval  | lexical {retr['lexical']:.3f} | learned {retr['learned']:.3f} | "
          f"HYBRID {retr['hybrid']:.3f}  ({routed_learned}/{n} routed to learned)", flush=True)

    # ---- end-to-end open-book answers (hybrid) + speed ----
    sample = random.Random(SEED + 1).sample(range(n), N_OPEN)
    ans_ok = 0
    tot_tokens = 0
    tot_time = 0.0
    for s in sample:
        true_i, hi, kind = hyb_pred[s]
        out, ntok, dt = _answer_timed(model, tok, f"{facts[hi]['fact']}\n{facts[true_i]['stem']}")
        ans_ok += int(facts[true_i]["value"] in out)
        tot_tokens += ntok
        tot_time += dt
    answer_acc = ans_ok / len(sample)
    tok_per_sec = tot_tokens / tot_time if tot_time else 0.0
    print(f"answers    | hybrid end-to-end accuracy {answer_acc:.3f}", flush=True)
    print(f"speed      | {tok_per_sec:.1f} tok/sec (reasoner, CPU)", flush=True)

    # ---- the system bill ----
    reasoner_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in W.parameters())
    mem_blob = "\n".join(f["fact"] for f in facts).encode("utf-8")
    mem_gzip = len(gzip.compress(mem_blob, 9))
    bill = {
        "reasoner_params": reasoner_params,
        "reasoner_fp32_MB": round(reasoner_params * 4 / 1e6, 1),
        "learned_head_params": head_params,
        "learned_head_fp32_KB": round(head_params * 4 / 1e3, 1),
        "memory_facts": K,
        "memory_gzip_bytes": mem_gzip,
        "memory_bits_per_fact": round(mem_gzip * 8 / K, 1),
    }
    _log({"bill": bill, "answer_acc": answer_acc, "tok_per_sec": round(tok_per_sec, 1)})
    print("system bill:", json.dumps(bill), flush=True)

    payload = {
        "model": _MODEL, "kb_size": K, "lex_threshold": LEX_THRESHOLD,
        "retrieval": retr, "answer_acc_hybrid": answer_acc,
        "tok_per_sec": round(tok_per_sec, 1), "bill": bill,
        "caveats": [
            "GPT-2-small reasoner (only offline model); synthetic low-entropy facts",
            "answers are copy-from-context; speed is greedy single-stream CPU",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
