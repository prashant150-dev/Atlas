"""Day-5 R3: the retrieval boss fight.

R2 showed storage is cheap and the open problem is RETRIEVAL. R1/R2 used distinct
keys + canonical wording, where lexical TF-IDF is perfect. R3 builds the hard
case and measures who survives:

  * HARD data: keys reuse a small token pool (collisions) and queries are
    PARAPHRASED (different surface words from the stored fact), so bag-of-words
    overlap stops being decisive.
  * Three retrievers: lexical TF-IDF, raw GPT-2 embeddings, and a small LEARNED
    projection trained contrastively (InfoNCE) on (paraphrase, fact) pairs — the
    learned-retrieval mini-proof. The learned head is evaluated on a HELD-OUT
    paraphrase template it never trained on.

Metrics: retrieval@1 on canonical vs paraphrased queries for each retriever, plus
open-book answer accuracy (lexical vs learned) to show it flows to answers.

Run from repo root::

    python projects/day5_reasoner_memory/r3_retrieval_stress.py
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

_HERE = Path(__file__).resolve().parent
try:
    from projects.day5_reasoner_memory.r1_keystone import _answer, TfidfRetriever  # noqa: E402
except ModuleNotFoundError:
    sys.path.insert(0, str(_HERE))
    from r1_keystone import _answer, TfidfRetriever  # type: ignore  # noqa: E402

OUT = _HERE / "r3_results.json"
LOG = _HERE / "r3_log.jsonl"
_MODEL = "models/gpt2"
K = 240
N_OPEN = 60          # questions for the (slow) open-book answer check
PROJ_DIM = 128
PROJ_STEPS = 600
SEED = 0

# Small token pools -> keys COLLIDE (each token reused ~13x).
_CRE = ["zor", "quen", "dro", "mire", "fen", "voss", "pin", "wex", "gar", "thren",
        "blor", "marr", "dwil", "pharn", "uxel", "krem", "snod", "yax"]
_PLC = ["mintar", "volgard", "eskfell", "dunmoor", "lyrvale", "bramspire", "fjordeep",
        "azkwatch", "umbarrow", "peltford", "rhohaven", "sceywick", "tornmere", "virholt",
        "glendrake", "yondcrest", "holcorr", "weldrin"]
_ATTRS = {
    "colour": (["cerulean", "crimson", "amber", "violet", "emerald", "scarlet", "indigo", "ochre"],
               ["hue", "shade", "tint"]),
    "material": (["granite", "copper", "obsidian", "ivory", "bronze", "marble", "quartz", "slate"],
                 ["substance", "composition", "make"]),
    "element": (["fire", "water", "stone", "wind", "frost", "shadow", "salt", "ember"],
                ["aspect", "essence", "nature"]),
}

# Paraphrase templates: 3 for TRAIN, 1 HELD-OUT for eval. All reference the keys
# (only way to identify the fact) but use different surrounding words.
_TRAIN_TPL = [
    "In {plc}, the {cre} is most noted for which {syn}?",
    "People recall the {cre} dwelling in {plc} for a certain {syn} — which one?",
    "Which {syn} does the {plc} {cre} carry with it?",
]
_EVAL_TPL = "Among the wonders of {plc}, what {syn} marks the {cre}?"
_CANON_TPL = "What {attr} is the {cre} of {plc} known for?"


def _build(rng: random.Random):
    seen = set()
    facts = []
    while len(facts) < K:
        cre, plc = rng.choice(_CRE), rng.choice(_PLC)
        if (cre, plc) in seen:
            continue
        seen.add((cre, plc))
        attr = rng.choice(list(_ATTRS))
        values, syns = _ATTRS[attr]
        value = rng.choice(values)
        syn = rng.choice(syns)
        facts.append({
            "cre": cre, "plc": plc, "attr": attr, "value": value, "syn": syn,
            "fact": f"The {cre} of {plc} is known for its {value} {attr}.",
            "stem": f"The {cre} of {plc} is known for its",
            "canon": _CANON_TPL.format(attr=attr, cre=cre, plc=plc),
            "eval_q": _EVAL_TPL.format(plc=plc, cre=cre, syn=syn),
            "train_qs": [t.format(cre=cre, plc=plc, syn=syn) for t in _TRAIN_TPL],
        })
    return facts


@torch.inference_mode()
def _embed(model, tokenizer, texts):
    vecs = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt")
        h = model.transformer(ids.input_ids).last_hidden_state[0]
        vecs.append(h.mean(dim=0))
    return torch.stack(vecs).float()


def _ret_acc(query_vecs, db_vecs, gold):
    q = F.normalize(query_vecs, dim=-1)
    d = F.normalize(db_vecs, dim=-1)
    pred = (q @ d.t()).argmax(dim=-1)
    return float((pred == gold).float().mean().item()), pred


def _tfidf_acc(facts, query_key):
    r = TfidfRetriever([f["fact"] for f in facts])
    hits = 0
    pred = []
    for i, f in enumerate(facts):
        top = r.top1(f[query_key])
        pred.append(top)
        hits += int(top == i)
    return hits / len(facts), torch.tensor(pred)


def _train_projection(train_q, train_fact_idx, fact_vecs):
    """Contrastive (InfoNCE) linear projection: pull paraphrase->its fact."""

    torch.manual_seed(SEED)
    W = torch.nn.Linear(fact_vecs.size(1), PROJ_DIM, bias=False)
    opt = torch.optim.Adam(W.parameters(), lr=1e-3)
    tgt = torch.tensor(train_fact_idx)
    qn = F.normalize(train_q, dim=-1)
    fn = F.normalize(fact_vecs, dim=-1)
    for _ in range(PROJ_STEPS):
        opt.zero_grad(set_to_none=True)
        qp = F.normalize(W(qn), dim=-1)
        fp = F.normalize(W(fn), dim=-1)
        logits = qp @ fp.t() / 0.05
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        opt.step()
    return W, float(loss.item())


def _log(row):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    facts = _build(random.Random(SEED))
    print(f"loaded + built {K} hard facts in {time.perf_counter()-t0:.1f}s", flush=True)

    te = time.perf_counter()
    fact_vecs = _embed(model, tok, [f["fact"] for f in facts])
    canon_vecs = _embed(model, tok, [f["canon"] for f in facts])
    eval_vecs = _embed(model, tok, [f["eval_q"] for f in facts])
    train_texts, train_fidx = [], []
    for i, f in enumerate(facts):
        for q in f["train_qs"]:
            train_texts.append(q)
            train_fidx.append(i)
    train_vecs = _embed(model, tok, train_texts)
    print(f"embedded {len(facts)*2 + len(eval_vecs) + len(train_vecs)} texts in "
          f"{time.perf_counter()-te:.1f}s", flush=True)

    gold = torch.arange(K)

    # --- lexical TF-IDF ---
    lex_canon, _ = _tfidf_acc(facts, "canon")
    lex_para, lex_pred = _tfidf_acc(facts, "eval_q")
    # --- raw GPT-2 embeddings ---
    raw_canon, _ = _ret_acc(canon_vecs, fact_vecs, gold)
    raw_para, _ = _ret_acc(eval_vecs, fact_vecs, gold)
    # --- learned projection (trained on TRAIN paraphrases, eval on held-out tpl) ---
    W, final_loss = _train_projection(train_vecs, train_fidx, fact_vecs)
    with torch.inference_mode():
        fp = W(F.normalize(fact_vecs, dim=-1))
        lrn_canon, _ = _ret_acc(W(F.normalize(canon_vecs, dim=-1)), fp, gold)
        lrn_para, lrn_pred = _ret_acc(W(F.normalize(eval_vecs, dim=-1)), fp, gold)

    for name, c, p in [("lexical", lex_canon, lex_para),
                       ("gpt2_raw", raw_canon, raw_para),
                       ("learned", lrn_canon, lrn_para)]:
        row = {"retriever": name, "canonical_acc": round(c, 4), "paraphrase_acc": round(p, 4)}
        _log(row)
        print(f"{name:9s} | canonical {c:.3f} | paraphrase {p:.3f}", flush=True)

    # --- open-book answer accuracy on paraphrase: lexical vs learned ---
    open_idx = random.Random(SEED + 1).sample(range(K), N_OPEN)

    def open_acc(pred):
        ok = 0
        for i in open_idx:
            ri = int(pred[i].item())
            out = _answer(model, tok, f"{facts[ri]['fact']}\n{facts[i]['stem']}")
            ok += int(facts[i]["value"] in out)
        return ok / len(open_idx)

    lex_open = open_acc(lex_pred)
    lrn_open = open_acc(lrn_pred)
    print(f"open-book (paraphrase) | lexical {lex_open:.3f} | learned {lrn_open:.3f}", flush=True)
    _log({"open_book_paraphrase": {"lexical": round(lex_open, 4), "learned": round(lrn_open, 4)}})

    payload = {
        "model": _MODEL, "kb_size": K, "proj_dim": PROJ_DIM, "proj_steps": PROJ_STEPS,
        "proj_final_loss": round(final_loss, 4), "n_open": N_OPEN,
        "retrieval": {
            "lexical": {"canonical": lex_canon, "paraphrase": lex_para},
            "gpt2_raw": {"canonical": raw_canon, "paraphrase": raw_para},
            "learned": {"canonical": lrn_canon, "paraphrase": lrn_para},
        },
        "open_book_paraphrase": {"lexical": lex_open, "learned": lrn_open},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
