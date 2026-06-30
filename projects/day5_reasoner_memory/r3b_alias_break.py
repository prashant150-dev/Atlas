"""Day-5 R3b: actually break lexical retrieval (the real boss).

R3 found lexical TF-IDF is unbreakable while the query shares the fact's exact
rare key tokens. So here the query refers to each entity by an ALIAS that does
NOT appear in the stored fact -> lexical overlap with the correct fact collapses
to ~0. A retriever now needs a *learned association* alias<->entity that string
matching cannot have. We train a small projection on (alias-query, fact) pairs
and test on a HELD-OUT alias template.

Expectation: lexical canonical high, lexical alias ~chance (boss appears),
learned alias recovers -> learned retrieval does something lexical cannot.

Run from repo root::

    python projects/day5_reasoner_memory/r3b_alias_break.py
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
    from projects.day5_reasoner_memory.r3_retrieval_stress import (  # noqa: E402
        _embed, _ret_acc, _tfidf_acc, _train_projection)
    from projects.day5_reasoner_memory.r1_keystone import _answer
except ModuleNotFoundError:
    sys.path.insert(0, str(_HERE))
    from r3_retrieval_stress import _embed, _ret_acc, _tfidf_acc, _train_projection  # type: ignore
    from r1_keystone import _answer  # type: ignore

OUT = _HERE / "r3b_results.json"
LOG = _HERE / "r3b_log.jsonl"
_MODEL = "models/gpt2"
K = 240
N_OPEN = 60
SEED = 0

# primary key words (go into the stored fact)
_CRE = ["zor", "quen", "dro", "mire", "fen", "voss", "pin", "wex", "gar", "thren",
        "blor", "marr", "dwil", "pharn", "uxel", "krem", "snod", "yax"]
_PLC = ["mintar", "volgard", "eskfell", "dunmoor", "lyrvale", "bramspire", "fjordeep",
        "azkwatch", "umbarrow", "peltford", "rhohaven", "sceywick", "tornmere", "virholt",
        "glendrake", "yondcrest", "holcorr", "weldrin"]
# ALIAS words (used only in queries; NEVER stored in the fact)
_ACRE = ["duskmane", "gloamfin", "ashbeak", "rimecoat", "embertail", "voidcrest",
         "paleclaw", "stormhide", "brackmaw", "soothfang", "dimwing", "frostlope",
         "cindermal", "warthrush", "umbervex", "kreelspur", "snagtooth", "yarrowback"]
_APLC = ["the salt city", "the iron reach", "the fen wastes", "the high downs",
         "the song vale", "the thorn spire", "the cold fjord", "the watch keep",
         "the shadow barrow", "the pelt ford", "the grey haven", "the sea wick",
         "the still mere", "the green holt", "the dragon lake", "the far crest",
         "the hollow corr", "the weld marsh"]
_ATTRS = {
    "colour": (["cerulean", "crimson", "amber", "violet", "emerald", "scarlet", "indigo", "ochre"],
               ["hue", "shade", "tint"]),
    "material": (["granite", "copper", "obsidian", "ivory", "bronze", "marble", "quartz", "slate"],
                 ["substance", "composition", "make"]),
    "element": (["fire", "water", "stone", "wind", "frost", "shadow", "salt", "ember"],
                ["aspect", "essence", "nature"]),
}
_TRAIN_TPL = [
    "In {aplc}, the {acre} is most noted for which {syn}?",
    "People recall {acre} of {aplc} for a certain {syn} — which one?",
    "Which {syn} does the {acre} from {aplc} carry?",
]
_EVAL_TPL = "Among the wonders of {aplc}, what {syn} marks the {acre}?"
_CANON_TPL = "What {attr} is the {cre} of {plc} known for?"


def _build(rng):
    seen, facts = set(), []
    ci = {c: _ACRE[i] for i, c in enumerate(_CRE)}
    pi = {p: _APLC[i] for i, p in enumerate(_PLC)}
    while len(facts) < K:
        cre, plc = rng.choice(_CRE), rng.choice(_PLC)
        if (cre, plc) in seen:
            continue
        seen.add((cre, plc))
        attr = rng.choice(list(_ATTRS))
        values, syns = _ATTRS[attr]
        value, syn = rng.choice(values), rng.choice(syns)
        acre, aplc = ci[cre], pi[plc]
        facts.append({
            "cre": cre, "plc": plc, "value": value, "attr": attr,
            "fact": f"The {cre} of {plc} is known for its {value} {attr}.",
            "stem": f"The {cre} of {plc} is known for its",
            "canon": _CANON_TPL.format(attr=attr, cre=cre, plc=plc),
            "eval_q": _EVAL_TPL.format(aplc=aplc, acre=acre, syn=syn),
            "train_qs": [t.format(acre=acre, aplc=aplc, syn=syn) for t in _TRAIN_TPL],
        })
    return facts


def _log(row):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n"); fh.flush()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()
    facts = _build(random.Random(SEED))
    print(f"built {K} alias facts (query keys NOT in stored fact)", flush=True)

    te = time.perf_counter()
    fact_vecs = _embed(model, tok, [f["fact"] for f in facts])
    canon_vecs = _embed(model, tok, [f["canon"] for f in facts])
    eval_vecs = _embed(model, tok, [f["eval_q"] for f in facts])
    train_texts, train_fidx = [], []
    for i, f in enumerate(facts):
        for q in f["train_qs"]:
            train_texts.append(q); train_fidx.append(i)
    train_vecs = _embed(model, tok, train_texts)
    print(f"embedded in {time.perf_counter()-te:.1f}s", flush=True)

    gold = torch.arange(K)
    lex_canon, _ = _tfidf_acc(facts, "canon")
    lex_alias, lex_pred = _tfidf_acc(facts, "eval_q")
    raw_alias, _ = _ret_acc(eval_vecs, fact_vecs, gold)
    W, loss = _train_projection(train_vecs, train_fidx, fact_vecs)
    with torch.inference_mode():
        fp = W(F.normalize(fact_vecs, dim=-1))
        lrn_alias, lrn_pred = _ret_acc(W(F.normalize(eval_vecs, dim=-1)), fp, gold)

    print(f"lexical   | canonical {lex_canon:.3f} | ALIAS {lex_alias:.3f}", flush=True)
    print(f"gpt2_raw  | ALIAS {raw_alias:.3f}", flush=True)
    print(f"learned   | ALIAS {lrn_alias:.3f}  (train loss {loss:.3f})", flush=True)
    for r, c, a in [("lexical", lex_canon, lex_alias), ("gpt2_raw", None, raw_alias),
                    ("learned", None, lrn_alias)]:
        _log({"retriever": r, "canonical_acc": c, "alias_acc": round(a, 4)})

    open_idx = random.Random(SEED + 1).sample(range(K), N_OPEN)

    def open_acc(pred):
        ok = 0
        for i in open_idx:
            ri = int(pred[i].item())
            out = _answer(model, tok, f"{facts[ri]['fact']}\n{facts[i]['stem']}")
            ok += int(facts[i]["value"] in out)
        return ok / len(open_idx)

    lex_open, lrn_open = open_acc(lex_pred), open_acc(lrn_pred)
    print(f"open-book (alias) | lexical {lex_open:.3f} | learned {lrn_open:.3f}", flush=True)
    _log({"open_book_alias": {"lexical": round(lex_open, 4), "learned": round(lrn_open, 4)}})

    OUT.write_text(json.dumps({
        "model": _MODEL, "kb_size": K, "proj_final_loss": round(loss, 4),
        "retrieval": {"lexical": {"canonical": lex_canon, "alias": lex_alias},
                      "gpt2_raw": {"alias": raw_alias}, "learned": {"alias": lrn_alias}},
        "open_book_alias": {"lexical": lex_open, "learned": lrn_open},
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
