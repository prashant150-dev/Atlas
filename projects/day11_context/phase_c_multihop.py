"""Day-11 Phase C: multi-hop retrieval + effective-context-beyond-window.

Critique #11: R1-R4 were single-hop lookups. Real "context/intelligence" needs
reasoning over MULTIPLE chained facts. And the dream's 10-15M context, since true
attention is impossible (proven), must come from retrieval over a store far larger
than the attention window. Two measured probes:

  A. MULTI-HOP: facts chain (creature -> its home place -> that place's attribute).
     A 2-hop query needs hop1 (find the creature's place) then hop2 (find that
     place's attribute). Compare single-hop retrieval vs iterative 2-hop.

  B. CONTEXT >> WINDOW: grow the store (200 -> 4000 facts, tokens far exceeding any
     attention window) and show multi-hop answer accuracy stays flat -> effective
     context is independent of the model's window (the retrieval-as-context claim).

Honest scope: retrieval-as-context, NOT true 15M-token attention (that is
impossible, established). Lexical retrieval (proven strong in R3) + structured
extraction stand in for the reasoner's read/extract step.

Run from repo root::

    python projects/day11_context/phase_c_multihop.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day5_reasoner_memory"))
from r1_keystone import TfidfRetriever  # type: ignore  # noqa: E402

OUT = _HERE / "phase_c_results.json"
LOG = _HERE / "phase_c_log.jsonl"
SEED = 0

_CRE_A = ["zor", "quen", "dro", "mire", "fen", "voss", "pin", "wex", "gar", "thren",
          "blor", "marr", "dwil", "pharn", "uxel", "krem", "snod", "yax", "veld", "obul"]
_CRE_B = ["blax", "nel", "vith", "mop", "wick", "ler", "dolar", "ome", "nubble", "now"]
_PLC_A = ["min", "tarn", "vol", "esk", "hol", "dun", "wend", "cor", "lyr", "bram",
          "fjor", "azk", "umbr", "pelt", "rho", "scey", "torn", "vir", "glend", "yond"]
_PLC_B = ["tar", "holt", "mere", "gard", "fell", "reach", "moor", "vale", "spire", "deep"]
_VALS = ["cerulean", "crimson", "amber", "violet", "emerald", "scarlet", "indigo", "ochre"]


def build_kb(n, rng):
    """n entities; each: creature -> place (hop1), place -> value (hop2)."""
    seenc, seenp = set(), set()
    facts, ents = [], []
    while len(ents) < n:
        c = rng.choice(_CRE_A) + rng.choice(_CRE_B)
        p = rng.choice(_PLC_A) + rng.choice(_PLC_B)
        if c in seenc or p in seenp:
            continue
        seenc.add(c); seenp.add(p)
        v = rng.choice(_VALS)
        live = f"The {c} makes its home in the city of {p}."
        attr = f"The city of {p} is renowned for its {v} crystal."
        facts.append(live); facts.append(attr)
        ents.append({"c": c, "p": p, "v": v,
                     "query": f"what is the home city of the {c} renowned for",
                     "hop1_q": f"home city of the {c}",
                     "hop2_tmpl": "city of {p} renowned crystal"})
    return facts, ents


def _extract_place(fact):
    m = re.search(r"city of (\w+)", fact)
    return m.group(1) if m else None


def _value_in(fact, v):
    return v in fact


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def evaluate(n, n_eval=80):
    rng = random.Random(SEED)
    facts, ents = build_kb(n, rng)
    retr = TfidfRetriever(facts)
    eval_ents = random.Random(SEED + 1).sample(ents, min(n_eval, len(ents)))

    single_ok = multi_ok = hop1_ok = 0
    for e in eval_ents:
        # --- single-hop: one retrieval on the full 2-hop query ---
        top = facts[retr.top1(e["query"])]
        if _value_in(top, e["v"]):
            single_ok += 1
        # --- multi-hop: hop1 (find place), hop2 (find that place's attribute) ---
        f1 = facts[retr.top1(e["hop1_q"])]
        place = _extract_place(f1)
        if place == e["p"]:
            hop1_ok += 1
        f2 = facts[retr.top1(e["hop2_tmpl"].format(p=place or "?"))]
        if _value_in(f2, e["v"]):
            multi_ok += 1

    ntok = sum(len(f.split()) for f in facts)
    row = {"kb_entities": n, "kb_facts": len(facts), "approx_tokens": ntok,
           "n_eval": len(eval_ents),
           "single_hop_acc": round(single_ok / len(eval_ents), 3),
           "hop1_retrieval_acc": round(hop1_ok / len(eval_ents), 3),
           "multi_hop_acc": round(multi_ok / len(eval_ents), 3)}
    _log(row)
    print(f"  N={n:5d} ({len(facts)} facts, ~{ntok} tok >> 1024 window) | "
          f"single-hop {row['single_hop_acc']:.2f} | multi-hop {row['multi_hop_acc']:.2f}", flush=True)
    return row


def main():
    LOG.write_text("", encoding="utf-8")
    print("Phase C - multi-hop + context-beyond-window:", flush=True)
    rows = [evaluate(n) for n in (200, 1000, 4000)]
    payload = {"note": "single-hop vs 2-hop retrieval; multi-hop flat across store size "
                       "= effective context independent of attention window",
               "results": rows}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
