"""Day-5 R1 keystone: does external memory give a same-size model capability
that a parametric-only model cannot hold?

We invent facts about nonsense subjects (so GPT-2 pretraining can't know them),
then ask the SAME local GPT-2 to answer questions:
  * closed-book  -> "Q: ... A:"            (weights only -> should fail)
  * open-book    -> "<retrieved fact>\nQ: ... A:"  (reads memory -> should answer)

A tiny offline TF-IDF retriever fetches the most relevant fact for each question.
We sweep KB size {20, 100, 500}: closed-book stays ~0 at every size, open-book
should stay high and roughly FLAT -> capability scales with external memory, not
with parameters. That asymmetry is the keystone of the reasoner+memory design.

Run from repo root::

    python projects/day5_reasoner_memory/r1_keystone.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "r1_results.json"
LOG = _HERE / "r1_log.jsonl"
_MODEL = "models/gpt2"

KB_SIZES: tuple[int, ...] = (20, 100, 500)
N_EVAL = 60  # held-out questions per KB size
SEED = 0

# Invented subjects (nonsense -> not in GPT-2 pretraining), built from ascii
# syllables. Real attribute values (colours/materials) so the reasoner can read
# and copy them from the retrieved context.
_SYL_A = ["zor", "quen", "dro", "mire", "fen", "voss", "pin", "wex", "gar", "thren",
          "blor", "marr", "dwil", "pharn", "uxel", "krem", "snod", "yax", "veld", "obul"]
_SYL_B = ["blax", "nel", "vith", "mop", "wick", "ler", "dolar", "ome", "nubble", "now",
          "van", "owkin", "tch", "oxen", "lin", "lot", "rin", "une", "arn", "eron"]
_PLACES_A = ["min", "tarn", "vol", "esk", "hol", "dun", "wend", "cor", "lyr", "bram",
             "fjor", "azk", "umbr", "pelt", "rho", "scey", "torn", "vir", "glend", "yond"]
_PLACES_B = ["tar", "holt", "mere", "gard", "fell", "reach", "moor", "vale", "spire",
             "hollow", "marsh", "crest", "ford", "watch", "barrow", "deep", "haven",
             "rake", "wick", "thwaite"]

_ATTRS = [
    ("colour", ["cerulean", "crimson", "amber", "violet", "emerald", "scarlet",
                "indigo", "magenta", "turquoise", "ochre"]),
    ("material", ["granite", "copper", "obsidian", "ivory", "bronze", "marble",
                  "quartz", "oak", "slate", "tin"]),
    ("element", ["fire", "water", "stone", "wind", "frost", "shadow", "iron",
                 "salt", "ember", "mist"]),
]

_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _make_kb(n: int, rng: random.Random) -> list[dict]:
    """Build n unique invented facts with question + gold value."""

    seen: set[tuple[str, str]] = set()
    facts: list[dict] = []
    while len(facts) < n:
        creature = rng.choice(_SYL_A) + rng.choice(_SYL_B)
        place = rng.choice(_PLACES_A) + rng.choice(_PLACES_B)
        key = (creature, place)
        if key in seen:
            continue
        seen.add(key)
        attr, values = rng.choice(_ATTRS)
        value = rng.choice(values)
        fact_text = f"The {creature} of {place} is known for its {value} {attr}."
        # cloze stem: a base LM continues it by naming the value (copying from
        # context when the fact is present, guessing when it is not).
        stem = f"The {creature} of {place} is known for its"
        question = f"What {attr} is the {creature} of {place} known for?"
        facts.append(
            {"creature": creature, "place": place, "attr": attr, "value": value,
             "fact": fact_text, "question": question, "stem": stem}
        )
    return facts


class TfidfRetriever:
    """Minimal offline TF-IDF cosine retriever over the fact strings."""

    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.doc_toks = [_tok(d) for d in docs]
        df: Counter[str] = Counter()
        for toks in self.doc_toks:
            df.update(set(toks))
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.doc_vecs = [self._vec(toks) for toks in self.doc_toks]
        self.doc_norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in self.doc_vecs]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: (tf[t] / len(toks)) * self.idf.get(t, 0.0) for t in tf}

    def top1_scored(self, query: str) -> tuple[int, float]:
        """Return (best_doc_index, cosine_score) for the query."""

        qvec = self._vec(_tok(query))
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        best_i, best_s = 0, -1.0
        for i, dvec in enumerate(self.doc_vecs):
            dot = sum(qvec.get(t, 0.0) * v for t, v in dvec.items())
            score = dot / (qnorm * self.doc_norms[i])
            if score > best_s:
                best_s, best_i = score, i
        return best_i, best_s

    def top1(self, query: str) -> int:
        return self.top1_scored(query)[0]


def _answer(model, tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).lower()


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
    print(f"loaded gpt2 in {time.perf_counter() - t0:.1f}s", flush=True)

    rows = []
    for K in KB_SIZES:
        rng = random.Random(SEED)
        kb = _make_kb(K, rng)
        retriever = TfidfRetriever([f["fact"] for f in kb])

        eval_rng = random.Random(SEED + 1)
        eval_idx = eval_rng.sample(range(K), min(N_EVAL, K))

        closed_ok = open_ok = retr_ok = 0
        ts = time.perf_counter()
        for j, idx in enumerate(eval_idx):
            q = kb[idx]
            gold = q["value"]

            # closed-book: weights only must continue the stem.
            closed = _answer(model, tokenizer, q["stem"])
            if gold in closed:
                closed_ok += 1

            # open-book: retrieve a fact, place it in context, then the same stem.
            ri = retriever.top1(q["question"])
            if ri == idx:
                retr_ok += 1
            open_prompt = f"{kb[ri]['fact']}\n{q['stem']}"
            opened = _answer(model, tokenizer, open_prompt)
            if gold in opened:
                open_ok += 1

        n = len(eval_idx)
        row = {
            "kb_size": K, "n_eval": n,
            "closed_acc": round(closed_ok / n, 4),
            "open_acc": round(open_ok / n, 4),
            "retrieval_acc": round(retr_ok / n, 4),
            "elapsed_sec": round(time.perf_counter() - ts, 1),
        }
        rows.append(row)
        _log(row)
        print(
            f"KB={K:4d} | closed {row['closed_acc']:.3f} | open {row['open_acc']:.3f} | "
            f"retrieval {row['retrieval_acc']:.3f} | {row['elapsed_sec']}s",
            flush=True,
        )

    payload = {
        "model": _MODEL, "seed": SEED, "n_eval": N_EVAL,
        "kb_sizes": list(KB_SIZES), "results": rows,
        "note": "closed-book = weights only; open-book = retrieved fact in context",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
