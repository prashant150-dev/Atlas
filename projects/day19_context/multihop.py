"""Part-3 BEAST CONTEXT — step 2: multi-hop retrieval (the lever beyond naive RAG).

Step-1 proved single-fact needle retrieval scales to 15M tokens. But real "beast
context" questions need information COMBINED across chunks that are far apart and don't
share tokens with the question. Naive (single-hop) RAG fails these; iterative multi-hop
retrieval is the lever.

Setup at scale (chunks buried in a 1M / 15M-token filler haystack):
  chunk A:  "the secret property of <ent> is <mid>"        (links ent -> mid)
  chunk B:  "the path from <mid> leads to <answer>"        (links mid -> answer; far away,
                                                            shares NO token with <ent>)
  query:    relates to <ent>, asks for the final <answer>.

  * single-hop: retrieve with the query's entity token only -> finds A, never B
    (B doesn't contain <ent>) -> cannot answer.
  * multi-hop : retrieve A, EXTRACT <mid>, retrieve B with <mid>, read <answer>.

Measure answer-accuracy single-hop vs multi-hop, plus latency (2 lookups), at scale.

Run:  python projects/day19_context/multihop.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from needle_scale import CHUNK, _FILLER, _rand_entity, build_index  # type: ignore

OUT = _HERE / "multihop_results.json"
SEED = 0
SIZES = [1_000_000, 15_000_000]
N_CHAINS = 200
TOPK = 5


def build_corpus(n_tokens, rng):
    n_chunks = n_tokens // CHUNK
    chunks = ["" for _ in range(n_chunks)]
    slots = rng.sample(range(n_chunks), 2 * N_CHAINS)        # distinct chunks for A and B
    chains = []
    for i in range(N_CHAINS):
        ent, mid, ans = _rand_entity(rng), _rand_entity(rng), _rand_entity(rng)
        a_ci, b_ci = slots[2 * i], slots[2 * i + 1]          # A and B far apart, random
        a = f"the secret property of {ent} is {mid}"
        b = f"the path from {mid} leads to {ans}"
        chunks[a_ci] = a + " " + " ".join(rng.choice(_FILLER) for _ in range(CHUNK - len(a.split())))
        chunks[b_ci] = b + " " + " ".join(rng.choice(_FILLER) for _ in range(CHUNK - len(b.split())))
        chains.append((ent, mid, ans, a_ci, b_ci))
    for ci in range(n_chunks):
        if not chunks[ci]:
            chunks[ci] = " ".join(rng.choice(_FILLER) for _ in range(CHUNK))
    return chunks, chains


def _retrieve(inv, query_tokens, topk):
    score = defaultdict(int)
    for t in query_tokens:
        for ci in inv.get(t, ()):
            score[ci] += 1
    return sorted(score, key=lambda c: -score[c])[:topk]


def _extract_mid(chunk):
    """parse 'the secret property of <ent> is <mid> ...' -> <mid>."""
    toks = chunk.split()
    if "is" in toks:
        i = toks.index("is")
        if i + 1 < len(toks):
            return toks[i + 1]
    return None


def _extract_answer(chunk):
    """parse 'the path from <mid> leads to <answer> ...' -> <answer>."""
    toks = chunk.split()
    if "to" in toks:
        i = toks.index("to")
        if i + 1 < len(toks):
            return toks[i + 1]
    return None


def main():
    print(f"{'tokens':>12} | {'single-hop acc':>14} | {'multi-hop acc':>13} | "
          f"{'multihop ms':>12}", flush=True)
    rows = []
    for n in SIZES:
        rng = random.Random(SEED)
        chunks, chains = build_corpus(n, rng)
        inv = build_index(chunks)

        single_ok, multi_ok, lat = 0, 0, 0.0
        for ent, mid, ans, a_ci, b_ci in chains:
            # retrieve on the RARE discriminative token only (common words have huge
            # postings and make lookup O(N); rare entity tokens keep latency flat).
            # --- single-hop: only the entity is known; one retrieval ---
            got1 = _retrieve(inv, [ent], TOPK)
            # best it can do is read A; B is unreachable without <mid>
            s_ans = None
            for ci in got1:
                if _extract_answer(chunks[ci]):     # B would have an answer; A won't
                    s_ans = _extract_answer(chunks[ci]); break
            if s_ans == ans:
                single_ok += 1

            # --- multi-hop: A -> mid -> B -> answer ---
            t = time.perf_counter()
            hopA = _retrieve(inv, [ent], TOPK)
            got_mid = None
            for ci in hopA:
                m = _extract_mid(chunks[ci])
                if m == mid or (m and f"property of {ent}" in chunks[ci]):
                    got_mid = m; break
            m_ans = None
            if got_mid:
                hopB = _retrieve(inv, [got_mid], TOPK)
                for ci in hopB:
                    # only the B-chunk ('the path from <mid> leads to <answer>') answers;
                    # the A-chunk also contains <mid> but is not a 'leads to' fact.
                    if "leads" in chunks[ci] and f"from {got_mid}" in chunks[ci]:
                        m_ans = _extract_answer(chunks[ci]); break
            lat += time.perf_counter() - t
            if m_ans == ans:
                multi_ok += 1

        s_acc = single_ok / len(chains)
        m_acc = multi_ok / len(chains)
        ms = lat / len(chains) * 1e3
        rows.append({"tokens": n, "chains": len(chains),
                     "single_hop_acc": round(s_acc, 4), "multi_hop_acc": round(m_acc, 4),
                     "multihop_ms": round(ms, 3)})
        print(f"{n:12,} | {s_acc:14.3f} | {m_acc:13.3f} | {ms:12.3f}", flush=True)

    payload = {"chunk_tokens": CHUNK, "topk": TOPK, "n_chains": N_CHAINS, "rows": rows,
               "note": "multi-hop vs single-hop retrieval over a 1M/15M-token haystack; "
                       "answer needs 2 chunks chained (ent->mid->answer). Single-hop cannot "
                       "reach the 2nd chunk; iterative retrieval can. The lever beyond naive RAG."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
