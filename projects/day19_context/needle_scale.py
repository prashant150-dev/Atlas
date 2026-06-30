"""Part-3 BEAST CONTEXT — step 1: does retrieval give 15M-token context on this PC?

True attention over 15M tokens is physically impossible here (KV cache ~2.9 TB). The
dream's only path is RETRIEVAL: store the context on disk, pull the few relevant chunks
per query. This probe measures whether that path actually SCALES to 15M tokens:

  * needle-in-a-haystack: bury N rare "fact" chunks in a huge filler corpus,
  * index with an INVERTED INDEX (token -> chunk-ids), the structure that scales,
  * query each needle and measure recall@k, retrieval latency, index storage,
  * sweep context size 100K -> 1M -> 4M -> 15M tokens.

If recall stays high and latency/storage stay small as tokens -> 15M, the context
axis of the dream is reachable on this PC. Measure, don't assume.

Run:  python projects/day19_context/needle_scale.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "needle_scale_results.json"

CHUNK = 60                      # tokens per chunk (retrieval granularity)
SEED = 0
SIZES = [100_000, 1_000_000, 4_000_000, 15_000_000]
N_NEEDLES = 200                 # buried facts to query
TOPK = 5

# filler vocabulary (common words) + rare needle entities (unique, low-collision)
_FILLER = ("the of and to in a is that it was for on are as with his they at be this "
           "have from or one had by word but not what all were we when your can said there "
           "use an each which she do how their if will up other about out many then them these "
           "so some her would make like him into time has look two more write go see number no").split()


def _rand_entity(rng):
    cons, vow = "bcdfghjklmnpqrstvwxz", "aeiou"
    return "".join(rng.choice(cons) + rng.choice(vow) for _ in range(4))


def build_corpus(n_tokens, rng):
    """list of chunks (strings). Inject N_NEEDLES needle chunks with unique entities."""
    n_chunks = n_tokens // CHUNK
    chunks = []
    # positions for needles spread across the whole context
    needle_pos = set(rng.sample(range(n_chunks), min(N_NEEDLES, n_chunks)))
    needles = {}                                  # entity -> (chunk_id, attribute)
    for ci in range(n_chunks):
        if ci in needle_pos:
            ent = _rand_entity(rng)
            attr = _rand_entity(rng)
            # the fact sentence + filler padding to CHUNK tokens
            body = f"the secret property of {ent} is {attr}"
            pad = " ".join(rng.choice(_FILLER) for _ in range(CHUNK - len(body.split())))
            chunks.append(body + " " + pad)
            needles[ent] = (ci, attr)
        else:
            chunks.append(" ".join(rng.choice(_FILLER) for _ in range(CHUNK)))
    return chunks, needles


def build_index(chunks):
    """inverted index token -> list[chunk_id]. The structure that scales to 15M."""
    inv = defaultdict(list)
    for ci, ch in enumerate(chunks):
        for tok in set(ch.split()):
            inv[tok].append(ci)
    return inv


def retrieve(inv, chunks, query_tokens, topk):
    """candidate chunks = union of postings for query tokens; rank by overlap count."""
    score = defaultdict(int)
    for t in query_tokens:
        for ci in inv.get(t, ()):                 # rare entity token -> short postings
            score[ci] += 1
    ranked = sorted(score, key=lambda c: -score[c])[:topk]
    return ranked


def main():
    rng = random.Random(SEED)
    print(f"{'tokens':>12} | {'chunks':>9} | {'recall@%d' % TOPK:>9} | "
          f"{'index MB':>9} | {'build s':>8} | {'query ms':>9}", flush=True)
    rows = []
    for n in SIZES:
        r = random.Random(SEED)                   # same needles layout base per size
        chunks, needles = build_corpus(n, r)
        t0 = time.perf_counter()
        inv = build_index(chunks)
        build_s = time.perf_counter() - t0

        # query every needle: "the secret property of <ent> is" -> expect its chunk
        hits, lat = 0, 0.0
        for ent, (true_ci, _attr) in needles.items():
            q = ["secret", "property", ent]       # entity is the discriminative token
            t = time.perf_counter()
            got = retrieve(inv, chunks, q, TOPK)
            lat += time.perf_counter() - t
            if true_ci in got:
                hits += 1
        recall = hits / max(1, len(needles))
        query_ms = lat / max(1, len(needles)) * 1e3

        # index storage estimate: total postings * ~4 bytes (chunk-id int)
        postings = sum(len(v) for v in inv.values())
        index_mb = postings * 4 / 1e6

        row = {"tokens": n, "chunks": len(chunks), "needles": len(needles),
               "recall_at_k": round(recall, 4), "index_mb": round(index_mb, 1),
               "build_s": round(build_s, 2), "query_ms": round(query_ms, 3)}
        rows.append(row)
        print(f"{n:12,} | {len(chunks):9,} | {recall:9.3f} | {index_mb:9.1f} | "
              f"{build_s:8.2f} | {query_ms:9.3f}", flush=True)

    payload = {"chunk_tokens": CHUNK, "topk": TOPK, "n_needles": N_NEEDLES, "rows": rows,
               "note": "needle-in-haystack retrieval over a growing context via inverted index; "
                       "shows recall/latency/storage scaling toward 15M tokens (true attention "
                       "would need ~2.9TB KV — impossible here)."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
