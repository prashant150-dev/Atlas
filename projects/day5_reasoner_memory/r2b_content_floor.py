"""Day-5 R2b: lossless storage floor of the knowledge CONTENT.

R2 found that a dense GPT-2-embedding index is a poor (and uncompressible) memory
index here — lexical TF-IDF retrieves perfectly and is essentially free. So the
real memory-storage cost is the CONTENT (the fact text the reasoner reads). This
measures that content's lossless floor (gzip ~ entropy) and how bits-per-fact
scale, to project "how much knowledge fits on a small disk".

Run from repo root::

    python projects/day5_reasoner_memory/r2b_content_floor.py
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_HERE = Path(__file__).resolve().parent
try:
    from projects.day5_reasoner_memory.r1_keystone import _make_kb
except ModuleNotFoundError:
    sys.path.insert(0, str(_HERE))
    from r1_keystone import _make_kb  # type: ignore

OUT = _HERE / "r2b_results.json"
SIZES = (500, 5000, 50000)
SEED = 0


def main() -> None:
    rows = []
    for K in SIZES:
        kb = _make_kb(K, random.Random(SEED))
        blob = "\n".join(f["fact"] for f in kb).encode("utf-8")
        gz = gzip.compress(blob, 9)
        raw_bpf = len(blob) * 8 / K
        gz_bpf = len(gz) * 8 / K
        rows.append({
            "facts": K,
            "raw_bytes": len(blob),
            "gzip_bytes": len(gz),
            "raw_bits_per_fact": round(raw_bpf, 1),
            "gzip_bits_per_fact": round(gz_bpf, 1),
            "compression_x": round(raw_bpf / gz_bpf, 2),
        })
        print(f"K={K:6d} | raw {raw_bpf:6.1f} b/fact | gzip {gz_bpf:6.1f} b/fact "
              f"| {rows[-1]['compression_x']}x", flush=True)

    # project the largest measured gzip bits/fact to big knowledge bases
    bpf = rows[-1]["gzip_bits_per_fact"]
    proj = {f"{n:,}_facts_MB": round(n * bpf / 8 / 1e6, 1)
            for n in (1_000_000, 100_000_000, 1_000_000_000)}
    print("projection (lossless content):", proj, flush=True)

    OUT.write_text(json.dumps({"sizes": list(SIZES), "results": rows,
                               "projection_MB": proj}, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
