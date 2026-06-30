"""Day-5 R6: native low-bit reasoner vs the 0.000 post-hoc wall.

R5 showed a post-hoc ternary GPT-2 reasoner collapses to 0.000 answer accuracy.
D3's thesis: a model trained *natively* in ternary+sparse space survives where
post-hoc dies. R6 tests that thesis on a reasoner-shaped task: **in-context
associative recall** — the sequence holds several (key, value) pairs with a
RANDOM mapping per example, then a query key; the model must read the context and
output that key's value. The mapping is fresh every example, so nothing can be
memorised in the weights — it is the abstract form of "read the retrieved fact
and answer".

Three variants share one config (the D3 comparison):
  * DenseFP            — fp baseline
  * PostHocTernary     — the fp model ternarized with NO retraining (the wall)
  * AetherNet          — natively trained ternary + sparse MoE

If AetherNet >> PostHocTernary (which sits near chance), native low-bit breaks
the wall, and Lever 1 of the dream has a measured proof.

Run from repo root::

    python projects/day5_reasoner_memory/r6_native_reasoner.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.architecture.aethernet import (  # noqa: E402
    AetherNetConfig, build_aethernet, build_dense_fp, ternarize_dense_to_posthoc)
from src.architecture.experiment import _evaluate, _train  # noqa: E402

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "r6_results.json"
LOG = _HERE / "r6_log.jsonl"

N_SLOTS = 5                        # how many value-fields in the context
N_VALS = 12                        # value symbols
SEQ_LEN = N_SLOTS + 3              # values.. , sep, query_index, answer
VOCAB = 1 + N_VALS + N_SLOTS       # 0=sep, 1..V values, then N_SLOTS index tokens
CHANCE = 1.0 / N_VALS
N_TRAIN = 6000
N_TEST = 2000
DENSE_STEPS = 1500
AETHER_STEPS = 2500
BATCH = 128


def make_recall(n: int, gen: torch.Generator):
    """Indexed value retrieval (1-layer solvable, still must read context).

    Sequence = [v_0 .. v_{S-1}, 0, qi, v_qi]. The values are random per example
    (nothing memorisable); qi is an index token pointing at one slot; the model
    must output that slot's value. Only the qi position is scored.
    """

    val_off = 1                        # values occupy 1..N_VALS
    idx_off = 1 + N_VALS               # index tokens occupy idx_off .. idx_off+N_SLOTS-1
    inputs = torch.zeros(n, SEQ_LEN, dtype=torch.long)
    vals = torch.randint(0, N_VALS, (n, N_SLOTS), generator=gen) + val_off
    inputs[:, :N_SLOTS] = vals
    qsel = torch.randint(0, N_SLOTS, (n,), generator=gen)   # which slot to fetch
    qval = vals[torch.arange(n), qsel]
    inputs[:, N_SLOTS] = 0                                  # separator
    inputs[:, N_SLOTS + 1] = qsel + idx_off                # query index token
    inputs[:, N_SLOTS + 2] = qval                          # the answer

    targets = torch.full_like(inputs, -100)
    targets[:, SEQ_LEN - 2] = qval     # at the query-index position, predict value
    return inputs, targets


def _log(row):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n"); fh.flush()


def main():
    torch.manual_seed(0)
    g_tr = torch.Generator().manual_seed(0)
    g_te = torch.Generator().manual_seed(1)
    Xtr, Ytr = make_recall(N_TRAIN, g_tr)
    Xte, Yte = make_recall(N_TEST, g_te)
    cfg = AetherNetConfig(vocab_size=VOCAB, seq_len=SEQ_LEN, seed=0)
    print(f"recall task: {N_SLOTS} slots, vocab {VOCAB}, seq {SEQ_LEN}, chance {CHANCE:.3f}", flush=True)

    t0 = time.perf_counter()
    rows = []

    # DenseFP
    dense = build_dense_fp(cfg)
    df, dl = _train(dense, Xtr, Ytr, DENSE_STEPS, 3e-3, BATCH, seed=0)
    dacc, _ = _evaluate(dense, Xte, Yte)
    da = dense.bit_account()
    rows.append(("DenseFP", dacc, da, df, dl))
    print(f"DenseFP        | test acc {dacc:.3f} | stored {da.stored_bits:.0f} bits", flush=True)

    # PostHoc ternary (no retrain) — the wall
    posthoc = ternarize_dense_to_posthoc(dense)
    pacc, _ = _evaluate(posthoc, Xte, Yte)
    pa = posthoc.bit_account()
    rows.append(("PostHocTernary", pacc, pa, float("nan"), float("nan")))
    print(f"PostHocTernary | test acc {pacc:.3f}", flush=True)

    # AetherNet native ternary+sparse
    aether = build_aethernet(cfg)
    af, al = _train(aether, Xtr, Ytr, AETHER_STEPS, 3e-3, BATCH, seed=1)
    aacc, _ = _evaluate(aether, Xte, Yte)
    aa = aether.bit_account()
    rows.append(("AetherNet", aacc, aa, af, al))
    print(f"AetherNet      | test acc {aacc:.3f} | stored {aa.stored_bits:.0f} bits "
          f"({da.stored_bits/aa.stored_bits:.1f}x smaller, active "
          f"{da.active_bits_per_token/aa.active_bits_per_token:.1f}x)", flush=True)

    payload = {
        "task": "in_context_associative_recall",
        "n_slots": N_SLOTS, "vocab": VOCAB, "seq_len": SEQ_LEN, "chance": CHANCE,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "variants": [],
    }
    for name, acc, acct, fl, ll in rows:
        d = {"name": name, "test_acc": round(acc, 4),
             "stored_bits": round(acct.stored_bits, 1),
             "active_bits_per_token": round(acct.active_bits_per_token, 1),
             "stored_ratio_vs_fp": round(rows[0][2].stored_bits / acct.stored_bits, 2),
             "active_ratio_vs_fp": round(rows[0][2].active_bits_per_token / acct.active_bits_per_token, 2)}
        payload["variants"].append(d)
        _log(d)

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nverdict: post-hoc {rows[1][1]:.3f} (chance {CHANCE:.3f}) vs native {rows[2][1]:.3f}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
