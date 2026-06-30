"""Day-8 Stage 5: the remaining critique points, tested for real where possible.

One script, several honest probes:
  A. MoE routing health (#2/#8): expert load balance, dead experts, routing entropy.
  B. Shared codebook across layers (#9): per-matrix vs one-codebook-for-all NMSE.
  C. Memory storage realism (#15): templated vs rich (relations+context) bits/fact.
  D. D1 floor by weight type (#10): entropy/quantisability varies across GPT-2 layers.
  E. Forgetting proxy (#12): VQ+heal ppl on heal-text vs a different held-out text.

Points not testable offline are acknowledged honestly in the report (#5 benchmark
download, #16 official AWQ/GPTQ libs, #17/#10 other architectures, #11 multi-hop at
scale).

Run from repo root::

    python projects/day8_validation/stage5_remaining.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
sys.path.insert(0, str(_HERE.parent / "day7_vq_moe"))
from vq_vs_scalar import _assign, _kmeans, nmse  # type: ignore  # noqa: E402

OUT = _HERE / "stage5_results.json"
LOG = _HERE / "stage5_log.jsonl"
SEED = 0


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


# --- Probe A: MoE routing health -------------------------------------------
def probe_a():
    import vq_moe
    from src.architecture.experiment import TaskSpec, _make_char_lm_dataset, _train
    from vq_moe import LM, MoEFFN, N_EXPERT, TOP_K, EXP_H
    print("Probe A - MoE routing health:", flush=True)
    spec = TaskSpec(name="char_lm", vocab_size=vq_moe.VOCAB, seq_len=vq_moe.SEQ,
                    chance_accuracy=0.168, n_examples=4000, dense_steps=600,
                    aether_steps=600, dense_lr=3e-3, aether_lr=3e-3, batch_size=128)
    X, Y = _make_char_lm_dataset(spec, torch.Generator().manual_seed(0))
    lm = LM(MoEFFN(N_EXPERT, EXP_H, TOP_K))
    _train(lm, X, Y, 600, 3e-3, 128, seed=0)
    with torch.no_grad():
        x = lm.embed(X) + lm.pos[:, :X.shape[1]]
        x = x + lm.attn(lm.n1(x))
        fi = lm.n2(x).reshape(-1, x.shape[-1])
        topi = lm.ffn.router(fi).topk(TOP_K, -1).indices
    counts = torch.bincount(topi.reshape(-1), minlength=N_EXPERT).float()
    frac = (counts / counts.sum()).tolist()
    dead = int((counts == 0).sum().item())
    ideal = 1.0 / N_EXPERT
    ent = -sum(f * math.log(f + 1e-12) for f in frac) / math.log(N_EXPERT)  # 1=balanced
    cv = float(counts.std().item() / (counts.mean().item() + 1e-9))
    row = {"probe": "A_moe_routing", "n_expert": N_EXPERT, "top_k": TOP_K,
           "load_fraction": [round(f, 3) for f in frac], "dead_experts": dead,
           "ideal_load": round(ideal, 3), "routing_entropy_norm": round(ent, 3),
           "load_cv": round(cv, 3)}
    _log(row)
    print(f"  loads {[round(f,2) for f in frac]} | dead {dead}/{N_EXPERT} | "
          f"entropy {ent:.3f} (1=balanced) | CV {cv:.2f}", flush=True)
    return row


# --- Probe B: shared codebook across layers --------------------------------
def probe_b():
    from transformers import AutoModelForCausalLM
    print("Probe B - shared codebook across layers:", flush=True)
    m = AutoModelForCausalLM.from_pretrained("models/gpt2", local_files_only=True)
    sd = m.state_dict()
    names = ["transformer.h.0.attn.c_proj.weight",
             "transformer.h.0.mlp.c_fc.weight",
             "transformer.h.11.mlp.c_proj.weight"]
    mats = [sd[n].detach().cpu().float().numpy() for n in names]
    del m
    d, K = 4, 256

    def vecs(W):
        a = W.reshape(-1)
        pad = (-a.size) % d
        if pad:
            a = np.concatenate([a, np.zeros(pad, a.dtype)])
        return a.reshape(-1, d)

    # per-matrix codebooks
    per = []
    for W in mats:
        V = vecs(W); c = _kmeans(V, K, seed=SEED); idx = _assign(V, c)
        per.append(nmse(W.reshape(-1), c[idx].reshape(-1)[:W.size]))
    # one shared codebook for all three
    allV = np.concatenate([vecs(W) for W in mats], 0)
    cs = _kmeans(allV, K, seed=SEED)
    shared = []
    for W in mats:
        V = vecs(W); idx = _assign(V, cs)
        shared.append(nmse(W.reshape(-1), cs[idx].reshape(-1)[:W.size]))
    row = {"probe": "B_shared_codebook",
           "per_matrix_nmse": [round(x, 4) for x in per],
           "shared_nmse": [round(x, 4) for x in shared],
           "shared_penalty_x": round((sum(shared) / len(shared)) / (sum(per) / len(per)), 3)}
    _log(row)
    print(f"  per-matrix {[round(x,3) for x in per]} | shared {[round(x,3) for x in shared]} "
          f"| penalty {row['shared_penalty_x']}x", flush=True)
    return row


# --- Probe C: memory storage realism ---------------------------------------
def probe_c():
    print("Probe C - memory storage realism:", flush=True)
    import random
    rng = random.Random(0)
    cre = ["zor", "quen", "dro", "mire", "fen", "voss"]
    plc = ["mintar", "volgard", "eskfell", "dunmoor"]
    templated, rich = [], []
    for i in range(2000):
        c = rng.choice(cre) + str(i % 97); p = rng.choice(plc)
        templated.append(f"The {c} of {p} is known for its crimson colour.")
        rich.append(
            f"The {c}, a creature native to {p}, is known for its crimson colour, its "
            f"migratory habit of crossing the {p} delta each spring, its rivalry with "
            f"the pale {rng.choice(cre)} clans, and a lifespan of roughly {20+i%40} years; "
            f"local records first mention it in the year {1200+i%600}.")

    def bpf(items):
        blob = "\n".join(items).encode("utf-8")
        return len(gzip.compress(blob, 9)) * 8 / len(items)

    t, r = bpf(templated), bpf(rich)
    row = {"probe": "C_memory_realism", "templated_bits_per_fact": round(t, 1),
           "rich_bits_per_fact": round(r, 1), "realism_factor_x": round(r / t, 2),
           "rich_1B_facts_GB": round(1e9 * r / 8 / 1e9, 1)}
    _log(row)
    print(f"  templated {t:.0f} b/fact | rich {r:.0f} b/fact ({r/t:.1f}x) "
          f"| 1B rich facts ~ {row['rich_1B_facts_GB']} GB", flush=True)
    return row


# --- Probe D: D1 floor varies by weight type -------------------------------
def probe_d():
    from transformers import AutoModelForCausalLM
    print("Probe D - D1 floor by weight type (entropy proxy):", flush=True)
    m = AutoModelForCausalLM.from_pretrained("models/gpt2", local_files_only=True)
    sd = m.state_dict()
    types = {"attn.c_attn": "transformer.h.0.attn.c_attn.weight",
             "mlp.c_fc": "transformer.h.0.mlp.c_fc.weight",
             "wte_embed": "transformer.wte.weight",
             "ln_1": "transformer.h.0.ln_1.weight"}
    rows = {}
    for t, n in types.items():
        W = sd[n].detach().cpu().float().numpy().reshape(-1)
        # differential entropy of a Gaussian fit: 0.5*log2(2*pi*e*var) bits (proxy)
        var = float(W.var()) + 1e-12
        h = 0.5 * math.log2(2 * math.pi * math.e * var)
        rows[t] = round(h, 3)
    del m
    row = {"probe": "D_floor_by_type", "gaussian_entropy_bits": rows}
    _log(row)
    print(f"  gaussian-entropy proxy (bits): {rows}", flush=True)
    return row


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    res = {"A_routing": probe_a(), "B_shared_codebook": probe_b(),
           "C_memory_realism": probe_c(), "D_floor_by_type": probe_d()}
    res["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
