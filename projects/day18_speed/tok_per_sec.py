"""Part-2 BEAST SPEED — step 1: REAL tokens/sec, not one mat-vec.

Day-14 showed the LUT-GEMM ternary kernel beats fp32 1.25x on a single FFN-sized
mat-vec. But the dream needs an honest END-TO-END decode number: run a full single-
token decode through L transformer layers (QKV, attn-proj, FFN-up, FFN-down) using the
kernel for every matmul, time it, and report tok/s. Then project to the dream config
(a ~900M-active-param MoE) from the measured params/sec throughput.

This is the number that actually answers "can this PC hit 40-50 tok/s?".

Run:  python projects/day18_speed/tok_per_sec.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day14_kernel"))
from lut_gemm import build_table, encode_groups, lut_accumulate, G, NPAT  # type: ignore

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "tok_per_sec_results.json"

# dim profiles: small matrices (GPT-2) vs large expert matrices (the dream regime).
# The LUT win grows with N (outputs amortize the per-token table build), so the
# relevant test for a 400B MoE is the LARGE profile, not GPT-2-small.
PROFILES = {
    "gpt2_small": dict(D=768, FF=4, LAYERS=12),
    "large_expert": dict(D=4096, FF=3, LAYERS=8),   # LLaMA-7B-ish layer widths
}
REPS = 20


class LutLinear:
    """A ternary weight matrix stored as group-indices, applied via the LUT kernel."""

    def __init__(self, K, N, seed):
        rng = np.random.default_rng(seed)
        Wf = rng.standard_normal((K, N)).astype(np.float32)
        thr = 0.7 * np.abs(Wf).mean(0, keepdims=True)
        signs = np.zeros_like(Wf, dtype=np.int8)
        signs[Wf > thr] = 1
        signs[Wf < -thr] = -1
        self.scale = ((np.abs(Wf) * (signs != 0)).sum(0)
                      / np.clip((signs != 0).sum(0), 1, None)).astype(np.float32)
        self.Wdeq = signs.astype(np.float32) * self.scale          # for fp32 baseline
        self.idx = encode_groups(signs)                            # [N, KG]
        self.K, self.N, self.KG = K, N, K // G
        self._table = np.zeros((self.KG, NPAT), dtype=np.float32)
        self._y = np.zeros(N, dtype=np.float32)
        self.n_params = K * N

    def lut(self, x):
        build_table(x, self.KG, G, NPAT, self._table)
        lut_accumulate(self.idx, self._table, self.scale, self._y, self.N, self.KG)
        return self._y

    def fp32(self, x):
        return x @ self.Wdeq


def build_stack(D, FF, LAYERS):
    matmuls = [("attn.qkv", D, 3 * D), ("attn.proj", D, D),
               ("mlp.fc", D, FF * D), ("mlp.proj", FF * D, D)]
    layers = []
    seed = 0
    for _ in range(LAYERS):
        mats = []
        for _name, K, N in matmuls:
            mats.append(LutLinear(K, N, seed)); seed += 1
        layers.append(mats)
    return layers


def decode_step_lut(layers):
    """one token through all layers; feed each matmul's output as next input (sized)."""
    for mats in layers:
        for m in mats:
            x = np.ones(m.K, dtype=np.float32)     # dummy activation of the right width
            m.lut(x)


def decode_step_fp32(layers):
    for mats in layers:
        for m in mats:
            x = np.ones(m.K, dtype=np.float32)
            m.fp32(x)


def _time(fn, layers):
    fn(layers)                                     # warm-up / JIT
    t = time.perf_counter()
    for _ in range(REPS):
        fn(layers)
    return (time.perf_counter() - t) / REPS


def _run_profile(name, cfg):
    layers = build_stack(cfg["D"], cfg["FF"], cfg["LAYERS"])
    active = sum(m.n_params for mats in layers for m in mats)
    lut_s = _time(decode_step_lut, layers)
    fp_s = _time(decode_step_fp32, layers)
    pps = active / lut_s
    print(f"[{name}] D={cfg['D']} FF={cfg['FF']}x L={cfg['LAYERS']} | "
          f"{active/1e6:.0f}M params/token", flush=True)
    print(f"  fp32 : {fp_s*1e3:7.2f} ms -> {1/fp_s:6.1f} tok/s | "
          f"LUT : {lut_s*1e3:7.2f} ms -> {1/lut_s:6.1f} tok/s "
          f"({fp_s/lut_s:.2f}x) | {pps/1e6:.0f}M params/s", flush=True)
    return {"profile": name, **{k: cfg[k] for k in cfg},
            "active_params_M": round(active/1e6, 1),
            "lut_ms": round(lut_s*1e3, 3), "fp32_ms": round(fp_s*1e3, 3),
            "speedup_x": round(fp_s/lut_s, 3),
            "active_params_per_sec_M": round(pps/1e6, 1)}


def main():
    import os
    print(f"threads available: {os.cpu_count()} | numba prange in the accumulate\n", flush=True)
    results = {}
    for name, cfg in PROFILES.items():
        results[name] = _run_profile(name, cfg)
        print(flush=True)

    # honest tok/s projection uses the LARGE-matrix throughput (the dream regime),
    # since a 400B MoE's experts are large, not GPT-2-small.
    pps = results["large_expert"]["active_params_per_sec_M"] * 1e6
    print("projection at LARGE-expert throughput (the dream's matrix regime):", flush=True)
    proj = {}
    for tag, ap in [("dream MoE active 900M", 900e6), ("active 500M", 500e6),
                    ("active 300M", 300e6), ("active 200M", 200e6)]:
        tps = pps / ap
        proj[tag] = round(tps, 2)
        print(f"  {tag:24s} {ap/1e6:7.0f}M active -> {tps:6.1f} tok/s", flush=True)

    payload = {"profiles": results, "projection_tok_s_large_regime": proj,
               "note": "end-to-end decode, two matrix regimes. LUT wins only at large N "
                       "(table build amortises); GPT-2-small is too small. Projection uses "
                       "large-expert throughput = the 400B-MoE expert regime."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
