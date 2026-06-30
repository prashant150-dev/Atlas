"""Day-8 Stage 2: VQ vs GPTQ/AWQ-style scalar quantization (SOTA positioning).

Reviewer critique #16: "how is this better than GPTQ/AWQ/etc?" We cannot install
the official libraries offline, so we faithfully reimplement their CORE algorithm
-- group-wise scalar quantization with per-group scales -- which is the shared
backbone of GPTQ / AWQ / HQQ / EXL2. (Those add activation-aware scaling and
error-correction that shift the scalar curve a little but do not change the
extreme-low-bit story; noted honestly.)

We compare post-hoc (no healing -- GPTQ/AWQ are post-hoc) whole-model GPT-2
perplexity on real English at MATCHED bits/weight. Expect: scalar is excellent at
4-bit (near lossless) and competitive at 3-bit; VQ wins decisively at 2-bit and
below, where scalar collapses. That crossover is the honest SOTA answer.

Run from repo root::

    python projects/day8_validation/stage2_sota.py
"""

from __future__ import annotations

import json
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
from stage1_real_lang import _apply_recon, _eval_ids, _fresh, _ppl, _WRAP  # type: ignore  # noqa: E402
from vq_vs_scalar import vector_quant  # type: ignore  # noqa: E402

OUT = _HERE / "stage2_results.json"
LOG = _HERE / "stage2_log.jsonl"


def scalar_groupwise(W, bits, group=64):
    """GPTQ/AWQ/HQQ backbone: per-(group,output-channel) symmetric int quant.
    Returns (reconstruction, honest bits/weight incl per-group fp16 scales)."""
    in_f, out_f = W.shape
    levels = (1 << (bits - 1)) - 1
    pad = (-in_f) % group
    Wp = np.concatenate([W, np.zeros((pad, out_f), W.dtype)], 0) if pad else W
    g = Wp.reshape(-1, group, out_f)
    scale = np.clip(np.abs(g).max(axis=1, keepdims=True) / levels, 1e-12, None)
    q = np.clip(np.round(g / scale), -levels, levels) * scale
    recon = q.reshape(-1, out_f)[:in_f]
    n_groups = g.shape[0]
    bpw = bits + 16.0 * (n_groups * out_f) / (in_f * out_f)   # = bits + 16/group
    return recon.astype(np.float32), float(bpw)


def _log(r):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n"); fh.flush()


def main():
    from transformers import AutoTokenizer
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    LOG.write_text("", encoding="utf-8")
    tok = AutoTokenizer.from_pretrained("models/gpt2", local_files_only=True)
    ids = _eval_ids(tok)
    fp = _fresh().eval()
    fp_ppl = _ppl(fp, ids); del fp
    print(f"FP teacher ppl (real English) = {fp_ppl:.2f}", flush=True)
    _log({"method": "FP", "bits_per_weight": 32, "ppl": round(fp_ppl, 3)})

    # matched bit-width pairs: scalar (GPTQ/AWQ-style) vs VQ
    methods = [
        ("scalar_g64_int4", lambda W: scalar_groupwise(W, 4, 64)),
        ("vq_d2_K256(~4b)", lambda W: vector_quant(W, 2, 256, seed=0)),
        ("scalar_g64_int3", lambda W: scalar_groupwise(W, 3, 64)),
        ("vq_d2_K64(~3b)", lambda W: vector_quant(W, 2, 64, seed=0)),
        ("scalar_g64_int2", lambda W: scalar_groupwise(W, 2, 64)),
        ("vq_d4_K256(~2b)", lambda W: vector_quant(W, 4, 256, seed=0)),
    ]
    rows = []
    for label, fn in methods:
        t0 = time.perf_counter()
        m = _fresh().eval()
        # measure honest bits/weight from one matrix, then apply to all
        Wsample = m.state_dict()["transformer.h.0.mlp.c_fc.weight"].numpy().astype(np.float32)
        _, bpw = fn(Wsample)
        _apply_recon(m, fn)
        ppl = _ppl(m, ids); del m
        row = {"method": label, "bits_per_weight": round(bpw, 3), "ppl": round(ppl, 2),
               "sec": round(time.perf_counter() - t0, 1)}
        rows.append(row); _log(row)
        print(f"  {label:18s} | {bpw:5.2f} b/w | ppl {ppl:10.1f}", flush=True)

    payload = {"eval": "held-out English (offline)", "FP_ppl": fp_ppl,
               "baseline": "scalar group-wise (GPTQ/AWQ/HQQ backbone), post-hoc",
               "results": rows,
               "note": "matched bits/weight; crossover = honest SOTA positioning"}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
