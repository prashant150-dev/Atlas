"""Part-1 BEAST SIZE, lever 1: impact-weighted mixed-precision VQ.

Goal: close the gap to FP. Not all weights matter equally. Quantize MOST weight-
vectors at 2-bit VQ, but PROTECT the small fraction whose quantization error is
largest (the 'critical' vectors) at higher precision. A few extra bits on the
critical few should buy a big quality jump -> toward FP quality at low average bits.

Fast probe on a real GPT-2 matrix: average bits vs reconstruction NMSE, for
protect-fractions p = 0%, 1%, 2%, 5%, 10%. Baseline p=0 = plain VQ (our 2-bit).

Run:  python projects/day17_beast_size/mixed_precision_probe.py
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day6_vector_quant"))
from vq_vs_scalar import _assign, _kmeans, nmse  # type: ignore

d, K = 4, 256


def mixed_vq(W, protect_frac, seed=0):
    """VQ all d-vectors; keep the top-`protect_frac` highest-error vectors at FP."""
    a = W.reshape(-1).astype(np.float32)
    pad = (-a.size) % d
    if pad:
        a = np.concatenate([a, np.zeros(pad, np.float32)])
    V = a.reshape(-1, d)
    c = _kmeans(V, K, seed=seed)
    idx = _assign(V, c)
    recon = c[idx].copy()
    err = ((V - recon) ** 2).sum(1)          # per-vector squared error
    nprot = int(len(V) * protect_frac)
    if nprot:
        worst = np.argpartition(err, -nprot)[-nprot:]
        # protect at int8 (near-lossless for a 4-vector, 4x cheaper than fp32)
        Vw = V[worst]
        scale = np.abs(Vw).max(1, keepdims=True) / 127 + 1e-12
        recon[worst] = np.round(Vw / scale).clip(-127, 127) * scale
    out = recon.reshape(-1)[:W.size].reshape(W.shape)
    # average bits/weight: (1-p) at index(log2K/d) + p at int8(8)
    base_bpw = math.log2(K) / d + K * d * 32 / W.size
    avg_bpw = (1 - protect_frac) * base_bpw + protect_frac * 8
    return out, avg_bpw


def main():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("models/gpt2", local_files_only=True)
    W = m.state_dict()["transformer.h.0.mlp.c_fc.weight"].detach().cpu().float().numpy()
    del m
    print(f"matrix mlp.c_fc {W.shape}\n  protect%  avg-bits   NMSE     vs-plain", flush=True)
    base = None
    for p in [0.0, 0.01, 0.02, 0.05, 0.10]:
        r, bpw = mixed_vq(W, p)
        e = nmse(W, r)
        if base is None:
            base = e
        print(f"   {p*100:4.0f}%   {bpw:6.2f}    {e:.4f}   {base/e:5.2f}x better", flush=True)


if __name__ == "__main__":
    main()
