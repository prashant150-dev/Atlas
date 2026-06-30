"""Probe 2: can ONE codebook serve ALL layers (shared vocabulary)?
Even if layers are orthogonal, they may share a vocabulary of weight-vectors.
If a single codebook for all 12 layers matches per-layer codebooks, we save the
codebook overhead 12x (a real bits/weight win at scale)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day6_vector_quant"))
import numpy as np
from vq_vs_scalar import _assign, _kmeans, nmse
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained("models/gpt2", local_files_only=True)
sd = m.state_dict(); nl = m.config.n_layer
Ws = [sd[f"transformer.h.{i}.mlp.c_fc.weight"].detach().cpu().float().numpy() for i in range(nl)]
d, K = 4, 256


def vecs(W):
    a = W.reshape(-1); pad = (-a.size) % d
    if pad:
        a = np.concatenate([a, np.zeros(pad, a.dtype)])
    return a.reshape(-1, d)


per = []
for W in Ws:
    V = vecs(W); c = _kmeans(V, K, seed=0); idx = _assign(V, c)
    per.append(nmse(W.reshape(-1), c[idx].reshape(-1)[:W.size]))
allV = np.concatenate([vecs(W) for W in Ws], 0)
cs = _kmeans(allV, K, seed=0)
sh = []
for W in Ws:
    V = vecs(W); idx = _assign(V, cs)
    sh.append(nmse(W.reshape(-1), cs[idx].reshape(-1)[:W.size]))
print(f"per-layer codebooks NMSE avg: {np.mean(per):.4f}")
print(f"ONE shared codebook NMSE avg: {np.mean(sh):.4f}")
print(f"penalty {np.mean(sh)/np.mean(per):.3f}x | codebook overhead saved 12x")
# net bits/weight: per-layer = log2K/d + K*d*32/Nlayer_weights ; shared = log2K/d + K*d*32/(12*N)
Nw = Ws[0].size
import math
per_bpw = math.log2(K)/d + K*d*32/Nw
sh_bpw = math.log2(K)/d + K*d*32/(nl*Nw)
print(f"bits/weight: per-layer {per_bpw:.3f} -> shared {sh_bpw:.3f} (saves {per_bpw-sh_bpw:.3f} b/w)")
