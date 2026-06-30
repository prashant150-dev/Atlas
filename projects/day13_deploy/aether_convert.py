"""AetherConvert — turn ANY Hugging Face causal-LM into our VQ-compressed format,
then load it back and use it like a normal model.

What it does:
  convert : load a model, vector-quantize every large 2D weight matrix (d=4, K=256,
            ~2 bits/weight, the P-A recipe), pack to disk + manifest, copy the
            tokenizer/config. Architecture-agnostic — works on GPT-2, Llama, Qwen,
            Phi, etc. (anything AutoModelForCausalLM loads).
  load    : reconstruct the weights and return a normal AutoModelForCausalLM you can
            generate with exactly like the original.
  chat    : quick interactive generation from a converted folder.

Honest notes:
  - DISK shrinks for real (~2 bits/weight on the big matrices). On CPU, RAM/speed do
    NOT improve (weights dequantise to fp32; no low-bit kernel — see Phase B).
  - Quality drops at ~2 bits, more on weak/old models (e.g. GPT-2). Bigger, newer
    models are more redundant and tolerate it better. Healing (the GPT-2 path)
    recovers quality but is model-specific; this general converter is post-hoc.

CLI:
  python projects/day13_deploy/aether_convert.py convert models/gpt2 experiments/gpt2_aether
  python projects/day13_deploy/aether_convert.py chat experiments/gpt2_aether
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
from vq_vs_scalar import _assign, _kmeans  # type: ignore  # noqa: E402

DEFAULT_D = 4
DEFAULT_K = 256
MIN_DIM = 256          # only quantize matrices whose both dims >= this (overhead amortises)
# embedding / output-head weights ARE token identity — 2-bit VQ on them destroys
# generation (measured: garbage output). Keep them raw.
_SKIP = ("wte", "wpe", "embed", "lm_head", "shared", "embeddings", "tok_embeddings")


# --------------------------------------------------------------------------
def _vq_encode(W: np.ndarray, d: int, K: int, seed: int):
    flat = W.reshape(-1)
    pad = int((-flat.size) % d)
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, flat.dtype)])
    V = flat.reshape(-1, d)
    cent = _kmeans(V, K, seed=seed).astype(np.float32)
    idx = _assign(V, cent)
    idx = idx.astype(np.uint8 if K <= 256 else np.int32)
    return cent, idx, pad


def _vq_decode(cent, idx, shape, pad):
    rec = cent[idx.astype(np.int64)].reshape(-1)
    n = int(np.prod(shape))
    return torch.from_numpy(rec[:n].reshape(shape).astype(np.float32))


def convert(model_path: str, out_dir: str, d=DEFAULT_D, K=DEFAULT_K, seed=0):
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"loading {model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    sd = model.state_dict()

    store: dict[str, torch.Tensor] = {}
    manifest = {"format": "aethercore.vq", "d": d, "K": K, "tensors": {}}
    orig_bits = comp_bits = 0
    nq = nr = 0
    seen_ptr: dict[int, str] = {}        # dedup tied/shared weights (e.g. GPT-2 wte<->lm_head)
    for i, (name, t) in enumerate(sd.items()):
        W = t.detach().cpu().float().numpy()
        orig_bits += W.size * 32
        skip = any(s in name.lower() for s in _SKIP)
        if W.ndim == 2 and min(W.shape) >= MIN_DIM and not skip:
            cent, idx, pad = _vq_encode(W, d, K, seed)
            store[f"{i}.cb"] = torch.from_numpy(cent)
            store[f"{i}.idx"] = torch.from_numpy(idx)
            manifest["tensors"][name] = {"kind": "vq", "shape": list(W.shape), "pad": pad,
                                         "cb": f"{i}.cb", "idx": f"{i}.idx",
                                         "dtype": str(t.dtype)}
            comp_bits += math.log2(K) * (W.size + pad) / d + K * d * 32
            nq += 1
        else:
            ptr = t.detach().cpu().untyped_storage().data_ptr()
            if ptr in seen_ptr:           # shared storage -> store once, reference it
                manifest["tensors"][name] = {"kind": "ref", "target": seen_ptr[ptr]}
                continue
            key = f"{i}.raw"
            store[key] = t.detach().cpu().clone().contiguous()
            seen_ptr[ptr] = name
            manifest["tensors"][name] = {"kind": "raw", "raw": key, "dtype": str(t.dtype)}
            comp_bits += W.size * 32
            nr += 1

    save_file(store, str(out / "weights.safetensors"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # copy config + tokenizer sidecars (everything except big weight files)
    src = Path(model_path)
    for f in src.iterdir():
        if f.is_file() and f.suffix not in {".safetensors", ".bin", ".pt", ".h5", ".msgpack"}:
            shutil.copy2(f, out / f.name)

    ratio = orig_bits / comp_bits
    print(f"  quantized {nq} matrices, kept {nr} raw", flush=True)
    print(f"  size: {orig_bits/8/1e6:.1f} MB -> {comp_bits/8/1e6:.1f} MB  ({ratio:.1f}x smaller)", flush=True)
    print(f"  written {out}", flush=True)
    return {"orig_MB": round(orig_bits/8/1e6, 1), "comp_MB": round(comp_bits/8/1e6, 1),
            "ratio": round(ratio, 2), "quantized": nq, "raw": nr}


def load(out_dir: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors.torch import load_file
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    store = load_file(str(out / "weights.safetensors"))
    state = {}
    refs = {}
    for name, e in manifest["tensors"].items():
        if e["kind"] == "vq":
            cent = store[e["cb"]].numpy()
            idx = store[e["idx"]].numpy()
            state[name] = _vq_decode(cent, idx, tuple(e["shape"]), e["pad"]).to(
                getattr(torch, e["dtype"].split(".")[-1]))
        elif e["kind"] == "ref":
            refs[name] = e["target"]
        else:
            state[name] = store[e["raw"]]
    for name, target in refs.items():
        state[name] = state[target]
    config = AutoConfig.from_pretrained(out, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def chat(out_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out_dir, local_files_only=True)
    model = load(out_dir)
    print(f"loaded VQ-compressed model from {out_dir}. Type a prompt ('exit' to quit).")
    while True:
        try:
            p = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!"); break
        if p.lower() in {"exit", "quit", "q"}:
            break
        if not p:
            continue
        inp = tok(p, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        print(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip())


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "convert":
        convert(sys.argv[2], sys.argv[3])
    elif cmd == "load":
        m = load(sys.argv[2]); print(f"loaded OK: {m.num_parameters():,} params")
    elif cmd == "chat":
        chat(sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
