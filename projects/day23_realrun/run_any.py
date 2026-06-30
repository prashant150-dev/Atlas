"""Run ANY converted model end-to-end on our tech, vs FP — memory-safe (one at a time).

Loads the FP model, generates, FREES it, then loads OUR 2-bit-reconstructed model and
generates. Peak RAM ~ one model (not two), and reconstructed weights kept in bf16.

Run:  python projects/day23_realrun/run_any.py models/qwen2.5-1.5b experiments/qwen_streamed "Write a haiku about the sea"
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _dequant(c, z):
    if c["kind"] != "mixed_vq":
        return z["raw"].astype(np.float32)
    cent = z["codebook"].astype(np.float32)
    rec = cent[z["idx"]].copy()
    if len(z["prot_pos"]):
        rec[z["prot_pos"]] = z["prot_q"].astype(np.float32) * z["prot_scale"][:, None].astype(np.float32)
    n = int(np.prod(c["shape"]))
    return rec.reshape(-1)[:n].reshape(c["shape"])


def _load_our_sd(streamed: Path):
    man = json.loads((streamed / "manifest.json").read_text())
    sd = {}
    for key, meta in man["tensors"].items():
        z = np.load(streamed / meta["file"])
        W = _dequant(meta, z)
        sd[key] = torch.from_numpy(np.ascontiguousarray(W)).to(torch.bfloat16)
    return sd


@torch.inference_mode()
def _gen(model, tok, prompt, n=60):
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    ids = tok(text, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = Path(sys.argv[1]); streamed = Path(sys.argv[2])
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Write a short Python function for factorial."
    tok = AutoTokenizer.from_pretrained(str(src), local_files_only=True)
    print("=" * 72 + f"\nPROMPT: {prompt!r}\n" + "=" * 72, flush=True)

    # 1) FP model -> generate -> free
    print("\n[1] FP model loading + generating...", flush=True)
    fp = AutoModelForCausalLM.from_pretrained(str(src), local_files_only=True,
                                              torch_dtype=torch.bfloat16).eval()
    t = time.perf_counter(); fp_out = _gen(fp, tok, prompt); fp_dt = time.perf_counter() - t
    del fp; gc.collect()

    # 2) OUR 2-bit model -> generate
    print("[2] OUR 2-bit model loading (reconstruct) + generating...", flush=True)
    ours = AutoModelForCausalLM.from_pretrained(str(src), local_files_only=True,
                                                torch_dtype=torch.bfloat16).eval()
    sd = _load_our_sd(streamed)
    miss, unexp = ours.load_state_dict(sd, strict=False)
    t = time.perf_counter(); our_out = _gen(ours, tok, prompt); our_dt = time.perf_counter() - t

    print("\n" + "=" * 72, flush=True)
    print(f"FP (bf16):\n  {fp_out.strip()}\n  [{fp_dt:.1f}s]", flush=True)
    print("-" * 72, flush=True)
    print(f"OURS (2-bit AetherCore, no-heal):\n  {our_out.strip()}\n  [{our_dt:.1f}s]", flush=True)
    print("=" * 72, flush=True)
    print(f"reconstructed {len(sd)} tensors ({len(miss)} missing, {len(unexp)} unexpected)", flush=True)


if __name__ == "__main__":
    main()
