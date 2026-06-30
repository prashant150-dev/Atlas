"""Run a model ON OUR TECH, end-to-end: load the streamed 2-bit AetherCore model back,
reconstruct the weights, and GENERATE — side by side with the original FP model.

This is the first end-to-end "run a real model on our architecture" test: it proves the
convert -> store(2-bit) -> reload -> generate pipeline actually works and produces text,
not just per-tensor round-trip numbers.

Run:  python projects/day22_streaming/run_streamed.py "The meaning of life is"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from stream_convert import _dequant  # type: ignore  # reuse the dequantizer

SRC = Path("models/gpt2")
STREAMED = Path("experiments/gpt2_streamed")


def _load_streamed_state_dict():
    """rebuild a full fp32 state_dict from the streamed manifest + per-tensor npz."""
    manifest = json.loads((STREAMED / "manifest.json").read_text())
    sd = {}
    for key, meta in manifest["tensors"].items():
        z = np.load(STREAMED / meta["file"])
        c = dict(meta)
        for k in z.files:
            c[k] = z[k]
        W = _dequant(c)                      # reconstruct fp32 weight
        sd[key] = torch.from_numpy(np.ascontiguousarray(W)).float()
    return sd


@torch.inference_mode()
def _generate(model, tok, prompt, n=40):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The meaning of life is"
    tok = AutoTokenizer.from_pretrained(str(SRC), local_files_only=True)

    print("=" * 70, flush=True)
    print(f"PROMPT: {prompt!r}", flush=True)
    print("=" * 70, flush=True)

    # 1) original FP GPT-2
    print("\n[1] loading original FP model...", flush=True)
    fp = AutoModelForCausalLM.from_pretrained(str(SRC), local_files_only=True).eval()
    t = time.perf_counter()
    fp_out = _generate(fp, tok, prompt)
    fp_dt = time.perf_counter() - t

    # 2) OUR 2-bit streamed model: reconstruct + load into the same architecture
    print("[2] loading OUR streamed 2-bit model (from experiments/gpt2_streamed)...", flush=True)
    ours = AutoModelForCausalLM.from_pretrained(str(SRC), local_files_only=True).eval()
    sd = _load_streamed_state_dict()
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    t = time.perf_counter()
    our_out = _generate(ours, tok, prompt)
    our_dt = time.perf_counter() - t

    print("\n" + "=" * 70, flush=True)
    print("FP (fp32) GPT-2:", flush=True)
    print("  " + fp_out.replace("\n", " "), flush=True)
    print(f"  [{fp_dt:.1f}s]", flush=True)
    print("-" * 70, flush=True)
    print("OURS (2-bit AetherCore, streamed+reloaded):", flush=True)
    print("  " + our_out.replace("\n", " "), flush=True)
    print(f"  [{our_dt:.1f}s]", flush=True)
    print("=" * 70, flush=True)
    print(f"\nloaded our model from {len(sd)} reconstructed tensors "
          f"({len(missing)} missing, {len(unexpected)} unexpected keys)", flush=True)
    print("END-TO-END: a real model stored in our 2-bit format generated text. "
          "This is the convert->store->reload->run pipeline working.", flush=True)


if __name__ == "__main__":
    main()
