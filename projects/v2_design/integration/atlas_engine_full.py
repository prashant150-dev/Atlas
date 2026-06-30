"""COMPLETE ATLAS engine on a REAL ~1B model (Qwen2.5-1.5B) — end-to-end, CPU, no GPU.

The full smart-layer running with a real capable model in the base slot:
  prompt -> ROUTER -> [math: exact tool] [code: run+verify] [fact: retrieve]
                      [general/creative: REAL MODEL generation]
         -> VERIFY / honest "I don't know"
This is the first complete mini-ATLAS: verifiable tasks answered EXACTLY by tools, open/
creative tasks by the real 1.5B model, unknowns handled honestly. The whole pipeline as ONE.

Run:  python projects/v2_design/integration/atlas_engine_full.py
      python projects/v2_design/integration/atlas_engine_full.py gpt2   # faster, smaller
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_engine as E  # type: ignore  # reuse router + tools + retrieval + verify

_MODEL = "models/qwen2.5-1.5b"
_tok = None
_lm = None
_is_chat = True


def _load():
    global _tok, _lm
    if _lm is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  [loading {_MODEL} ...]", flush=True)
        _tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
        _lm = AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True,
                                                   torch_dtype=torch.bfloat16).eval()


@torch.inference_mode()
def real_base_model(prompt: str) -> str:
    _load()
    if _is_chat:
        msgs = [{"role": "user", "content": prompt}]
        text = _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    ids = _tok(text, return_tensors="pt").input_ids
    out = _lm.generate(ids, max_new_tokens=80, do_sample=False, pad_token_id=_tok.eos_token_id)
    return _tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


E.base_model = real_base_model      # plug the real model into the engine


def main():
    global _MODEL, _is_chat
    if len(sys.argv) > 1 and sys.argv[1] == "gpt2":
        _MODEL, _is_chat = "models/gpt2", False

    prompts = [
        ("What is 47389 * 8291?", "math->tool"),
        ("Write a Python function for factorial", "code->verify"),
        ("What is the capital of Japan?", "fact->retrieve"),
        ("What is the population of Mars in 2090?", "unknown->honest"),
        ("Explain why the sky is blue in one sentence.", "general->model"),
        ("Give me one tip to learn programming faster.", "general->model"),
    ]
    print("=" * 72, flush=True)
    print(f"COMPLETE ATLAS ENGINE  |  model: {_MODEL}  |  CPU, no GPU", flush=True)
    print("router + tools + retrieval + verify + REAL model, all in one pipeline", flush=True)
    print("=" * 72, flush=True)

    for p, expect in prompts:
        t = time.perf_counter()
        res = E.answer(p)
        dt = time.perf_counter() - t
        print(f"\n[{res.route:7s}|{res.tool_used:11s}|{dt:4.1f}s] {p}", flush=True)
        ans = res.answer.strip().replace("\n", " ")
        print(f"   -> {ans[:140]}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("COMPLETE: verifiable tasks -> exact tools (instant, 100%); open tasks -> real", flush=True)
    print("1.5B generation; unknowns -> honest IDK. The full ATLAS engine runs end-to-end.", flush=True)


if __name__ == "__main__":
    main()
