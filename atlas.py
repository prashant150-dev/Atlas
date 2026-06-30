#!/usr/bin/env python3
"""ATLAS — AetherCore-V2 command-line tool.

The "select a model -> ATLAS -> run" flow, as a CLI (the backbone a UI would call).

  python atlas.py convert models/qwen2.5-1.5b      # fp16 model -> ATLAS format (CPU)
  python atlas.py ask "What is 47389 * 8291?"       # one question
  python atlas.py chat --model qwen                 # interactive
  python atlas.py "Explain why the sky is blue"     # shorthand for `ask`

Status (honest): the orchestration layer (router + exact tools + retrieval + verify) is
production-usable. Conversion does post-hoc low-bit (4-bit usable, no GPU); the 0.15-bit
native path + the LUT-kernel live inference are the remaining integration (need GPU / more
work). This CLI is a working prototype, not yet hardened for untrusted input.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENG = ROOT / "projects" / "v2_design" / "integration"
CONV = ROOT / "projects" / "day23_realrun" / "convert_any.py"
sys.path.insert(0, str(ENG))

BANNER = "=" * 60


def cmd_convert(args):
    src = Path(args.model)
    if not src.exists():
        print(f"  error: model dir not found: {src}")
        return 1
    out = Path(args.out) if args.out else ROOT / "experiments" / (src.name + "_atlas")
    print(BANNER)
    print(f"  ATLAS convert  ·  {src}  ->  {out}")
    print(f"  streaming low-bit (bounded RAM, CPU, no GPU)")
    print(BANNER)
    t = time.perf_counter()
    rc = subprocess.call([sys.executable, str(CONV), str(src), str(out)])
    if rc == 0:
        print(f"\n  done in {time.perf_counter()-t:.0f}s  ->  {out}")
        print(f"  run it:  python atlas.py ask \"hello\" --atlas {out}")
    return rc


def _engine(model_key: str):
    import atlas_engine as E
    import atlas_engine_full as F
    F._MODEL = "models/qwen2.5-1.5b" if model_key == "qwen" else "models/gpt2"
    F._is_chat = model_key == "qwen"
    E.base_model = F.real_base_model
    return E


def _ask(E, prompt: str):
    t = time.perf_counter()
    res = E.answer(prompt)
    dt = time.perf_counter() - t
    print(f"\n[{res.route} | {res.tool_used} | {dt:.1f}s]")
    print(f"   {res.answer.strip()}\n")


def cmd_ask(args):
    print(BANNER)
    print(f"  ATLAS  ·  base model: {args.model}  ·  verifiable->tools, open->model")
    print(BANNER)
    E = _engine(args.model)
    _ask(E, " ".join(args.prompt) if isinstance(args.prompt, list) else args.prompt)
    return 0


def cmd_chat(args):
    print(BANNER)
    print(f"  ATLAS chat  ·  base model: {args.model}  ·  type 'exit' to quit")
    print(BANNER)
    E = _engine(args.model)
    while True:
        try:
            q = input("\natlas> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!"); break
        if q.lower() in ("exit", "quit", "q"):
            print("bye!"); break
        if q:
            _ask(E, q)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="atlas", description="ATLAS (AetherCore-V2) CLI")
    sub = ap.add_subparsers(dest="cmd")

    pc = sub.add_parser("convert", help="convert an fp16 model -> ATLAS format")
    pc.add_argument("model"); pc.add_argument("--out", default=None)

    pa = sub.add_parser("ask", help="ask one question")
    pa.add_argument("prompt", nargs="+")
    pa.add_argument("--model", choices=["gpt2", "qwen"], default="gpt2")

    ch = sub.add_parser("chat", help="interactive chat")
    ch.add_argument("--model", choices=["gpt2", "qwen"], default="gpt2")

    # shorthand: `atlas.py "prompt"` -> ask
    args, extra = ap.parse_known_args()
    if args.cmd is None:
        if extra:
            ns = argparse.Namespace(prompt=extra, model="gpt2")
            return cmd_ask(ns)
        ap.print_help(); return 0
    return {"convert": cmd_convert, "ask": cmd_ask, "chat": cmd_chat}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
