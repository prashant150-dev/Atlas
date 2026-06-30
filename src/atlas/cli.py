"""Command-line interface for the ATLAS engine.

    python atlas.py ask "What is 47389 * 8291?"        # one question
    python atlas.py chat --model qwen                   # interactive
    python atlas.py convert models/qwen2.5-1.5b         # fp16 model -> ATLAS low-bit
    python atlas.py "Explain why the sky is blue"       # shorthand for `ask`

The CLI is a thin shell over ``AtlasEngine`` so the engine stays UI-agnostic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONVERT = ROOT / "projects" / "day23_realrun" / "convert_any.py"

BANNER = "=" * 60


def _engine(model_key: str):
    """Build an engine for a known model key (lazy heavy imports)."""

    from .config import AtlasConfig
    from .engine import AtlasEngine

    return AtlasEngine(config=AtlasConfig(model_key=model_key))


def _ask(engine, prompt: str) -> None:
    """Run one prompt and print a compact, labelled answer."""

    t = time.perf_counter()
    res = engine.answer(prompt)
    dt = time.perf_counter() - t
    print(f"\n[{res.route} | {res.tool_used} | {dt:.1f}s | conf {res.confidence:.2f}]")
    print(f"   {res.answer.strip()}\n")


def cmd_ask(args) -> int:
    """Answer a single prompt."""

    prompt = " ".join(args.prompt) if isinstance(args.prompt, list) else args.prompt
    print(BANNER)
    print(f"  ATLAS  ·  model: {args.model}  ·  verifiable->tools, open->model")
    print(BANNER)
    _ask(_engine(args.model), prompt)
    return 0


def cmd_chat(args) -> int:
    """Interactive chat loop."""

    print(BANNER)
    print(f"  ATLAS chat  ·  model: {args.model}  ·  type 'exit' to quit")
    print(BANNER)
    engine = _engine(args.model)
    while True:
        try:
            q = input("\natlas> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if q.lower() in ("exit", "quit", "q"):
            print("bye!")
            break
        if q:
            _ask(engine, q)
    return 0


def cmd_convert(args) -> int:
    """Convert an fp16/fp32 HF model to ATLAS low-bit format (streaming, CPU)."""

    src = Path(args.model)
    if not src.exists():
        print(f"  error: model dir not found: {src}")
        return 1
    out = Path(args.out) if args.out else ROOT / "experiments" / (src.name + "_atlas")
    print(BANNER)
    print(f"  ATLAS convert  ·  {src}  ->  {out}  ·  streaming low-bit (bounded RAM, CPU)")
    print(BANNER)
    t = time.perf_counter()
    rc = subprocess.call([sys.executable, str(CONVERT), str(src), str(out)])
    if rc == 0:
        print(f"\n  done in {time.perf_counter() - t:.0f}s  ->  {out}")
    return rc


_SUBCOMMANDS = ("ask", "chat", "convert")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand.

    Shorthand: ``atlas.py "any prompt"`` (no subcommand) is treated as ``ask``.
    """

    raw = list(sys.argv[1:] if argv is None else argv)
    # Shorthand BEFORE argparse: a bare prompt that is not a subcommand or help flag.
    if raw and raw[0] not in _SUBCOMMANDS and raw[0] not in ("-h", "--help"):
        return cmd_ask(argparse.Namespace(prompt=raw, model="qwen"))

    ap = argparse.ArgumentParser(prog="atlas", description="ATLAS unified engine CLI")
    sub = ap.add_subparsers(dest="cmd")

    pa = sub.add_parser("ask", help="ask one question")
    pa.add_argument("prompt", nargs="+")
    pa.add_argument("--model", choices=["gpt2", "qwen"], default="qwen")

    ch = sub.add_parser("chat", help="interactive chat")
    ch.add_argument("--model", choices=["gpt2", "qwen"], default="qwen")

    pc = sub.add_parser("convert", help="convert an fp16 model -> ATLAS low-bit format")
    pc.add_argument("model")
    pc.add_argument("--out", default=None)

    args = ap.parse_args(raw)
    if args.cmd is None:
        ap.print_help()
        return 0
    return {"ask": cmd_ask, "chat": cmd_chat, "convert": cmd_convert}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
