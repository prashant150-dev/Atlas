#!/usr/bin/env python3
"""ATLAS — entry point. Thin wrapper over the engine package ``src.atlas.cli``.

    python atlas.py ask "What is 47389 * 8291?"
    python atlas.py chat --model qwen
    python atlas.py convert models/qwen2.5-1.5b
    python atlas.py "Explain why the sky is blue"     # shorthand for `ask`

The real engine lives in ``src/atlas/`` (config, models, tools, engine, cli) so
the same code backs the CLI and the future UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.atlas import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli.main())
