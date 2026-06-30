"""ATLAS — the unified efficiency-first inference engine (intelligence layer).

Public API (lazy-imported so ``import src.atlas`` stays cheap and torch-free):

    from src.atlas import AtlasEngine, AtlasConfig
    engine = AtlasEngine(config=AtlasConfig(model_key="qwen"))
    print(engine.answer("What is 47389 * 8291?").answer)
"""

from __future__ import annotations

from typing import Any

__all__ = ["AtlasEngine", "AtlasResult", "AtlasConfig", "route"]


def __getattr__(name: str) -> Any:
    """Import engine/config symbols on first access only."""

    if name in ("AtlasEngine", "AtlasResult", "route"):
        from . import engine

        return getattr(engine, name)
    if name == "AtlasConfig":
        from . import config

        return config.AtlasConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
