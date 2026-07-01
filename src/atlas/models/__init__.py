"""ATLAS base-model implementations (lazy imports)."""

from __future__ import annotations

from typing import Any

__all__ = ["BaseModel", "EchoModel", "HFModel"]


def __getattr__(name: str) -> Any:
    """Import model classes lazily to keep ``import`` cheap and torch-free."""

    if name in ("BaseModel", "EchoModel"):
        from . import base

        return getattr(base, name)
    if name == "HFModel":
        from . import hf_model

        return hf_model.HFModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
