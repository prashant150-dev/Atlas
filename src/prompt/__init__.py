"""Prompt refinement components for AetherCore v3."""

__all__ = [
    "AutoPromptRefiner",
    "Intent",
    "IntentDetector",
    "IntentType",
    "PromptEnhancer",
    "QueryDecomposer",
    "ResponseValidator",
    "SimpleQuery",
]


def __getattr__(name: str) -> object:
    """Lazily expose prompt refiner classes."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import refiner

    return getattr(refiner, name)
