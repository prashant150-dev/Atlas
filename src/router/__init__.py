"""Routing components for AetherCore v3."""

__all__ = [
    "BeastRouter",
    "Domain",
    "DomainFlashClassifier",
    "ExpertPredictor",
    "PrecisionDecider",
    "PrecisionLevel",
    "RoutingDecision",
]


def __getattr__(name: str) -> object:
    """Lazily expose router classes."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import beast_router

    return getattr(beast_router, name)
