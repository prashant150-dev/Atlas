"""Math engine components for AetherCore v3."""

__all__ = [
    "BeastMathEngine",
    "MathExpression",
    "MathExplainer",
    "MathParser",
    "MathResult",
    "MathVerifier",
    "SymbolicSolver",
]


def __getattr__(name: str) -> object:
    """Lazily expose math engine classes."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import symbolic

    return getattr(symbolic, name)
