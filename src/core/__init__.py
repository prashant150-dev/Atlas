"""Core runtime components for AetherCore v3."""

__all__ = [
    "AetherCoreV3",
    "AetherConfig",
    "ExpertPool",
    "PipelineOrchestrator",
    "PipelineResult",
    "TernaryExpert",
    "TokenGenerator",
]


def __getattr__(name: str) -> object:
    """Lazily expose core classes."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in {"ExpertPool", "TernaryExpert"}:
        from . import expert

        return getattr(expert, name)

    from . import inference_engine

    return getattr(inference_engine, name)
