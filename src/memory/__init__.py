"""Memory management components for AetherCore v3."""

__all__ = [
    "AsyncSSDLoader",
    "ContextTier",
    "ExpertMemoryManager",
    "Fact",
    "HierarchicalKVCache",
    "HotColdBalancer",
    "ImportanceScorer",
    "KVCacheStats",
    "MemoryConsolidator",
    "MemoryStats",
    "MemoryTier",
]


def __getattr__(name: str) -> object:
    """Lazily expose memory classes."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in {"AsyncSSDLoader", "ExpertMemoryManager", "HotColdBalancer", "MemoryStats", "MemoryTier"}:
        from . import god_manager

        return getattr(god_manager, name)

    from . import infinite_context

    return getattr(infinite_context, name)
