"""Compression primitives for AetherCore v3."""

__all__ = [
    "CompressedLayer",
    "CompressionStats",
    "CorrectionTableExtractor",
    "DeltaCompressor",
    "DynamicBitAllocator",
    "GodCompressionEngine",
    "PackedTernaryCompressor",
    "QualityMetrics",
    "SemanticDeduplicator",
    "TernaryQuantizer",
]


def __getattr__(name: str) -> object:
    """Lazily expose engine classes without eager submodule execution."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import engine

    return getattr(engine, name)
