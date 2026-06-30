"""Quality and hallucination-control components for AetherCore v3."""

__all__ = [
    "GenerationCheck",
    "HallucinationKiller",
    "KnowledgeBoundaryEnforcer",
    "LocalFact",
    "UncertaintyBand",
    "UncertaintyDetector",
    "VerificationResult",
    "FactVerifier",
    "common_knowledge_facts",
]


def __getattr__(name: str) -> object:
    """Lazily expose quality classes."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import hallucination_killer

    return getattr(hallucination_killer, name)
