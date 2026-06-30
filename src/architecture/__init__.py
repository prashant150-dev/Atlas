"""Native ternary + sparse-MoE architecture experiments for AetherCore v3.

This package holds the Day-3 *co-design* proof: instead of compressing a dense
floating-point model after the fact, we design a model that is natively ternary
and sparse-MoE from scratch (``AetherNet``), train it, and measure
capability-per-bit against a dense FP baseline (``DenseFP``) and a
post-hoc-ternary baseline (the dense FP weights ternarized with no retraining).
"""

__all__ = [
    "AetherNet",
    "AetherNetConfig",
    "DenseFP",
    "PostHocTernary",
    "BitAccount",
    "build_dense_fp",
    "build_aethernet",
    "ternarize_dense_to_posthoc",
]


def __getattr__(name: str) -> object:
    """Lazily expose architecture classes without eager submodule execution."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import aethernet

    return getattr(aethernet, name)
