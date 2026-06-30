"""Code generation and execution components for AetherCore v3."""

__all__ = [
    "BeastCodeEngine",
    "CodeGenerator",
    "CodeResult",
    "ExecutionResult",
    "Issue",
    "SandboxExecutor",
    "SelfHealingLoop",
    "StaticAnalyzer",
    "TestCase",
    "TestGenerator",
]


def __getattr__(name: str) -> object:
    """Lazily expose code engine classes."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import executor

    return getattr(executor, name)
