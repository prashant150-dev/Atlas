"""Base-model interface for the ATLAS engine.

The engine never talks to a specific model implementation directly; it depends on
the small ``BaseModel`` protocol below. Today the only implementation is
``HFModel`` (a real Hugging Face model in bf16). In Phase 2 a ``LowBitModel`` that
runs on packed 2-bit weights + the LUT kernel will implement the same interface,
so the engine can swap the efficiency core in with zero call-site changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseModel(Protocol):
    """Minimal text-generation contract the engine relies on.

    ``sample``/``temperature``/``seed`` enable self-consistency (T6): the reasoning
    tier draws several diverse samples and majority-votes the answer. Deterministic
    implementations may ignore them.
    """

    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 sample: bool = False, temperature: float = 0.8,
                 seed: int | None = None) -> str:
        """Return generated text for ``prompt`` (the prompt itself is not echoed)."""
        ...


class EchoModel:
    """Deterministic stand-in model for tests (no weights, no torch).

    Useful so the orchestration layer can be exercised without loading a real
    model. It never claims knowledge — it simply echoes a bounded marker, which
    keeps engine self-tests fast and offline.
    """

    def __init__(self, tag: str = "echo") -> None:
        """Create an echo model with a short identifying tag."""

        self.tag = str(tag)

    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 sample: bool = False, temperature: float = 0.8,
                 seed: int | None = None) -> str:
        """Return a deterministic bounded echo of the prompt (sampling ignored)."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        snippet = prompt.strip().replace("\n", " ")[:60]
        return f"[{self.tag}] {snippet}"


def _self_test() -> None:
    """Check the protocol and the echo stand-in."""

    model = EchoModel()
    if not isinstance(model, BaseModel):
        raise RuntimeError("EchoModel should satisfy the BaseModel protocol")
    out = model.generate("hello world")
    if "hello world" not in out:
        raise RuntimeError(f"unexpected echo: {out}")
    print("BaseModel self-test")
    print(f"  echo: {out}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
