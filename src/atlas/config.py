"""Configuration for the unified ATLAS engine.

A single ``AtlasConfig`` dataclass holds every knob the engine and its tiers read:
which base model to load, generation limits, and per-tier feature toggles. The
toggles let later stages (retrieval, reasoning, safety, memory) be turned on or
off without touching call sites — the engine reads the config, not globals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Known local models. ``chat`` selects whether to wrap prompts in a chat template.
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "qwen": {"path": "models/qwen2.5-1.5b", "chat": True},
    "gpt2": {"path": "models/gpt2", "chat": False},
}


@dataclass(frozen=True, slots=True)
class AtlasConfig:
    """Immutable engine configuration.

    Attributes:
        model_key: Short name of a known model (``"qwen"`` or ``"gpt2"``) or
            empty when ``model_path`` is given directly.
        model_path: Local directory of the base model (overrides ``model_key``).
        chat: Whether the base model uses a chat template.
        max_new_tokens: Generation cap for the base model path.
        retrieval_threshold: Minimum match score for a confident fact retrieval.
        use_retrieval: Enable the grounded retrieval tier (T4/T8).
        use_reasoning: Enable test-time compute on hard prompts (T6/T7).
        use_safety: Enable the input/output safety filter (T17).
        use_memory: Enable per-user memory (T14/T15).
        use_speculative: Enable speculative decoding (T10).
    """

    model_key: str = "qwen"
    model_path: str = ""
    chat: bool = True
    max_new_tokens: int = 80
    retrieval_threshold: float = 0.6
    reasoning_samples: int = 3
    use_retrieval: bool = True
    use_reasoning: bool = True
    use_safety: bool = True
    use_memory: bool = False
    use_speculative: bool = False

    def __post_init__(self) -> None:
        """Resolve ``model_key`` into a concrete path and chat flag."""

        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(self.reasoning_samples, int) or self.reasoning_samples <= 0:
            raise ValueError("reasoning_samples must be a positive integer")
        if not 0.0 <= self.retrieval_threshold <= 1.0:
            raise ValueError("retrieval_threshold must be in [0, 1]")
        if not self.model_path:
            spec = KNOWN_MODELS.get(self.model_key)
            if spec is None:
                raise ValueError(
                    f"unknown model_key {self.model_key!r}; give model_path or one of {sorted(KNOWN_MODELS)}"
                )
            # frozen dataclass: set resolved fields via object.__setattr__
            object.__setattr__(self, "model_path", spec["path"])
            object.__setattr__(self, "chat", bool(spec["chat"]))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


def _self_test() -> None:
    """Validate config resolution and guards."""

    cfg = AtlasConfig(model_key="qwen")
    if cfg.model_path != "models/qwen2.5-1.5b" or cfg.chat is not True:
        raise RuntimeError(f"qwen config wrong: {cfg.to_dict()}")
    gpt2 = AtlasConfig(model_key="gpt2")
    if gpt2.chat is not False:
        raise RuntimeError("gpt2 should not be chat")
    direct = AtlasConfig(model_key="", model_path="models/foo", chat=True)
    if direct.model_path != "models/foo":
        raise RuntimeError("direct path not honored")
    for bad in (lambda: AtlasConfig(max_new_tokens=0), lambda: AtlasConfig(retrieval_threshold=2.0),
                lambda: AtlasConfig(model_key="nope")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise RuntimeError("expected ValueError for invalid config")
    print("AtlasConfig self-test")
    print(f"  qwen: {cfg.to_dict()}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
