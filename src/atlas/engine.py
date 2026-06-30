"""The unified ATLAS engine: one orchestrator over every tier.

``AtlasEngine.answer(prompt)`` runs the smart layer that wraps any base model:

    prompt -> ROUTE -> dispatch:
        math  -> exact calculator / symbolic engine   (T9, 100% on verifiable)
        code  -> generate + sandbox-run + self-heal    (T9)
        fact  -> retrieve from knowledge base           (T4/T8, grounded)
        else  -> base model (general / creative text)   (the quality path)
      -> VERIFY -> AtlasResult (with honest "I don't know" on unknowns)

The engine wires routing + exact tools + grounded retrieval + the real base model.
Both ``fact`` and factual ``general`` prompts are grounded against a real retrieval
index (T4/T8) before falling back to the model, so unknowns get an honest "I don't
know" instead of a hallucination. Test-time reasoning (A3) and safety + memory (A4)
plug into the same ``answer`` interface without changing call sites.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .config import AtlasConfig
from .models.base import BaseModel, EchoModel
from . import tools

_CODE_WORDS = ("function", "code", "python", "def ", "write a program", "algorithm", "script")
_FACT_WORDS = ("capital", "who is", "who wrote", "speed of", "symbol for",
               "author of", "planet", "boiling", "melting", "currency of", "largest", "tallest")


@dataclass(frozen=True, slots=True)
class AtlasResult:
    """Structured engine answer."""

    route: str
    answer: str
    verified: bool
    tool_used: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


def route(prompt: str) -> str:
    """Classify a prompt into one of: ``math``, ``code``, ``fact``, ``general``."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    p = prompt.lower()
    s = prompt.replace("x", "*").replace("X", "*")
    # math: a real operator (incl. ** and //) between digits, 'calculate', or algebra words
    if re.search(r"\d\s*[+\-*/%]{1,2}\s*\d", s) or "calculate" in p or tools.looks_algebraic(prompt):
        return "math"
    if any(w in p for w in _CODE_WORDS):
        return "code"
    if any(w in p for w in _FACT_WORDS):
        return "fact"
    return "general"


class AtlasEngine:
    """The complete ATLAS smart layer wrapping a pluggable base model."""

    def __init__(self, model: BaseModel | None = None, config: AtlasConfig | None = None) -> None:
        """Create the engine.

        Args:
            model: A ``BaseModel`` for the general/creative path. Defaults to a
                lazily-loaded ``HFModel`` built from ``config``.
            config: Engine configuration; defaults to ``AtlasConfig()``.
        """

        self.config = config or AtlasConfig()
        self.model = model if model is not None else self._default_model()
        self._grounder = None  # built lazily on first factual lookup

    def _ground(self):
        """Lazily build the grounding tier (real retriever over the corpus)."""

        if self._grounder is None:
            from .grounding import Grounder

            self._grounder = Grounder(threshold=self.config.retrieval_threshold)
        return self._grounder

    def _default_model(self) -> BaseModel:
        """Build the configured real model (lazy — weights load on first use)."""

        from .models.hf_model import HFModel

        return HFModel(self.config.model_path, chat=self.config.chat,
                       max_new_tokens=self.config.max_new_tokens)

    # ---------------- dispatch ----------------
    def answer(self, prompt: str) -> AtlasResult:
        """Route ``prompt``, run the right tier, and return a verified result."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        r = route(prompt)
        if r == "math":
            return self._do_math(prompt)
        if r == "code":
            return self._do_code(prompt)
        if r == "fact":
            return self._do_fact(prompt)
        if self.config.use_reasoning:
            from .reasoning import looks_like_reasoning

            if looks_like_reasoning(prompt):
                return self._do_reasoning(prompt)
        return self._do_general(prompt)

    def _do_math(self, prompt: str) -> AtlasResult:
        """Exact arithmetic first; fall back to the symbolic engine for algebra."""

        if not tools.looks_algebraic(prompt):
            expr = tools.extract_arithmetic(prompt)
            if expr:
                try:
                    val = tools.arith_eval(expr)
                    return AtlasResult("math", tools.format_number(val), True, "calculator", 1.0)
                except ValueError:
                    pass
        try:
            res = tools.solve_math(prompt)
            return AtlasResult("math", str(res.answer), bool(res.verification),
                               "symbolic", float(res.confidence))
        except Exception:  # noqa: BLE001 — symbolic parse can fail on free text
            return self._do_general(prompt)

    def _do_code(self, prompt: str) -> AtlasResult:
        """Generate + verify code; return the verified solution or honest fallback."""

        try:
            res = tools.solve_code(prompt)
        except Exception:  # noqa: BLE001
            return self._do_general(prompt)
        answer = res.code
        if res.verified:
            answer = res.code + "\n# verified: generated tests passed"
        return AtlasResult("code", answer, bool(res.verified), "code-run",
                           0.95 if res.verified else 0.4)

    def _do_fact(self, prompt: str) -> AtlasResult:
        """Retrieve a grounded fact, or answer an honest 'I don't know'."""

        if self.config.use_retrieval:
            hit = self._ground().lookup(prompt)
            if hit.grounded:
                return AtlasResult("fact", hit.answer or "", True, "retrieval", round(0.9 + 0.1 * hit.score, 3))
        return AtlasResult("fact", "I don't know based on my knowledge base.", True, "honest-IDK", 0.9)

    def _do_reasoning(self, prompt: str) -> AtlasResult:
        """Answer a quantitative word problem with self-consistency (T6)."""

        from . import reasoning

        res = reasoning.self_consistency(
            self.model, prompt, k=self.config.reasoning_samples,
            max_new_tokens=max(self.config.max_new_tokens, 160),
        )
        if res.numeric is None:  # no numeric consensus — fall back honestly
            return self._do_general(prompt)
        return AtlasResult("reasoning", res.answer, False, "self-consistency", res.confidence)

    def _do_general(self, prompt: str) -> AtlasResult:
        """Ground when possible (kills hallucination), else use the base model."""

        if self.config.use_retrieval:
            hit = self._ground().lookup(prompt)
            if hit.grounded:
                return AtlasResult("general", hit.answer or "", True, "retrieval", round(0.9 + 0.1 * hit.score, 3))
        text = self.model.generate(prompt, self.config.max_new_tokens)
        return AtlasResult("general", text, False, "base-model", 0.5)


def _self_test() -> None:
    """Exercise routing + every tool path with an offline EchoModel (no weights)."""

    engine = AtlasEngine(model=EchoModel(), config=AtlasConfig(model_key="gpt2"))

    cases = {
        "What is 47389 * 8291?": ("math", "392,902,199", True),
        "Calculate (12 + 8) * 3 - 10": ("math", "50", True),
        "2 ** 16": ("math", "65,536", True),
        "solve x^2 - 4 = 0": ("math", None, True),
        "Write a Python function for factorial": ("code", None, True),
        "What is the capital of Japan?": ("fact", "Tokyo", True),
        "What is the population of Mars in 2090?": ("general", None, False),
        "Tell me a story about a dragon": ("general", None, False),
    }
    for prompt, (exp_route, exp_answer, exp_verified) in cases.items():
        res = engine.answer(prompt)
        if res.route != exp_route:
            raise RuntimeError(f"{prompt!r}: route {res.route} != {exp_route}")
        if exp_answer is not None and exp_answer not in res.answer:
            raise RuntimeError(f"{prompt!r}: {exp_answer!r} not in {res.answer!r}")
        if res.verified != exp_verified:
            raise RuntimeError(f"{prompt!r}: verified {res.verified} != {exp_verified}")

    # algebra solve really used the symbolic engine and verified the roots
    algebra = engine.answer("solve x^2 - 4 = 0")
    if algebra.tool_used != "symbolic" or not algebra.verified:
        raise RuntimeError(f"algebra not solved symbolically: {algebra.to_dict()}")

    # reasoning: a word problem routes through self-consistency (offline numbered model)
    class _NumModel:
        def generate(self, prompt, max_new_tokens=None, sample=False, temperature=0.8, seed=None):
            return "3*12=36, minus 17 leaves 19. The answer is 19."

    rengine = AtlasEngine(model=_NumModel(), config=AtlasConfig(model_key="gpt2"))
    wp = rengine.answer("A shop has 3 boxes of 12 apples, sells 17. How many remain?")
    if wp.route != "reasoning" or wp.tool_used != "self-consistency" or "19" not in wp.answer:
        raise RuntimeError(f"reasoning route failed: {wp.to_dict()}")

    print("AtlasEngine self-test (offline EchoModel)")
    for prompt in cases:
        res = engine.answer(prompt)
        print(f"  [{res.route:7s}|{res.tool_used:10s}|{'OK' if res.verified else '..'}] "
              f"{prompt[:40]:40s} -> {res.answer.splitlines()[0][:46]}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
