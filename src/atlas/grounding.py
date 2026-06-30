"""Grounding tier (T8) — answer from retrieved evidence or honestly refuse.

The single most important reliability rule in ATLAS: never confidently state a
fact the system cannot ground. ``Grounder.lookup`` retrieves the best passage and
returns it only when the match clears a confidence threshold; otherwise it returns
``grounded=False`` so the engine can fall back to an honest "I don't know" (for
factual questions) or to the language model (for open/creative prompts).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .retrieval import Retriever


@dataclass(frozen=True, slots=True)
class GroundResult:
    """Outcome of a grounding lookup."""

    grounded: bool
    answer: str | None
    score: float
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


class Grounder:
    """Decide whether a query can be answered from the knowledge corpus."""

    def __init__(self, retriever: Retriever | None = None, threshold: float = 0.6) -> None:
        """Create a grounder.

        Args:
            retriever: A built ``Retriever``; defaults to the bundled corpus.
            threshold: Minimum coverage score in [0, 1] for a confident answer.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.retriever = retriever or Retriever.from_corpus()
        self.threshold = float(threshold)

    def lookup(self, query: str) -> GroundResult:
        """Return a grounded answer when a passage clears the threshold."""

        hit = self.retriever.top(query)
        if hit is not None and hit.score >= self.threshold:
            return GroundResult(True, hit.text, hit.score, "knowledge-base")
        score = hit.score if hit is not None else 0.0
        return GroundResult(False, None, score, None)


def _self_test() -> None:
    """Check grounded hits and honest misses on the bundled corpus."""

    g = Grounder(threshold=0.6)

    known = g.lookup("What is the capital of Japan?")
    if not known.grounded or "Tokyo" not in (known.answer or ""):
        raise RuntimeError(f"known fact not grounded: {known.to_dict()}")

    unknown = g.lookup("What is the population of Mars in 2090?")
    if unknown.grounded:
        raise RuntimeError(f"unknown should not be grounded: {unknown.to_dict()}")

    creative = g.lookup("Tell me a story about a dragon")
    if creative.grounded:
        raise RuntimeError(f"creative prompt should not be grounded: {creative.to_dict()}")

    print("ATLAS grounding self-test")
    print(f"  capital of Japan -> grounded={known.grounded} ({known.score:.2f}) {known.answer}")
    print(f"  Mars 2090        -> grounded={unknown.grounded} ({unknown.score:.2f}) -> honest IDK")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
