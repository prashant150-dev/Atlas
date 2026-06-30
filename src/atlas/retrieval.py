"""Grounded retrieval (T4) — a real inverted-index retriever over a text corpus.

This replaces the engine's 8-entry placeholder dict with a genuine retriever:
an inverted index (term -> passages) for fast candidate generation, IDF-weighted
scoring so rare/specific words count more, and an interpretable coverage score in
[0, 1] that the grounding tier (T8) can threshold. The bundled corpus lives at
``data/atlas_kb.jsonl`` and is user-extensible; a small built-in is used as a
fallback if the file is missing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CORPUS = _ROOT / "data" / "atlas_kb.jsonl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "of", "is", "are", "a", "an", "what", "who", "for", "to", "in", "at",
    "on", "and", "or", "by", "with", "as", "was", "were", "be", "this", "that",
    "it", "its", "me", "tell", "which", "how", "many", "much", "do", "does", "can",
    "you", "your", "i", "my", "about", "give", "name", "list",
}

_FALLBACK_CORPUS: tuple[str, ...] = (
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "The speed of light in vacuum is 299,792,458 metres per second.",
    "The largest planet in the solar system is Jupiter.",
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens, and drop stopwords."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True, slots=True)
class Passage:
    """A retrieved corpus passage with its match score."""

    text: str
    score: float
    index: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


class Retriever:
    """IDF-weighted inverted-index retriever returning a [0, 1] coverage score."""

    def __init__(self, passages: list[str]) -> None:
        """Build the inverted index and IDF table from raw passage strings."""

        if not isinstance(passages, list) or not passages:
            raise ValueError("passages must be a non-empty list of strings")
        self.passages: list[str] = [str(p) for p in passages]
        self._tokens: list[set[str]] = [set(tokenize(p)) for p in self.passages]
        self._index: dict[str, list[int]] = {}
        for i, toks in enumerate(self._tokens):
            for tok in toks:
                self._index.setdefault(tok, []).append(i)
        n = len(self.passages)
        # smoothed idf: rare terms weigh more
        self._idf: dict[str, float] = {
            tok: math.log((n + 1) / (len(docs) + 1)) + 1.0 for tok, docs in self._index.items()
        }
        # a query term absent from the corpus is maximally specific (df -> 0): it must
        # weigh MORE than any seen term, so unmatched query words tank the coverage score
        # (precision guard: don't ground "population of Mars" on "the red planet is Mars").
        self._idf_unseen: float = math.log((n + 1) / 1) + 1.0

    @classmethod
    def from_corpus(cls, path: str | Path | None = None) -> "Retriever":
        """Load passages from a JSONL corpus (``{"text": ...}`` per line)."""

        corpus_path = Path(path) if path else _DEFAULT_CORPUS
        passages: list[str] = []
        if corpus_path.exists():
            for line in corpus_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    text = None
                if text:
                    passages.append(str(text))
        if not passages:
            passages = list(_FALLBACK_CORPUS)
        return cls(passages)

    def search(self, query: str, k: int = 3) -> list[Passage]:
        """Return up to ``k`` passages ranked by IDF-weighted query coverage."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        q_terms = [t for t in dict.fromkeys(tokenize(query))]  # de-dup, keep order
        if not q_terms:
            return []
        total = sum(self._idf.get(t, self._idf_unseen) for t in q_terms)
        if total <= 0:
            return []
        # candidate passages = any that share a query term (inverted index)
        candidates: set[int] = set()
        for t in q_terms:
            candidates.update(self._index.get(t, ()))
        scored: list[Passage] = []
        for i in candidates:
            matched = sum(self._idf.get(t, self._idf_unseen) for t in q_terms if t in self._tokens[i])
            scored.append(Passage(self.passages[i], matched / total, i))
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:k]

    def top(self, query: str) -> Passage | None:
        """Return the single best passage, or ``None`` if nothing matched."""

        hits = self.search(query, k=1)
        return hits[0] if hits else None


def _self_test() -> None:
    """Exercise tokenization, IDF ranking, and the bundled corpus."""

    if tokenize("What is the Capital of Japan?") != ["capital", "japan"]:
        raise RuntimeError(f"tokenize wrong: {tokenize('What is the Capital of Japan?')}")

    r = Retriever.from_corpus()
    if len(r.passages) < 20:
        raise RuntimeError(f"corpus too small ({len(r.passages)}); did data/atlas_kb.jsonl load?")

    japan = r.top("What is the capital of Japan?")
    if japan is None or "Tokyo" not in japan.text or japan.score < 0.99:
        raise RuntimeError(f"Japan retrieval failed: {japan}")
    light = r.top("speed of light")
    if light is None or "299,792,458" not in light.text:
        raise RuntimeError(f"speed-of-light retrieval failed: {light}")
    unknown = r.top("What is the GDP of Atlantis?")
    if unknown is not None and unknown.score >= 0.6:
        raise RuntimeError(f"unknown should score low, got {unknown}")

    print("ATLAS retrieval self-test")
    print(f"  corpus passages: {len(r.passages)}")
    print(f"  'capital of Japan' -> {japan.text} (score {japan.score:.2f})")
    print(f"  'GDP of Atlantis'  -> {'(no confident hit)' if unknown is None else f'{unknown.text} ({unknown.score:.2f})'}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
