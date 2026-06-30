"""Lookup-table router for AetherCore v3.

The router is intentionally deterministic and CPU-friendly. It uses compact
lookup tables and vectorized tensor reductions instead of neural classifiers,
then returns a small routing decision that downstream inference code can act on.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from operator import itemgetter
from typing import Any, Mapping, Sequence

import torch


class Domain(str, Enum):
    """Supported expert routing domains."""

    MATH = "MATH"
    CODE = "CODE"
    LANGUAGE = "LANGUAGE"
    LOGIC = "LOGIC"
    SCIENCE = "SCIENCE"
    CREATIVE = "CREATIVE"


class PrecisionLevel(str, Enum):
    """Precision policy selected for the next token."""

    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Final routing output consumed by the inference pipeline."""

    domain: Domain
    expert_ids: list[str]
    precision: PrecisionLevel
    confidence: float
    prefetch_list: list[str]
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this decision."""

        return {
            "domain": self.domain.value,
            "expert_ids": list(self.expert_ids),
            "precision": self.precision.value,
            "confidence": float(self.confidence),
            "prefetch_list": list(self.prefetch_list),
            "latency_ms": float(self.latency_ms),
        }


@dataclass(frozen=True, slots=True)
class RoutingRecord:
    """Compact record stored by the expert predictor."""

    domain: Domain
    token_hash: int
    expert_ids: tuple[str, ...]
    confidence: float


DomainLike = Domain | str
TokenInput = torch.Tensor | Sequence[int] | int | str


_DOMAINS: tuple[Domain, ...] = (
    Domain.MATH,
    Domain.CODE,
    Domain.LANGUAGE,
    Domain.LOGIC,
    Domain.SCIENCE,
    Domain.CREATIVE,
)


_DEFAULT_EXPERTS: dict[Domain, tuple[str, ...]] = {
    Domain.MATH: ("math.core", "math.algebra", "math.symbolic", "math.verify"),
    Domain.CODE: ("code.core", "code.python", "code.static", "code.repair"),
    Domain.LANGUAGE: ("language.core", "language.context", "language.syntax", "language.summary"),
    Domain.LOGIC: ("logic.core", "logic.proof", "logic.consistency", "logic.planner"),
    Domain.SCIENCE: ("science.core", "science.physics", "science.bio", "science.chem"),
    Domain.CREATIVE: ("creative.core", "creative.style", "creative.story", "creative.brainstorm"),
}


def _normalize_domain(domain: DomainLike) -> Domain:
    """Convert strings and enums to a Domain value."""

    if isinstance(domain, Domain):
        return domain
    if not isinstance(domain, str):
        raise TypeError(f"domain must be a Domain or string, got {type(domain)!r}")
    normalized = domain.strip().upper()
    try:
        return Domain(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in _DOMAINS)
        raise ValueError(f"Unsupported domain {domain!r}; allowed domains: {allowed}") from exc


def _domain_index(domain: DomainLike) -> int:
    """Return the stable index for a routing domain."""

    return int(_DOMAINS.index(_normalize_domain(domain)))


def _as_token_tensor(tokens: TokenInput, name: str = "tokens") -> torch.Tensor:
    """Convert accepted token inputs into a flat int64 CPU tensor."""

    if isinstance(tokens, torch.Tensor):
        tensor = tokens.detach().to(device="cpu").flatten()
        if tensor.numel() == 0:
            raise ValueError(f"{name} must contain at least one token")
        if torch.is_floating_point(tensor):
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} contains non-finite values")
            tensor = torch.round(tensor)
        return tensor.to(dtype=torch.long)

    if isinstance(tokens, str):
        encoded = tokens.encode("utf-8")
        if not encoded:
            raise ValueError(f"{name} must not be an empty string")
        return torch.tensor(tuple(encoded), dtype=torch.long)

    if isinstance(tokens, int):
        return torch.tensor((int(tokens),), dtype=torch.long)

    try:
        tensor = torch.as_tensor(tokens, dtype=torch.long).flatten()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a tensor, int, string, or sequence of ints") from exc
    if tensor.numel() == 0:
        raise ValueError(f"{name} must contain at least one token")
    return tensor.to(device="cpu", dtype=torch.long)


def _as_token_tuple(tokens: TokenInput, name: str = "tokens") -> tuple[int, ...]:
    """Convert accepted token inputs into a flat tuple of integers."""

    if isinstance(tokens, torch.Tensor):
        tensor = tokens.detach().to(device="cpu").flatten()
        if tensor.numel() == 0:
            raise ValueError(f"{name} must contain at least one token")
        if torch.is_floating_point(tensor):
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} contains non-finite values")
            tensor = torch.round(tensor)
        return tuple(int(value) for value in tensor.to(dtype=torch.long).tolist())

    if isinstance(tokens, str):
        encoded = tokens.encode("utf-8")
        if not encoded:
            raise ValueError(f"{name} must not be an empty string")
        return tuple(int(value) for value in encoded)

    if isinstance(tokens, int):
        return (int(tokens),)

    try:
        tensor = torch.as_tensor(tokens, dtype=torch.long).flatten()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a tensor, int, string, or sequence of ints") from exc
    if tensor.numel() == 0:
        raise ValueError(f"{name} must contain at least one token")
    return tuple(int(value) for value in tensor.to(device="cpu", dtype=torch.long).tolist())


def _confidence_from_counts(counts: torch.Tensor) -> float:
    """Return a stable confidence from domain vote counts."""

    total = float(counts.sum().item())
    if total <= 0.0:
        return 0.0
    top_two = torch.topk(counts.float(), k=min(2, counts.numel())).values
    top = float(top_two[0].item())
    second = float(top_two[1].item()) if top_two.numel() > 1 else 0.0
    dominance = (top - second) / total
    coverage = top / total
    return float(max(0.0, min(1.0, 0.55 * coverage + 0.45 * dominance)))


def _confidence_from_count_values(counts: Sequence[float]) -> float:
    """Return a stable confidence from Python count values."""

    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    ordered = sorted((float(value) for value in counts), reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    dominance = (top - second) / total
    coverage = top / total
    return float(max(0.0, min(1.0, 0.55 * coverage + 0.45 * dominance)))


def _unique_keep_order(values: Sequence[str], blocked: set[str] | None = None) -> list[str]:
    """Return unique values in insertion order."""

    seen = set(blocked or ())
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


class DomainFlashClassifier:
    """Classify domains with token and embedding lookup tables."""

    def __init__(
        self,
        vocab_size: int = 65_536,
        domains: Sequence[DomainLike] = _DOMAINS,
        token_domain_overrides: Mapping[int, DomainLike] | None = None,
        embedding_buckets: int = 4_096,
    ) -> None:
        """Create lookup tables for fast non-neural domain classification."""

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_buckets <= 0:
            raise ValueError("embedding_buckets must be positive")

        normalized_domains = tuple(_normalize_domain(domain) for domain in domains)
        if tuple(normalized_domains) != _DOMAINS:
            raise ValueError("domains must contain MATH, CODE, LANGUAGE, LOGIC, SCIENCE, CREATIVE in that order")

        self.vocab_size = int(vocab_size)
        self.domains = normalized_domains
        self._token_lookup = torch.full((self.vocab_size,), _domain_index(Domain.LANGUAGE), dtype=torch.long)
        self._token_lookup_py = [_domain_index(Domain.LANGUAGE)] * self.vocab_size
        self._embedding_lookup = torch.remainder(
            torch.arange(int(embedding_buckets), dtype=torch.long),
            len(self.domains),
        )
        self._keyword_domains = {
            Domain.MATH: ("solve", "integral", "derivative", "matrix", "equation", "algebra", "theorem"),
            Domain.CODE: ("def ", "class ", "return", "import ", "from ", "for ", "while ", "print(", "function"),
            Domain.LANGUAGE: ("summarize", "translate", "grammar", "rewrite", "sentence", "paragraph"),
            Domain.LOGIC: ("therefore", "prove", "because", "implies", "contradiction", "if and only if"),
            Domain.SCIENCE: ("physics", "biology", "chemistry", "experiment", "hypothesis", "molecule"),
            Domain.CREATIVE: ("story", "poem", "scene", "character", "dialogue", "brainstorm"),
        }
        self._install_default_token_overrides()
        if token_domain_overrides:
            self.update_token_domains(token_domain_overrides)

    def classify(self, input_embedding: TokenInput) -> Domain:
        """Return the most likely domain from tokens or an embedding-like vector."""

        domain_id, _confidence = self.classify_with_confidence(input_embedding)
        return self.domains[domain_id]

    def classify_with_confidence(self, input_embedding: TokenInput) -> tuple[int, float]:
        """Return ``(domain_id, confidence)`` using lookup-table votes."""

        if isinstance(input_embedding, str):
            return self._classify_text(input_embedding)

        if isinstance(input_embedding, torch.Tensor) and torch.is_floating_point(input_embedding):
            return self._classify_float_tensor(input_embedding)

        return self.classify_token_tuple(_as_token_tuple(input_embedding, name="input_embedding"))

    def classify_token_tuple(self, tokens: Sequence[int]) -> tuple[int, float]:
        """Classify integer tokens using the Python lookup table."""

        if not tokens:
            raise ValueError("tokens must contain at least one token")
        counts = [0.0] * len(self.domains)
        for token in tokens:
            counts[self._token_lookup_py[int(token) % self.vocab_size]] += 1.0
        domain_id = max(range(len(counts)), key=counts.__getitem__)
        return int(domain_id), _confidence_from_count_values(counts)

    def update_token_domains(self, overrides: Mapping[int, DomainLike]) -> None:
        """Update token-id to domain mappings."""

        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be a mapping from token id to domain")
        for token_id, domain in overrides.items():
            self._set_token_domain(int(token_id), domain)

    def _classify_float_tensor(self, input_embedding: torch.Tensor) -> tuple[int, float]:
        """Classify a floating point vector without learned weights."""

        vector = input_embedding.detach().to(device="cpu", dtype=torch.float32).flatten()
        if vector.numel() == 0:
            raise ValueError("input_embedding must contain at least one value")
        if not torch.isfinite(vector).all().item():
            raise ValueError("input_embedding contains non-finite values")

        if vector.numel() >= len(self.domains):
            scores = vector[: len(self.domains)].abs()
            counts = scores + 1.0e-6
            domain_id = int(torch.argmax(counts).item())
            return domain_id, _confidence_from_counts(counts)

        summary = torch.stack(
            (
                vector.mean().abs(),
                vector.std(unbiased=False),
                vector.abs().mean(),
            )
        )
        bucket = int(torch.remainder(torch.round(summary.mul(torch.tensor((997.0, 313.0, 73.0))).sum()).long(), self._embedding_lookup.numel()).item())
        domain_id = int(self._embedding_lookup[bucket].item())
        confidence = float(max(0.35, min(0.75, 0.35 + float(summary.mean().item() % 0.40))))
        return domain_id, confidence

    def _classify_text(self, text: str) -> tuple[int, float]:
        """Classify text with byte lookup plus keyword lookup tables."""

        tokens = _as_token_tuple(text, name="input_embedding")
        counts = [0.0] * len(self.domains)
        for token in tokens:
            counts[self._token_lookup_py[int(token) % self.vocab_size]] += 1.0

        lowered = text.lower()
        for domain, keywords in self._keyword_domains.items():
            score = sum(lowered.count(keyword) for keyword in keywords)
            if score:
                counts[_domain_index(domain)] += float(score * 16)

        domain_id = max(range(len(counts)), key=counts.__getitem__)
        return int(domain_id), _confidence_from_count_values(counts)

    def _install_default_token_overrides(self) -> None:
        """Install byte-level defaults for string inputs and byte tokenizers."""

        for char_code in range(ord("0"), ord("9") + 1):
            self._set_token_domain(char_code, Domain.MATH)

        for char in "+-*/=%^":
            self._set_token_domain(ord(char), Domain.MATH)

        for char in "{}[];:_#`\\\n\t":
            self._set_token_domain(ord(char), Domain.CODE)

        for char in "<>?&|!":
            self._set_token_domain(ord(char), Domain.LOGIC)

        for char in "~@$":
            self._set_token_domain(ord(char), Domain.SCIENCE)

        for char in "\"'":
            self._set_token_domain(ord(char), Domain.CREATIVE)

    def _set_token_domain(self, token_id: int, domain: DomainLike) -> None:
        """Set one token-domain mapping in both lookup tables."""

        index = int(token_id) % self.vocab_size
        domain_id = _domain_index(domain)
        self._token_lookup[index] = domain_id
        self._token_lookup_py[index] = domain_id


class ExpertPredictor:
    """Predict active and prefetched experts from the last eight tokens."""

    def __init__(
        self,
        experts_by_domain: Mapping[DomainLike, Sequence[str]] | None = None,
        top_k: int = 2,
        prefetch_count: int = 3,
        history_size: int = 128,
    ) -> None:
        """Create a deterministic expert predictor."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if prefetch_count < 0:
            raise ValueError("prefetch_count must be non-negative")
        if history_size <= 0:
            raise ValueError("history_size must be positive")

        source = experts_by_domain or _DEFAULT_EXPERTS
        self.experts_by_domain: dict[Domain, tuple[str, ...]] = {}
        for domain, experts in source.items():
            normalized = _normalize_domain(domain)
            expert_tuple = tuple(str(expert_id).strip() for expert_id in experts if str(expert_id).strip())
            if not expert_tuple:
                raise ValueError(f"Domain {normalized.value} must have at least one expert id")
            self.experts_by_domain[normalized] = expert_tuple

        missing = set(_DOMAINS).difference(self.experts_by_domain)
        if missing:
            missing_names = ", ".join(sorted(domain.value for domain in missing))
            raise ValueError(f"Missing expert lists for domains: {missing_names}")

        self.top_k = int(top_k)
        self.prefetch_count = int(prefetch_count)
        self.routing_history: deque[RoutingRecord] = deque(maxlen=int(history_size))
        self.last_prefetch_list: list[str] = []
        self.last_confidence: float = 0.5

    def predict(self, last_8_tokens: TokenInput, domain: DomainLike) -> list[str]:
        """Return expert ids and update history plus prefetch state."""

        normalized = _normalize_domain(domain)
        token_tail = _as_token_tuple(last_8_tokens, name="last_8_tokens")[-8:]
        token_hash = self._hash_tokens(token_tail)
        experts = self.experts_by_domain[normalized]
        expert_count = len(experts)

        active_count = min(self.top_k, expert_count)
        active_indexes = tuple((token_hash + offset) % expert_count for offset in range(active_count))
        expert_ids = self._select_experts(experts, active_indexes)

        prefetch_total = min(self.prefetch_count + active_count, expert_count)
        prefetch_indexes = tuple((token_hash + offset) % expert_count for offset in range(active_count, prefetch_total))
        self.last_prefetch_list = _unique_keep_order(self._select_experts(experts, prefetch_indexes), blocked=set(expert_ids))
        self.last_confidence = self._history_confidence(normalized)
        self.routing_history.append(
            RoutingRecord(
                domain=normalized,
                token_hash=int(token_hash),
                expert_ids=tuple(expert_ids),
                confidence=float(self.last_confidence),
            )
        )
        return expert_ids

    def expand_experts(self, domain: DomainLike, seed: int, min_count: int = 3) -> list[str]:
        """Return more experts for low-confidence routing."""

        normalized = _normalize_domain(domain)
        experts = self.experts_by_domain[normalized]
        count = min(max(int(min_count), self.top_k), len(experts))
        indexes = tuple((int(seed) + offset) % len(experts) for offset in range(count))
        return self._select_experts(experts, indexes)

    def _hash_tokens(self, tokens: Sequence[int]) -> int:
        """Hash the last eight tokens with a fixed integer lookup pattern."""

        if not tokens:
            return 0
        weights = (3, 5, 7, 11, 13, 17, 19, 23)[-len(tokens) :]
        hashed = sum((int(token) % 65_537) * weight for token, weight in zip(tokens, weights))
        return int(hashed % 2_147_483_647)

    def _select_experts(self, experts: tuple[str, ...], indexes: Sequence[int]) -> list[str]:
        """Select expert ids by integer indexes."""

        index_tuple = tuple(int(index) for index in indexes)
        if not index_tuple:
            return []
        if len(index_tuple) == 1:
            return [experts[index_tuple[0]]]
        selected = itemgetter(*index_tuple)(experts)
        if isinstance(selected, str):
            return [selected]
        return list(selected)

    def _history_confidence(self, domain: Domain) -> float:
        """Estimate routing stability from recent history."""

        if not self.routing_history:
            return 0.5
        sample_size = min(16, len(self.routing_history))
        recent = tuple(self.routing_history)[-sample_size:]
        matches = sum(1 for record in recent if record.domain == domain)
        return float(max(0.2, min(1.0, matches / sample_size)))


class PrecisionDecider:
    """Select precision level from token confidence."""

    def __init__(self, easy_threshold: float = 0.90, hard_threshold: float = 0.50) -> None:
        """Create precision thresholds."""

        if not 0.0 <= hard_threshold <= easy_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= hard_threshold <= easy_threshold <= 1")
        self.easy_threshold = float(easy_threshold)
        self.hard_threshold = float(hard_threshold)

    def decide(self, token: int | torch.Tensor, confidence: float) -> PrecisionLevel:
        """Return EASY, MEDIUM, or HARD for a token and confidence."""

        if isinstance(token, torch.Tensor):
            if token.numel() == 0:
                raise ValueError("token tensor must contain at least one value")
            token_value = int(token.flatten()[-1].item())
        else:
            token_value = int(token)

        confidence_value = float(confidence)
        if not math.isfinite(confidence_value):
            raise ValueError("confidence must be finite")
        confidence_value = max(0.0, min(1.0, confidence_value))

        if token_value < 0:
            return PrecisionLevel.HARD
        if confidence_value > self.easy_threshold:
            return PrecisionLevel.EASY
        if confidence_value < self.hard_threshold:
            return PrecisionLevel.HARD
        return PrecisionLevel.MEDIUM


class BeastRouter:
    """Orchestrate domain, expert, precision, and prefetch decisions."""

    def __init__(
        self,
        classifier: DomainFlashClassifier | None = None,
        predictor: ExpertPredictor | None = None,
        decider: PrecisionDecider | None = None,
        hard_expert_count: int = 3,
    ) -> None:
        """Create a complete three-level router."""

        if hard_expert_count <= 0:
            raise ValueError("hard_expert_count must be positive")
        self.classifier = classifier or DomainFlashClassifier()
        self.predictor = predictor or ExpertPredictor()
        self.decider = decider or PrecisionDecider()
        self.hard_expert_count = int(hard_expert_count)

    def route(self, input_tokens: TokenInput) -> RoutingDecision:
        """Route input tokens to domain, experts, precision, and prefetch list."""

        started = time.perf_counter_ns()
        tokens = _as_token_tuple(input_tokens, name="input_tokens")
        if isinstance(input_tokens, str):
            domain_id, domain_confidence = self.classifier.classify_with_confidence(input_tokens)
        else:
            domain_id, domain_confidence = self.classifier.classify_token_tuple(tokens)
        domain = self.classifier.domains[domain_id]
        tail = tokens[-8:]
        predicted = self.predictor.predict(tail, domain)
        confidence = self._blend_confidence(domain_confidence, self.predictor.last_confidence, tokens)
        precision = self.decider.decide(tokens[-1], confidence)

        token_seed = self.predictor._hash_tokens(tail)
        if precision == PrecisionLevel.EASY:
            expert_ids = predicted[:1]
        elif precision == PrecisionLevel.HARD:
            expert_ids = self.predictor.expand_experts(domain, seed=token_seed, min_count=self.hard_expert_count)
        else:
            expert_ids = predicted

        prefetch_list = _unique_keep_order(self.predictor.last_prefetch_list, blocked=set(expert_ids))
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return RoutingDecision(
            domain=domain,
            expert_ids=expert_ids,
            precision=precision,
            confidence=confidence,
            prefetch_list=prefetch_list,
            latency_ms=latency_ms,
        )

    def _blend_confidence(self, domain_confidence: float, predictor_confidence: float, tokens: Sequence[int]) -> float:
        """Blend classifier certainty, predictor stability, and token focus."""

        token_count = max(1, len(tokens))
        focus_bonus = min(0.15, 1.0 / token_count)
        confidence = 0.72 * float(domain_confidence) + 0.23 * float(predictor_confidence) + focus_bonus
        return float(max(0.0, min(1.0, confidence)))


def _self_test() -> None:
    """Run a small CPU sanity check for the beast router."""

    router = BeastRouter()
    math_decision = router.route("2+2=4")
    code_decision = router.route("def f(x):\n    return x+1")
    hard_decision = router.route(torch.tensor((17, 23, 42, -1), dtype=torch.long))

    classifier = DomainFlashClassifier()
    language_domain = classifier.classify(torch.tensor((0.1, 0.2, 3.0, 0.2, 0.1, 0.1), dtype=torch.float32))

    if math_decision.domain != Domain.MATH:
        raise RuntimeError(f"Expected math route, got {math_decision.domain}")
    if code_decision.domain != Domain.CODE:
        raise RuntimeError(f"Expected code route, got {code_decision.domain}")
    if hard_decision.precision != PrecisionLevel.HARD:
        raise RuntimeError(f"Expected hard precision, got {hard_decision.precision}")
    if language_domain != Domain.LANGUAGE:
        raise RuntimeError(f"Expected language classification, got {language_domain}")

    print("AetherCore beast router self-test")
    print(f"  math decision: {math_decision.to_dict()}")
    print(f"  code decision: {code_decision.to_dict()}")
    print(f"  hard decision: {hard_decision.to_dict()}")
    print(f"  history size: {len(router.predictor.routing_history)}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
