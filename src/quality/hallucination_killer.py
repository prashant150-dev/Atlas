"""Local uncertainty and hallucination-control system for AetherCore v3.

The component is deliberately bounded: it verifies claims against a local JSONL
knowledge base and says "I don't know" when evidence is absent. That makes the
module useful in offline inference and honest about the knowledge boundary.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


_EPS = 1.0e-12
_FACT_VERBS = (
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "contains",
    "contain",
    "uses",
    "use",
    "equals",
    "equal",
)
_NEGATION_WORDS = {"not", "never", "no", "none", "false", "cannot", "can't", "isn't", "aren't", "wasn't", "weren't"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


class UncertaintyBand(str, Enum):
    """Human-readable uncertainty bands."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result returned by the local fact verifier."""

    verified: bool
    confidence: float
    source: str
    claim: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationCheck:
    """Safety decision for a generated token."""

    token: int
    safe_token: int
    confidence: float
    band: UncertaintyBand
    allowed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        payload = asdict(self)
        payload["band"] = self.band.value
        return payload


@dataclass(frozen=True, slots=True)
class LocalFact:
    """One local knowledge-base fact."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "local"
    created_at_ns: int = field(default_factory=time.perf_counter_ns)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_claim(self) -> str:
        """Render the fact as a simple natural-language claim."""

        verb = "is" if self.predicate.lower() in {"is", "states"} else self.predicate
        return f"{self.subject} {verb} {self.object}".strip()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalFact":
        """Create a fact from a dictionary payload."""

        if "claim" in payload and not {"subject", "predicate", "object"}.issubset(payload):
            fact = _parse_claim_to_fact(str(payload["claim"]))
            return cls(
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                confidence=float(payload.get("confidence", fact.confidence)),
                source=str(payload.get("source", fact.source)),
                created_at_ns=int(payload.get("created_at_ns", time.perf_counter_ns())),
                metadata=dict(payload.get("metadata", {})),
            )

        required = {"subject", "predicate", "object"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"LocalFact payload missing keys: {sorted(missing)}")
        return cls(
            subject=_clean_text(str(payload["subject"])),
            predicate=_clean_text(str(payload["predicate"])).lower() or "states",
            object=_clean_text(str(payload["object"])),
            confidence=_clamp01(float(payload.get("confidence", 1.0))),
            source=str(payload.get("source", "local")),
            created_at_ns=int(payload.get("created_at_ns", time.perf_counter_ns())),
            metadata=dict(payload.get("metadata", {})),
        )


def common_knowledge_facts() -> list[LocalFact]:
    """Return the small built-in common knowledge base used offline."""

    return [
        LocalFact("AetherCore", "is", "a local inference prototype", 0.95, "common"),
        LocalFact("AetherCore v3", "has", "compression, routing, memory, math, code, prompt, and quality components", 0.90, "common"),
        LocalFact("Artificial intelligence", "is", "the field of building computer systems that perform tasks associated with human intelligence", 0.92, "common"),
        LocalFact("Machine learning", "is", "a branch of artificial intelligence where systems learn patterns from data", 0.90, "common"),
        LocalFact("Deep learning", "is", "a machine learning approach that uses neural networks with multiple layers", 0.88, "common"),
        LocalFact("Neural networks", "are", "machine learning models inspired by connected layers of simple computational units", 0.88, "common"),
        LocalFact("Natural language processing", "is", "a field of artificial intelligence focused on understanding and generating human language", 0.88, "common"),
        LocalFact("Large language models", "are", "neural networks trained on large text datasets to predict and generate language", 0.86, "common"),
        LocalFact("Transformers", "are", "neural network architectures that use attention mechanisms to process sequences", 0.88, "common"),
        LocalFact("GPT-2", "is", "a transformer-based language model released by OpenAI in 2019", 0.86, "common"),
        LocalFact("Python", "is", "a general-purpose programming language known for readability and a large ecosystem", 0.90, "common"),
        LocalFact("JavaScript", "is", "a programming language commonly used to build interactive web applications", 0.88, "common"),
        LocalFact("HTTP", "is", "the application protocol used by web clients and servers to exchange resources", 0.88, "common"),
        LocalFact("APIs", "are", "interfaces that let software systems communicate through defined inputs and outputs", 0.88, "common"),
        LocalFact("Databases", "are", "organized systems for storing, querying, and managing data", 0.88, "common"),
        LocalFact("Encryption", "is", "the process of transforming information so only authorized parties can read it", 0.88, "common"),
        LocalFact("Photosynthesis", "is", "the process plants, algae, and some bacteria use to convert light energy into chemical energy", 0.90, "common"),
        LocalFact("Gravity", "is", "the physical attraction between masses", 0.88, "common"),
        LocalFact("The Earth", "is", "the third planet from the Sun", 0.92, "common"),
        LocalFact("The Moon", "is", "Earth's natural satellite", 0.92, "common"),
        LocalFact("Water", "is", "a chemical compound made of hydrogen and oxygen with the formula H2O", 0.92, "common"),
        LocalFact("A triangle", "has", "three sides", 0.94, "common"),
        LocalFact("A circle", "is", "a set of points in a plane at a fixed distance from a center point", 0.90, "common"),
        LocalFact("Democracy", "is", "a system of government where people participate directly or through elected representatives", 0.86, "common"),
    ]


def _clamp01(value: float) -> float:
    """Clamp a finite float into [0, 1]."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return float(max(0.0, min(1.0, number)))


def _clean_text(value: str) -> str:
    """Normalize whitespace and edge punctuation."""

    return re.sub(r"\s+", " ", value.strip().strip(" .!?;:,")).strip()


def _normalize_text(value: str) -> str:
    """Return lowercase alphanumeric text with normalized spaces."""

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _tokens(value: str) -> set[str]:
    """Tokenize text for fuzzy matching."""

    return {token for token in _normalize_text(value).split() if token and token not in _STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    """Return Jaccard similarity for token sets."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right)) / len(left.union(right)))


def _contains_negation(value: str) -> bool:
    """Return true when text contains negation."""

    return any(token in _NEGATION_WORDS for token in _normalize_text(value).split())


def _looks_like_claim(value: str) -> bool:
    """Return true for sentence-like factual claims."""

    normalized = f" {_normalize_text(value)} "
    if any(f" {verb} " in normalized for verb in _FACT_VERBS):
        return True
    return bool(re.search(r"\d", value))


def _split_sentences(text: str) -> list[str]:
    """Split text into simple sentence chunks."""

    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", cleaned) if chunk.strip()]
    return chunks or [cleaned]


def _parse_claim_to_fact(claim: str) -> LocalFact:
    """Parse a simple claim into a LocalFact-shaped record."""

    cleaned = _clean_text(claim)
    if not cleaned:
        raise ValueError("claim must not be empty")

    patterns = (
        (r"^(.{1,100}?)\s+(is|are|was|were|equals|equal)\s+(.{1,220})$", "is"),
        (r"^(.{1,100}?)\s+(has|have|contains|contain|uses|use)\s+(.{1,220})$", "has"),
        (r"^(fact|remember)\s*:\s*(.{1,260})$", "states"),
    )
    for pattern, predicate in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        if predicate == "states":
            return LocalFact(subject="memory", predicate=predicate, object=_clean_text(match.group(2)), confidence=0.8, source=claim)
        return LocalFact(
            subject=_clean_text(match.group(1)),
            predicate=predicate,
            object=_clean_text(match.group(3)),
            confidence=0.8,
            source=claim,
        )

    words = cleaned.split()
    subject = " ".join(words[: min(5, len(words))])
    obj = cleaned if len(words) <= 5 else " ".join(words[min(5, len(words)) :])
    return LocalFact(subject=subject, predicate="states", object=obj, confidence=0.4, source=claim)


class UncertaintyDetector:
    """Estimate model confidence from next-token logits."""

    def __init__(self, high_threshold: float = 0.90, low_threshold: float = 0.50) -> None:
        """Create a detector with HIGH/MEDIUM/LOW thresholds."""

        if not 0.0 <= low_threshold <= high_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low_threshold <= high_threshold <= 1")
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)

    def score(self, logits: torch.Tensor, top_k: int = 10) -> float:
        """Return confidence in [0, 1] for the latest token distribution."""

        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.numel() == 0:
            raise ValueError("logits must contain at least one value")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        row = logits.detach().float()
        if row.ndim == 0:
            row = row.reshape(1)
        if row.ndim > 1:
            row = row.reshape(-1, row.shape[-1])[-1]
        if not torch.isfinite(row).all().item():
            raise ValueError("logits contain non-finite values")

        shifted = row - row.max()
        probs = torch.softmax(shifted, dim=-1)
        k = min(int(top_k), int(probs.numel()))
        top_values = torch.topk(probs, k=k).values
        top1 = float(top_values[0].item())
        top2 = float(top_values[1].item()) if top_values.numel() > 1 else 0.0
        margin = max(0.0, top1 - top2)
        concentration = float(top_values.sum().item())
        entropy = float(-(probs * probs.clamp_min(_EPS).log()).sum().item())
        max_entropy = math.log(max(1, int(probs.numel())))
        entropy_confidence = 1.0 if max_entropy <= _EPS else 1.0 - min(1.0, entropy / max_entropy)

        confidence = 0.55 * top1 + 0.25 * margin + 0.15 * entropy_confidence + 0.05 * concentration
        return _clamp01(confidence)

    def band(self, confidence: float) -> UncertaintyBand:
        """Convert a confidence score to a band."""

        value = _clamp01(confidence)
        if value > self.high_threshold:
            return UncertaintyBand.HIGH
        if value < self.low_threshold:
            return UncertaintyBand.LOW
        return UncertaintyBand.MEDIUM


class FactVerifier:
    """Verify simple claims against a local JSONL knowledge base."""

    def __init__(
        self,
        knowledge_base_path: str | Path = "data/quality/knowledge_base.jsonl",
        facts: Sequence[LocalFact | Mapping[str, Any] | str] | None = None,
        similarity_threshold: float = 0.72,
        include_common_facts: bool = True,
    ) -> None:
        """Create a verifier backed by local facts."""

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self.knowledge_base_path = Path(knowledge_base_path)
        self.similarity_threshold = float(similarity_threshold)
        self._facts: list[LocalFact] = []
        self._signatures: set[tuple[str, str, str]] = set()
        self._load_knowledge_base()
        if include_common_facts:
            for fact in common_knowledge_facts():
                self.add_fact(fact, persist=False)
        for fact in facts or ():
            self.add_fact(fact, persist=False)

    def verify(self, claim: str) -> VerificationResult:
        """Verify a claim against the local knowledge base."""

        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        cleaned = _clean_text(claim)
        if not cleaned:
            raise ValueError("claim must not be empty")
        if not self._facts:
            return VerificationResult(False, 0.0, "local knowledge base is empty", cleaned, "no local facts available")

        claim_fact = _parse_claim_to_fact(cleaned)
        claim_negated = _contains_negation(cleaned)
        best_fact: LocalFact | None = None
        best_score = 0.0
        best_contradiction = False

        for fact in self._facts:
            score = self._match_score(claim_fact, fact, cleaned)
            contradiction = score >= self.similarity_threshold and claim_negated != _contains_negation(fact.to_claim())
            if score > best_score:
                best_score = score
                best_fact = fact
                best_contradiction = contradiction

        if best_fact is None:
            return VerificationResult(False, 0.0, "local knowledge base: no evidence", cleaned, "no comparable fact found")

        confidence = _clamp01(best_score * best_fact.confidence)
        if best_contradiction:
            return VerificationResult(False, confidence, best_fact.source, cleaned, f"contradicted by: {best_fact.to_claim()}")
        if best_score >= self.similarity_threshold:
            return VerificationResult(True, confidence, best_fact.source, cleaned, f"matched local fact: {best_fact.to_claim()}")
        return VerificationResult(False, confidence, best_fact.source, cleaned, f"weak match below threshold: {best_fact.to_claim()}")

    def add_fact(self, fact: LocalFact | Mapping[str, Any] | str, persist: bool = True) -> None:
        """Add a fact to the local knowledge base."""

        local_fact = self._coerce_fact(fact)
        signature = self._signature(local_fact)
        if signature in self._signatures:
            return
        self._facts.append(local_fact)
        self._signatures.add(signature)
        if persist:
            self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            with self.knowledge_base_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(local_fact.to_dict(), sort_keys=True) + "\n")

    def facts(self) -> list[LocalFact]:
        """Return known local facts."""

        return list(self._facts)

    def relevant_facts(self, topic: str, limit: int = 3, minimum_score: float = 0.12) -> list[LocalFact]:
        """Return locally known facts most relevant to a topic or question."""

        if not isinstance(topic, str):
            raise TypeError("topic must be a string")
        if limit <= 0:
            raise ValueError("limit must be positive")
        topic_tokens = _tokens(topic)
        if not topic_tokens:
            return []

        scored: list[tuple[float, LocalFact]] = []
        for fact in self._facts:
            fact_tokens = _tokens(f"{fact.subject} {fact.object}")
            score = _jaccard(topic_tokens, fact_tokens) * fact.confidence
            if score >= minimum_score:
                scored.append((score, fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in scored[:limit]]

    def topic_confidence(self, topic: str) -> float:
        """Return confidence that a topic exists in local knowledge."""

        if not isinstance(topic, str):
            raise TypeError("topic must be a string")
        topic_tokens = _tokens(topic)
        if not topic_tokens:
            return 0.0
        best = 0.0
        for fact in self._facts:
            fact_tokens = _tokens(f"{fact.subject} {fact.object}")
            similarity = _jaccard(topic_tokens, fact_tokens)
            best = max(best, similarity * fact.confidence)
        return _clamp01(best)

    def _match_score(self, claim_fact: LocalFact, fact: LocalFact, raw_claim: str) -> float:
        """Return fuzzy evidence score for claim versus local fact."""

        claim_tokens = _tokens(raw_claim)
        fact_tokens = _tokens(fact.to_claim())
        subject_score = _jaccard(_tokens(claim_fact.subject), _tokens(fact.subject))
        object_score = _jaccard(_tokens(claim_fact.object), _tokens(fact.object))
        claim_score = _jaccard(claim_tokens, fact_tokens)
        predicate_score = 1.0 if claim_fact.predicate == fact.predicate else 0.45

        exactish = 1.0 if _normalize_text(claim_fact.to_claim()) == _normalize_text(fact.to_claim()) else 0.0
        return _clamp01(0.42 * claim_score + 0.24 * subject_score + 0.24 * object_score + 0.06 * predicate_score + 0.04 * exactish)

    def _coerce_fact(self, fact: LocalFact | Mapping[str, Any] | str) -> LocalFact:
        """Normalize a supported fact input."""

        if isinstance(fact, LocalFact):
            return fact
        if isinstance(fact, str):
            return _parse_claim_to_fact(fact)
        if isinstance(fact, Mapping):
            return LocalFact.from_dict(fact)
        raise TypeError("fact must be a LocalFact, mapping, or string")

    def _signature(self, fact: LocalFact) -> tuple[str, str, str]:
        """Return a dedupe signature."""

        return _normalize_text(fact.subject), _normalize_text(fact.predicate), _normalize_text(fact.object)

    def _load_knowledge_base(self) -> None:
        """Load local facts from JSONL if present."""

        if not self.knowledge_base_path.exists():
            return
        with self.knowledge_base_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    self.add_fact(LocalFact.from_dict(json.loads(stripped)), persist=False)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue


class KnowledgeBoundaryEnforcer:
    """Decide whether a topic is inside local knowledge boundaries."""

    def __init__(
        self,
        verifier: FactVerifier,
        known_topics: Sequence[str] | None = None,
        confidence_threshold: float = 0.45,
    ) -> None:
        """Create a boundary enforcer."""

        if not isinstance(verifier, FactVerifier):
            raise TypeError("verifier must be a FactVerifier")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.verifier = verifier
        self.known_topics = {_normalize_text(topic) for topic in (known_topics or ()) if _normalize_text(topic)}
        self.confidence_threshold = float(confidence_threshold)

    def knows(self, topic: str) -> bool:
        """Return true when a topic is sufficiently covered locally."""

        return self.confidence(topic) >= self.confidence_threshold

    def confidence(self, topic: str) -> float:
        """Return local boundary confidence for a topic."""

        if not isinstance(topic, str):
            raise TypeError("topic must be a string")
        normalized = _normalize_text(topic)
        if not normalized:
            return 0.0
        if normalized in self.known_topics:
            return 1.0
        topic_tokens = _tokens(normalized)
        known_topic_score = 0.0
        for known in self.known_topics:
            known_topic_score = max(known_topic_score, _jaccard(topic_tokens, _tokens(known)))
        return _clamp01(max(known_topic_score, self.verifier.topic_confidence(topic)))

    def extract_topic(self, text: str) -> str:
        """Extract a conservative topic from a sentence."""

        fact = _parse_claim_to_fact(text)
        subject = fact.subject if fact.subject != "memory" else fact.object
        words = subject.split()
        return " ".join(words[: min(5, len(words))])


class HallucinationKiller:
    """Orchestrate uncertainty, local verification, and knowledge boundaries."""

    def __init__(
        self,
        uncertainty_detector: UncertaintyDetector | None = None,
        fact_verifier: FactVerifier | None = None,
        boundary_enforcer: KnowledgeBoundaryEnforcer | None = None,
        fallback_token: int = -1,
        fallback_response: str = "I don't know based on the local knowledge base.",
    ) -> None:
        """Create the hallucination-control orchestrator."""

        self.uncertainty_detector = uncertainty_detector or UncertaintyDetector()
        self.fact_verifier = fact_verifier or FactVerifier()
        self.boundary_enforcer = boundary_enforcer or KnowledgeBoundaryEnforcer(self.fact_verifier)
        self.fallback_token = int(fallback_token)
        self.fallback_response = _clean_text(fallback_response)
        self.last_generation_check: GenerationCheck | None = None

    def check_generation(self, token: int | torch.Tensor, logits: torch.Tensor) -> int | torch.Tensor:
        """Return a safe token based on model uncertainty."""

        token_value = self._token_to_int(token)
        confidence = self.uncertainty_detector.score(logits)
        band = self.uncertainty_detector.band(confidence)
        allowed = band != UncertaintyBand.LOW
        safe_token = token_value if allowed else self.fallback_token
        reason = "model confidence acceptable" if allowed else "low confidence; substituted fallback token"
        self.last_generation_check = GenerationCheck(
            token=token_value,
            safe_token=safe_token,
            confidence=confidence,
            band=band,
            allowed=allowed,
            reason=reason,
        )

        if isinstance(token, torch.Tensor):
            return torch.as_tensor(safe_token, dtype=token.dtype, device=token.device)
        return safe_token

    def verify_response(self, response: str) -> str:
        """Verify factual sentences and replace unsupported claims with uncertainty."""

        if not isinstance(response, str):
            raise TypeError("response must be a string")
        sentences = _split_sentences(response)
        if not sentences:
            return self.fallback_response

        verified_sentences: list[str] = []
        for sentence in sentences:
            stripped = _clean_text(sentence)
            if not stripped:
                continue
            if not _looks_like_claim(stripped):
                verified_sentences.append(stripped)
                continue

            topic = self.boundary_enforcer.extract_topic(stripped)
            if not self.boundary_enforcer.knows(topic):
                verified_sentences.append(self.fallback_response)
                continue

            verification = self.fact_verifier.verify(stripped)
            if verification.verified:
                verified_sentences.append(stripped)
            else:
                verified_sentences.append(self.fallback_response)

        return self._dedupe_fallbacks(verified_sentences) or self.fallback_response

    def _dedupe_fallbacks(self, sentences: Sequence[str]) -> str:
        """Collapse repeated fallback sentences."""

        output: list[str] = []
        previous = ""
        for sentence in sentences:
            if sentence == self.fallback_response and previous == self.fallback_response:
                continue
            output.append(sentence)
            previous = sentence
        return ". ".join(sentence.rstrip(".") for sentence in output).strip() + ("." if output else "")

    def _token_to_int(self, token: int | torch.Tensor) -> int:
        """Normalize token input to an integer."""

        if isinstance(token, torch.Tensor):
            if token.numel() == 0:
                raise ValueError("token tensor must contain at least one value")
            return int(token.detach().flatten()[-1].item())
        return int(token)


def _self_test() -> None:
    """Run a small CPU sanity check for hallucination control."""

    base_dir = Path.cwd() / "experiments" / "_quality_selftest"
    base_dir.mkdir(parents=True, exist_ok=True)
    kb_path = base_dir / "knowledge_base.jsonl"
    if kb_path.exists():
        kb_path.unlink()

    facts = [
        LocalFact("AetherCore", "is", "a local inference prototype", 0.95, "self-test"),
        LocalFact("Component six", "has", "a local fact verifier", 0.90, "self-test"),
        LocalFact("The hallucination killer", "uses", "local knowledge boundaries", 0.88, "self-test"),
    ]
    verifier = FactVerifier(kb_path, facts=facts)
    for fact in facts:
        verifier.add_fact(fact, persist=True)
    boundary = KnowledgeBoundaryEnforcer(verifier, known_topics=["AetherCore", "Component six"])
    detector = UncertaintyDetector()
    killer = HallucinationKiller(detector, verifier, boundary, fallback_token=0)

    high_logits = torch.tensor([10.0, -2.0, -3.0, -4.0])
    low_logits = torch.zeros(4)
    high_confidence = detector.score(high_logits)
    low_confidence = detector.score(low_logits)
    safe_high = killer.check_generation(1, high_logits)
    safe_low = killer.check_generation(1, low_logits)

    verified = verifier.verify("AetherCore is a local inference prototype.")
    unsupported = verifier.verify("AetherCore is not a local inference prototype.")
    response = killer.verify_response(
        "AetherCore is a local inference prototype. "
        "The moon is made of cheese. "
        "Component six has a local fact verifier."
    )

    if high_confidence <= 0.90:
        raise RuntimeError(f"Expected high confidence, got {high_confidence}")
    if low_confidence >= 0.50:
        raise RuntimeError(f"Expected low confidence, got {low_confidence}")
    if safe_high != 1 or safe_low != 0:
        raise RuntimeError("Token safety gate produced unexpected output")
    if not verified.verified:
        raise RuntimeError("Expected AetherCore claim to verify")
    if unsupported.verified:
        raise RuntimeError("Expected negated AetherCore claim to fail")
    if "moon is made of cheese" in response.lower():
        raise RuntimeError("Unsupported claim was not filtered")

    print("AetherCore hallucination killer self-test")
    print(f"  high confidence: {high_confidence:.4f}")
    print(f"  low confidence: {low_confidence:.4f}")
    print(f"  verified: {verified.to_dict()}")
    print(f"  unsupported: {unsupported.to_dict()}")
    print(f"  safe response: {response}")
    print(f"  last generation check: {killer.last_generation_check.to_dict() if killer.last_generation_check else None}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
