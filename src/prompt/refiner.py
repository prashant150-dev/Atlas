"""Automatic prompt refinement for AetherCore v3.

This module turns vague user input into explicit task instructions, decomposes
multi-part requests, and validates whether a response matches the detected
intent. It is deterministic and local, designed for use before routing or
generation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    """Supported high-level user intents."""

    QUESTION = "QUESTION"
    TASK = "TASK"
    CREATIVE = "CREATIVE"
    ANALYSIS = "ANALYSIS"
    DEBUG = "DEBUG"


@dataclass(frozen=True, slots=True)
class Intent:
    """Detected user intent."""

    type: IntentType
    domain: str
    complexity: str
    format_needed: str
    confidence: float
    signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        payload = asdict(self)
        payload["type"] = self.type.value
        return payload


@dataclass(frozen=True, slots=True)
class SimpleQuery:
    """One decomposed query part."""

    text: str
    index: int
    domain: str
    intent_type: IntentType

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        payload = asdict(self)
        payload["intent_type"] = self.intent_type.value
        return payload


def _clean_text(text: str) -> str:
    """Normalize user text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        raise ValueError("text must not be empty")
    return cleaned


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    """Return true when text contains any word/phrase."""

    lowered = text.lower()
    return any(word in lowered for word in words)


class IntentDetector:
    """Detect task type, domain, complexity, and expected format."""

    def detect(self, user_input: str) -> Intent:
        """Detect intent from user input."""

        text = _clean_text(user_input)
        lowered = text.lower()
        signals: list[str] = []

        domain = self._detect_domain(lowered, signals)
        intent_type = self._detect_type(lowered, signals)
        complexity = self._detect_complexity(text, lowered, signals)
        format_needed = self._detect_format(lowered, intent_type, signals)
        confidence = min(1.0, 0.45 + 0.10 * len(signals))

        return Intent(
            type=intent_type,
            domain=domain,
            complexity=complexity,
            format_needed=format_needed,
            confidence=confidence,
            signals=tuple(signals),
        )

    def _detect_domain(self, lowered: str, signals: list[str]) -> str:
        """Detect content domain."""

        domain_keywords = (
            ("MATH", ("solve", "equation", "integral", "derivative", "mean", "matrix", "probability")),
            ("CODE", ("code", "python", "javascript", "bug", "function", "class", "api", "compile", "error")),
            ("SCIENCE", ("physics", "biology", "chemistry", "experiment", "molecule", "energy")),
            ("LOGIC", ("prove", "logic", "therefore", "implies", "argument", "contradiction")),
            ("CREATIVE", ("story", "poem", "scene", "character", "brand", "creative")),
        )
        for domain, keywords in domain_keywords:
            if _contains_any(lowered, keywords):
                signals.append(f"domain:{domain}")
                return domain
        signals.append("domain:LANGUAGE")
        return "LANGUAGE"

    def _detect_type(self, lowered: str, signals: list[str]) -> IntentType:
        """Detect intent type."""

        if _contains_any(lowered, ("debug", "fix", "error", "traceback", "fails", "bug")):
            signals.append("type:DEBUG")
            return IntentType.DEBUG
        if _contains_any(lowered, ("compare", "analyze", "review", "explain why", "evaluate", "tradeoff")):
            signals.append("type:ANALYSIS")
            return IntentType.ANALYSIS
        if _contains_any(lowered, ("write a story", "poem", "brainstorm", "draft", "creative")):
            signals.append("type:CREATIVE")
            return IntentType.CREATIVE
        if lowered.endswith("?") or lowered.startswith(("what", "why", "how", "when", "where", "who")):
            signals.append("type:QUESTION")
            return IntentType.QUESTION
        signals.append("type:TASK")
        return IntentType.TASK

    def _detect_complexity(self, text: str, lowered: str, signals: list[str]) -> str:
        """Detect rough complexity."""

        separators = len(re.findall(r"\b(?:and|then|also|plus)\b|[;]", lowered))
        if len(text) > 240 or separators >= 3:
            signals.append("complexity:HIGH")
            return "HIGH"
        if len(text) > 90 or separators >= 1:
            signals.append("complexity:MEDIUM")
            return "MEDIUM"
        signals.append("complexity:LOW")
        return "LOW"

    def _detect_format(self, lowered: str, intent_type: IntentType, signals: list[str]) -> str:
        """Detect requested response format."""

        if _contains_any(lowered, ("json", "schema")):
            signals.append("format:JSON")
            return "JSON"
        if _contains_any(lowered, ("table", "columns")):
            signals.append("format:TABLE")
            return "TABLE"
        if _contains_any(lowered, ("bullet", "list", "steps")):
            signals.append("format:STEPS")
            return "STEPS"
        if intent_type == IntentType.DEBUG:
            signals.append("format:DIAGNOSIS")
            return "DIAGNOSIS"
        if intent_type == IntentType.CREATIVE:
            signals.append("format:PROSE")
            return "PROSE"
        signals.append("format:CONCISE")
        return "CONCISE"


class PromptEnhancer:
    """Enhance vague prompts into explicit instructions."""

    def enhance(self, original: str, intent: Intent) -> str:
        """Return an enhanced prompt."""

        text = _clean_text(original)
        if not isinstance(intent, Intent):
            raise TypeError("intent must be an Intent")

        lines = [
            f"User request: {text}",
            f"Intent: {intent.type.value}",
            f"Domain: {intent.domain}",
            f"Complexity: {intent.complexity}",
            f"Expected format: {intent.format_needed}",
            "Instructions:",
            "- Answer the request directly.",
            "- State assumptions only when needed.",
            "- Prefer verified local computation for math, code, and factual claims.",
        ]
        if intent.type == IntentType.DEBUG:
            lines.append("- Identify the likely root cause before proposing a fix.")
        elif intent.type == IntentType.ANALYSIS:
            lines.append("- Compare options using concrete criteria.")
        elif intent.type == IntentType.CREATIVE:
            lines.append("- Preserve the requested tone and audience.")
        elif intent.type == IntentType.QUESTION:
            lines.append("- Give the answer first, then the supporting detail.")
        else:
            lines.append("- Produce a complete, actionable result.")
        return "\n".join(lines)


class QueryDecomposer:
    """Split multi-part queries into simple units."""

    def decompose(self, complex_query: str) -> list[SimpleQuery]:
        """Return decomposed query parts."""

        text = _clean_text(complex_query)
        detector = IntentDetector()
        chunks = self._split_query(text)
        queries: list[SimpleQuery] = []
        for index, chunk in enumerate(chunks, start=1):
            intent = detector.detect(chunk)
            queries.append(SimpleQuery(text=chunk, index=index, domain=intent.domain, intent_type=intent.type))
        return queries

    def _split_query(self, text: str) -> list[str]:
        """Split text on strong separators."""

        raw_parts = re.split(r"\s*(?:;|\bthen\b|\balso\b|\band then\b)\s*", text, flags=re.IGNORECASE)
        parts = [part.strip(" ,") for part in raw_parts if part.strip(" ,")]
        if len(parts) <= 1 and "?" in text:
            parts = [part.strip() + "?" for part in text.split("?") if part.strip()]
        return parts or [text]


class ResponseValidator:
    """Validate response shape against the detected intent."""

    def validates(self, response: str, intent: Intent) -> bool:
        """Return true when response appears to satisfy intent."""

        if not isinstance(response, str):
            raise TypeError("response must be a string")
        if not isinstance(intent, Intent):
            raise TypeError("intent must be an Intent")
        cleaned = response.strip()
        if not cleaned:
            return False
        lowered = cleaned.lower()

        if intent.format_needed == "JSON":
            return cleaned.startswith("{") or cleaned.startswith("[")
        if intent.format_needed == "TABLE":
            return "|" in cleaned and "\n" in cleaned
        if intent.format_needed == "STEPS":
            return bool(re.search(r"(^|\n)\s*(?:\d+\.|-)\s+", cleaned))
        if intent.type == IntentType.DEBUG:
            return any(word in lowered for word in ("cause", "fix", "error", "issue"))
        if intent.type == IntentType.QUESTION:
            return len(cleaned.split()) >= 3
        if intent.type == IntentType.CREATIVE:
            return len(cleaned.split()) >= 10
        return len(cleaned) >= 2


class AutoPromptRefiner:
    """Orchestrate prompt detection, enhancement, decomposition, and validation."""

    def __init__(
        self,
        detector: IntentDetector | None = None,
        enhancer: PromptEnhancer | None = None,
        decomposer: QueryDecomposer | None = None,
        validator: ResponseValidator | None = None,
    ) -> None:
        """Create a prompt refiner."""

        self.detector = detector or IntentDetector()
        self.enhancer = enhancer or PromptEnhancer()
        self.decomposer = decomposer or QueryDecomposer()
        self.validator = validator or ResponseValidator()

    def refine(self, user_input: str) -> str:
        """Return an enhanced prompt string."""

        intent = self.detector.detect(user_input)
        queries = self.decomposer.decompose(user_input)
        enhanced = self.enhancer.enhance(user_input, intent)
        if len(queries) > 1:
            query_lines = ["Sub-queries:"]
            query_lines.extend(f"{query.index}. [{query.domain}/{query.intent_type.value}] {query.text}" for query in queries)
            return enhanced + "\n" + "\n".join(query_lines)
        return enhanced

    def validate_response(self, response: str, original: str) -> str:
        """Return response if valid, otherwise a concise regeneration request."""

        intent = self.detector.detect(original)
        if self.validator.validates(response, intent):
            return response
        return (
            "Response did not match the detected intent. "
            f"Regenerate as {intent.format_needed} for a {intent.type.value} request in {intent.domain}."
        )


def _self_test() -> None:
    """Run a small CPU sanity check for prompt refinement."""

    refiner = AutoPromptRefiner()
    original = "Debug this Python error and then explain the fix in steps"
    intent = refiner.detector.detect(original)
    refined = refiner.refine(original)
    queries = refiner.decomposer.decompose(original)
    valid = refiner.validate_response("1. Cause: bad input\n2. Fix: validate it", original)
    invalid = refiner.validate_response("ok", original)

    if intent.type != IntentType.DEBUG:
        raise RuntimeError(f"Unexpected intent: {intent}")
    if len(queries) < 2:
        raise RuntimeError("Expected decomposed query parts")
    if "Sub-queries:" not in refined:
        raise RuntimeError("Expected refined prompt to include sub-queries")
    if valid.startswith("Response did not match"):
        raise RuntimeError("Expected response to validate")
    if not invalid.startswith("Response did not match"):
        raise RuntimeError("Expected invalid response warning")

    print("AetherCore prompt refiner self-test")
    print(f"  intent: {intent.to_dict()}")
    print(f"  queries: {[query.to_dict() for query in queries]}")
    print(f"  refined first line: {refined.splitlines()[0]}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
