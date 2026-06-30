"""Main local inference orchestrator for AetherCore v3.

This is a verified local scaffold rather than a pretend 400B model. It ties the
implemented components together, routes requests, performs exact math/code
workflows when appropriate, and provides a token-generation interface for later
model integration.
"""

from __future__ import annotations

import time
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.code_engine import BeastCodeEngine
from src.core.expert import TernaryExpert
from src.math_engine import BeastMathEngine
from src.memory import HierarchicalKVCache, ImportanceScorer
from src.prompt import AutoPromptRefiner
from src.quality import FactVerifier, HallucinationKiller, KnowledgeBoundaryEnforcer, LocalFact
from src.router import BeastRouter, Domain, RoutingDecision


@dataclass(frozen=True, slots=True)
class AetherConfig:
    """Configuration for AetherCoreV3."""

    context_dir: str = "data/context/runtime_kv"
    knowledge_base_path: str = "data/quality/runtime_knowledge.jsonl"
    max_context_tokens: int = 2_048
    fallback_response: str = "I don't know based on the local knowledge base."
    enable_quality_filter: bool = True
    seed: int = 37

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "AetherConfig":
        """Create a config from a mapping."""

        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping or None")
        fields = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Result from one pipeline orchestration call."""

    response: str
    routing: dict[str, Any]
    refined_prompt: str
    latency_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


def _text_to_tokens(text: str) -> list[int]:
    """Encode text into byte tokens."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    encoded = text.encode("utf-8")
    if not encoded:
        return [0]
    return [int(value) for value in encoded]


def _normalize_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Convert chat messages into a compact prompt."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of mappings")
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("each message must be a mapping")
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        raise ValueError("messages must contain at least one non-empty content")
    return "\n".join(lines)


class TokenGenerator:
    """Small deterministic next-token generator with confidence gating."""

    def __init__(self, vocab_size: int = 256, hallucination_killer: HallucinationKiller | None = None) -> None:
        """Create a token generator."""

        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        self.vocab_size = int(vocab_size)
        self.hallucination_killer = hallucination_killer or HallucinationKiller()

    def next_token(self, context: Sequence[int] | torch.Tensor, experts: Sequence[TernaryExpert] | None = None) -> int:
        """Return the next token id using optional sparse expert activation."""

        tokens = self._normalize_context(context)
        logits = torch.full((self.vocab_size,), -8.0, dtype=torch.float32)
        preferred = self._preferred_token(tokens)
        logits[preferred] = 8.0

        for expert in experts or ():
            if not isinstance(expert, TernaryExpert) or not expert.active:
                continue
            if expert.in_features > self.vocab_size:
                continue
            feature = torch.zeros(expert.in_features, dtype=torch.float32)
            for index, token in enumerate(tokens[-expert.in_features :]):
                feature[index] = (token % 256) / 255.0
            output = expert(feature, use_correction=True).flatten()
            limit = min(output.numel(), logits.numel())
            logits[:limit] = logits[:limit] + output[:limit].float()
            break

        candidate = int(torch.argmax(logits).item())
        safe = self.hallucination_killer.check_generation(candidate, logits)
        return int(safe)

    def _normalize_context(self, context: Sequence[int] | torch.Tensor) -> list[int]:
        """Normalize context tokens."""

        if isinstance(context, torch.Tensor):
            if context.numel() == 0:
                return [0]
            return [int(value) for value in context.detach().cpu().flatten().tolist()]
        if not isinstance(context, Sequence):
            raise TypeError("context must be a sequence or tensor")
        values = [int(value) for value in context]
        return values or [0]

    def _preferred_token(self, tokens: Sequence[int]) -> int:
        """Choose a deterministic local next token."""

        last = int(tokens[-1]) if tokens else 0
        if last in {ord("."), ord("!"), ord("?")}:
            return 0
        if 32 <= last <= 126:
            return ord(" ")
        return (last + 1) % self.vocab_size


class PipelineOrchestrator:
    """Run router, inference, loader placeholder, and memory tasks in parallel."""

    def __init__(self, workers: int = 4) -> None:
        """Create a pipeline orchestrator."""

        if workers < 4:
            raise ValueError("workers must be at least 4")
        self.workers = int(workers)

    def run(self, engine: "AetherCoreV3", prompt: str, max_tokens: int = 500) -> PipelineResult:
        """Run one generation through four logical pipeline workers."""

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            route_future = executor.submit(engine.router.route, prompt)
            refine_future = executor.submit(engine.prompt_refiner.refine, prompt)
            memory_future = executor.submit(engine._update_context_memory, prompt)
            loader_future = executor.submit(engine._prefetch_from_prompt, prompt)

            routing = route_future.result()
            refined = refine_future.result()
            memory_info = memory_future.result()
            loader_info = loader_future.result()
            response = engine._generate_with_routing(prompt, routing, refined, max_tokens=max_tokens)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return PipelineResult(
            response=response,
            routing=routing.to_dict(),
            refined_prompt=refined,
            latency_ms=latency_ms,
            diagnostics={"memory": memory_info, "prefetch": loader_info},
        )


class AetherCoreV3:
    """Main local AetherCore v3 inference class."""

    def __init__(self, model_path: str | Path | None = None, config: Mapping[str, Any] | AetherConfig | None = None) -> None:
        """Create the local inference engine."""

        self.model_path = None if model_path is None else Path(model_path)
        self.config = config if isinstance(config, AetherConfig) else AetherConfig.from_mapping(config)
        torch.manual_seed(int(self.config.seed))

        runtime_facts = [
            LocalFact("AetherCore", "is", "a local inference prototype", 0.95, "runtime"),
            LocalFact("AetherCore v3", "has", "compression, routing, memory, math, code, prompt, and quality components", 0.90, "runtime"),
        ]
        verifier = FactVerifier(self.config.knowledge_base_path, facts=runtime_facts)
        boundary = KnowledgeBoundaryEnforcer(
            verifier,
            known_topics=[fact.subject for fact in verifier.facts()],
        )
        self.hallucination_killer = HallucinationKiller(
            fact_verifier=verifier,
            boundary_enforcer=boundary,
            fallback_response=self.config.fallback_response,
        )
        self.router = BeastRouter()
        self.prompt_refiner = AutoPromptRefiner()
        self.math_engine = BeastMathEngine()
        self.code_engine = BeastCodeEngine()
        self.context_cache = HierarchicalKVCache(
            working_limit_tokens=min(128, self.config.max_context_tokens),
            short_term_limit_tokens=max(128, min(512, self.config.max_context_tokens)),
            long_term_limit_tokens=max(512, self.config.max_context_tokens),
            storage_dir=self.config.context_dir,
        )
        self.importance_scorer = ImportanceScorer()
        self.token_generator = TokenGenerator(hallucination_killer=self.hallucination_killer)
        self.pipeline = PipelineOrchestrator()
        self.last_pipeline_result: PipelineResult | None = None

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a response for a prompt."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        result = self.pipeline.run(self, prompt, max_tokens=max_tokens)
        self.last_pipeline_result = result
        return result.response

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate a response from chat-style messages."""

        prompt = _normalize_messages(messages)
        return self.generate(prompt)

    def _generate_with_routing(self, prompt: str, routing: RoutingDecision, refined: str, max_tokens: int) -> str:
        """Generate using the selected route."""

        if routing.domain == Domain.MATH or self._looks_math(prompt):
            result = self.math_engine.solve(prompt)
            return self._truncate(f"Answer: {result.answer}\nVerified: {result.verification}\nSteps:\n" + "\n".join(result.steps), max_tokens)

        if routing.domain == Domain.CODE or self._looks_code(prompt):
            code_result = self.code_engine.solve(prompt, language="python")
            status = "verified" if code_result.verified else "unverified"
            return self._truncate(
                f"Code result: {status}\n```python\n{code_result.code.rstrip()}\n```\nOutput:\n{code_result.output.output or code_result.output.errors}",
                max_tokens,
            )

        if "aethercore" in prompt.lower():
            response = "AetherCore v3 is a local inference prototype with compression, routing, memory, math, code, prompt, and quality components."
            return self._truncate(response, max_tokens)

        local_answer = self._answer_known_topic(prompt)
        if local_answer:
            return self._truncate(local_answer, max_tokens)

        response = self._language_fallback(prompt, routing)
        return self._truncate(response, max_tokens)

    def _update_context_memory(self, prompt: str) -> dict[str, Any]:
        """Store prompt byte tokens in the KV cache."""

        tokens = _text_to_tokens(prompt)
        for index, token in enumerate(tokens[-64:]):
            key = torch.tensor([token / 255.0, index / max(1, len(tokens))], dtype=torch.float32)
            value = torch.tensor([index / max(1, len(tokens)), token / 255.0], dtype=torch.float32)
            attention = torch.linspace(0.1, 1.0, steps=4)
            importance = self.importance_scorer.score(token, attention)
            self.context_cache.store(index, key, value, importance)
        return self.context_cache.stats().to_dict()

    def _prefetch_from_prompt(self, prompt: str) -> dict[str, Any]:
        """Produce router prefetch diagnostics."""

        decision = self.router.route(prompt)
        return {"prefetch_list": decision.prefetch_list, "expert_ids": decision.expert_ids}

    def _looks_math(self, prompt: str) -> bool:
        """Return true for obvious math prompts."""

        lowered = prompt.lower()
        return any(word in lowered for word in ("solve", "integrate", "differentiate", "derivative", "mean", "equation")) or bool(re.search(r"\d+\s*[\+\-\*/=]\s*\d+", prompt))

    def _looks_code(self, prompt: str) -> bool:
        """Return true for obvious code prompts."""

        lowered = prompt.lower()
        return any(word in lowered for word in ("code", "python", "function", "debug", "traceback", "class", "javascript"))

    def _answer_known_topic(self, prompt: str) -> str:
        """Answer small built-in knowledge topics."""

        lowered = prompt.lower()
        if "artificial intelligence" in lowered or re.search(r"\bai\b", lowered):
            return self._answer_from_facts("Artificial intelligence")
        for topic in ("machine learning", "deep learning", "neural network", "natural language processing", "large language model", "transformer", "gpt-2", "photosynthesis", "gravity", "water"):
            if topic in lowered:
                return self._answer_from_facts(topic)
        return ""

    def _language_fallback(self, prompt: str, routing: RoutingDecision) -> str:
        """Return a helpful general-language fallback from local facts."""

        cleaned = " ".join(prompt.strip().split())
        lowered = cleaned.lower()
        if any(greeting == lowered.strip(" .!?") for greeting in ("hi", "hello", "hey", "hii")):
            return "Hi. I can help with local math, Python/code tasks, AetherCore details, and common concepts from the built-in knowledge base."
        if any(phrase in lowered for phrase in ("what can you do", "help me", "capabilities")):
            return (
                "I can solve symbolic math, draft and check small Python code, explain AetherCore components, "
                "and answer common factual questions from the local knowledge base."
            )

        facts = self.hallucination_killer.fact_verifier.relevant_facts(cleaned, limit=2)
        if facts:
            return " ".join(self._fact_to_sentence(fact) for fact in facts)

        if self._looks_general_question(cleaned):
            return (
                "I do not have a verified local fact for that specific question yet. "
                "Ask about a narrower term, add the fact to the local knowledge base, or ask for math/code help and I can work from local tools."
            )

        return (
            "I can work on this locally. The router classified it as "
            f"{routing.domain.value} with {routing.precision.value} precision; add a clearer question or local facts for a more direct answer."
        )

    def _answer_from_facts(self, topic: str) -> str:
        """Render the best local facts for a known topic."""

        facts = self.hallucination_killer.fact_verifier.relevant_facts(topic, limit=2, minimum_score=0.05)
        if not facts:
            return ""
        return " ".join(self._fact_to_sentence(fact) for fact in facts)

    def _fact_to_sentence(self, fact: LocalFact) -> str:
        """Render a local fact as a readable sentence."""

        sentence = fact.to_claim().rstrip(".")
        return sentence[:1].upper() + sentence[1:] + "."

    def _looks_general_question(self, prompt: str) -> bool:
        """Return true for broad natural-language questions."""

        lowered = prompt.lower().strip()
        return lowered.endswith("?") or lowered.startswith(("what ", "who ", "why ", "how ", "when ", "where ", "explain ", "define ", "tell me "))

    def _truncate(self, response: str, max_tokens: int) -> str:
        """Truncate by approximate whitespace tokens."""

        words = response.split()
        if len(words) <= max_tokens:
            return response
        return " ".join(words[:max_tokens])


def _self_test() -> None:
    """Run a small CPU sanity check for the inference engine."""

    engine = AetherCoreV3(config={"context_dir": "experiments/_inference_selftest/kv", "knowledge_base_path": "experiments/_inference_selftest/kb.jsonl"})
    math_response = engine.generate("solve x^2 - 4 = 0")
    code_response = engine.generate("write a factorial function")
    chat_response = engine.chat([{"role": "user", "content": "What is AetherCore v3?"}])
    next_token = engine.token_generator.next_token(_text_to_tokens("hello"))

    if "[-2, 2]" not in math_response:
        raise RuntimeError(f"Unexpected math response: {math_response}")
    if "factorial" not in code_response or "verified" not in code_response:
        raise RuntimeError(f"Unexpected code response: {code_response}")
    if "AetherCore v3" not in chat_response:
        raise RuntimeError(f"Unexpected chat response: {chat_response}")
    if not isinstance(next_token, int):
        raise RuntimeError("Token generator did not return an int")

    print("AetherCore inference engine self-test")
    print(f"  math response first line: {math_response.splitlines()[0]}")
    print(f"  code response first line: {code_response.splitlines()[0]}")
    print(f"  chat response: {chat_response}")
    print(f"  next token: {next_token}")
    print(f"  last routing: {engine.last_pipeline_result.routing if engine.last_pipeline_result else None}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
