"""Hierarchical context memory for AetherCore v3.

This module provides a practical infinite-context substrate:

* recent key/value tensors stay in a small FP16 working tier,
* less recent tensors are kept in RAM using 4-bit quantization,
* old tensors are serialized to SSD with 1-bit quantization,
* durable text facts are consolidated into a JSONL permanent store.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


_EPS = 1.0e-12


class ContextTier(str, Enum):
    """Context storage tiers ordered from freshest to most durable."""

    WORKING = "WORKING"
    SHORT_TERM = "SHORT_TERM"
    LONG_TERM = "LONG_TERM"
    PERMANENT = "PERMANENT"


@dataclass(frozen=True, slots=True)
class Fact:
    """A durable fact extracted from text context."""

    subject: str
    predicate: str
    object: str
    confidence: float
    source: str = ""
    created_at_ns: int = field(default_factory=time.perf_counter_ns)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Fact":
        """Create a fact from a dictionary."""

        required = {"subject", "predicate", "object", "confidence"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Fact payload missing keys: {sorted(missing)}")

        return cls(
            subject=str(payload["subject"]),
            predicate=str(payload["predicate"]),
            object=str(payload["object"]),
            confidence=float(payload["confidence"]),
            source=str(payload.get("source", "")),
            created_at_ns=int(payload.get("created_at_ns", time.perf_counter_ns())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    """Snapshot of hierarchical KV cache usage."""

    working_tokens: int
    short_term_tokens: int
    long_term_tokens: int
    permanent_facts: int
    working_bytes: int
    short_term_bytes: int
    long_term_bytes: int
    total_ram_bytes: int
    latest_token_id: int | None
    storage_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(slots=True)
class _WorkingEntry:
    """In-memory FP16 KV entry."""

    key: torch.Tensor
    value: torch.Tensor
    importance: float
    original_dtype: str
    last_access_ns: int


QuantizedPayload = dict[str, Any]


def _validate_token_id(token_id: int) -> int:
    """Validate and normalize a token id."""

    if isinstance(token_id, bool):
        raise TypeError("token_id must be an integer, not bool")
    normalized = int(token_id)
    if normalized < 0:
        raise ValueError("token_id must be non-negative")
    return normalized


def _clamp01(value: float) -> float:
    """Clamp a finite float into [0, 1]."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return float(max(0.0, min(1.0, number)))


def _ensure_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a tensor input."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must contain at least one value")
    if torch.is_floating_point(value) and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().cpu()


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    """Convert a string such as ``torch.float32`` back to a dtype."""

    name = str(dtype_name).split(".")[-1]
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.float32


def _tensor_bytes(tensor: torch.Tensor) -> int:
    """Return tensor byte count."""

    return int(tensor.numel() * tensor.element_size())


def _payload_bytes(payload: QuantizedPayload) -> int:
    """Estimate RAM bytes for a quantized payload."""

    total = 0
    for value in payload.values():
        if isinstance(value, torch.Tensor):
            total += _tensor_bytes(value)
        elif isinstance(value, Mapping):
            total += _payload_bytes(dict(value))
        elif isinstance(value, (int, float, bool)):
            total += 8
    return total


def _quantize_tensor(tensor: torch.Tensor, bit_width: int) -> QuantizedPayload:
    """Symmetrically quantize one tensor."""

    source = _ensure_tensor(tensor, "tensor")
    if not torch.is_floating_point(source):
        source = source.float()
    data = source.float()

    if bit_width == 1:
        scale = float(data.abs().mean().clamp_min(_EPS).item())
        quantized = torch.sign(data).to(torch.int8)
    elif 2 <= bit_width <= 8:
        max_level = (2 ** (bit_width - 1)) - 1
        scale = float((data.abs().max() / max_level).clamp_min(_EPS).item())
        quantized = torch.round(data / scale).clamp(-max_level, max_level).to(torch.int8)
    else:
        raise ValueError("bit_width must be 1 or between 2 and 8")

    return {
        "shape": tuple(int(dim) for dim in source.shape),
        "dtype": str(source.dtype),
        "bit_width": int(bit_width),
        "scale": scale,
        "values": quantized.contiguous(),
    }


def _dequantize_tensor(payload: QuantizedPayload) -> torch.Tensor:
    """Reconstruct a tensor from a quantized payload."""

    required = {"shape", "dtype", "scale", "values"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Quantized payload missing keys: {sorted(missing)}")

    dtype = _dtype_from_name(str(payload["dtype"]))
    values = payload["values"]
    if not isinstance(values, torch.Tensor):
        raise TypeError("payload['values'] must be a torch.Tensor")
    shape = tuple(int(dim) for dim in payload["shape"])
    return values.float().mul(float(payload["scale"])).reshape(shape).to(dtype=dtype)


def _serialize_token_id(token_id: int) -> str:
    """Return a stable filename stem for a token id."""

    return f"token_{int(token_id):012d}"


def _safe_torch_load(path: Path) -> dict[str, Any]:
    """Load a trusted local torch payload with PyTorch-version compatibility."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dictionary payload in {path}")
    return payload


class HierarchicalKVCache:
    """Hierarchical KV cache with FP16, 4-bit, and 1-bit tiers."""

    def __init__(
        self,
        working_limit_tokens: int = 2_048,
        short_term_limit_tokens: int = 20_000,
        long_term_limit_tokens: int = 1_000_000,
        storage_dir: str | Path = "data/context/kv",
        permanent_facts: Sequence[Fact] | None = None,
    ) -> None:
        """Create a hierarchical KV cache."""

        if working_limit_tokens <= 0:
            raise ValueError("working_limit_tokens must be positive")
        if short_term_limit_tokens < working_limit_tokens:
            raise ValueError("short_term_limit_tokens must be >= working_limit_tokens")
        if long_term_limit_tokens < short_term_limit_tokens:
            raise ValueError("long_term_limit_tokens must be >= short_term_limit_tokens")

        self.working_limit_tokens = int(working_limit_tokens)
        self.short_term_limit_tokens = int(short_term_limit_tokens)
        self.long_term_limit_tokens = int(long_term_limit_tokens)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.working_tier: dict[int, _WorkingEntry] = {}
        self.short_term_tier: dict[int, dict[str, QuantizedPayload | float | int | str]] = {}
        self.long_term_tier: dict[int, Path] = {}
        self.permanent_tier: list[Fact] = list(permanent_facts or [])

        self._importance: dict[int, float] = {}
        self._tier_index: dict[int, ContextTier] = {}
        self._latest_token_id: int | None = None
        self._lock = threading.RLock()

    def store(self, token_id: int, key: torch.Tensor, value: torch.Tensor, importance: float) -> None:
        """Store key/value tensors with automatic tier placement."""

        normalized_id = _validate_token_id(token_id)
        key_tensor = _ensure_tensor(key, "key")
        value_tensor = _ensure_tensor(value, "value")
        if key_tensor.shape != value_tensor.shape:
            raise ValueError("key and value must have identical shapes")

        importance_value = _clamp01(importance)
        with self._lock:
            self._latest_token_id = normalized_id if self._latest_token_id is None else max(self._latest_token_id, normalized_id)
            self._remove_unlocked(normalized_id, delete_file=True)
            self._importance[normalized_id] = importance_value
            target_tier = self.auto_tier(normalized_id, importance_value)

            if target_tier == ContextTier.WORKING:
                self._store_working_unlocked(normalized_id, key_tensor, value_tensor, importance_value)
            elif target_tier == ContextTier.SHORT_TERM:
                self._store_short_unlocked(normalized_id, key_tensor, value_tensor, importance_value)
            else:
                self._store_long_unlocked(normalized_id, key_tensor, value_tensor, importance_value)

            self._enforce_limits_unlocked()

    def retrieve(self, token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve key/value tensors for a token id."""

        normalized_id = _validate_token_id(token_id)
        with self._lock:
            tier = self._tier_index.get(normalized_id)
            if tier is None:
                raise KeyError(f"Token id {normalized_id} not found in KV cache")

            if tier == ContextTier.WORKING:
                entry = self.working_tier[normalized_id]
                entry.last_access_ns = time.perf_counter_ns()
                return entry.key.clone(), entry.value.clone()

            if tier == ContextTier.SHORT_TERM:
                payload = self.short_term_tier[normalized_id]
                key_tensor = _dequantize_tensor(payload["key"])
                value_tensor = _dequantize_tensor(payload["value"])
                payload["last_access_ns"] = time.perf_counter_ns()
                if float(payload["importance"]) >= 0.85:
                    self._promote_short_to_working_unlocked(normalized_id, key_tensor, value_tensor)
                return key_tensor, value_tensor

            if tier == ContextTier.LONG_TERM:
                payload = _safe_torch_load(self.long_term_tier[normalized_id])
                key_tensor = _dequantize_tensor(payload["key"])
                value_tensor = _dequantize_tensor(payload["value"])
                if float(payload.get("importance", 0.0)) >= 0.90:
                    self._store_working_unlocked(normalized_id, key_tensor, value_tensor, float(payload["importance"]))
                    self._enforce_limits_unlocked()
                return key_tensor, value_tensor

            raise KeyError(f"Token id {normalized_id} is not a KV cache token")

    def auto_tier(self, token_id: int, importance: float | None = None) -> ContextTier:
        """Choose a tier from token recency and importance."""

        normalized_id = _validate_token_id(token_id)
        importance_value = _clamp01(self._importance.get(normalized_id, 0.0) if importance is None else importance)
        latest = normalized_id if self._latest_token_id is None else max(self._latest_token_id, normalized_id)
        age = max(0, latest - normalized_id)

        if importance_value >= 0.85 or age < self.working_limit_tokens:
            return ContextTier.WORKING
        if importance_value >= 0.35 or age < self.short_term_limit_tokens:
            return ContextTier.SHORT_TERM
        return ContextTier.LONG_TERM

    def compress_tier(self, tier: ContextTier | str) -> threading.Thread:
        """Compress or demote one tier in a background thread."""

        context_tier = self._normalize_tier(tier)
        thread = threading.Thread(
            target=self._compress_tier_worker,
            args=(context_tier,),
            name=f"aether-context-compress-{context_tier.value.lower()}",
            daemon=True,
        )
        thread.start()
        return thread

    def add_permanent_fact(self, fact: Fact) -> None:
        """Add a durable fact to the permanent tier."""

        if not isinstance(fact, Fact):
            raise TypeError("fact must be a Fact")
        with self._lock:
            signature = (fact.subject.lower(), fact.predicate.lower(), fact.object.lower())
            existing = {
                (item.subject.lower(), item.predicate.lower(), item.object.lower())
                for item in self.permanent_tier
            }
            if signature not in existing:
                self.permanent_tier.append(fact)

    def stats(self) -> KVCacheStats:
        """Return cache usage statistics."""

        with self._lock:
            working_bytes = sum(_tensor_bytes(entry.key) + _tensor_bytes(entry.value) for entry in self.working_tier.values())
            short_bytes = sum(_payload_bytes(dict(payload)) for payload in self.short_term_tier.values())
            long_bytes = 0
            for path in self.long_term_tier.values():
                try:
                    long_bytes += int(path.stat().st_size)
                except OSError:
                    continue
            return KVCacheStats(
                working_tokens=len(self.working_tier),
                short_term_tokens=len(self.short_term_tier),
                long_term_tokens=len(self.long_term_tier),
                permanent_facts=len(self.permanent_tier),
                working_bytes=working_bytes,
                short_term_bytes=short_bytes,
                long_term_bytes=long_bytes,
                total_ram_bytes=working_bytes + short_bytes,
                latest_token_id=self._latest_token_id,
                storage_dir=str(self.storage_dir),
            )

    def _store_working_unlocked(self, token_id: int, key: torch.Tensor, value: torch.Tensor, importance: float) -> None:
        """Store tensors in the FP16 working tier."""

        key_dtype = str(key.dtype)
        self.working_tier[token_id] = _WorkingEntry(
            key=key.to(torch.float16).contiguous(),
            value=value.to(torch.float16).contiguous(),
            importance=float(importance),
            original_dtype=key_dtype,
            last_access_ns=time.perf_counter_ns(),
        )
        self._tier_index[token_id] = ContextTier.WORKING

    def _store_short_unlocked(self, token_id: int, key: torch.Tensor, value: torch.Tensor, importance: float) -> None:
        """Store tensors in the 4-bit short-term tier."""

        self.short_term_tier[token_id] = {
            "key": _quantize_tensor(key, bit_width=4),
            "value": _quantize_tensor(value, bit_width=4),
            "importance": float(importance),
            "created_at_ns": time.perf_counter_ns(),
            "last_access_ns": time.perf_counter_ns(),
        }
        self._tier_index[token_id] = ContextTier.SHORT_TERM

    def _store_long_unlocked(self, token_id: int, key: torch.Tensor, value: torch.Tensor, importance: float) -> None:
        """Store tensors in the 1-bit SSD-backed long-term tier."""

        path = self.storage_dir / f"{_serialize_token_id(token_id)}.pt"
        payload = {
            "format": "aethercore_v3.long_context_kv",
            "token_id": int(token_id),
            "key": _quantize_tensor(key, bit_width=1),
            "value": _quantize_tensor(value, bit_width=1),
            "importance": float(importance),
            "created_at_ns": time.perf_counter_ns(),
        }
        torch.save(payload, path)
        self.long_term_tier[token_id] = path
        self._tier_index[token_id] = ContextTier.LONG_TERM

    def _remove_unlocked(self, token_id: int, delete_file: bool) -> None:
        """Remove one token from all tiers."""

        self.working_tier.pop(token_id, None)
        self.short_term_tier.pop(token_id, None)
        path = self.long_term_tier.pop(token_id, None)
        self._tier_index.pop(token_id, None)
        if delete_file and path is not None and path.exists():
            path.unlink()

    def _enforce_limits_unlocked(self) -> None:
        """Keep tier sizes inside configured token limits."""

        while len(self.working_tier) > self.working_limit_tokens:
            victim_id = min(self.working_tier, key=self._eviction_score_unlocked)
            victim = self.working_tier.pop(victim_id)
            self._store_short_unlocked(victim_id, victim.key.float(), victim.value.float(), victim.importance)

        max_short_entries = max(0, self.short_term_limit_tokens - self.working_limit_tokens)
        while len(self.short_term_tier) > max_short_entries:
            victim_id = min(self.short_term_tier, key=self._eviction_score_unlocked)
            payload = self.short_term_tier.pop(victim_id)
            key = _dequantize_tensor(payload["key"])
            value = _dequantize_tensor(payload["value"])
            self._store_long_unlocked(victim_id, key, value, float(payload["importance"]))

        while len(self.long_term_tier) > self.long_term_limit_tokens:
            victim_id = min(self.long_term_tier, key=self._eviction_score_unlocked)
            self._remove_unlocked(victim_id, delete_file=True)
            self._importance.pop(victim_id, None)

    def _eviction_score_unlocked(self, token_id: int) -> tuple[float, int]:
        """Return an eviction score: lower means easier to demote."""

        importance = self._importance.get(token_id, 0.0)
        latest = self._latest_token_id if self._latest_token_id is not None else token_id
        age = max(0, latest - token_id)
        return importance - min(0.25, age / max(1, self.short_term_limit_tokens)), token_id

    def _promote_short_to_working_unlocked(self, token_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Promote a short-term item into working memory."""

        payload = self.short_term_tier.pop(token_id, None)
        if payload is None:
            return
        self._store_working_unlocked(token_id, key, value, float(payload["importance"]))
        self._enforce_limits_unlocked()

    def _compress_tier_worker(self, tier: ContextTier) -> None:
        """Worker used by compress_tier."""

        with self._lock:
            if tier == ContextTier.WORKING:
                self._enforce_limits_unlocked()
            elif tier == ContextTier.SHORT_TERM:
                self._enforce_limits_unlocked()
            elif tier == ContextTier.LONG_TERM:
                self._enforce_limits_unlocked()
            elif tier == ContextTier.PERMANENT:
                self.permanent_tier = list({
                    (fact.subject.lower(), fact.predicate.lower(), fact.object.lower()): fact
                    for fact in self.permanent_tier
                }.values())

    def _normalize_tier(self, tier: ContextTier | str) -> ContextTier:
        """Normalize a tier value."""

        if isinstance(tier, ContextTier):
            return tier
        if not isinstance(tier, str):
            raise TypeError("tier must be a ContextTier or string")
        try:
            return ContextTier(tier.strip().upper())
        except ValueError as exc:
            raise ValueError(f"Unsupported context tier: {tier!r}") from exc


class MemoryConsolidator:
    """Extract and persist durable facts from text context."""

    def __init__(self, permanent_path: str | Path = "data/context/permanent_facts.jsonl") -> None:
        """Create a fact consolidator backed by a JSONL file."""

        self.permanent_path = Path(permanent_path)
        self.permanent_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._facts: list[Fact] = []
        self._signatures: set[tuple[str, str, str]] = set()
        self._load_existing()

    def consolidate(self, tokens_1000: Sequence[int] | str) -> list[Fact]:
        """Extract facts from around 1000 tokens and store them permanently."""

        text = tokens_1000 if isinstance(tokens_1000, str) else self._tokens_to_text(tokens_1000)
        facts = self.extract_facts(text)
        for fact in facts:
            self.store_permanent(fact)
        return facts

    def extract_facts(self, text: str) -> list[Fact]:
        """Extract simple factual statements from text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return []

        facts: list[Fact] = []
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]
        for sentence in sentences:
            facts.extend(self._extract_sentence_facts(sentence))
        return facts

    def store_permanent(self, fact: Fact) -> None:
        """Persist a fact if it has not already been stored."""

        if not isinstance(fact, Fact):
            raise TypeError("fact must be a Fact")
        signature = self._signature(fact)
        with self._lock:
            if signature in self._signatures:
                return
            self._facts.append(fact)
            self._signatures.add(signature)
            with self.permanent_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fact.to_dict(), sort_keys=True) + "\n")

    def facts(self) -> list[Fact]:
        """Return all stored facts."""

        with self._lock:
            return list(self._facts)

    def search(self, topic: str) -> list[Fact]:
        """Return facts whose subject or object mentions a topic."""

        if not isinstance(topic, str):
            raise TypeError("topic must be a string")
        needle = topic.strip().lower()
        if not needle:
            return []
        with self._lock:
            return [
                fact
                for fact in self._facts
                if needle in fact.subject.lower() or needle in fact.object.lower()
            ]

    def _extract_sentence_facts(self, sentence: str) -> list[Fact]:
        """Extract facts from one sentence."""

        patterns = (
            (r"^(.{2,80}?)\s+(is|are|was|were)\s+(.{2,160})$", "is"),
            (r"^(.{2,80}?)\s+(has|have|contains|contain|uses|use)\s+(.{2,160})$", "has"),
            (r"^(remember|fact)\s*:\s*(.{2,200})$", "states"),
        )
        output: list[Fact] = []
        normalized = sentence.strip().strip(".!?")

        for pattern, predicate in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            if predicate == "states":
                subject = "memory"
                obj = match.group(2).strip()
            else:
                subject = match.group(1).strip()
                obj = match.group(3).strip()
            if subject and obj:
                output.append(
                    Fact(
                        subject=self._clean_fact_part(subject),
                        predicate=predicate,
                        object=self._clean_fact_part(obj),
                        confidence=0.80 if predicate == "states" else 0.72,
                        source=sentence,
                    )
                )
        return output

    def _clean_fact_part(self, value: str) -> str:
        """Normalize one fact text field."""

        return re.sub(r"\s+", " ", value.strip(" ,;:")).strip()

    def _signature(self, fact: Fact) -> tuple[str, str, str]:
        """Return a deduplication signature."""

        return fact.subject.lower(), fact.predicate.lower(), fact.object.lower()

    def _tokens_to_text(self, tokens: Sequence[int]) -> str:
        """Best-effort conversion of token ids to text for consolidation demos."""

        if isinstance(tokens, str):
            return tokens
        if not isinstance(tokens, Sequence):
            raise TypeError("tokens_1000 must be a string or sequence of ints")
        values = [int(token) for token in tokens]
        if not values:
            return ""
        if all(9 <= value <= 126 for value in values):
            return bytes(values).decode("utf-8", errors="ignore")
        return " ".join(str(value) for value in values)

    def _load_existing(self) -> None:
        """Load existing JSONL facts from disk."""

        if not self.permanent_path.exists():
            return
        with self.permanent_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    fact = Fact.from_dict(json.loads(stripped))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                signature = self._signature(fact)
                if signature not in self._signatures:
                    self._facts.append(fact)
                    self._signatures.add(signature)


class ImportanceScorer:
    """Score token importance from attention weights."""

    def __init__(self, token_salience_weight: float = 0.10) -> None:
        """Create an importance scorer."""

        if token_salience_weight < 0:
            raise ValueError("token_salience_weight must be non-negative")
        self.token_salience_weight = float(token_salience_weight)

    def score(self, token: int | torch.Tensor, attention_weights: torch.Tensor | Sequence[float]) -> float:
        """Return an importance score in [0, 1]."""

        token_id = self._token_to_int(token)
        weights = torch.as_tensor(attention_weights, dtype=torch.float32).detach().flatten()
        if weights.numel() == 0:
            raise ValueError("attention_weights must contain at least one value")
        if not torch.isfinite(weights).all().item():
            raise ValueError("attention_weights contains non-finite values")

        abs_weights = weights.abs()
        total = abs_weights.sum().clamp_min(_EPS)
        normalized = abs_weights / total
        focus = float(normalized.max().item())
        mean_strength = float(abs_weights.mean().clamp(0.0, 1.0).item())
        entropy = float(-(normalized * normalized.clamp_min(_EPS).log()).sum().item())
        max_entropy = math.log(max(1, int(normalized.numel())))
        concentration = 1.0 if max_entropy <= _EPS else 1.0 - min(1.0, entropy / max_entropy)
        token_salience = self._token_salience(token_id)

        score = 0.45 * focus + 0.30 * mean_strength + 0.15 * concentration + self.token_salience_weight * token_salience
        return _clamp01(score)

    def _token_to_int(self, token: int | torch.Tensor) -> int:
        """Normalize a token input to int."""

        if isinstance(token, torch.Tensor):
            if token.numel() == 0:
                raise ValueError("token tensor must contain at least one value")
            return int(token.detach().flatten()[-1].item())
        return int(token)

    def _token_salience(self, token_id: int) -> float:
        """Heuristic salience for punctuation, digits, and uncommon ids."""

        if token_id < 0:
            return 1.0
        if 48 <= token_id <= 57:
            return 0.75
        if token_id in {10, 35, 40, 41, 42, 43, 45, 47, 58, 61, 91, 93, 123, 125}:
            return 0.85
        return min(1.0, math.log1p(token_id % 10_000) / math.log1p(10_000))


def _self_test() -> None:
    """Run a small CPU sanity check for infinite context."""

    torch.manual_seed(23)
    base_dir = Path.cwd() / "experiments" / "_context_selftest"
    kv_dir = base_dir / "kv"
    facts_path = base_dir / "facts.jsonl"
    base_dir.mkdir(parents=True, exist_ok=True)

    cache = HierarchicalKVCache(
        working_limit_tokens=3,
        short_term_limit_tokens=5,
        long_term_limit_tokens=10,
        storage_dir=kv_dir,
    )
    scorer = ImportanceScorer()

    for token_id in range(8):
        key = torch.randn(2, 4) + token_id * 0.01
        value = torch.randn(2, 4) - token_id * 0.01
        attention = torch.linspace(0.05, 0.95, steps=8)
        importance = scorer.score(token_id + 48, attention.roll(token_id))
        cache.store(token_id, key, value, importance)

    key, value = cache.retrieve(7)
    compression_thread = cache.compress_tier(ContextTier.WORKING)
    compression_thread.join(timeout=5.0)

    consolidator = MemoryConsolidator(permanent_path=facts_path)
    facts = consolidator.consolidate(
        "AetherCore is a local inference prototype. "
        "Fact: component five stores hierarchical context. "
        "The cache has working memory."
    )
    for fact in facts:
        cache.add_permanent_fact(fact)
    stats = cache.stats()

    if key.shape != value.shape:
        raise RuntimeError("Retrieved key/value shapes differ")
    if stats.working_tokens > 3:
        raise RuntimeError("Working tier exceeded configured limit")
    if not facts:
        raise RuntimeError("Expected at least one extracted fact")

    print("AetherCore infinite context self-test")
    print(f"  retrieved shape: {tuple(key.shape)}")
    print(f"  stats: {stats.to_dict()}")
    print(f"  facts extracted: {[fact.to_dict() for fact in facts]}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
