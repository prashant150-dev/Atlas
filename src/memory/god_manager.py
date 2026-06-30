"""Tiered expert memory manager for AetherCore v3.

This module manages ternary experts across RAM and SSD-backed cold storage.
It favors predictable behavior over theatrics: resident memory is counted,
async loads return real ``Future`` objects, and evictions are driven by a small
LFU plus recency score.
"""

from __future__ import annotations

import itertools
import queue
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


try:
    from src.core.expert import TernaryExpert
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.core.expert import TernaryExpert


GB = 1024 * 1024 * 1024
MB = 1024 * 1024


class MemoryTier(str, Enum):
    """Memory tier labels ordered from fastest to slowest."""

    L1_CPU_CACHE = "L1_CPU_CACHE"
    L2_RAM_HOT = "L2_RAM_HOT"
    L3_RAM_WARM = "L3_RAM_WARM"
    L4_SSD_COLD = "L4_SSD_COLD"


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Snapshot of memory manager state."""

    l1_bytes: int
    hot_bytes: int
    warm_bytes: int
    cold_bytes: int
    total_resident_bytes: int
    l1_count: int
    hot_count: int
    warm_count: int
    cold_count: int
    hot_capacity_bytes: int
    warm_capacity_bytes: int
    l1_capacity_bytes: int
    pending_loads: int
    inflight_loads: int
    loader_bandwidth_mb_s: float
    total_accesses: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _LoadRequest:
    """Internal async load request."""

    path: Path
    future: Future[TernaryExpert]
    expert_id: str | None
    requested_at_ns: int


def _validate_expert_id(expert_id: str) -> str:
    """Validate and normalize an expert identifier."""

    if not isinstance(expert_id, str):
        raise TypeError("expert_id must be a string")
    normalized = expert_id.strip()
    if not normalized:
        raise ValueError("expert_id must not be empty")
    return normalized


def _expert_nbytes(expert: TernaryExpert) -> int:
    """Estimate resident tensor bytes for an expert."""

    if not isinstance(expert, TernaryExpert):
        raise TypeError("expert must be a TernaryExpert")

    total = 0
    seen: set[int] = set()
    for tensor in itertools.chain(expert.parameters(recurse=True), expert.buffers(recurse=True)):
        if tensor is None:
            continue
        storage_id = id(tensor.untyped_storage()) if hasattr(tensor, "untyped_storage") else id(tensor)
        if storage_id in seen:
            continue
        seen.add(storage_id)
        total += int(tensor.numel() * tensor.element_size())
    return total


def _completed_future(value: TernaryExpert) -> Future[TernaryExpert]:
    """Return a Future that has already completed with a value."""

    future: Future[TernaryExpert] = Future()
    future.set_result(value)
    return future


class AsyncSSDLoader:
    """Priority-queue based asynchronous SSD expert loader."""

    def __init__(self, worker_count: int = 1) -> None:
        """Start background loader workers."""

        if worker_count <= 0:
            raise ValueError("worker_count must be positive")

        self._queue: queue.PriorityQueue[tuple[int, int, _LoadRequest | None]] = queue.PriorityQueue()
        self._counter = itertools.count()
        self._shutdown = threading.Event()
        self._stats_lock = threading.Lock()
        self._bytes_loaded = 0
        self._seconds_spent = 0.0
        self._workers = [
            threading.Thread(target=self._worker_loop, name=f"aether-ssd-loader-{index}", daemon=True)
            for index in range(int(worker_count))
        ]
        for worker in self._workers:
            worker.start()

    @property
    def queue_size(self) -> int:
        """Return the number of pending load requests."""

        return int(self._queue.qsize())

    def load_async(self, path: str | Path, priority: int = 100, expert_id: str | None = None) -> Future[TernaryExpert]:
        """Queue an expert load from SSD and return a Future."""

        input_path = Path(path)
        future: Future[TernaryExpert] = Future()
        if expert_id is not None:
            _validate_expert_id(expert_id)
        if not input_path.exists():
            future.set_exception(FileNotFoundError(f"Expert file not found: {input_path}"))
            return future
        if not input_path.is_file():
            future.set_exception(ValueError(f"Expert path is not a file: {input_path}"))
            return future
        if self._shutdown.is_set():
            future.set_exception(RuntimeError("AsyncSSDLoader has been shut down"))
            return future

        request = _LoadRequest(
            path=input_path,
            future=future,
            expert_id=expert_id,
            requested_at_ns=time.perf_counter_ns(),
        )
        self._queue.put((int(priority), next(self._counter), request))
        return future

    def bandwidth_monitor(self) -> float:
        """Return observed loader bandwidth in MB/s."""

        with self._stats_lock:
            if self._seconds_spent <= 0.0:
                return 0.0
            return float((self._bytes_loaded / MB) / self._seconds_spent)

    def shutdown(self, wait: bool = True) -> None:
        """Stop background loader workers."""

        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for _worker in self._workers:
            self._queue.put((10**12, next(self._counter), None))
        if wait:
            for worker in self._workers:
                worker.join(timeout=5.0)

    def _worker_loop(self) -> None:
        """Load experts until a sentinel request is received."""

        while True:
            _priority, _sequence, request = self._queue.get()
            try:
                if request is None:
                    return
                self._run_request(request)
            finally:
                self._queue.task_done()

    def _run_request(self, request: _LoadRequest) -> None:
        """Execute one load request and update bandwidth stats."""

        future = request.future
        if not future.set_running_or_notify_cancel():
            return

        started = time.perf_counter()
        try:
            file_size = int(request.path.stat().st_size)
            expert = TernaryExpert.load_from_file(request.path)
            if request.expert_id is not None:
                expert.expert_id = request.expert_id
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            with self._stats_lock:
                self._bytes_loaded += file_size
                self._seconds_spent += elapsed
            future.set_result(expert)
        except BaseException as exc:
            future.set_exception(exc)


class HotColdBalancer:
    """LFU plus recency tracker for expert promotion and eviction."""

    def __init__(self, promotion_threshold: int = 2, recency_weight: float = 0.25) -> None:
        """Create a balancer with promotion and eviction knobs."""

        if promotion_threshold <= 0:
            raise ValueError("promotion_threshold must be positive")
        if recency_weight < 0:
            raise ValueError("recency_weight must be non-negative")

        self.promotion_threshold = int(promotion_threshold)
        self.recency_weight = float(recency_weight)
        self.access_counts: dict[str, int] = {}
        self.last_access_ns: dict[str, int] = {}
        self._sequence = 0
        self._lock = threading.Lock()

    def track_access(self, expert_id: str) -> None:
        """Record one access for an expert."""

        normalized_id = _validate_expert_id(expert_id)
        with self._lock:
            self._sequence += 1
            self.access_counts[normalized_id] = self.access_counts.get(normalized_id, 0) + 1
            self.last_access_ns[normalized_id] = self._sequence

    def suggest_promotion(self, expert_id: str) -> bool:
        """Return true when an expert has enough access pressure to promote."""

        normalized_id = _validate_expert_id(expert_id)
        with self._lock:
            count = self.access_counts.get(normalized_id, 0)
            if count >= self.promotion_threshold:
                return True
            if not self.last_access_ns:
                return False
            newest = max(self.last_access_ns.values())
            return self.last_access_ns.get(normalized_id, -1) == newest and count > 0

    def suggest_eviction(self, candidates: Iterable[str] | None = None) -> str | None:
        """Return the lowest-value candidate by LFU plus recency score."""

        with self._lock:
            candidate_list = tuple(_validate_expert_id(candidate) for candidate in (candidates or self.access_counts.keys()))
            if not candidate_list:
                return None

            newest = max(self.last_access_ns.values(), default=0)

            def score(expert_id: str) -> tuple[float, str]:
                frequency = self.access_counts.get(expert_id, 0)
                recency_age = newest - self.last_access_ns.get(expert_id, 0)
                value = float(frequency) - self.recency_weight * float(recency_age)
                return value, expert_id

            return min(candidate_list, key=score)

    def total_accesses(self) -> int:
        """Return total tracked accesses."""

        with self._lock:
            return int(sum(self.access_counts.values()))


class ExpertMemoryManager:
    """Manage experts across hot RAM, warm RAM, and SSD cold tiers."""

    def __init__(
        self,
        hot_max_bytes: int = 2 * GB,
        warm_max_bytes: int = 1 * GB,
        l1_max_bytes: int = 32 * MB,
        cold_storage_dir: str | Path = "models/experts",
        loader: AsyncSSDLoader | None = None,
        balancer: HotColdBalancer | None = None,
    ) -> None:
        """Create a tiered expert memory manager."""

        if hot_max_bytes <= 0:
            raise ValueError("hot_max_bytes must be positive")
        if warm_max_bytes < 0:
            raise ValueError("warm_max_bytes must be non-negative")
        if l1_max_bytes < 0:
            raise ValueError("l1_max_bytes must be non-negative")

        self.hot_max_bytes = int(hot_max_bytes)
        self.warm_max_bytes = int(warm_max_bytes)
        self.l1_max_bytes = int(l1_max_bytes)
        self.cold_storage_dir = Path(cold_storage_dir)
        self.cold_storage_dir.mkdir(parents=True, exist_ok=True)

        self.l1_tier: dict[str, TernaryExpert] = {}
        self.hot_tier: dict[str, TernaryExpert] = {}
        self.warm_tier: dict[str, TernaryExpert] = {}
        self.cold_tier: dict[str, Path] = {}

        self._l1_bytes = 0
        self._hot_bytes = 0
        self._warm_bytes = 0
        self._inflight: dict[str, Future[TernaryExpert]] = {}
        self._lock = threading.RLock()
        self.loader = loader or AsyncSSDLoader()
        self._owns_loader = loader is None
        self.balancer = balancer or HotColdBalancer()

    def register_hot_expert(self, expert_id: str, expert: TernaryExpert) -> None:
        """Register an expert directly in the hot RAM tier."""

        normalized_id = _validate_expert_id(expert_id)
        if not isinstance(expert, TernaryExpert):
            raise TypeError("expert must be a TernaryExpert")
        with self._lock:
            self._remove_resident_unlocked(normalized_id)
            self._ensure_hot_capacity_unlocked(_expert_nbytes(expert), protected={normalized_id})
            expert.expert_id = normalized_id
            expert.wake()
            self.hot_tier[normalized_id] = expert
            self._hot_bytes += _expert_nbytes(expert)
            self.balancer.track_access(normalized_id)
            self._refresh_l1_unlocked()

    def register_warm_expert(self, expert_id: str, expert: TernaryExpert) -> None:
        """Register an expert directly in the warm RAM tier."""

        normalized_id = _validate_expert_id(expert_id)
        if not isinstance(expert, TernaryExpert):
            raise TypeError("expert must be a TernaryExpert")
        with self._lock:
            self._remove_resident_unlocked(normalized_id)
            expert.expert_id = normalized_id
            expert.sleep()
            self.warm_tier[normalized_id] = expert
            self._warm_bytes += _expert_nbytes(expert)
            self._ensure_warm_capacity_unlocked(protected={normalized_id})

    def register_cold_expert(self, expert_id: str, path: str | Path) -> None:
        """Register a cold SSD path for an expert."""

        normalized_id = _validate_expert_id(expert_id)
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Expert file not found: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Expert path is not a file: {input_path}")
        with self._lock:
            self.cold_tier[normalized_id] = input_path

    def get_expert(self, expert_id: str) -> TernaryExpert | None:
        """Return a resident expert, promoting warm experts when appropriate."""

        normalized_id = _validate_expert_id(expert_id)
        with self._lock:
            if normalized_id in self.hot_tier:
                self.balancer.track_access(normalized_id)
                self.hot_tier[normalized_id].wake()
                self._refresh_l1_unlocked()
                return self.hot_tier[normalized_id]
            if normalized_id in self.warm_tier:
                self.balancer.track_access(normalized_id)
                expert = self.warm_tier[normalized_id]
                if self.balancer.suggest_promotion(normalized_id):
                    return self._promote_warm_to_hot_unlocked(normalized_id)
                return expert
            return None

    def load_expert_async(self, expert_id: str) -> Future[TernaryExpert]:
        """Load an expert asynchronously from cold storage into hot RAM."""

        normalized_id = _validate_expert_id(expert_id)
        with self._lock:
            resident = self.get_expert(normalized_id)
            if resident is not None and normalized_id in self.hot_tier:
                return _completed_future(resident)
            if normalized_id in self._inflight:
                return self._inflight[normalized_id]
            if normalized_id not in self.cold_tier:
                future: Future[TernaryExpert] = Future()
                future.set_exception(KeyError(f"Expert {normalized_id!r} has no cold-tier path"))
                return future

            priority = self._load_priority_unlocked(normalized_id)
            future = self.loader.load_async(self.cold_tier[normalized_id], priority=priority, expert_id=normalized_id)
            self._inflight[normalized_id] = future
            future.add_done_callback(lambda completed, eid=normalized_id: self._install_loaded_callback(eid, completed))
            return future

    def evict_lfu(self) -> str | None:
        """Evict the lowest-value hot expert into warm or cold storage."""

        with self._lock:
            evicted_id = self.balancer.suggest_eviction(self.hot_tier.keys())
            if evicted_id is None:
                return None
            expert = self.hot_tier.pop(evicted_id)
            self._hot_bytes -= _expert_nbytes(expert)
            expert.sleep()
            self._add_to_warm_unlocked(evicted_id, expert)
            self._refresh_l1_unlocked()
            return evicted_id

    def prefetch(self, expert_ids_list: Iterable[str]) -> None:
        """Schedule asynchronous loads for likely-next experts."""

        if isinstance(expert_ids_list, str):
            raise TypeError("expert_ids_list must be an iterable of expert ids, not a single string")
        for expert_id in expert_ids_list:
            try:
                self.load_expert_async(expert_id)
            except (KeyError, ValueError, TypeError):
                continue

    def memory_stats(self) -> MemoryStats:
        """Return a snapshot of tier usage and loader state."""

        with self._lock:
            cold_bytes = 0
            for path in self.cold_tier.values():
                try:
                    cold_bytes += int(path.stat().st_size)
                except OSError:
                    continue
            return MemoryStats(
                l1_bytes=self._l1_bytes,
                hot_bytes=self._hot_bytes,
                warm_bytes=self._warm_bytes,
                cold_bytes=cold_bytes,
                total_resident_bytes=self._l1_bytes + self._hot_bytes + self._warm_bytes,
                l1_count=len(self.l1_tier),
                hot_count=len(self.hot_tier),
                warm_count=len(self.warm_tier),
                cold_count=len(self.cold_tier),
                hot_capacity_bytes=self.hot_max_bytes,
                warm_capacity_bytes=self.warm_max_bytes,
                l1_capacity_bytes=self.l1_max_bytes,
                pending_loads=self.loader.queue_size,
                inflight_loads=len(self._inflight),
                loader_bandwidth_mb_s=self.loader.bandwidth_monitor(),
                total_accesses=self.balancer.total_accesses(),
            )

    def shutdown(self) -> None:
        """Shut down owned background loader workers."""

        if self._owns_loader:
            self.loader.shutdown(wait=True)

    def _install_loaded_callback(self, expert_id: str, completed: Future[TernaryExpert]) -> None:
        """Install a loaded expert into hot memory once its Future completes."""

        with self._lock:
            self._inflight.pop(expert_id, None)
        if completed.cancelled():
            return
        try:
            expert = completed.result()
        except BaseException:
            return

        with self._lock:
            self._remove_resident_unlocked(expert_id)
            self._ensure_hot_capacity_unlocked(_expert_nbytes(expert), protected={expert_id})
            expert.expert_id = expert_id
            expert.wake()
            self.hot_tier[expert_id] = expert
            self._hot_bytes += _expert_nbytes(expert)
            self.balancer.track_access(expert_id)
            self._refresh_l1_unlocked()

    def _load_priority_unlocked(self, expert_id: str) -> int:
        """Lower priority numbers load sooner."""

        count = self.balancer.access_counts.get(expert_id, 0)
        return max(0, 100 - count)

    def _promote_warm_to_hot_unlocked(self, expert_id: str) -> TernaryExpert:
        """Move a warm expert into hot RAM."""

        expert = self.warm_tier.pop(expert_id)
        self._warm_bytes -= _expert_nbytes(expert)
        self._ensure_hot_capacity_unlocked(_expert_nbytes(expert), protected={expert_id})
        expert.wake()
        self.hot_tier[expert_id] = expert
        self._hot_bytes += _expert_nbytes(expert)
        self._refresh_l1_unlocked()
        return expert

    def _add_to_warm_unlocked(self, expert_id: str, expert: TernaryExpert) -> None:
        """Add an expert to warm memory, spilling warm overflow to cold."""

        self.warm_tier[expert_id] = expert
        self._warm_bytes += _expert_nbytes(expert)
        self._ensure_warm_capacity_unlocked(protected={expert_id})

    def _ensure_hot_capacity_unlocked(self, extra_bytes: int, protected: set[str] | None = None) -> None:
        """Evict hot experts until enough capacity exists."""

        protected_ids = set(protected or ())
        while self.hot_tier and self._hot_bytes + int(extra_bytes) > self.hot_max_bytes:
            candidates = [expert_id for expert_id in self.hot_tier if expert_id not in protected_ids]
            evict_id = self.balancer.suggest_eviction(candidates)
            if evict_id is None:
                break
            expert = self.hot_tier.pop(evict_id)
            self._hot_bytes -= _expert_nbytes(expert)
            expert.sleep()
            self._add_to_warm_unlocked(evict_id, expert)

    def _ensure_warm_capacity_unlocked(self, protected: set[str] | None = None) -> None:
        """Spill warm experts to cold storage until capacity fits."""

        protected_ids = set(protected or ())
        while self.warm_tier and self._warm_bytes > self.warm_max_bytes:
            candidates = [expert_id for expert_id in self.warm_tier if expert_id not in protected_ids]
            spill_id = self.balancer.suggest_eviction(candidates)
            if spill_id is None:
                break
            expert = self.warm_tier.pop(spill_id)
            self._warm_bytes -= _expert_nbytes(expert)
            self._spill_to_cold_unlocked(spill_id, expert)

    def _spill_to_cold_unlocked(self, expert_id: str, expert: TernaryExpert) -> None:
        """Persist an expert to cold storage when no cold path exists."""

        if expert_id in self.cold_tier and self.cold_tier[expert_id].exists():
            return
        output_path = self.cold_storage_dir / f"{self._safe_filename(expert_id)}.pt"
        expert.save_to_file(output_path)
        self.cold_tier[expert_id] = output_path

    def _remove_resident_unlocked(self, expert_id: str) -> None:
        """Remove an expert from resident tiers without touching cold storage."""

        if expert_id in self.l1_tier:
            self.l1_tier.pop(expert_id)
            self._refresh_l1_unlocked()
        if expert_id in self.hot_tier:
            expert = self.hot_tier.pop(expert_id)
            self._hot_bytes -= _expert_nbytes(expert)
        if expert_id in self.warm_tier:
            expert = self.warm_tier.pop(expert_id)
            self._warm_bytes -= _expert_nbytes(expert)

    def _refresh_l1_unlocked(self) -> None:
        """Mirror the hottest experts into the L1 metadata tier budget."""

        self.l1_tier.clear()
        self._l1_bytes = 0
        if self.l1_max_bytes <= 0:
            return

        candidates = sorted(
            self.hot_tier,
            key=lambda expert_id: (
                self.balancer.access_counts.get(expert_id, 0),
                self.balancer.last_access_ns.get(expert_id, 0),
            ),
            reverse=True,
        )
        for expert_id in candidates:
            expert = self.hot_tier[expert_id]
            size = _expert_nbytes(expert)
            if self._l1_bytes + size > self.l1_max_bytes:
                continue
            self.l1_tier[expert_id] = expert
            self._l1_bytes += size

    def _safe_filename(self, expert_id: str) -> str:
        """Return a filesystem-safe expert filename stem."""

        return "".join(char if char.isalnum() or char in "._-" else "_" for char in expert_id).strip("._") or "expert"


def _self_test() -> None:
    """Run a small CPU sanity check for the memory manager."""

    torch.manual_seed(19)
    temp_path = Path.cwd() / "experiments" / "_memory_selftest"
    temp_path.mkdir(parents=True, exist_ok=True)
    cold_dir = temp_path / "cold"
    cold_dir.mkdir(parents=True, exist_ok=True)
    expert_paths: dict[str, Path] = {}
    experts: dict[str, TernaryExpert] = {}

    for index in range(4):
        expert_id = f"expert.{index}"
        expert = TernaryExpert.from_float_weight(torch.randn(16, 16), expert_id=expert_id)
        path = temp_path / f"{expert_id}.pt"
        expert.save_to_file(path)
        expert_paths[expert_id] = path
        experts[expert_id] = expert

    sample_size = _expert_nbytes(next(iter(experts.values())))
    manager = ExpertMemoryManager(
        hot_max_bytes=max(sample_size * 2, 1),
        warm_max_bytes=max(sample_size, 1),
        l1_max_bytes=max(sample_size, 1),
        cold_storage_dir=cold_dir,
    )

    try:
        for expert_id, path in expert_paths.items():
            manager.register_cold_expert(expert_id, path)

        manager.register_hot_expert("expert.0", experts["expert.0"])
        first_stats = manager.memory_stats()
        future = manager.load_expert_async("expert.1")
        loaded = future.result(timeout=10.0)
        manager.prefetch(["expert.2", "expert.3"])
        for expert_id in ("expert.2", "expert.3"):
            inflight = manager._inflight.get(expert_id)
            if inflight is not None:
                inflight.result(timeout=10.0)

        resident = manager.get_expert("expert.1")
        evicted_id = manager.evict_lfu()
        final_stats = manager.memory_stats()
    finally:
        manager.shutdown()

    if loaded.expert_id != "expert.1":
        raise RuntimeError(f"Unexpected loaded expert id: {loaded.expert_id}")
    if resident is None:
        raise RuntimeError("Loaded expert was not resident")
    if evicted_id is None:
        raise RuntimeError("Expected one hot expert to evict")
    if final_stats.hot_count < 1:
        raise RuntimeError("Expected at least one hot expert")

    print("AetherCore memory manager self-test")
    print(f"  initial hot count: {first_stats.hot_count}")
    print(f"  loaded expert: {loaded.expert_id}")
    print(f"  evicted expert: {evicted_id}")
    print(f"  final stats: {final_stats.to_dict()}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
