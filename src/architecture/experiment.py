"""Day-3 co-design experiment: native ternary+sparse vs post-hoc ternary.

For each of two deterministic synthetic tasks we

1. train a :class:`~src.architecture.aethernet.DenseFP` baseline,
2. build a :class:`~src.architecture.aethernet.PostHocTernary` view of it
   (ternarize the trained weights, *no retraining*), and
3. train an :class:`~src.architecture.aethernet.AetherNet` natively in ternary +
   sparse space,

then measure accuracy and the stored / active bit budgets of each variant and
write ``projects/day3_aethernet/results.json``.

The thesis the numbers must show: on the harder ``char_lm`` task the post-hoc
ternary model collapses while the natively-trained AetherNet nearly matches the
dense FP baseline at a comparable stored-bit budget and far fewer active bits.

CPU only.  Deterministic given ``seed=0``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.architecture.aethernet import (
    AetherNet,
    AetherNetConfig,
    BitAccount,
    DenseFP,
    build_aethernet,
    build_dense_fp,
    ternarize_dense_to_posthoc,
)

_RESULTS_PATH = _PROJECT_ROOT / "projects" / "day3_aethernet" / "results.json"


# ---------------------------------------------------------------------------
# Task / training specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Specification of one synthetic sequence task and its training budget."""

    name: str
    vocab_size: int
    seq_len: int
    chance_accuracy: float
    n_examples: int
    dense_steps: int
    aether_steps: int
    dense_lr: float
    aether_lr: float
    batch_size: int

    def __post_init__(self) -> None:
        """Validate the task specification."""

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        for n in ("vocab_size", "seq_len", "n_examples", "dense_steps",
                  "aether_steps", "batch_size"):
            v = getattr(self, n)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{n} must be a positive int, got {v!r}")
        for n in ("chance_accuracy", "dense_lr", "aether_lr"):
            v = getattr(self, n)
            if not isinstance(v, float) or v <= 0.0:
                raise ValueError(f"{n} must be a positive float, got {v!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view of the task spec."""

        return {
            "name": self.name,
            "vocab_size": int(self.vocab_size),
            "seq_len": int(self.seq_len),
            "chance_accuracy": float(self.chance_accuracy),
            "n_examples": int(self.n_examples),
            "dense_steps": int(self.dense_steps),
            "aether_steps": int(self.aether_steps),
            "dense_lr": float(self.dense_lr),
            "aether_lr": float(self.aether_lr),
            "batch_size": int(self.batch_size),
        }


@dataclass(frozen=True, slots=True)
class VariantResult:
    """Measured outcome for one model variant on one task."""

    name: str
    accuracy: float
    stored_bits: float
    active_bits_per_token: float
    stored_ratio_vs_fp: float
    active_ratio_vs_fp: float
    first_loss: float
    last_loss: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view matching the results.json schema."""

        return {
            "accuracy": float(self.accuracy),
            "active_bits_per_token": float(self.active_bits_per_token),
            "active_ratio_vs_fp": float(self.active_ratio_vs_fp),
            "first_loss": float(self.first_loss),
            "last_loss": float(self.last_loss),
            "name": self.name,
            "stored_bits": float(self.stored_bits),
            "stored_ratio_vs_fp": float(self.stored_ratio_vs_fp),
        }


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _make_copy_dataset(spec: TaskSpec, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the copy/recall task.

    The first half of the sequence is a random pattern over symbols
    ``1..vocab-1`` (0 is reserved as a separator); the model must reproduce the
    pattern in the second half.  Targets in the non-recall region are masked with
    ``-100`` so only the recall span is scored.
    """

    v = spec.vocab_size
    half = spec.seq_len // 2
    n = spec.n_examples
    pattern = torch.randint(1, v, (n, half), generator=generator)
    sep = torch.zeros(n, spec.seq_len - 2 * half, dtype=torch.long)
    inputs = torch.cat([pattern, sep, pattern], dim=1)[:, : spec.seq_len]

    targets = torch.full_like(inputs, -100)
    # Predict the next token; the recall region is the second copy.
    recall_start = half + sep.shape[1]
    targets[:, recall_start - 1 : spec.seq_len - 1] = inputs[:, recall_start:spec.seq_len]
    return inputs, targets


def _make_char_lm_dataset(spec: TaskSpec, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a tiny character language-model task with learnable structure.

    Sequences are sampled from a fixed random first-order Markov chain over
    ``vocab`` symbols.  The chain has sharp (low-entropy) transitions, so a
    capable model reaches high next-token accuracy while a collapsed (post-hoc
    ternary) model falls back toward the chain's stationary accuracy.
    """

    v = spec.vocab_size
    n = spec.n_examples
    # Fixed peaky transition matrix: each state strongly prefers a couple of
    # successors.  Built from the generator so it is deterministic with seed 0.
    logits = torch.randn(v, v, generator=generator) * 3.0
    trans = torch.softmax(logits, dim=-1)

    seqs = torch.zeros(n, spec.seq_len, dtype=torch.long)
    state = torch.randint(0, v, (n,), generator=generator)
    seqs[:, 0] = state
    for t in range(1, spec.seq_len):
        probs = trans[state]
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        seqs[:, t] = nxt
        state = nxt

    inputs = seqs
    targets = torch.full_like(inputs, -100)
    targets[:, :-1] = inputs[:, 1:]
    return inputs, targets


_DATASET_BUILDERS: dict[str, Callable[[TaskSpec, torch.Generator], tuple[torch.Tensor, torch.Tensor]]] = {
    "copy_m6": _make_copy_dataset,
    "char_lm": _make_char_lm_dataset,
}


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def _iterate_batches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    steps: int,
    generator: torch.Generator,
) -> Any:
    """Yield ``steps`` random mini-batches (with replacement) from the dataset."""

    n = inputs.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=generator)
        yield inputs[idx], targets[idx]


def _train(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    """Train ``model`` with Adam, returning (first_loss, last_loss)."""

    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    first_loss = float("nan")
    last_loss = float("nan")
    for step, (xb, yb) in enumerate(_iterate_batches(inputs, targets, batch_size, steps, gen)):
        optim.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), ignore_index=-100)
        loss.backward()
        optim.step()
        if step == 0:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
    return first_loss, last_loss


@torch.no_grad()
def _evaluate(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Return (accuracy, loss) over scored (non -100) positions."""

    model.eval()
    logits = model(inputs)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    mask = flat_targets != -100
    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
    pred = flat_logits.argmax(dim=-1)
    correct = (pred[mask] == flat_targets[mask]).float().mean()
    return float(correct.item()), float(loss.item())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _config_for(spec: TaskSpec) -> AetherNetConfig:
    """Build the shared model configuration for a task."""

    return AetherNetConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len, seed=0)


def run_task(spec: TaskSpec) -> dict[str, Any]:
    """Run all three variants on one task and return its results dict."""

    if not isinstance(spec, TaskSpec):
        raise TypeError("spec must be a TaskSpec")

    start = time.perf_counter()
    torch.manual_seed(0)
    data_gen = torch.Generator().manual_seed(0)
    builder = _DATASET_BUILDERS[spec.name]
    inputs, targets = builder(spec, data_gen)

    cfg = _config_for(spec)

    # --- DenseFP baseline -------------------------------------------------
    dense = build_dense_fp(cfg)
    d_first, d_last = _train(dense, inputs, targets, spec.dense_steps, spec.dense_lr,
                             spec.batch_size, seed=0)
    dense_acc, _ = _evaluate(dense, inputs, targets)
    dense_acct = dense.bit_account()

    # --- PostHoc ternary (no retraining) ---------------------------------
    posthoc = ternarize_dense_to_posthoc(dense)
    posthoc_acc, posthoc_loss = _evaluate(posthoc, inputs, targets)
    posthoc_acct = posthoc.bit_account()

    # --- AetherNet (native ternary + sparse MoE) -------------------------
    aether = build_aethernet(cfg)
    a_first, a_last = _train(aether, inputs, targets, spec.aether_steps, spec.aether_lr,
                             spec.batch_size, seed=1)
    aether_acc, _ = _evaluate(aether, inputs, targets)
    aether_acct = aether.bit_account()

    def ratios(a: BitAccount) -> tuple[float, float]:
        return (dense_acct.stored_bits / a.stored_bits,
                dense_acct.active_bits_per_token / a.active_bits_per_token)

    results = [
        VariantResult("DenseFP", dense_acc, dense_acct.stored_bits,
                      dense_acct.active_bits_per_token, 1.0, 1.0, d_first, d_last),
        VariantResult("PostHocTernary", posthoc_acc, posthoc_acct.stored_bits,
                      posthoc_acct.active_bits_per_token, *ratios(posthoc_acct),
                      posthoc_loss, posthoc_loss),
        VariantResult("AetherNet", aether_acc, aether_acct.stored_bits,
                      aether_acct.active_bits_per_token, *ratios(aether_acct),
                      a_first, a_last),
    ]
    elapsed = time.perf_counter() - start

    return {
        "chance_accuracy": float(spec.chance_accuracy),
        "elapsed_sec": float(elapsed),
        "results": [r.to_dict() for r in results],
        "task": spec.name,
        "vocab_size": int(spec.vocab_size),
    }


def default_specs() -> list[TaskSpec]:
    """Return the two production task specifications."""

    return [
        TaskSpec(
            name="copy_m6", vocab_size=13, seq_len=6, chance_accuracy=1.0 / 12.0,
            n_examples=384, dense_steps=1400, aether_steps=2000,
            dense_lr=3.0e-3, aether_lr=4.0e-3, batch_size=64,
        ),
        TaskSpec(
            name="char_lm", vocab_size=29, seq_len=24, chance_accuracy=0.168,
            n_examples=1024, dense_steps=2200, aether_steps=2600,
            dense_lr=3.0e-3, aether_lr=4.0e-3, batch_size=64,
        ),
    ]


def run_all(specs: list[TaskSpec] | None = None, write: bool = True) -> list[dict[str, Any]]:
    """Run every task, optionally writing ``results.json``, and return the list."""

    specs = specs if specs is not None else default_specs()
    outputs = [run_task(spec) for spec in specs]
    if write:
        _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _RESULTS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2, sort_keys=True)
    return outputs


def main() -> None:
    """Run the full experiment, write results.json, and print a summary table."""

    outputs = run_all(write=True)
    for task in outputs:
        print(f"\n## {task['task']}  (vocab {task['vocab_size']}, "
              f"chance {task['chance_accuracy']:.3f})  elapsed {task['elapsed_sec']:.1f}s")
        for r in task["results"]:
            print(f"  {r['name']:<16} acc {r['accuracy']*100:6.2f}%  "
                  f"stored {r['stored_bits']:>10,.0f}  active {r['active_bits_per_token']:>10,.0f}  "
                  f"sr {r['stored_ratio_vs_fp']:.2f}x  ar {r['active_ratio_vs_fp']:.2f}x")
    print(f"\nwrote {_RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Self-test (fast, reduced task)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Fast reduced run asserting the co-design direction holds."""

    spec = TaskSpec(
        name="char_lm", vocab_size=29, seq_len=12, chance_accuracy=0.168,
        n_examples=192, dense_steps=120, aether_steps=160,
        dense_lr=5.0e-3, aether_lr=6.0e-3, batch_size=48,
    )
    task = run_task(spec)
    by_name = {r["name"]: r for r in task["results"]}
    dense = by_name["DenseFP"]
    posthoc = by_name["PostHocTernary"]
    aether = by_name["AetherNet"]

    if not (aether["accuracy"] > posthoc["accuracy"]):
        raise RuntimeError(
            f"AetherNet acc {aether['accuracy']:.3f} must beat PostHoc {posthoc['accuracy']:.3f}"
        )
    if not (aether["active_ratio_vs_fp"] > 1.0):
        raise RuntimeError("AetherNet active_ratio_vs_fp must exceed 1.0")
    if not (posthoc["stored_ratio_vs_fp"] > 1.0):
        raise RuntimeError("PostHoc stored_ratio_vs_fp must exceed 1.0")
    if not (dense["accuracy"] >= aether["accuracy"] - 0.20):
        raise RuntimeError("DenseFP accuracy should be in range of AetherNet")

    print("Day-3 experiment self-test (reduced char_lm)")
    print(f"  DenseFP    acc {dense['accuracy']*100:.1f}%  sr {dense['stored_ratio_vs_fp']:.2f}x")
    print(f"  PostHoc    acc {posthoc['accuracy']*100:.1f}%  sr {posthoc['stored_ratio_vs_fp']:.2f}x "
          f"ar {posthoc['active_ratio_vs_fp']:.2f}x")
    print(f"  AetherNet  acc {aether['accuracy']*100:.1f}%  sr {aether['stored_ratio_vs_fp']:.2f}x "
          f"ar {aether['active_ratio_vs_fp']:.2f}x")
    print(f"  gap AetherNet - PostHoc: {(aether['accuracy']-posthoc['accuracy'])*100:.1f} pts")
    print("  status: ok")


if __name__ == "__main__":
    main()
