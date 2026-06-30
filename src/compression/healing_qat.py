"""Healing / Quantization-Aware-Training on a ternary GPT-2.

Day 1 (``compression_limit.py`` / ``rate_distortion_limit.py``) showed that
*post-hoc* ternary quantization of GPT-2-small collapses the model: the
next-token distribution is destroyed and top-1 agreement crashes to ~0%. The
Day-1 interpretation was blunt -- below ~4 bits, matching the FP weights
exactly is information-theoretically hopeless for a model with little
redundancy.

This module demonstrates the Day-2 lever: **change the goal**. Instead of
asking the ternary weights to *match the FP weights*, we ask the ternary
student to *preserve the behaviour* of the FP teacher. We distil the frozen FP
teacher into a ternary student whose latent FP "shadow" weights are trained
through a straight-through estimator (STE). The forward pass uses ternarized
weights; the backward pass flows gradients straight to the shadow weights. A
handful of distillation steps recovers a meaningful chunk of the collapse --
not because we beat information theory, but because the irreducible content of
the *behaviour* is lower than that of the exact FP weights, so the
rate-distortion floor that applied to the weights simply does not apply to the
function.

Run from the repo root::

    python -m src.compression.healing_qat          # full heal (~270s on CPU)
    python src/compression/healing_qat.py          # same

The full run writes ``projects/day2_healing_qat/healing_results.json``. The
``_self_test()`` is a fast smoke test on a tiny ``nn.Linear`` stack and does
*not* load GPT-2.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_EPS = 1.0e-12

# GPT-2 transformer-block linear layers we wrap. GPT-2 uses ``Conv1D`` (a
# transposed linear with weight shape [in_features, out_features]) for these.
_WRAPPED_SUFFIXES: tuple[str, ...] = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)

# A tiny "healing set": short, generic English so distillation has gradient
# signal on realistic next-token distributions. Kept small on purpose -- this
# demonstrates the mechanism, not a production heal.
_HEALING_TEXT: tuple[str, ...] = (
    "The most important idea in science is that observation guides theory.",
    "Knowledge grows when people share what they have learned with others.",
    "A good explanation makes a complex thing feel simple and clear.",
    "History teaches us that progress is slow but rarely stops entirely.",
    "The universe is vast, and yet every part of it follows the same laws.",
    "Language lets us carry an idea from one mind into another mind.",
    "Curiosity is the engine that has driven every great discovery so far.",
    "Mathematics is the language we use to describe patterns in the world.",
    "A small experiment, repeated carefully, can overturn a large belief.",
    "The future depends on the questions we are brave enough to ask today.",
)

_DEFAULT_PROMPT = "The most important idea in science is"
_DEFAULT_MODEL_PATH = "models/gpt2"
_DEFAULT_OUTPUT = "projects/day2_healing_qat/healing_results.json"


@dataclass(frozen=True, slots=True)
class HealingConfig:
    """Hyper-parameters for the healing / QAT run.

    Attributes:
        steps: Number of distillation training steps for the healed student.
        learning_rate: Adam learning rate on the shadow weights.
        kl_temperature: Softmax temperature ``T`` for the distillation KL term
            (the KL is scaled by ``T**2`` per the standard Hinton recipe).
        ce_weight: Weight of the hard next-token cross-entropy term.
        grad_clip: Max global gradient norm (``0`` disables clipping).
        seq_len: Token length of each training / eval window.
        threshold_factor: Per-output-channel ternarization threshold factor;
            ``threshold = threshold_factor * mean(|W_shadow|)``.
        seed: RNG seed for reproducibility.
    """

    steps: int = 30
    learning_rate: float = 2.0e-4
    kl_temperature: float = 2.0
    ce_weight: float = 0.1
    grad_clip: float = 1.0
    seq_len: int = 64
    threshold_factor: float = 0.7
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.kl_temperature <= 0:
            raise ValueError("kl_temperature must be positive")
        if self.ce_weight < 0:
            raise ValueError("ce_weight must be non-negative")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.threshold_factor < 0:
            raise ValueError("threshold_factor must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration dictionary."""

        return {
            "ce_weight": self.ce_weight,
            "grad_clip": self.grad_clip,
            "kl_temperature": self.kl_temperature,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "steps": self.steps,
            "threshold_factor": self.threshold_factor,
        }


@dataclass(frozen=True, slots=True)
class HealingResults:
    """Measured outcome of a healing / QAT run (the report contract)."""

    teacher_perplexity: float
    naive_perplexity: float
    healed_perplexity: float
    naive_top1_agreement: float
    healed_top1_agreement: float
    naive_kl: float
    healed_kl: float
    first_step_loss: float
    last_step_loss: float
    layers_wrapped: int
    ternary_params: int
    bits_per_weight: float
    teacher_sample: str
    naive_sample: str
    healed_sample: str
    elapsed_sec: float
    model_path: str
    config: HealingConfig

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload written to ``healing_results.json``."""

        return {
            "bits_per_weight": self.bits_per_weight,
            "config": self.config.to_dict(),
            "elapsed_sec": self.elapsed_sec,
            "first_step_loss": self.first_step_loss,
            "healed_kl": self.healed_kl,
            "healed_perplexity": self.healed_perplexity,
            "healed_sample": self.healed_sample,
            "healed_top1_agreement": self.healed_top1_agreement,
            "last_step_loss": self.last_step_loss,
            "layers_wrapped": self.layers_wrapped,
            "model_path": self.model_path,
            "naive_kl": self.naive_kl,
            "naive_perplexity": self.naive_perplexity,
            "naive_sample": self.naive_sample,
            "naive_top1_agreement": self.naive_top1_agreement,
            "teacher_perplexity": self.teacher_perplexity,
            "teacher_sample": self.teacher_sample,
            "ternary_params": self.ternary_params,
        }


def _ternarize_ste(
    shadow: torch.Tensor,
    threshold_factor: float,
) -> torch.Tensor:
    """Ternarize ``shadow`` per output channel with a straight-through gradient.

    The forward value is ``scale * sign-with-threshold(shadow)`` where, for each
    output channel (the last dimension of a GPT-2 ``Conv1D`` weight of shape
    ``[in, out]``):

      * ``threshold = threshold_factor * mean(|shadow|)`` over the channel,
      * a weight is mapped to ``+1`` / ``-1`` if it exceeds ``+threshold`` /
        ``-threshold`` and to ``0`` otherwise,
      * ``scale = mean(|shadow|)`` over the *kept* (nonzero) entries of the
        channel.

    The straight-through estimator replaces the (a.e. zero) derivative of the
    ternarization with the identity, so ``d loss / d shadow`` equals the
    gradient that arrives at the ternary value. This is what lets a few
    distillation steps move the latent FP shadow weights.

    Args:
        shadow: Latent FP weight of shape ``[in_features, out_features]``.
        threshold_factor: Threshold scaling applied to per-channel ``mean|W|``.

    Returns:
        The ternarized weight (same shape/dtype as ``shadow``) with gradients
        wired straight through to ``shadow``.
    """

    if shadow.ndim != 2:
        raise ValueError("shadow must be a 2-D [in_features, out_features] tensor")

    abs_w = shadow.detach().abs()
    # Per output channel (dim=0 collapses the input dimension).
    abs_mean = abs_w.mean(dim=0, keepdim=True).clamp_min(_EPS)
    threshold = threshold_factor * abs_mean

    sign = torch.zeros_like(shadow.detach())
    sign = torch.where(shadow.detach() > threshold, torch.ones_like(sign), sign)
    sign = torch.where(shadow.detach() < -threshold, -torch.ones_like(sign), sign)

    kept = sign != 0
    kept_sum = (abs_w * kept).sum(dim=0, keepdim=True)
    kept_count = kept.sum(dim=0, keepdim=True).clamp_min(1)
    scale = (kept_sum / kept_count).clamp_min(_EPS)

    quantized = sign * scale
    # Straight-through: forward = quantized, backward = identity to shadow.
    return shadow + (quantized - shadow).detach()


class TernaryShadowConv1D(nn.Module):
    """STE-wrapped replacement for a GPT-2 ``Conv1D`` layer.

    Holds a trainable FP *shadow* weight. The forward pass computes
    ``y = x @ ternarize(shadow) + bias`` exactly like ``Conv1D`` (which stores
    its weight as ``[in_features, out_features]``), but the matmul uses the
    ternarized weight. With ``train_shadow=False`` the shadow is frozen, which
    -- when the shadow equals the original FP weight -- reproduces the Day-1
    post-hoc collapse ("naive ternary"). With ``train_shadow=True`` the shadow
    is an ``nn.Parameter`` that distillation can heal.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        threshold_factor: float = 0.7,
        train_shadow: bool = False,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must have shape [in_features, out_features]")
        if threshold_factor < 0:
            raise ValueError("threshold_factor must be non-negative")

        self.in_features = int(weight.shape[0])
        self.out_features = int(weight.shape[1])
        self.threshold_factor = float(threshold_factor)

        shadow = weight.detach().clone().float()
        if train_shadow:
            self.shadow = nn.Parameter(shadow)
        else:
            self.register_buffer("shadow", shadow, persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone().float(), persistent=True)

    def ternary_weight(self) -> torch.Tensor:
        """Return the current ternarized weight (with STE gradient wiring)."""

        return _ternarize_ste(self.shadow, self.threshold_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_t = self.ternary_weight()
        size_out = x.size()[:-1] + (self.out_features,)
        flat = x.reshape(-1, x.size(-1)).float()
        output = flat.matmul(weight_t)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(size_out).to(dtype=x.dtype)


def _iter_wrapped_modules(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    """Find GPT-2 block linears to wrap.

    Returns a list of ``(parent_qualified_name, parent_module, child_attr)``
    for every module whose qualified name ends with one of
    ``_WRAPPED_SUFFIXES``.
    """

    targets: list[tuple[str, nn.Module, str]] = []
    module_by_name = dict(model.named_modules())
    for name, _module in model.named_modules():
        if not name.endswith(_WRAPPED_SUFFIXES):
            continue
        parent_name, _, child_attr = name.rpartition(".")
        parent = module_by_name.get(parent_name)
        if parent is None:
            continue
        targets.append((name, parent, child_attr))
    return targets


def wrap_ternary_student(
    model: nn.Module,
    threshold_factor: float = 0.7,
    train_shadow: bool = False,
) -> tuple[int, int, int]:
    """Replace GPT-2 block linears with :class:`TernaryShadowConv1D`.

    Args:
        model: A GPT-2 ``AutoModelForCausalLM`` (mutated in place).
        threshold_factor: Per-channel ternarization threshold factor.
        train_shadow: If ``True`` the shadow weights become trainable params
            (healed student). If ``False`` they are frozen buffers initialized
            to the original FP weights (naive student -> reproduces collapse).

    Returns:
        ``(layers_wrapped, ternary_params, scale_channels)`` where
        ``ternary_params`` is the total number of ternarized weights and
        ``scale_channels`` is the total number of per-output-channel scales.
    """

    targets = _iter_wrapped_modules(model)
    if not targets:
        raise RuntimeError("no GPT-2 block linears found to wrap")

    layers_wrapped = 0
    ternary_params = 0
    scale_channels = 0
    for _name, parent, child_attr in targets:
        child = getattr(parent, child_attr)
        weight = getattr(child, "weight")
        if weight.ndim != 2:
            raise RuntimeError(f"expected 2-D weight for {child_attr}, got {tuple(weight.shape)}")
        bias = getattr(child, "bias", None)
        wrapped = TernaryShadowConv1D(
            weight=weight,
            bias=bias,
            threshold_factor=threshold_factor,
            train_shadow=train_shadow,
        )
        setattr(parent, child_attr, wrapped)
        layers_wrapped += 1
        ternary_params += int(weight.numel())
        scale_channels += int(weight.shape[1])  # one scale per output channel

    return layers_wrapped, ternary_params, scale_channels


def compute_bits_per_weight(ternary_params: int, scale_channels: int) -> float:
    """Honest bits/weight: 2-bit ternary codes + fp16 per-channel scales.

    Each ternary weight costs 2 bits (3 values packed into a 2-bit code). Each
    output channel additionally stores one fp16 scale (16 bits) amortized over
    the weights in that channel. The total is

        (2 * ternary_params + 16 * scale_channels) / ternary_params.

    For GPT-2-small's wrapped layers this lands at ~2.0156 bits/weight.
    """

    if ternary_params <= 0:
        raise ValueError("ternary_params must be positive")
    if scale_channels < 0:
        raise ValueError("scale_channels must be non-negative")
    code_bits = 2 * ternary_params
    scale_bits = 16 * scale_channels
    return float((code_bits + scale_bits) / ternary_params)


def _build_eval_batch(
    tokenizer: Any,
    seq_len: int,
    seed: int,
) -> torch.Tensor:
    """Build a fixed evaluation window of ``seq_len`` tokens from healing text."""

    torch.manual_seed(seed)
    joined = " ".join(_HEALING_TEXT)
    ids = tokenizer(joined, return_tensors="pt").input_ids[0]
    if ids.numel() < seq_len:
        reps = math.ceil(seq_len / max(1, ids.numel()))
        ids = ids.repeat(reps)
    return ids[:seq_len].unsqueeze(0)


def _build_training_batches(
    tokenizer: Any,
    seq_len: int,
    steps: int,
    seed: int,
) -> list[torch.Tensor]:
    """Build ``steps`` token windows that cycle through the healing text."""

    windows: list[torch.Tensor] = []
    for index, text in enumerate(_HEALING_TEXT):
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.numel() < seq_len:
            reps = math.ceil(seq_len / max(1, ids.numel()))
            ids = ids.repeat(reps)
        windows.append(ids[:seq_len].unsqueeze(0))
    if not windows:
        raise RuntimeError("no healing windows could be built")
    return [windows[i % len(windows)] for i in range(steps)]


def _perplexity(model: nn.Module, input_ids: torch.Tensor) -> float:
    """Causal-LM perplexity of ``model`` on a single token window."""

    with torch.inference_mode():
        logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
    shift_labels = input_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels)
    return float(torch.exp(loss).item())


def _agreement_and_kl(
    teacher: nn.Module,
    student: nn.Module,
    input_ids: torch.Tensor,
) -> tuple[float, float]:
    """Top-1 next-token agreement and mean KL(student||teacher) vs the teacher."""

    with torch.inference_mode():
        teacher_logits = teacher(input_ids).logits[0].float()
        student_logits = student(input_ids).logits[0].float()

    teacher_top1 = teacher_logits.argmax(dim=-1)
    student_top1 = student_logits.argmax(dim=-1)
    agreement = float((teacher_top1 == student_top1).float().mean().item())

    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1)
    # KL(student || teacher) = sum p_s * (logp_s - logp_t).
    kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1).mean()
    return agreement, float(kl.item())


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float,
    ce_weight: float,
) -> torch.Tensor:
    """KL distillation (scaled by ``T**2``) plus a hard next-token CE term."""

    t_logits = teacher_logits[:, :-1, :].reshape(-1, teacher_logits.size(-1)).float()
    s_logits = student_logits[:, :-1, :].reshape(-1, student_logits.size(-1)).float()

    teacher_soft = F.log_softmax(t_logits / temperature, dim=-1)
    student_soft = F.log_softmax(s_logits / temperature, dim=-1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean", log_target=True)
    soft_loss = kl * (temperature * temperature)

    labels = input_ids[:, 1:].reshape(-1)
    hard_loss = F.cross_entropy(s_logits, labels)
    return soft_loss + ce_weight * hard_loss


def _generate_sample(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 20) -> str:
    """Greedy-decode a short continuation for qualitative comparison."""

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_healing(
    config: HealingConfig | None = None,
    model_path: str | Path = _DEFAULT_MODEL_PATH,
    prompt: str = _DEFAULT_PROMPT,
) -> HealingResults:
    """Run the full healing / QAT experiment on real local GPT-2.

    Builds a frozen FP teacher, a frozen "naive" ternary student (reproduces
    the Day-1 collapse), and a "healed" ternary student trained for
    ``config.steps`` distillation steps through the STE. Measures perplexity,
    top-1 agreement and KL for all three, and returns the report contract.
    """

    config = config or HealingConfig()
    if not isinstance(config, HealingConfig):
        raise TypeError("config must be a HealingConfig")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model_path = Path(model_path)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    teacher = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    # Naive ternary student: frozen shadow == original FP weights -> collapse.
    naive = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    naive.eval()
    layers_wrapped, ternary_params, scale_channels = wrap_ternary_student(
        naive, threshold_factor=config.threshold_factor, train_shadow=False
    )

    # Healed ternary student: trainable shadow weights, distilled from teacher.
    student = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    wrap_ternary_student(student, threshold_factor=config.threshold_factor, train_shadow=True)

    eval_ids = _build_eval_batch(tokenizer, config.seq_len, config.seed)
    train_batches = _build_training_batches(tokenizer, config.seq_len, config.steps, config.seed)

    # ----- Train (heal) the student ---------------------------------------
    shadow_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(shadow_params, lr=config.learning_rate)
    student.train()

    first_step_loss = float("nan")
    last_step_loss = float("nan")
    for step, batch in enumerate(train_batches):
        with torch.inference_mode():
            teacher_logits = teacher(batch).logits
        teacher_logits = teacher_logits.clone()  # leave inference-mode guard
        optimizer.zero_grad(set_to_none=True)
        student_logits = student(batch).logits
        loss = _distillation_loss(
            student_logits,
            teacher_logits,
            batch,
            temperature=config.kl_temperature,
            ce_weight=config.ce_weight,
        )
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(shadow_params, config.grad_clip)
        optimizer.step()
        loss_value = float(loss.item())
        if step == 0:
            first_step_loss = loss_value
        last_step_loss = loss_value

    student.eval()

    # ----- Evaluate teacher / naive / healed ------------------------------
    teacher_ppl = _perplexity(teacher, eval_ids)
    naive_ppl = _perplexity(naive, eval_ids)
    healed_ppl = _perplexity(student, eval_ids)

    naive_top1, naive_kl = _agreement_and_kl(teacher, naive, eval_ids)
    healed_top1, healed_kl = _agreement_and_kl(teacher, student, eval_ids)

    teacher_sample = _generate_sample(teacher, tokenizer, prompt)
    naive_sample = _generate_sample(naive, tokenizer, prompt)
    healed_sample = _generate_sample(student, tokenizer, prompt)

    bits_per_weight = compute_bits_per_weight(ternary_params, scale_channels)
    elapsed = time.perf_counter() - started

    return HealingResults(
        teacher_perplexity=teacher_ppl,
        naive_perplexity=naive_ppl,
        healed_perplexity=healed_ppl,
        naive_top1_agreement=naive_top1,
        healed_top1_agreement=healed_top1,
        naive_kl=naive_kl,
        healed_kl=healed_kl,
        first_step_loss=first_step_loss,
        last_step_loss=last_step_loss,
        layers_wrapped=layers_wrapped,
        ternary_params=ternary_params,
        bits_per_weight=bits_per_weight,
        teacher_sample=teacher_sample,
        naive_sample=naive_sample,
        healed_sample=healed_sample,
        elapsed_sec=elapsed,
        model_path=str(model_path.resolve()),
        config=config,
    )


def main(
    model_path: str | Path = _DEFAULT_MODEL_PATH,
    output_path: str | Path = _DEFAULT_OUTPUT,
) -> HealingResults:
    """Run the full heal and write ``healing_results.json``."""

    results = run_healing(model_path=model_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    print("AetherCore v3 Healing / QAT (Day 2)")
    print("-" * 60)
    print(f"  teacher perplexity     : {results.teacher_perplexity:.2f}")
    print(f"  naive   perplexity     : {results.naive_perplexity:.2f}  top1={results.naive_top1_agreement:.3f}  KL={results.naive_kl:.3f}")
    print(f"  healed  perplexity     : {results.healed_perplexity:.2f}  top1={results.healed_top1_agreement:.3f}  KL={results.healed_kl:.3f}")
    print(f"  loss   {results.first_step_loss:.3f} -> {results.last_step_loss:.3f}")
    print(f"  layers wrapped         : {results.layers_wrapped}")
    print(f"  ternary params         : {results.ternary_params:,}")
    print(f"  bits/weight            : {results.bits_per_weight:.4f}")
    print(f"  elapsed                : {results.elapsed_sec:.1f}s")
    print(f"  written                : {out}")
    return results


def _self_test() -> None:
    """Fast structural smoke test of the STE healing mechanism.

    Does NOT load GPT-2. Validates the mechanism on a tiny linear stack:
      1. ternarized weights take only 3 distinct values per channel-scale,
      2. STE gradient flows to the trainable shadow (grad not None, nonzero),
      3. distillation loss decreases over a couple of steps,
      4. bits/weight lands in (2.0, 2.1).
    """

    torch.manual_seed(0)

    # --- 1. Ternarization is genuinely ternary per channel scale ----------
    in_features, out_features = 32, 8
    shadow = torch.randn(in_features, out_features)
    quantized = _ternarize_ste(shadow, threshold_factor=0.7)
    for channel in range(out_features):
        column = quantized[:, channel]
        nonzero = column[column != 0]
        if nonzero.numel():
            abs_values = nonzero.abs()
            if float((abs_values - abs_values[0]).abs().max().item()) > 1.0e-5:
                raise RuntimeError(
                    f"channel {channel} has multiple scale magnitudes, expected 1"
                )
            signs = torch.unique(torch.sign(column))
            if not set(signs.tolist()).issubset({-1.0, 0.0, 1.0}):
                raise RuntimeError(f"channel {channel} produced non-ternary signs: {signs.tolist()}")

    # --- 2. STE gradient flows to the shadow ------------------------------
    param = torch.nn.Parameter(torch.randn(in_features, out_features))
    x = torch.randn(4, in_features)
    out = x.matmul(_ternarize_ste(param, threshold_factor=0.7))
    out.sum().backward()
    if param.grad is None:
        raise RuntimeError("STE failed: shadow gradient is None")
    if float(param.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("STE failed: shadow gradient is all zeros")

    # --- 3. Distillation loss decreases on a tiny teacher/student ---------
    torch.manual_seed(0)
    teacher_net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
    for p in teacher_net.parameters():
        p.requires_grad_(False)

    # Student replaces the two Linear weights with STE-wrapped shadows.
    class _STELinear(nn.Module):
        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            # store as [in, out] to match the Conv1D convention used above
            self.shadow = nn.Parameter(linear.weight.detach().t().clone())
            self.bias = nn.Parameter(linear.bias.detach().clone())

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return t.matmul(_ternarize_ste(self.shadow, 0.7)) + self.bias

    student_net = nn.Sequential(
        _STELinear(teacher_net[0]), nn.GELU(), _STELinear(teacher_net[2])
    )
    optimizer = torch.optim.Adam(student_net.parameters(), lr=5.0e-3)
    inp = torch.randn(8, 16)
    with torch.no_grad():
        teacher_out = teacher_net(inp)

    losses: list[float] = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        student_out = student_net(inp)
        loss = F.mse_loss(student_out, teacher_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_net.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"healing did not reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # --- 4. bits/weight honesty -------------------------------------------
    # GPT-2-small wrapped-layer accounting (12 blocks x 4 linears).
    gpt2_params = 84_934_656
    gpt2_channels = 12 * (2304 + 768 + 3072 + 768)
    bpw = compute_bits_per_weight(gpt2_params, gpt2_channels)
    if not (2.0 < bpw < 2.1):
        raise RuntimeError(f"bits_per_weight {bpw} outside (2.0, 2.1)")

    # --- config / dataclass round-trips -----------------------------------
    config = HealingConfig()
    payload = config.to_dict()
    if payload["threshold_factor"] != 0.7 or payload["steps"] != 30:
        raise RuntimeError("HealingConfig.to_dict mismatch")

    print("AetherCore healing/QAT self-test")
    print(f"  ternary distinct values  : ok (3 per channel scale)")
    print(f"  STE shadow grad norm     : {float(param.grad.norm().item()):.4f}")
    print(f"  distill loss             : {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  gpt2 bits/weight         : {bpw:.4f}")
    print("  status: ok")


__all__ = [
    "HealingConfig",
    "HealingResults",
    "TernaryShadowConv1D",
    "wrap_ternary_student",
    "compute_bits_per_weight",
    "run_healing",
    "main",
]


if __name__ == "__main__":
    main()
