"""Information-theoretic rate-distortion floor for GPT-2 weights.

This module answers an architecture-independent question: given a fixed
bit-budget per weight, what is the *minimum* distortion that *any* compressor
plus correction plus healing scheme can achieve? Information theory supplies
this floor; no real-world method can cross it (the correction table and healing
themselves spend bits inside the same budget, so they can only *approach* it).

The floor is computed from real GPT-2 weights. We gather every 2D weight matrix
(the compressible matrices), measure their distribution, and report two floors
at each compression ratio:

* **Gaussian floor** -- the rate-distortion function of a memoryless Gaussian
  source, ``D = sigma^2 * 2^(-2R)``. In normalized MSE (NMSE = D / sigma^2)
  units this is simply ``NMSE = 2^(-2R)``.
* **Shannon-Lower-Bound (SLB) floor** -- the same bound but using the *measured*
  differential entropy ``h`` (on unit-variance weights), which is slightly below
  the Gaussian entropy because the weights have heavy tails. The SLB is
  ``NMSE = (1 / (2*pi*e)) * 2^(2h) * 2^(-2R)``.

Everything is reported in honest, reproducible numbers and written to
``projects/day1_compression_limit/rate_distortion_limit.json``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_EPS = 1.0e-12

# ``0.5 * log2(2 * pi * e)`` -- differential entropy (in bits) of a unit-variance
# Gaussian. Heavy-tailed real weights sit just under this value.
_GAUSSIAN_UNIT_ENTROPY_BITS = 0.5 * math.log2(2.0 * math.pi * math.e)

# ``1 / (2 * pi * e)`` ~ 0.0585 -- the constant in the Shannon Lower Bound that
# maps a differential entropy to a minimum achievable MSE.
_SLB_CONSTANT = 1.0 / (2.0 * math.pi * math.e)

# FP16 reference storage and the default extrapolation target (the "dream"
# 400B-parameter model the project is sizing against).
_FP16_BITS_PER_WEIGHT = 16.0
_DEFAULT_TARGET_PARAMS = 4.0e11
_BYTES_PER_GIB = float(1 << 30)

# Compression ratios and fidelity targets reported by the day-1 experiment.
_DEFAULT_RATIOS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 50.0, 100.0, 200.0, 400.0)
_DEFAULT_TARGET_NMSE: tuple[tuple[float, str], ...] = (
    (1.0e-1, "10% distortion"),
    (1.0e-2, "1% distortion"),
    (1.0e-3, "0.1% distortion"),
    (1.0e-4, "0.01% distortion"),
    (1.0e-5, "0.001% distortion"),
)

_DEFAULT_MODEL_PATH = "models/gpt2"
_DEFAULT_OUTPUT_DIR = Path("projects") / "day1_compression_limit"


@dataclass(frozen=True, slots=True)
class WeightStatistics:
    """Distributional statistics of the compressible GPT-2 weight matrices."""

    n_params: int
    n_matrices: int
    mean_abs: float
    variance: float
    differential_entropy_bits: float
    gaussianity_gap_bits: float
    kurtosis: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return {
            "n_params": int(self.n_params),
            "n_matrices": int(self.n_matrices),
            "mean_abs": float(self.mean_abs),
            "variance": float(self.variance),
            "differential_entropy_bits": float(self.differential_entropy_bits),
            "gaussianity_gap_bits": float(self.gaussianity_gap_bits),
            "kurtosis": float(self.kurtosis),
        }


@dataclass(frozen=True, slots=True)
class RateDistortionPoint:
    """The achievable distortion floor at one compression ratio."""

    label: str
    bits_per_weight: float
    compression_ratio: float
    size_gb_at_target_params: float
    gaussian_nmse_floor: float
    slb_nmse_floor: float
    gaussian_rms_rel_floor: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return {
            "label": self.label,
            "bits_per_weight": float(self.bits_per_weight),
            "compression_ratio": float(self.compression_ratio),
            "size_gb_at_target_params": float(self.size_gb_at_target_params),
            "gaussian_nmse_floor": float(self.gaussian_nmse_floor),
            "slb_nmse_floor": float(self.slb_nmse_floor),
            "gaussian_rms_rel_floor": float(self.gaussian_rms_rel_floor),
        }


@dataclass(frozen=True, slots=True)
class FidelityPoint:
    """The maximum compression physically permitted to hold a target distortion."""

    target_nmse: float
    target_label: str
    min_bits_per_weight_gaussian: float
    max_compression_gaussian: float
    min_bits_per_weight_slb: float
    max_compression_slb: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return {
            "target_nmse": float(self.target_nmse),
            "target_label": self.target_label,
            "min_bits_per_weight_gaussian": float(self.min_bits_per_weight_gaussian),
            "max_compression_gaussian": float(self.max_compression_gaussian),
            "min_bits_per_weight_slb": float(self.min_bits_per_weight_slb),
            "max_compression_slb": float(self.max_compression_slb),
        }


@dataclass(slots=True)
class RateDistortionReport:
    """Full rate-distortion floor report for a model."""

    model_path: str
    target_params: float
    fp16_bits_per_weight: float
    stats: WeightStatistics
    rate_distortion: list[RateDistortionPoint] = field(default_factory=list)
    fidelity: list[FidelityPoint] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary mirroring rate_distortion_limit.json."""

        return {
            "model_path": self.model_path,
            "target_params": float(self.target_params),
            "fp16_bits_per_weight": float(self.fp16_bits_per_weight),
            "stats": self.stats.to_dict(),
            "rate_distortion": [point.to_dict() for point in self.rate_distortion],
            "fidelity": [point.to_dict() for point in self.fidelity],
            "elapsed_sec": float(self.elapsed_sec),
        }


def _collect_weight_matrices(model: Any) -> list[torch.Tensor]:
    """Gather every 2D floating-point weight matrix from a model state dict.

    These are the *compressible* matrices: attention/MLP projections and the
    token/position embeddings. 1D tensors (biases, LayerNorm gains) are storage
    noise at scale and are excluded, matching the day-1 experiment.
    """

    if not hasattr(model, "state_dict"):
        raise TypeError("model must expose state_dict()")

    matrices: list[torch.Tensor] = []
    for tensor in model.state_dict().values():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not torch.is_floating_point(tensor):
            continue
        if tensor.ndim == 2:
            matrices.append(tensor.detach().cpu().float())
    if not matrices:
        raise ValueError("model contains no 2D floating-point weight matrices")
    return matrices


def _histogram_differential_entropy(
    samples: torch.Tensor,
    bins: int = 256,
    clip_sigma: float = 8.0,
) -> float:
    """Estimate the differential entropy (bits) of unit-variance samples.

    Uses a uniform histogram over a clipped range. For a continuous source the
    differential entropy is ``h = -sum p_i log2(p_i) + log2(dx)`` where ``p_i``
    is the probability mass in bin ``i`` and ``dx`` is the bin width.
    """

    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    if samples.numel() == 0:
        raise ValueError("samples must contain at least one value")

    flat = samples.detach().cpu().float().flatten()
    lo = -float(clip_sigma)
    hi = float(clip_sigma)
    clamped = flat.clamp(lo, hi)
    counts = torch.histc(clamped, bins=bins, min=lo, max=hi)
    total = float(counts.sum().item())
    if total <= 0.0:
        raise ValueError("histogram is empty after clipping")

    probs = (counts / total).clamp_min(_EPS)
    dx = (hi - lo) / bins
    # Differential entropy in bits: discrete entropy of the bins plus log2(dx).
    discrete_entropy = float((-probs * torch.log2(probs)).sum().item())
    return discrete_entropy + math.log2(dx)


def compute_weight_statistics(
    matrices: list[torch.Tensor],
    bins: int = 256,
) -> WeightStatistics:
    """Compute distributional statistics over a list of weight matrices.

    The differential entropy is estimated on the *normalized* (unit-variance)
    weights so it is comparable across layers and directly to the unit-variance
    Gaussian entropy. The non-Gaussianity gap is ``gaussian_entropy - measured``.
    """

    if not matrices:
        raise ValueError("matrices must be a non-empty list")

    flats = [m.detach().cpu().float().flatten() for m in matrices]
    all_weights = torch.cat(flats)
    n_params = int(all_weights.numel())

    mean = float(all_weights.mean().item())
    variance = float(all_weights.var(unbiased=False).item())
    mean_abs = float(all_weights.abs().mean().item())

    centered = all_weights - mean
    std = math.sqrt(max(variance, _EPS))
    normalized = centered / std

    # Excess-free (Pearson) kurtosis: E[(x-mu)^4] / sigma^4.
    fourth_moment = float((normalized.pow(4)).mean().item())
    kurtosis = fourth_moment

    differential_entropy_bits = _histogram_differential_entropy(normalized, bins=bins)
    gaussianity_gap_bits = _GAUSSIAN_UNIT_ENTROPY_BITS - differential_entropy_bits

    return WeightStatistics(
        n_params=n_params,
        n_matrices=len(matrices),
        mean_abs=mean_abs,
        variance=variance,
        differential_entropy_bits=differential_entropy_bits,
        gaussianity_gap_bits=gaussianity_gap_bits,
        kurtosis=kurtosis,
    )


def _gaussian_nmse_floor(bits_per_weight: float) -> float:
    """Gaussian rate-distortion floor in NMSE units: ``2^(-2R)``."""

    return float(2.0 ** (-2.0 * bits_per_weight))


def _slb_nmse_floor(bits_per_weight: float, differential_entropy_bits: float) -> float:
    """Shannon-Lower-Bound NMSE floor using the measured differential entropy."""

    return float(_SLB_CONSTANT * (2.0 ** (2.0 * differential_entropy_bits)) * (2.0 ** (-2.0 * bits_per_weight)))


def _size_gb_at_target_params(target_params: float, bits_per_weight: float) -> float:
    """Storage size in GiB for ``target_params`` weights at a bit-budget."""

    return float(target_params * bits_per_weight / 8.0 / _BYTES_PER_GIB)


def build_rate_distortion_table(
    stats: WeightStatistics,
    target_params: float,
    ratios: tuple[float, ...] = _DEFAULT_RATIOS,
    fp16_bits_per_weight: float = _FP16_BITS_PER_WEIGHT,
) -> list[RateDistortionPoint]:
    """Build the achievable-distortion floor at each compression ratio."""

    if target_params <= 0:
        raise ValueError("target_params must be positive")
    if fp16_bits_per_weight <= 0:
        raise ValueError("fp16_bits_per_weight must be positive")

    points: list[RateDistortionPoint] = []
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError("compression ratios must be positive")
        bits = fp16_bits_per_weight / ratio
        gaussian = _gaussian_nmse_floor(bits)
        slb = _slb_nmse_floor(bits, stats.differential_entropy_bits)
        points.append(
            RateDistortionPoint(
                label=_ratio_label(ratio),
                bits_per_weight=bits,
                compression_ratio=float(ratio),
                size_gb_at_target_params=_size_gb_at_target_params(target_params, bits),
                gaussian_nmse_floor=gaussian,
                slb_nmse_floor=slb,
                gaussian_rms_rel_floor=math.sqrt(gaussian),
            )
        )
    return points


def build_fidelity_table(
    stats: WeightStatistics,
    targets: tuple[tuple[float, str], ...] = _DEFAULT_TARGET_NMSE,
    fp16_bits_per_weight: float = _FP16_BITS_PER_WEIGHT,
) -> list[FidelityPoint]:
    """Build the inverse table: max compression permitted at a target distortion.

    Inverting ``NMSE = 2^(-2R)`` gives ``R = -0.5 * log2(NMSE)`` bits/weight for
    the Gaussian floor. The SLB variant subtracts the non-Gaussianity gain (real
    heavy-tailed weights need slightly fewer bits than a Gaussian).
    """

    if fp16_bits_per_weight <= 0:
        raise ValueError("fp16_bits_per_weight must be positive")

    points: list[FidelityPoint] = []
    for target_nmse, label in targets:
        if not 0.0 < target_nmse < 1.0:
            raise ValueError("target_nmse must be in (0, 1)")
        min_bits_gauss = -0.5 * math.log2(target_nmse)
        # SLB needs fewer bits by exactly the measured non-Gaussianity gain.
        min_bits_slb = min_bits_gauss - stats.gaussianity_gap_bits
        points.append(
            FidelityPoint(
                target_nmse=float(target_nmse),
                target_label=label,
                min_bits_per_weight_gaussian=float(min_bits_gauss),
                max_compression_gaussian=float(fp16_bits_per_weight / min_bits_gauss),
                min_bits_per_weight_slb=float(min_bits_slb),
                max_compression_slb=float(fp16_bits_per_weight / min_bits_slb),
            )
        )
    return points


def _ratio_label(ratio: float) -> str:
    """Format a compression ratio as a short label such as ``100x``."""

    if abs(ratio - round(ratio)) < 1.0e-9:
        return f"{int(round(ratio))}x"
    return f"{ratio:g}x"


def _load_gpt2(model_path: str) -> Any:
    """Load a local GPT-2 checkpoint for weight analysis (CPU only)."""

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return model


def compute_rate_distortion_limit(
    model_path: str = _DEFAULT_MODEL_PATH,
    target_params: float = _DEFAULT_TARGET_PARAMS,
    bins: int = 256,
) -> RateDistortionReport:
    """Load GPT-2, measure its weights, and compute the rate-distortion floor."""

    start = time.perf_counter()
    model = _load_gpt2(model_path)
    matrices = _collect_weight_matrices(model)
    stats = compute_weight_statistics(matrices, bins=bins)
    rate_distortion = build_rate_distortion_table(stats, target_params)
    fidelity = build_fidelity_table(stats)
    elapsed = time.perf_counter() - start

    return RateDistortionReport(
        model_path=model_path,
        target_params=float(target_params),
        fp16_bits_per_weight=_FP16_BITS_PER_WEIGHT,
        stats=stats,
        rate_distortion=rate_distortion,
        fidelity=fidelity,
        elapsed_sec=elapsed,
    )


def _format_markdown(report: RateDistortionReport) -> str:
    """Render the human-readable rate-distortion markdown report."""

    s = report.stats
    lines: list[str] = []
    lines.append("# Mathematical Compression Limit (rate-distortion floor)")
    lines.append("")
    lines.append(
        "**Question:** For any architecture, when a model's weights are stored "
        "in a fixed bit-budget, what is the *minimum* distortion that *any* "
        "compressor + correction + healing scheme can reach? Information theory "
        "gives this floor -- crossing it in the real world is impossible."
    )
    lines.append("")
    lines.append(
        "> **Key fact:** the correction table and healing spend bits that come "
        "out of the same budget. They can only *approach* this floor, never cross it."
    )
    lines.append("")
    lines.append("## Measured weight statistics (real GPT-2)")
    lines.append("")
    lines.append(f"- Compressible params: {s.n_params:,} across {s.n_matrices} matrices")
    lines.append(f"- Differential entropy (unit-variance): **{s.differential_entropy_bits:.3f} bits**")
    lines.append(f"- Gaussian entropy (unit-variance): {_GAUSSIAN_UNIT_ENTROPY_BITS:.3f} bits")
    lines.append(
        f"- Non-Gaussianity gain: **{s.gaussianity_gap_bits:.3f} bits/weight** "
        f"(kurtosis {s.kurtosis:.2f}; >3 = heavier tails => slightly more compressible than Gaussian)"
    )
    lines.append("")
    lines.append("## The rate-distortion floor (best case for ANY method)")
    lines.append("")
    lines.append(
        "`NMSE` = normalized weight MSE = D / sigma^2 (0 = perfect, 1 = signal "
        "fully lost). `RMS rel` = sqrt(NMSE) = best-possible relative RMS weight error."
    )
    lines.append("")
    lines.append("| Ratio | bits/weight | 400B size | NMSE floor (Gauss) | NMSE floor (SLB) | best RMS rel err |")
    lines.append("|------:|------------:|----------:|-------------------:|-----------------:|-----------------:|")
    for p in report.rate_distortion:
        lines.append(
            f"| {p.label} | {p.bits_per_weight:.3f} | {p.size_gb_at_target_params:.2f} GB | "
            f"{p.gaussian_nmse_floor:.4f} | {p.slb_nmse_floor:.4f} | {p.gaussian_rms_rel_floor * 100:.2f}% |"
        )
    lines.append("")
    lines.append("## Inverse view: high fidelity FORCES low compression")
    lines.append("")
    lines.append(
        "To hold weight distortion at or below a target, this is the *maximum* "
        "compression physically permitted -- no correction/healing can exceed it."
    )
    lines.append("")
    lines.append("| Target weight distortion | min bits/weight (Gauss) | max ratio (Gauss) | max ratio (SLB) |")
    lines.append("|:-------------------------|------------------------:|------------------:|----------------:|")
    for f in report.fidelity:
        lines.append(
            f"| {f.target_label} (NMSE {f.target_nmse:.0e}) | "
            f"{f.min_bits_per_weight_gaussian:.2f} | {f.max_compression_gaussian:.2f}x | "
            f"{f.max_compression_slb:.2f}x |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: RateDistortionReport,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    write_markdown: bool = True,
) -> Path:
    """Write the JSON report (and optionally markdown) to ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rate_distortion_limit.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if write_markdown:
        md_path = output_dir / "rate_distortion_limit.md"
        md_path.write_text(_format_markdown(report), encoding="utf-8")
    return json_path


def main(argv: list[str] | None = None) -> RateDistortionReport:
    """CLI entry point: compute the floor, print a summary, and write outputs."""

    import argparse

    parser = argparse.ArgumentParser(description="Rate-distortion floor for GPT-2 weights")
    parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--target-params", type=float, default=_DEFAULT_TARGET_PARAMS)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--bins", type=int, default=256)
    args = parser.parse_args(argv)

    report = compute_rate_distortion_limit(
        model_path=args.model_path,
        target_params=args.target_params,
        bins=args.bins,
    )
    json_path = write_outputs(
        report,
        output_dir=Path(args.output_dir),
        write_markdown=not args.no_markdown,
    )

    s = report.stats
    print("Rate-distortion floor (GPT-2 weights)")
    print(f"  compressible params : {s.n_params:,} across {s.n_matrices} matrices")
    print(f"  mean |w|            : {s.mean_abs:.6f}")
    print(f"  variance            : {s.variance:.6f}")
    print(f"  diff entropy (bits) : {s.differential_entropy_bits:.6f}")
    print(f"  gaussianity gap     : {s.gaussianity_gap_bits:.6f}")
    print(f"  kurtosis            : {s.kurtosis:.4f}")
    print("  ratio | bits/w | gauss NMSE | slb NMSE | rms rel")
    for p in report.rate_distortion:
        print(
            f"  {p.label:>5} | {p.bits_per_weight:6.3f} | {p.gaussian_nmse_floor:10.6f} | "
            f"{p.slb_nmse_floor:8.6f} | {p.gaussian_rms_rel_floor:7.4f}"
        )
    print(f"  elapsed: {report.elapsed_sec:.2f}s")
    print(f"  wrote: {json_path}")
    return report


def _self_test() -> None:
    """Validate formulas (synthetic) and end-to-end numbers against the contract."""

    # --- Formula checks on synthetic statistics (no model load). ---
    fake_stats = WeightStatistics(
        n_params=162_915_840,
        n_matrices=51,
        mean_abs=0.1048,
        variance=0.01876,
        differential_entropy_bits=2.040150499335402,
        gaussianity_gap_bits=_GAUSSIAN_UNIT_ENTROPY_BITS - 2.040150499335402,
        kurtosis=12.24,
    )
    rd = build_rate_distortion_table(fake_stats, _DEFAULT_TARGET_PARAMS)
    by_label = {p.label: p for p in rd}

    p2 = by_label["2x"]
    if not (1.52e-05 < p2.gaussian_nmse_floor < 1.53e-05):
        raise RuntimeError(f"2x gaussian floor off: {p2.gaussian_nmse_floor}")

    p100 = by_label["100x"]
    if not (0.79 < p100.gaussian_nmse_floor < 0.81):
        raise RuntimeError(f"100x gaussian floor off: {p100.gaussian_nmse_floor}")
    if not (0.78 < p100.slb_nmse_floor < 0.80):
        raise RuntimeError(f"100x slb floor off: {p100.slb_nmse_floor}")
    if abs(p100.bits_per_weight - 0.16) > 1.0e-9:
        raise RuntimeError(f"100x bits/weight off: {p100.bits_per_weight}")
    if not (7.4 < p100.size_gb_at_target_params < 7.5):
        raise RuntimeError(f"100x size GiB off: {p100.size_gb_at_target_params}")
    if not (0.89 < p100.gaussian_rms_rel_floor < 0.90):
        raise RuntimeError(f"100x rms rel off: {p100.gaussian_rms_rel_floor}")

    fid = build_fidelity_table(fake_stats)
    fid_by_nmse = {f.target_nmse: f for f in fid}
    f5 = fid_by_nmse[1.0e-5]
    if not (8.30 < f5.min_bits_per_weight_gaussian < 8.31):
        raise RuntimeError(f"1e-5 min bits off: {f5.min_bits_per_weight_gaussian}")
    if not (1.92 < f5.max_compression_gaussian < 1.93):
        raise RuntimeError(f"1e-5 max ratio off: {f5.max_compression_gaussian}")

    # --- End-to-end check on the real model (entropy + floors). ---
    report = compute_rate_distortion_limit()
    s = report.stats
    if not (2.0 < s.differential_entropy_bits < 2.1):
        raise RuntimeError(f"measured entropy out of range: {s.differential_entropy_bits}")
    if not (160_000_000 < s.n_params < 165_000_000):
        raise RuntimeError(f"n_params unexpected: {s.n_params}")
    if s.n_matrices != 51:
        raise RuntimeError(f"expected 51 matrices, got {s.n_matrices}")
    if not (10.0 < s.kurtosis < 14.0):
        raise RuntimeError(f"kurtosis out of range: {s.kurtosis}")
    real100 = {p.label: p for p in report.rate_distortion}["100x"]
    if not (0.79 < real100.gaussian_nmse_floor < 0.81):
        raise RuntimeError(f"real 100x gaussian floor off: {real100.gaussian_nmse_floor}")

    write_outputs(report)
    print("rate_distortion_limit self-test")
    print(f"  n_params={s.n_params:,} matrices={s.n_matrices}")
    print(f"  diff entropy={s.differential_entropy_bits:.4f} gap={s.gaussianity_gap_bits:.5f} kurtosis={s.kurtosis:.2f}")
    print(f"  100x gaussian NMSE={real100.gaussian_nmse_floor:.4f} slb={real100.slb_nmse_floor:.4f}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
