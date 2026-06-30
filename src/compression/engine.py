"""Compression engine for AetherCore v3.

This module implements CPU-friendly compression building blocks for tensor
weights. The code is intentionally practical: it stores enough metadata to
round-trip tensors, gives honest error and size estimates, and avoids claiming
hardware-independent miracles that cannot be verified from a local file.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


_EPS = 1.0e-12


@dataclass(slots=True)
class QualityMetrics:
    """Numerical quality report comparing original and reconstructed tensors."""

    mse: float
    mae: float
    max_abs_error: float
    relative_l2_error: float
    cosine_similarity: float
    quality_score: float


@dataclass(slots=True)
class CompressionStats:
    """Summary returned after compressing a model or tensor collection."""

    original_bytes: int
    estimated_compressed_bytes: int
    actual_serialized_bytes: int
    compression_ratio_estimated: float
    compression_ratio_serialized: float
    layers_compressed: int
    layers_deduplicated: int
    average_bits_per_weight: float
    average_relative_error: float
    quality_score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompressedLayer:
    """Serializable representation of a compressed tensor layer."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dynamic_data: dict[str, Any]
    correction_table: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a torch.save-friendly dictionary."""

        return {
            "format": "aethercore_v3.compressed_layer",
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "dynamic_data": self.dynamic_data,
            "correction_table": self.correction_table,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompressedLayer":
        """Build a compressed layer from a dictionary payload."""

        required = {"name", "shape", "dtype", "dynamic_data", "correction_table"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Compressed layer payload missing keys: {sorted(missing)}")

        return cls(
            name=str(payload["name"]),
            shape=tuple(int(v) for v in payload["shape"]),
            dtype=str(payload["dtype"]),
            dynamic_data=dict(payload["dynamic_data"]),
            correction_table=dict(payload["correction_table"]),
            metadata=dict(payload.get("metadata", {})),
        )


def _ensure_float_tensor(weight: torch.Tensor, name: str = "weight") -> torch.Tensor:
    """Validate that a tensor can be processed by the compression pipeline."""

    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(weight)!r}")
    if not torch.is_floating_point(weight):
        raise TypeError(f"{name} must be a floating point tensor, got {weight.dtype}")
    if weight.numel() == 0:
        raise ValueError(f"{name} must contain at least one value")
    return weight.detach()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    """Return the physical byte count for a tensor."""

    return int(tensor.numel() * tensor.element_size())


def _ceil_div(value: int, divisor: int) -> int:
    """Integer ceil division."""

    return (value + divisor - 1) // divisor


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    """Convert a string such as ``torch.float32`` back to a torch dtype."""

    name = dtype_name.split(".")[-1]
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.float32


def _safe_layer_filename(name: str) -> str:
    """Create a stable Windows-safe filename from a state-dict key."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "layer"


def _quantize_values(values: torch.Tensor, bit_width: int) -> tuple[torch.Tensor, float]:
    """Symmetrically quantize a 1-D tensor into signed integer levels."""

    if values.numel() == 0:
        return torch.empty(0, dtype=torch.int8), 1.0

    values = values.detach().float()
    if bit_width <= 1:
        scale = float(values.abs().mean().clamp_min(_EPS).item())
        quantized = torch.sign(values).to(torch.int8)
        return quantized, scale

    max_level = (2 ** (bit_width - 1)) - 1
    scale = float((values.abs().max() / max_level).clamp_min(_EPS).item())
    quantized = torch.round(values / scale).clamp(-max_level, max_level).to(torch.int8)
    return quantized, scale


def _dequantize_values(values: torch.Tensor, scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize signed integer levels with a scalar scale."""

    return values.to(dtype=torch.float32).mul(float(scale)).to(dtype=dtype)


def _recursive_tensor_bits(value: Any) -> int:
    """Estimate serialized tensor payload bits for nested dictionaries."""

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size() * 8)
    if isinstance(value, Mapping):
        return sum(_recursive_tensor_bits(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_recursive_tensor_bits(item) for item in value)
    if isinstance(value, (float, int, bool)):
        return 64
    return 0


def _pack_2bit(values: torch.Tensor) -> torch.Tensor:
    """Pack uint values in [0, 3] into bytes, four values per byte."""

    flat = values.detach().cpu().to(torch.uint8).flatten()
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    if int(flat.max().item()) > 3:
        raise ValueError("2-bit packing values must be in [0, 3]")
    padded_count = _ceil_div(int(flat.numel()), 4) * 4
    if padded_count != flat.numel():
        padding = torch.zeros(padded_count - flat.numel(), dtype=torch.uint8)
        flat = torch.cat((flat, padding), dim=0)
    chunks = flat.reshape(-1, 4)
    return (
        chunks[:, 0]
        | (chunks[:, 1] << 2)
        | (chunks[:, 2] << 4)
        | (chunks[:, 3] << 6)
    ).contiguous()


def _unpack_2bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack bytes into uint values in [0, 3]."""

    if count < 0:
        raise ValueError("count must be non-negative")
    data = packed.detach().cpu().to(torch.uint8).flatten()
    if data.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    unpacked = torch.empty(int(data.numel()) * 4, dtype=torch.uint8)
    unpacked[0::4] = data & 0b00000011
    unpacked[1::4] = (data >> 2) & 0b00000011
    unpacked[2::4] = (data >> 4) & 0b00000011
    unpacked[3::4] = (data >> 6) & 0b00000011
    return unpacked[:count].contiguous()


class TernaryQuantizer:
    """Quantize weights to ``{-1, 0, +1}`` plus a scale factor."""

    def __init__(self, threshold_factor: float = 0.7, min_scale: float = 1.0e-8) -> None:
        if threshold_factor < 0:
            raise ValueError("threshold_factor must be non-negative")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.threshold_factor = float(threshold_factor)
        self.min_scale = float(min_scale)

    def quantize(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ternary values and a scalar scale tensor."""

        weight = _ensure_float_tensor(weight).float()
        abs_mean = weight.abs().mean()
        threshold = self.threshold_factor * abs_mean

        ternary = torch.zeros_like(weight, dtype=torch.int8)
        ternary = torch.where(weight > threshold, torch.ones_like(ternary), ternary)
        ternary = torch.where(weight < -threshold, -torch.ones_like(ternary), ternary)

        selected = weight.abs()[ternary != 0]
        if selected.numel() == 0:
            scale_value = max(float(abs_mean.item()), self.min_scale)
        else:
            scale_value = max(float(selected.mean().item()), self.min_scale)

        scale = torch.tensor(scale_value, dtype=torch.float32, device=weight.device)
        return ternary, scale

    def dequantize(self, ternary: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
        """Reconstruct a floating tensor from ternary values and scale."""

        if not isinstance(ternary, torch.Tensor):
            raise TypeError("ternary must be a torch.Tensor")
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=ternary.device)
        return ternary.to(dtype=torch.float32).mul(scale_tensor)

    def sparsity(self, ternary: torch.Tensor) -> float:
        """Return the fraction of zero entries in a ternary tensor."""

        if not isinstance(ternary, torch.Tensor):
            raise TypeError("ternary must be a torch.Tensor")
        if ternary.numel() == 0:
            return 1.0
        return float((ternary == 0).sum().item() / ternary.numel())


class DynamicBitAllocator:
    """Assign 4-bit, 2-bit, and 1-bit storage by tensor importance."""

    def __init__(
        self,
        critical_fraction: float = 0.05,
        important_fraction: float = 0.10,
        min_scale: float = 1.0e-8,
    ) -> None:
        if not 0 <= critical_fraction <= 1:
            raise ValueError("critical_fraction must be in [0, 1]")
        if not 0 <= important_fraction <= 1:
            raise ValueError("important_fraction must be in [0, 1]")
        if critical_fraction + important_fraction > 1:
            raise ValueError("critical_fraction + important_fraction must be <= 1")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.critical_fraction = float(critical_fraction)
        self.important_fraction = float(important_fraction)
        self.min_scale = float(min_scale)

    def analyze_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """Return normalized importance scores in ``[0, 1]``."""

        weight = _ensure_float_tensor(weight).float()
        magnitude = weight.abs()
        if weight.ndim == 2:
            row_energy = magnitude.mean(dim=1, keepdim=True)
            col_energy = magnitude.mean(dim=0, keepdim=True)
            importance = 0.70 * magnitude + 0.15 * row_energy + 0.15 * col_energy
        else:
            importance = magnitude

        max_value = importance.max()
        if float(max_value.item()) <= _EPS:
            return torch.zeros_like(importance)
        return importance / max_value

    def allocate_bits(self, weight: torch.Tensor) -> dict[str, Any]:
        """Return boolean masks for critical, important, and normal values."""

        importance = self.analyze_importance(weight)
        flat = importance.flatten()
        total = int(flat.numel())
        if total == 0 or float(flat.max().item()) <= _EPS:
            false_mask = torch.zeros_like(flat, dtype=torch.bool).reshape_as(importance)
            return {
                "critical": false_mask.clone(),
                "important": false_mask.clone(),
                "normal": false_mask.clone(),
                "bit_map": torch.zeros_like(importance, dtype=torch.uint8),
                "average_bits": 0.0,
            }

        ranked = torch.argsort(flat, descending=True)
        critical_count = min(total, int(round(total * self.critical_fraction)))
        if self.critical_fraction > 0 and critical_count == 0:
            critical_count = 1
        important_count = min(total - critical_count, int(round(total * self.important_fraction)))
        normal_count = total - critical_count - important_count

        flat_bits = torch.zeros(total, dtype=torch.uint8, device=weight.device)
        if critical_count:
            flat_bits[ranked[:critical_count]] = 4
        if important_count:
            start = critical_count
            flat_bits[ranked[start : start + important_count]] = 2
        if normal_count:
            start = critical_count + important_count
            flat_bits[ranked[start:]] = 1

        bit_map = flat_bits.reshape_as(importance)
        average_bits = float(flat_bits.float().mean().item())
        return {
            "critical": bit_map == 4,
            "important": bit_map == 2,
            "normal": bit_map == 1,
            "bit_map": bit_map,
            "average_bits": average_bits,
        }

    def compress(self, weight: torch.Tensor) -> dict[str, Any]:
        """Compress a tensor into grouped signed integer values."""

        weight = _ensure_float_tensor(weight)
        allocations = self.allocate_bits(weight)
        flat_weight = weight.detach().float().flatten()
        groups: dict[str, dict[str, Any]] = {}

        for group_name, bit_width in (("critical", 4), ("important", 2), ("normal", 1)):
            mask = allocations[group_name].flatten()
            indices = torch.nonzero(mask, as_tuple=False).flatten().cpu().to(torch.int64)
            selected = flat_weight.index_select(0, indices.to(device=flat_weight.device))
            quantized, scale = _quantize_values(selected, bit_width)
            groups[group_name] = {
                "bit_width": bit_width,
                "indices": indices,
                "values": quantized.cpu(),
                "scale": max(float(scale), self.min_scale),
            }

        bit_estimate = self.estimated_bits_from_groups(groups, int(weight.numel()))
        return {
            "format": "aethercore_v3.dynamic_bits",
            "shape": tuple(int(v) for v in weight.shape),
            "dtype": str(weight.dtype),
            "groups": groups,
            "average_bits_data_only": float(allocations["average_bits"]),
            "estimated_bits": bit_estimate,
        }

    def decompress(
        self,
        compressed: Mapping[str, Any],
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Reconstruct a tensor from grouped dynamic-bit data."""

        shape = tuple(int(v) for v in compressed["shape"])
        target_dtype = dtype or _dtype_from_name(str(compressed.get("dtype", "torch.float32")))
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        total = math.prod(shape)
        flat = torch.zeros(total, dtype=target_dtype, device=target_device)

        groups = compressed.get("groups", {})
        for group in groups.values():
            indices = group["indices"].to(device=target_device, dtype=torch.long)
            values = group["values"].to(device=target_device)
            scale = float(group["scale"])
            restored = _dequantize_values(values, scale, target_dtype)
            if indices.numel() != restored.numel():
                raise ValueError("Compressed group has mismatched indices and values")
            flat.index_copy_(0, indices, restored)

        return flat.reshape(shape)

    def estimated_bits_from_groups(self, groups: Mapping[str, Any], total_values: int) -> dict[str, int]:
        """Estimate data and sparse-index bit usage for compressed groups."""

        index_width = max(1, math.ceil(math.log2(max(2, total_values))))
        data_bits = 0
        index_bits = 0
        for group in groups.values():
            values = group["values"]
            indices = group["indices"]
            bit_width = int(group["bit_width"])
            data_bits += int(values.numel() * bit_width)
            index_bits += int(indices.numel() * index_width)
        return {
            "data_bits": data_bits,
            "index_bits": index_bits,
            "total_bits": data_bits + index_bits,
        }


class PackedTernaryCompressor:
    """Dense packed ternary compressor with 2-bit values and block scales."""

    def __init__(self, block_size: int = 256, threshold_factor: float = 0.7, scale_dtype_bytes: int = 2) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if threshold_factor < 0:
            raise ValueError("threshold_factor must be non-negative")
        if scale_dtype_bytes not in {2, 4}:
            raise ValueError("scale_dtype_bytes must be 2 or 4")
        self.block_size = int(block_size)
        self.threshold_factor = float(threshold_factor)
        self.scale_dtype_bytes = int(scale_dtype_bytes)

    def compress(self, weight: torch.Tensor) -> dict[str, Any]:
        """Compress a floating tensor into dense packed ternary blocks."""

        source = _ensure_float_tensor(weight).detach().cpu().float()
        flat = source.flatten()
        total = int(flat.numel())
        block_count = _ceil_div(total, self.block_size)
        padded_total = block_count * self.block_size
        if padded_total != total:
            flat = torch.cat((flat, torch.zeros(padded_total - total, dtype=torch.float32)), dim=0)
        blocks = flat.reshape(block_count, self.block_size)

        abs_mean = blocks.abs().mean(dim=1).clamp_min(_EPS)
        thresholds = abs_mean * self.threshold_factor
        ternary = torch.zeros_like(blocks, dtype=torch.int8)
        ternary = torch.where(blocks > thresholds.view(-1, 1), torch.ones_like(ternary), ternary)
        ternary = torch.where(blocks < -thresholds.view(-1, 1), -torch.ones_like(ternary), ternary)

        selected_abs = torch.where(ternary != 0, blocks.abs(), torch.zeros_like(blocks))
        selected_count = (ternary != 0).sum(dim=1).clamp_min(1)
        scales = (selected_abs.sum(dim=1) / selected_count).clamp_min(_EPS)

        codes = (ternary + 1).to(torch.uint8).flatten()
        packed = _pack_2bit(codes)
        scale_tensor = scales.to(torch.float16 if self.scale_dtype_bytes == 2 else torch.float32).contiguous()
        data_bits = int(total * 2)
        scale_bits = int(block_count * self.scale_dtype_bytes * 8)
        return {
            "format": "aethercore_v3.packed_ternary",
            "shape": tuple(int(v) for v in source.shape),
            "dtype": str(weight.dtype),
            "block_size": self.block_size,
            "total_values": total,
            "padded_values": padded_total,
            "packed_values": packed,
            "scales": scale_tensor,
            "estimated_bits": {
                "data_bits": data_bits,
                "scale_bits": scale_bits,
                "metadata_bits": 256,
                "total_bits": data_bits + scale_bits + 256,
            },
            "average_bits_per_weight": (data_bits + scale_bits + 256) / max(1, total),
        }

    def decompress(
        self,
        compressed: Mapping[str, Any],
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Decompress a packed ternary tensor."""

        if compressed.get("format") != "aethercore_v3.packed_ternary":
            raise ValueError("Unsupported packed ternary format")
        total = int(compressed["total_values"])
        padded_total = int(compressed["padded_values"])
        block_size = int(compressed["block_size"])
        block_count = _ceil_div(padded_total, block_size)
        codes = _unpack_2bit(compressed["packed_values"], padded_total).to(torch.int8)
        ternary = (codes - 1).reshape(block_count, block_size).float()
        scales = compressed["scales"].detach().cpu().float().reshape(-1, 1)
        flat = (ternary * scales).flatten()[:total]
        shape = tuple(int(v) for v in compressed["shape"])
        target_dtype = dtype or _dtype_from_name(str(compressed.get("dtype", "torch.float32")))
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        return flat.reshape(shape).to(device=target_device, dtype=target_dtype)

    def estimate_model_size(self, parameter_count: int) -> dict[str, float]:
        """Estimate packed ternary model size for a parameter count."""

        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        blocks = _ceil_div(int(parameter_count), self.block_size)
        total_bits = int(parameter_count) * 2 + blocks * self.scale_dtype_bytes * 8
        total_bytes = _ceil_div(total_bits, 8)
        fp16_bytes = int(parameter_count) * 2
        return {
            "parameters": float(parameter_count),
            "fp16_gb": fp16_bytes / 1_000_000_000,
            "packed_gb": total_bytes / 1_000_000_000,
            "compression_ratio": fp16_bytes / max(1, total_bytes),
            "average_bits_per_weight": total_bits / parameter_count,
        }


class DeltaCompressor:
    """Quantize deltas between similarly shaped layer tensors."""

    def __init__(self, bit_width: int = 8) -> None:
        if bit_width < 2 or bit_width > 8:
            raise ValueError("bit_width must be between 2 and 8")
        self.bit_width = int(bit_width)

    def compute_deltas(self, layers_list: list[torch.Tensor]) -> dict[str, Any]:
        """Return a base tensor plus quantized deltas for subsequent layers."""

        if not layers_list:
            raise ValueError("layers_list must contain at least one tensor")

        base = _ensure_float_tensor(layers_list[0], "layers_list[0]").detach().cpu().float()
        previous = base
        entries: list[dict[str, Any]] = []

        for index, layer in enumerate(layers_list[1:], start=1):
            current = _ensure_float_tensor(layer, f"layers_list[{index}]").detach().cpu().float()
            if current.shape != previous.shape:
                entries.append(
                    {
                        "kind": "reset",
                        "shape": tuple(int(v) for v in current.shape),
                        "value": current,
                    }
                )
                previous = current
                continue

            delta = current - previous
            quantized, scale = _quantize_values(delta.flatten(), self.bit_width)
            entries.append(
                {
                    "kind": "delta",
                    "shape": tuple(int(v) for v in current.shape),
                    "values": quantized.reshape_as(current).cpu(),
                    "scale": float(scale),
                }
            )
            previous = current

        return {
            "format": "aethercore_v3.delta_layers",
            "bit_width": self.bit_width,
            "base": base,
            "entries": entries,
        }

    def reconstruct(self, delta_dict: Mapping[str, Any]) -> list[torch.Tensor]:
        """Reconstruct layers from a delta dictionary."""

        if "base" not in delta_dict:
            raise ValueError("delta_dict missing base tensor")

        layers = [delta_dict["base"].detach().clone().float()]
        previous = layers[0]
        for entry in delta_dict.get("entries", []):
            kind = entry.get("kind")
            if kind == "reset":
                current = entry["value"].detach().clone().float()
            elif kind == "delta":
                delta = _dequantize_values(entry["values"], float(entry["scale"]), torch.float32)
                current = previous + delta
            else:
                raise ValueError(f"Unknown delta entry kind: {kind!r}")
            layers.append(current)
            previous = current
        return layers


class SemanticDeduplicator:
    """Find nearly identical tensors using normalized pooled fingerprints."""

    def __init__(self, similarity_threshold: float = 0.95, fingerprint_size: int = 256) -> None:
        if not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if fingerprint_size <= 0:
            raise ValueError("fingerprint_size must be positive")
        self.similarity_threshold = float(similarity_threshold)
        self.fingerprint_size = int(fingerprint_size)

    def fingerprint(self, weight: torch.Tensor) -> torch.Tensor:
        """Return a fixed-size normalized tensor fingerprint."""

        weight = _ensure_float_tensor(weight).detach().cpu().float().flatten()
        if weight.numel() >= self.fingerprint_size:
            pooled = F.adaptive_avg_pool1d(
                weight.view(1, 1, -1),
                output_size=self.fingerprint_size,
            ).view(-1)
        else:
            pooled = torch.zeros(self.fingerprint_size, dtype=torch.float32)
            pooled[: weight.numel()] = weight

        pooled = pooled - pooled.mean()
        norm = pooled.norm()
        if float(norm.item()) <= _EPS:
            return torch.zeros_like(pooled)
        return pooled / norm

    def find_duplicates(self, weights_dict: Mapping[str, torch.Tensor]) -> dict[str, str]:
        """Map every tensor name to its canonical representative."""

        canonical_fingerprints: dict[str, torch.Tensor] = {}
        dedup_map: dict[str, str] = {}

        for name, weight in weights_dict.items():
            fingerprint = self.fingerprint(weight)
            best_name = name
            best_similarity = -1.0

            for canonical_name, canonical_fp in canonical_fingerprints.items():
                similarity = float(torch.dot(fingerprint, canonical_fp).item())
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = canonical_name

            if best_similarity >= self.similarity_threshold:
                dedup_map[name] = best_name
            else:
                canonical_fingerprints[name] = fingerprint
                dedup_map[name] = name

        return dedup_map


class CorrectionTableExtractor:
    """Extract a tiny residual table to improve compressed reconstructions."""

    def __init__(self, max_fraction: float = 0.01, max_rank: int = 16, max_svd_elements: int = 4_000_000) -> None:
        if max_fraction <= 0:
            raise ValueError("max_fraction must be positive")
        if max_rank < 0:
            raise ValueError("max_rank must be non-negative")
        if max_svd_elements <= 0:
            raise ValueError("max_svd_elements must be positive")
        self.max_fraction = float(max_fraction)
        self.max_rank = int(max_rank)
        self.max_svd_elements = int(max_svd_elements)

    def extract(self, original_weight: torch.Tensor, compressed_weight: torch.Tensor) -> dict[str, Any]:
        """Create a low-rank residual correction table when budget allows."""

        original = _ensure_float_tensor(original_weight, "original_weight").detach().cpu().float()
        compressed = _ensure_float_tensor(compressed_weight, "compressed_weight").detach().cpu().float()
        if original.shape != compressed.shape:
            raise ValueError("original_weight and compressed_weight must have identical shapes")

        residual = original - compressed
        table: dict[str, Any] = {
            "format": "aethercore_v3.correction_table",
            "shape": tuple(int(v) for v in original.shape),
            "max_fraction": self.max_fraction,
            "scalar_bias": float(residual.mean().item()),
            "kind": "scalar",
            "rank": 0,
        }

        if original.ndim != 2 or self.max_rank == 0:
            return table

        rows, cols = int(original.shape[0]), int(original.shape[1])
        allowed_parameters = max(1, int(original.numel() * self.max_fraction))
        parameters_per_rank = rows + cols + 1
        rank = min(self.max_rank, rows, cols, allowed_parameters // parameters_per_rank)
        if rank <= 0 or residual.numel() > self.max_svd_elements:
            return table

        try:
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        except RuntimeError:
            return table

        table.update(
            {
                "kind": "low_rank",
                "rank": int(rank),
                "u": u[:, :rank].contiguous().to(torch.float16),
                "s": s[:rank].contiguous().to(torch.float32),
                "v": vh[:rank, :].contiguous().to(torch.float16),
            }
        )
        return table

    def apply(
        self,
        compressed_output: torch.Tensor,
        correction_table: Mapping[str, Any],
        input_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply output correction.

        When ``input_tensor`` is provided for a linear layer ``x @ W.T``, a
        low-rank residual weight correction is projected into output space.
        Without input activations, a scalar bias correction is applied.
        """

        if not isinstance(compressed_output, torch.Tensor):
            raise TypeError("compressed_output must be a torch.Tensor")

        corrected = compressed_output
        if correction_table.get("kind") == "low_rank" and input_tensor is not None:
            residual = self._residual_from_table(correction_table).to(
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            correction = input_tensor.matmul(residual.t())
            corrected = corrected + correction.to(device=corrected.device, dtype=corrected.dtype)

        scalar_bias = float(correction_table.get("scalar_bias", 0.0))
        if scalar_bias:
            corrected = corrected + torch.as_tensor(scalar_bias, device=corrected.device, dtype=corrected.dtype)
        return corrected

    def apply_to_weight(self, compressed_weight: torch.Tensor, correction_table: Mapping[str, Any]) -> torch.Tensor:
        """Apply residual correction directly to a decompressed weight tensor."""

        if not isinstance(compressed_weight, torch.Tensor):
            raise TypeError("compressed_weight must be a torch.Tensor")

        corrected = compressed_weight
        if correction_table.get("kind") == "low_rank":
            residual = self._residual_from_table(correction_table).to(
                device=compressed_weight.device,
                dtype=compressed_weight.dtype,
            )
            corrected = corrected + residual

        scalar_bias = float(correction_table.get("scalar_bias", 0.0))
        if scalar_bias:
            corrected = corrected + torch.as_tensor(scalar_bias, device=corrected.device, dtype=corrected.dtype)
        return corrected

    def _residual_from_table(self, correction_table: Mapping[str, Any]) -> torch.Tensor:
        """Reconstruct a low-rank residual matrix from a correction table."""

        if correction_table.get("kind") != "low_rank":
            shape = tuple(int(v) for v in correction_table["shape"])
            return torch.zeros(shape, dtype=torch.float32)

        u = correction_table["u"].float()
        s = correction_table["s"].float()
        v = correction_table["v"].float()
        return (u * s.unsqueeze(0)).matmul(v)


class GodCompressionEngine:
    """Orchestrate tensor compression, saving, loading, and benchmarking."""

    def __init__(
        self,
        quantizer: TernaryQuantizer | None = None,
        bit_allocator: DynamicBitAllocator | None = None,
        delta_compressor: DeltaCompressor | None = None,
        deduplicator: SemanticDeduplicator | None = None,
        correction_extractor: CorrectionTableExtractor | None = None,
        packed_compressor: PackedTernaryCompressor | None = None,
    ) -> None:
        self.quantizer = quantizer or TernaryQuantizer()
        self.bit_allocator = bit_allocator or DynamicBitAllocator()
        self.delta_compressor = delta_compressor or DeltaCompressor()
        self.deduplicator = deduplicator or SemanticDeduplicator()
        self.correction_extractor = correction_extractor or CorrectionTableExtractor()
        self.packed_compressor = packed_compressor or PackedTernaryCompressor()

    def compress_layer_packed(self, layer_weights: torch.Tensor, name: str = "layer") -> CompressedLayer:
        """Compress a layer with dense packed ternary storage."""

        weight = _ensure_float_tensor(layer_weights, "layer_weights")
        packed_data = self.packed_compressor.compress(weight)
        reconstructed = self.packed_compressor.decompress(packed_data, dtype=torch.float32)
        correction_table = self.correction_extractor.extract(weight, reconstructed)
        quality = self.benchmark(weight, reconstructed)
        metadata = {
            "original_bytes": _tensor_bytes(weight),
            "estimated_bits": self._estimated_layer_bits(packed_data, correction_table),
            "packed_average_bits_per_weight": packed_data["average_bits_per_weight"],
            "quality": asdict(quality),
        }
        return CompressedLayer(
            name=name,
            shape=tuple(int(v) for v in weight.shape),
            dtype=str(weight.dtype),
            dynamic_data=packed_data,
            correction_table=correction_table,
            metadata=metadata,
        )

    def compress_layer(self, layer_weights: torch.Tensor, name: str = "layer") -> CompressedLayer:
        """Compress a single floating point tensor."""

        weight = _ensure_float_tensor(layer_weights, "layer_weights")
        dynamic_data = self.bit_allocator.compress(weight)
        reconstructed = self.bit_allocator.decompress(dynamic_data, dtype=torch.float32)
        correction_table = self.correction_extractor.extract(weight, reconstructed)

        ternary, ternary_scale = self.quantizer.quantize(weight)
        quality = self.benchmark(weight, CompressedLayer(
            name=name,
            shape=tuple(int(v) for v in weight.shape),
            dtype=str(weight.dtype),
            dynamic_data=dynamic_data,
            correction_table=correction_table,
            metadata={},
        ))

        metadata = {
            "original_bytes": _tensor_bytes(weight),
            "estimated_bits": self._estimated_layer_bits(dynamic_data, correction_table),
            "dynamic_average_bits_data_only": dynamic_data["average_bits_data_only"],
            "ternary_sparsity": self.quantizer.sparsity(ternary),
            "ternary_scale": float(ternary_scale.item()),
            "quality": asdict(quality),
        }
        return CompressedLayer(
            name=name,
            shape=tuple(int(v) for v in weight.shape),
            dtype=str(weight.dtype),
            dynamic_data=dynamic_data,
            correction_table=correction_table,
            metadata=metadata,
        )

    def decompress_layer(self, compressed_layer: CompressedLayer | Mapping[str, Any], apply_correction: bool = True) -> torch.Tensor:
        """Reconstruct a tensor from a compressed layer."""

        layer = self._coerce_layer(compressed_layer)
        dtype = _dtype_from_name(layer.dtype)
        if layer.dynamic_data.get("format") == "aethercore_v3.packed_ternary":
            reconstructed = self.packed_compressor.decompress(layer.dynamic_data, dtype=dtype)
        else:
            reconstructed = self.bit_allocator.decompress(layer.dynamic_data, dtype=dtype)
        if apply_correction:
            reconstructed = self.correction_extractor.apply_to_weight(reconstructed, layer.correction_table)
        return reconstructed.to(dtype=dtype)

    def save_experts(self, compressed: CompressedLayer | Mapping[str, CompressedLayer] | list[CompressedLayer], path: str | Path) -> None:
        """Save one compressed layer or a bundle of compressed layers."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(compressed, CompressedLayer):
            payload: dict[str, Any] = compressed.to_dict()
        elif isinstance(compressed, Mapping):
            payload = {
                "format": "aethercore_v3.expert_bundle",
                "layers": {str(name): self._coerce_layer(layer).to_dict() for name, layer in compressed.items()},
            }
        elif isinstance(compressed, list):
            payload = {
                "format": "aethercore_v3.expert_bundle",
                "layers": {layer.name: self._coerce_layer(layer).to_dict() for layer in compressed},
            }
        else:
            raise TypeError("compressed must be a CompressedLayer, mapping, or list")

        torch.save(payload, output_path)

    def load_expert(self, path: str | Path, expert_id: str | None = None) -> CompressedLayer:
        """Load one compressed layer from a file or bundle."""

        payload = torch.load(Path(path), map_location="cpu")
        if payload.get("format") == "aethercore_v3.compressed_layer":
            return CompressedLayer.from_dict(payload)

        if payload.get("format") == "aethercore_v3.expert_bundle":
            layers = payload.get("layers", {})
            if not layers:
                raise ValueError("Expert bundle contains no layers")
            selected_id = expert_id if expert_id is not None else next(iter(layers))
            if selected_id not in layers:
                raise KeyError(f"Expert id {selected_id!r} not found in bundle")
            return CompressedLayer.from_dict(layers[selected_id])

        raise ValueError("Unsupported expert file format")

    def compress_model(self, model: Any, output_dir: str | Path) -> CompressionStats:
        """Compress all matrix-like floating tensors in a model or state dict."""

        state_dict = self._extract_state_dict(model)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        candidates = {
            name: tensor
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor) and tensor.ndim >= 2
        }
        dedup_map = self.deduplicator.find_duplicates(candidates) if candidates else {}
        raw_tensors: dict[str, torch.Tensor] = {}
        manifest_layers: dict[str, dict[str, Any]] = {}

        original_bytes = sum(_tensor_bytes(tensor) for tensor in state_dict.values() if isinstance(tensor, torch.Tensor))
        estimated_bits = 0
        relative_errors: list[float] = []
        quality_scores: list[float] = []
        compressed_count = 0
        deduplicated_count = 0

        canonical_paths: dict[str, str] = {}
        for name, tensor in state_dict.items():
            if name not in candidates:
                if isinstance(tensor, torch.Tensor):
                    raw_tensors[name] = tensor.detach().cpu()
                continue

            canonical_name = dedup_map[name]
            if canonical_name != name:
                deduplicated_count += 1
                manifest_layers[name] = {
                    "kind": "reference",
                    "canonical": canonical_name,
                    "path": canonical_paths[canonical_name],
                }
                continue

            compressed = self.compress_layer(tensor, name=name)
            layer_filename = f"{_safe_layer_filename(name)}.pt"
            layer_path = output_path / layer_filename
            self.save_experts(compressed, layer_path)
            canonical_paths[name] = layer_filename
            manifest_layers[name] = {
                "kind": "compressed",
                "path": layer_filename,
                "shape": compressed.shape,
                "dtype": compressed.dtype,
            }
            estimated_bits += int(compressed.metadata["estimated_bits"]["total_bits"])
            relative_errors.append(float(compressed.metadata["quality"]["relative_l2_error"]))
            quality_scores.append(float(compressed.metadata["quality"]["quality_score"]))
            compressed_count += 1

        if raw_tensors:
            torch.save(raw_tensors, output_path / "raw_tensors.pt")

        actual_serialized_bytes = sum(
            int(file.stat().st_size)
            for file in output_path.glob("*.pt")
            if file.is_file()
        )
        estimated_compressed_bytes = _ceil_div(estimated_bits, 8)
        if raw_tensors:
            raw_path = output_path / "raw_tensors.pt"
            estimated_compressed_bytes += int(raw_path.stat().st_size)

        stats = CompressionStats(
            original_bytes=original_bytes,
            estimated_compressed_bytes=estimated_compressed_bytes,
            actual_serialized_bytes=actual_serialized_bytes,
            compression_ratio_estimated=self._ratio(original_bytes, estimated_compressed_bytes),
            compression_ratio_serialized=self._ratio(original_bytes, actual_serialized_bytes),
            layers_compressed=compressed_count,
            layers_deduplicated=deduplicated_count,
            average_bits_per_weight=self._average_bits(manifest_layers, output_path),
            average_relative_error=float(sum(relative_errors) / len(relative_errors)) if relative_errors else 0.0,
            quality_score=float(sum(quality_scores) / len(quality_scores)) if quality_scores else 1.0,
            details={
                "manifest": "manifest.pt",
                "raw_tensors": "raw_tensors.pt" if raw_tensors else None,
                "dedup_map": dedup_map,
            },
        )

        manifest = {
            "format": "aethercore_v3.compressed_model",
            "layers": manifest_layers,
            "stats": asdict(stats),
        }
        torch.save(manifest, output_path / "manifest.pt")
        stats.actual_serialized_bytes = sum(
            int(file.stat().st_size)
            for file in output_path.glob("*.pt")
            if file.is_file()
        )
        stats.compression_ratio_serialized = self._ratio(original_bytes, stats.actual_serialized_bytes)
        return stats

    def benchmark(self, original: torch.Tensor, compressed: CompressedLayer | Mapping[str, Any] | torch.Tensor) -> QualityMetrics:
        """Compare an original tensor with a compressed layer or tensor."""

        original_tensor = _ensure_float_tensor(original, "original").detach().cpu().float()
        if isinstance(compressed, torch.Tensor):
            restored = _ensure_float_tensor(compressed, "compressed").detach().cpu().float()
        else:
            restored = self.decompress_layer(compressed).detach().cpu().float()

        if original_tensor.shape != restored.shape:
            raise ValueError("original and reconstructed tensors must have identical shapes")

        difference = original_tensor - restored
        mse = float(torch.mean(difference.square()).item())
        mae = float(torch.mean(difference.abs()).item())
        max_abs_error = float(difference.abs().max().item())
        original_norm = original_tensor.norm().clamp_min(_EPS)
        relative_l2_error = float((difference.norm() / original_norm).item())
        cosine_similarity = float(F.cosine_similarity(original_tensor.flatten(), restored.flatten(), dim=0).item())
        quality_score = float(max(0.0, min(1.0, 1.0 - relative_l2_error)))

        return QualityMetrics(
            mse=mse,
            mae=mae,
            max_abs_error=max_abs_error,
            relative_l2_error=relative_l2_error,
            cosine_similarity=cosine_similarity,
            quality_score=quality_score,
        )

    def _extract_state_dict(self, model: Any) -> Mapping[str, torch.Tensor]:
        """Return a mapping of tensor names to tensors from a model-like object."""

        if isinstance(model, Mapping):
            return model
        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
            if isinstance(state_dict, Mapping):
                return state_dict
        raise TypeError("model must be a mapping or expose state_dict()")

    def _coerce_layer(self, compressed_layer: CompressedLayer | Mapping[str, Any]) -> CompressedLayer:
        """Normalize layer-like inputs to CompressedLayer."""

        if isinstance(compressed_layer, CompressedLayer):
            return compressed_layer
        if isinstance(compressed_layer, Mapping):
            return CompressedLayer.from_dict(compressed_layer)
        raise TypeError("compressed_layer must be CompressedLayer or mapping")

    def _estimated_layer_bits(self, dynamic_data: Mapping[str, Any], correction_table: Mapping[str, Any]) -> dict[str, int]:
        """Combine dynamic quantization and correction-table bit estimates."""

        dynamic_bits = dict(dynamic_data["estimated_bits"])
        correction_bits = _recursive_tensor_bits(correction_table)
        total_bits = int(dynamic_bits["total_bits"] + correction_bits)
        return {
            "dynamic_data_bits": int(dynamic_bits.get("data_bits", 0)),
            "dynamic_index_bits": int(dynamic_bits.get("index_bits", 0)),
            "scale_bits": int(dynamic_bits.get("scale_bits", 0)),
            "correction_bits": int(correction_bits),
            "total_bits": total_bits,
        }

    def _average_bits(self, manifest_layers: Mapping[str, Any], output_path: Path) -> float:
        """Compute average estimated bits per original compressed weight."""

        total_bits = 0
        total_values = 0
        for entry in manifest_layers.values():
            if entry.get("kind") != "compressed":
                continue
            layer = self.load_expert(output_path / entry["path"])
            total_bits += int(layer.metadata["estimated_bits"]["total_bits"])
            total_values += math.prod(layer.shape)
        if total_values == 0:
            return 0.0
        return float(total_bits / total_values)

    def _ratio(self, numerator: int, denominator: int) -> float:
        """Return a safe compression ratio."""

        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)


def _self_test() -> None:
    """Run a small CPU sanity check for the compression engine."""

    torch.manual_seed(7)
    base = torch.randn(64, 128, dtype=torch.float32) * 0.15
    structured = torch.linspace(-0.3, 0.3, steps=128).repeat(64, 1)
    weight = base + structured

    quantizer = TernaryQuantizer()
    ternary, scale = quantizer.quantize(weight)
    ternary_reconstruction = quantizer.dequantize(ternary, scale)

    allocator = DynamicBitAllocator()
    dynamic = allocator.compress(weight)
    dynamic_reconstruction = allocator.decompress(dynamic)

    delta = DeltaCompressor()
    delta_package = delta.compute_deltas([weight, weight + 0.01 * torch.randn_like(weight)])
    reconstructed_layers = delta.reconstruct(delta_package)

    deduplicator = SemanticDeduplicator(similarity_threshold=0.99)
    dedup_map = deduplicator.find_duplicates({"a": weight, "b": weight.clone(), "c": torch.randn_like(weight)})

    engine = GodCompressionEngine()
    compressed = engine.compress_layer(weight, name="demo.weight")
    restored = engine.decompress_layer(compressed)
    metrics = engine.benchmark(weight, compressed)

    temp_dir = Path.cwd() / "experiments" / "_compression_selftest"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "demo.pt"
    engine.save_experts(compressed, path)
    loaded = engine.load_expert(path)
    loaded_metrics = engine.benchmark(weight, loaded)

    print("AetherCore compression self-test")
    print(f"  ternary sparsity: {quantizer.sparsity(ternary):.3f}")
    print(f"  ternary reconstruction mean: {ternary_reconstruction.mean().item():.6f}")
    print(f"  dynamic reconstruction mean: {dynamic_reconstruction.mean().item():.6f}")
    print(f"  delta layers reconstructed: {len(reconstructed_layers)}")
    print(f"  dedup map: {dedup_map}")
    print(f"  restored shape: {tuple(restored.shape)}")
    print(f"  quality score: {metrics.quality_score:.4f}")
    print(f"  loaded quality score: {loaded_metrics.quality_score:.4f}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
