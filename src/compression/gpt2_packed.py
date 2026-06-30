"""Packed GPT-2 compression and reconstruction helpers.

This module is deliberately narrower than ``GodCompressionEngine``: it targets
Hugging Face GPT-2-style causal language models and stores matrix weights with
dense packed ternary values. The format is compact on disk and can be loaded
back into a normal Transformers model for honest generation tests.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from .engine import PackedTernaryCompressor


@dataclass(slots=True)
class PackedGPT2Stats:
    """Summary for a packed GPT-2 checkpoint."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    tensors_total: int
    tensors_packed: int
    tensors_raw: int
    tensors_referenced: int
    average_bits_per_packed_weight: float
    average_relative_error: float
    elapsed_sec: float
    output_dir: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


def compress_gpt2_packed(
    model_path: str | Path = "models/gpt2",
    output_dir: str | Path = "experiments/gpt2_packed",
    block_size: int = 256,
    threshold_factor: float = 0.7,
    compression: str = "int8",
) -> PackedGPT2Stats:
    """Compress a local Hugging Face GPT-2 checkpoint."""

    from transformers import AutoModelForCausalLM

    if compression not in {"int8", "ternary"}:
        raise ValueError("compression must be 'int8' or 'ternary'")

    started = time.perf_counter()
    source_dir = Path(model_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(source_dir, local_files_only=True)
    state = model.state_dict()
    compressor = PackedTernaryCompressor(block_size=block_size, threshold_factor=threshold_factor)

    tensor_store: dict[str, torch.Tensor] = {}
    manifest: dict[str, Any] = {
        "format": "aethercore_v3.packed_gpt2",
        "source_model": str(source_dir),
        "compression": compression,
        "block_size": int(block_size),
        "threshold_factor": float(threshold_factor),
        "tensors": {},
    }

    seen_storage: dict[tuple[int, int, tuple[int, ...]], str] = {}
    original_bytes = 0
    packed_weight_count = 0
    packed_bits = 0
    relative_errors: list[float] = []
    tensors_packed = 0
    tensors_raw = 0
    tensors_referenced = 0

    for index, (name, tensor) in enumerate(state.items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        cpu_tensor = tensor.detach().cpu().contiguous()
        original_bytes += int(cpu_tensor.numel() * cpu_tensor.element_size())
        storage_key = (int(cpu_tensor.untyped_storage().data_ptr()), int(cpu_tensor.storage_offset()), tuple(cpu_tensor.shape))

        if storage_key in seen_storage:
            manifest["tensors"][name] = {"kind": "reference", "target": seen_storage[storage_key]}
            tensors_referenced += 1
            continue
        seen_storage[storage_key] = name

        if torch.is_floating_point(cpu_tensor) and cpu_tensor.ndim >= 2 and compression == "ternary":
            packed = compressor.compress(cpu_tensor)
            values_key = f"packed.{index}.values"
            scales_key = f"packed.{index}.scales"
            tensor_store[values_key] = packed["packed_values"].contiguous()
            tensor_store[scales_key] = packed["scales"].contiguous()
            manifest["tensors"][name] = {
                "kind": "packed_ternary",
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype),
                "block_size": int(packed["block_size"]),
                "total_values": int(packed["total_values"]),
                "padded_values": int(packed["padded_values"]),
                "values": values_key,
                "scales": scales_key,
            }

            restored = compressor.decompress(
                {
                    "format": "aethercore_v3.packed_ternary",
                    "shape": tuple(cpu_tensor.shape),
                    "dtype": str(cpu_tensor.dtype),
                    "block_size": int(packed["block_size"]),
                    "total_values": int(packed["total_values"]),
                    "padded_values": int(packed["padded_values"]),
                    "packed_values": tensor_store[values_key],
                    "scales": tensor_store[scales_key],
                },
                dtype=torch.float32,
            )
            relative_errors.append(_relative_l2(cpu_tensor.float(), restored.float()))
            packed_weight_count += int(cpu_tensor.numel())
            packed_bits += int(packed["estimated_bits"]["total_bits"])
            tensors_packed += 1
        elif torch.is_floating_point(cpu_tensor) and cpu_tensor.ndim >= 2:
            quantized, scale = _quantize_int8(cpu_tensor)
            values_key = f"int8.{index}.values"
            scale_key = f"int8.{index}.scale"
            tensor_store[values_key] = quantized.contiguous()
            tensor_store[scale_key] = torch.tensor([scale], dtype=torch.float32)
            manifest["tensors"][name] = {
                "kind": "int8",
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype),
                "values": values_key,
                "scale": scale_key,
            }

            restored = _dequantize_int8(tensor_store[values_key], scale, cpu_tensor.dtype)
            relative_errors.append(_relative_l2(cpu_tensor.float(), restored.float()))
            packed_weight_count += int(cpu_tensor.numel())
            packed_bits += int(cpu_tensor.numel() * 8 + 32)
            tensors_packed += 1
        else:
            raw_key = f"raw.{index}.value"
            tensor_store[raw_key] = cpu_tensor
            manifest["tensors"][name] = {
                "kind": "raw",
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype),
                "value": raw_key,
            }
            tensors_raw += 1

    save_file(tensor_store, target_dir / "tensors.safetensors")
    _copy_model_sidecars(source_dir, target_dir)

    elapsed = time.perf_counter() - started
    compressed_bytes = _directory_size(target_dir)
    stats = PackedGPT2Stats(
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(1, compressed_bytes),
        tensors_total=len(manifest["tensors"]),
        tensors_packed=tensors_packed,
        tensors_raw=tensors_raw,
        tensors_referenced=tensors_referenced,
        average_bits_per_packed_weight=packed_bits / max(1, packed_weight_count),
        average_relative_error=sum(relative_errors) / len(relative_errors) if relative_errors else 0.0,
        elapsed_sec=elapsed,
        output_dir=str(target_dir),
        details={"tensor_file": "tensors.safetensors", "manifest": "manifest.json"},
    )
    manifest["stats"] = stats.to_dict()
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    stats.compressed_bytes = _directory_size(target_dir)
    stats.compression_ratio = original_bytes / max(1, stats.compressed_bytes)
    return stats


def load_packed_gpt2_model(compressed_dir: str | Path):
    """Load a packed GPT-2 folder back into a Transformers causal LM."""

    from transformers import AutoConfig, AutoModelForCausalLM

    folder = Path(compressed_dir)
    manifest = _read_manifest(folder)
    tensors = load_file(folder / "tensors.safetensors", device="cpu")
    compressor = PackedTernaryCompressor(block_size=int(manifest.get("block_size", 256)))
    state: dict[str, torch.Tensor] = {}
    references: dict[str, str] = {}

    for name, entry in manifest["tensors"].items():
        kind = entry["kind"]
        if kind == "reference":
            references[name] = str(entry["target"])
        elif kind == "raw":
            state[name] = tensors[entry["value"]]
        elif kind == "packed_ternary":
            state[name] = compressor.decompress(
                {
                    "format": "aethercore_v3.packed_ternary",
                    "shape": tuple(int(v) for v in entry["shape"]),
                    "dtype": str(entry["dtype"]),
                    "block_size": int(entry["block_size"]),
                    "total_values": int(entry["total_values"]),
                    "padded_values": int(entry["padded_values"]),
                    "packed_values": tensors[entry["values"]],
                    "scales": tensors[entry["scales"]],
                },
                dtype=_dtype_from_name(str(entry["dtype"])),
            )
        elif kind == "int8":
            scale_tensor = tensors[entry["scale"]].float().flatten()
            if scale_tensor.numel() != 1:
                raise ValueError(f"Invalid int8 scale for {name!r}")
            state[name] = _dequantize_int8(
                tensors[entry["values"]],
                float(scale_tensor.item()),
                _dtype_from_name(str(entry["dtype"])),
            ).reshape(tuple(int(v) for v in entry["shape"]))
        else:
            raise ValueError(f"Unsupported packed GPT-2 tensor kind: {kind!r}")

    for name, target in references.items():
        if target not in state:
            raise KeyError(f"Packed GPT-2 reference target {target!r} for {name!r} was not reconstructed")
        state[name] = state[target]

    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    model.eval()
    return model


def generate_with_packed_gpt2(
    prompt: str,
    compressed_dir: str | Path = "experiments/gpt2_packed",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
) -> str:
    """Generate text from a packed GPT-2 folder."""

    from transformers import AutoTokenizer

    folder = Path(compressed_dir)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = load_packed_gpt2_model(folder)
    inputs = tokenizer(prompt, return_tensors="pt")
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = max(float(temperature), 1.0e-6)
    output = model.generate(
        **inputs,
        **generation_kwargs,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def benchmark_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
) -> dict[str, float]:
    """Return a simple tokens/sec benchmark for a loaded model."""

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        started = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - started
    new_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
    return {
        "new_tokens": float(new_tokens),
        "elapsed_sec": elapsed,
        "tokens_per_sec": new_tokens / max(elapsed, 1.0e-9),
    }


def _copy_model_sidecars(source_dir: Path, target_dir: Path) -> None:
    """Copy tokenizer/config files needed to reconstruct and run GPT-2."""

    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)


def _directory_size(path: Path) -> int:
    """Return total file size under a directory."""

    return sum(int(file.stat().st_size) for file in path.rglob("*") if file.is_file())


def _read_manifest(folder: Path) -> Mapping[str, Any]:
    """Read and validate a packed GPT-2 manifest."""

    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Packed GPT-2 manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "aethercore_v3.packed_gpt2":
        raise ValueError("Unsupported packed GPT-2 manifest format")
    return manifest


def _relative_l2(original: torch.Tensor, restored: torch.Tensor) -> float:
    """Return relative L2 reconstruction error."""

    diff = original - restored
    return float((diff.norm() / original.norm().clamp_min(1.0e-12)).item())


def _quantize_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetrically quantize a floating tensor to int8 with one scale."""

    source = tensor.detach().cpu().float()
    max_abs = float(source.abs().max().clamp_min(1.0e-12).item())
    scale = max_abs / 127.0
    quantized = torch.round(source / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _dequantize_int8(values: torch.Tensor, scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct an int8-quantized tensor."""

    return values.detach().cpu().float().mul(float(scale)).to(dtype=dtype)


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    """Convert a string dtype name back to a torch dtype."""

    name = dtype_name.split(".")[-1]
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.float32
