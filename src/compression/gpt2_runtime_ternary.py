"""Runtime ternary GPT-2 inference experiment.

The packed GPT-2 path reconstructs normal floating-point weights before running
Transformers. This module keeps compressed matrix weights as ternary signs plus
scales in custom runtime modules. It is intentionally experimental: PyTorch CPU
still has to cast signs for matrix multiply because there is no custom bitpacked
kernel here, but the model does not store reconstructed fp32 weights.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass(slots=True)
class RuntimeTernaryStats:
    """Summary for a runtime ternary GPT-2 checkpoint."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    tensors_total: int
    tensors_ternary: int
    tensors_raw: int
    tensors_referenced: int
    average_bits_per_ternary_weight: float
    average_relative_error: float
    correction_rank: int
    elapsed_sec: float
    output_dir: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


class TernaryEmbedding(nn.Module):
    """Embedding table backed by ternary signs and per-feature scales."""

    def __init__(self, signs: torch.Tensor, scales: torch.Tensor) -> None:
        super().__init__()
        if signs.ndim != 2:
            raise ValueError("signs must be 2-D")
        if scales.ndim != 1 or scales.numel() != signs.shape[1]:
            raise ValueError("scales must be 1-D with embedding_dim entries")
        self.num_embeddings = int(signs.shape[0])
        self.embedding_dim = int(signs.shape[1])
        self.register_buffer("signs", signs.detach().cpu().to(torch.int8).contiguous(), persistent=True)
        self.register_buffer("scales", scales.detach().cpu().float().contiguous(), persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        selected = self.signs[input_ids].to(dtype=self.scales.dtype)
        return selected.mul(self.scales)


class TernaryLMHead(nn.Module):
    """Output projection tied to a ternary token embedding table."""

    def __init__(self, embedding: TernaryEmbedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape[:-1]
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        scaled = flat.mul(self.embedding.scales)
        logits = scaled.matmul(self.embedding.signs.t().float())
        return logits.reshape(*original_shape, self.embedding.num_embeddings)


class TernaryConv1D(nn.Module):
    """GPT-2 Conv1D replacement backed by ternary signs and correction tables."""

    def __init__(
        self,
        signs: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
        correction_u: torch.Tensor | None = None,
        correction_s: torch.Tensor | None = None,
        correction_v: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if signs.ndim != 2:
            raise ValueError("signs must be 2-D")
        if scales.ndim != 1 or scales.numel() != signs.shape[1]:
            raise ValueError("scales must be 1-D with output entries")
        self.nx = int(signs.shape[0])
        self.nf = int(signs.shape[1])
        self.register_buffer("signs", signs.detach().cpu().to(torch.int8).contiguous(), persistent=True)
        self.register_buffer("scales", scales.detach().cpu().float().contiguous(), persistent=True)
        if bias is None:
            bias = torch.zeros(self.nf, dtype=torch.float32)
        self.register_buffer("bias", bias.detach().cpu().float().contiguous(), persistent=True)

        rank = 0
        if correction_u is not None and correction_s is not None and correction_v is not None:
            rank = int(correction_s.numel())
            self.register_buffer("correction_u", correction_u.detach().cpu().float().contiguous(), persistent=True)
            self.register_buffer("correction_s", correction_s.detach().cpu().float().contiguous(), persistent=True)
            self.register_buffer("correction_v", correction_v.detach().cpu().float().contiguous(), persistent=True)
        else:
            self.register_buffer("correction_u", torch.empty(self.nx, 0), persistent=True)
            self.register_buffer("correction_s", torch.empty(0), persistent=True)
            self.register_buffer("correction_v", torch.empty(self.nf, 0), persistent=True)
        self.correction_rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        flat = x.reshape(-1, x.size(-1)).float()
        output = flat.matmul(self.signs.float()).mul(self.scales)
        if self.correction_rank:
            correction = flat.matmul(self.correction_u).mul(self.correction_s).matmul(self.correction_v.t())
            output = output + correction
        output = output + self.bias
        return output.reshape(size_out).to(dtype=x.dtype)


def compress_gpt2_runtime_ternary(
    model_path: str | Path = "models/gpt2",
    output_dir: str | Path = "experiments/gpt2_runtime_ternary",
    threshold_factor: float = 0.7,
    correction_rank: int = 0,
) -> RuntimeTernaryStats:
    """Create a compact runtime ternary GPT-2 checkpoint."""

    from transformers import AutoModelForCausalLM

    if threshold_factor < 0:
        raise ValueError("threshold_factor must be non-negative")
    if correction_rank < 0:
        raise ValueError("correction_rank must be non-negative")

    started = time.perf_counter()
    source_dir = Path(model_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(source_dir, local_files_only=True)
    state = model.state_dict()
    tensor_store: dict[str, torch.Tensor] = {}
    manifest: dict[str, Any] = {
        "format": "aethercore_v3.runtime_ternary_gpt2",
        "source_model": str(source_dir),
        "threshold_factor": float(threshold_factor),
        "correction_rank": int(correction_rank),
        "tensors": {},
    }

    seen_storage: dict[tuple[int, int, tuple[int, ...]], str] = {}
    original_bytes = 0
    ternary_weight_count = 0
    ternary_bits = 0
    relative_errors: list[float] = []
    tensors_ternary = 0
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

        if torch.is_floating_point(cpu_tensor) and cpu_tensor.ndim == 2:
            signs, scales = _ternarize_per_output(cpu_tensor.float(), threshold_factor)
            approx = signs.float().mul(scales)
            kind = "ternary_embedding" if name in {"transformer.wte.weight", "transformer.wpe.weight"} else "ternary_conv1d"
            values_key = f"ternary.{index}.values"
            scales_key = f"ternary.{index}.scales"
            tensor_store[values_key] = _pack_2bit((signs + 1).to(torch.uint8))
            tensor_store[scales_key] = scales.to(torch.float16).contiguous()

            entry: dict[str, Any] = {
                "kind": kind,
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype),
                "total_values": int(cpu_tensor.numel()),
                "padded_values": int(((cpu_tensor.numel() + 3) // 4) * 4),
                "values": values_key,
                "scales": scales_key,
            }

            if kind == "ternary_conv1d" and correction_rank > 0:
                correction = _low_rank_correction(cpu_tensor.float(), approx.float(), correction_rank)
                if correction:
                    for key, value in correction.items():
                        tensor_key = f"ternary.{index}.correction_{key}"
                        tensor_store[tensor_key] = value.to(torch.float16).contiguous()
                        entry[f"correction_{key}"] = tensor_key
                    corrected = approx + _apply_low_rank(torch.eye(cpu_tensor.shape[0]), correction).reshape_as(approx)
                    relative_errors.append(_relative_l2(cpu_tensor.float(), corrected.float()))
                else:
                    relative_errors.append(_relative_l2(cpu_tensor.float(), approx.float()))
            else:
                relative_errors.append(_relative_l2(cpu_tensor.float(), approx.float()))

            manifest["tensors"][name] = entry
            ternary_weight_count += int(cpu_tensor.numel())
            ternary_bits += int(cpu_tensor.numel() * 2 + scales.numel() * 16)
            tensors_ternary += 1
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
    stats = RuntimeTernaryStats(
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(1, compressed_bytes),
        tensors_total=len(manifest["tensors"]),
        tensors_ternary=tensors_ternary,
        tensors_raw=tensors_raw,
        tensors_referenced=tensors_referenced,
        average_bits_per_ternary_weight=ternary_bits / max(1, ternary_weight_count),
        average_relative_error=sum(relative_errors) / len(relative_errors) if relative_errors else 0.0,
        correction_rank=int(correction_rank),
        elapsed_sec=elapsed,
        output_dir=str(target_dir),
        details={"tensor_file": "tensors.safetensors", "manifest": "manifest.json"},
    )
    manifest["stats"] = stats.to_dict()
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    stats.compressed_bytes = _directory_size(target_dir)
    stats.compression_ratio = original_bytes / max(1, stats.compressed_bytes)
    return stats


def load_runtime_ternary_gpt2_model(compressed_dir: str | Path):
    """Load runtime ternary GPT-2 modules without fp32 weight reconstruction."""

    from transformers import AutoConfig, AutoModelForCausalLM

    folder = Path(compressed_dir)
    manifest = _read_manifest(folder)
    tensors = load_file(folder / "tensors.safetensors", device="cpu")
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)

    raw_state: dict[str, torch.Tensor] = {}
    entries: dict[str, Mapping[str, Any]] = dict(manifest["tensors"])
    for name, entry in entries.items():
        if entry["kind"] == "raw":
            raw_state[name] = tensors[entry["value"]]
    model.load_state_dict(raw_state, strict=False)

    embeddings: dict[str, TernaryEmbedding] = {}
    for name, entry in entries.items():
        kind = entry["kind"]
        if kind == "ternary_embedding":
            module_path = name.removesuffix(".weight")
            signs = _decode_signs(tensors[entry["values"]], tuple(int(v) for v in entry["shape"]), int(entry["padded_values"]))
            scales = tensors[entry["scales"]].float()
            embedding = TernaryEmbedding(signs, scales)
            _set_module(model, module_path, embedding)
            embeddings[name] = embedding

    for name, entry in entries.items():
        if entry["kind"] != "ternary_conv1d":
            continue
        module_path = name.removesuffix(".weight")
        signs = _decode_signs(tensors[entry["values"]], tuple(int(v) for v in entry["shape"]), int(entry["padded_values"]))
        scales = tensors[entry["scales"]].float()
        bias = raw_state.get(name.removesuffix(".weight") + ".bias")
        correction_u = tensors[entry["correction_u"]].float() if "correction_u" in entry else None
        correction_s = tensors[entry["correction_s"]].float() if "correction_s" in entry else None
        correction_v = tensors[entry["correction_v"]].float() if "correction_v" in entry else None
        _set_module(
            model,
            module_path,
            TernaryConv1D(signs, scales, bias=bias, correction_u=correction_u, correction_s=correction_s, correction_v=correction_v),
        )

    lm_head = entries.get("lm_head.weight")
    if lm_head and lm_head.get("kind") == "reference":
        target = str(lm_head["target"])
        if target in embeddings:
            model.lm_head = TernaryLMHead(embeddings[target])

    model.eval()
    return model


def generate_with_runtime_ternary_gpt2(
    prompt: str,
    compressed_dir: str | Path = "experiments/gpt2_runtime_ternary",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
) -> str:
    """Generate text from a runtime ternary GPT-2 folder."""

    from transformers import AutoTokenizer

    folder = Path(compressed_dir)
    tokenizer_dir = folder
    if not (folder / "tokenizer.json").exists():
        manifest = _read_manifest(folder)
        tokenizer_dir = Path(str(manifest.get("source_model", folder)))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    model = load_runtime_ternary_gpt2_model(folder)
    inputs = tokenizer(prompt, return_tensors="pt")
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = max(float(temperature), 1.0e-6)
    output = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def compare_gpt2_variants(
    model_path: str | Path = "models/gpt2",
    int8_path: str | Path = "experiments/gpt2_int8",
    ternary_path: str | Path = "experiments/gpt2_runtime_ternary",
    prompt: str = "The future of AI is",
    max_new_tokens: int = 50,
    quality_prompts: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare original, INT8-reconstructed, and runtime ternary GPT-2."""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.compression.gpt2_packed import load_packed_gpt2_model

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    original = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    int8 = load_packed_gpt2_model(int8_path)
    ternary = load_runtime_ternary_gpt2_model(ternary_path)
    prompts = list(quality_prompts or _DEFAULT_QUALITY_PROMPTS)

    original_speed = _benchmark_generation(original, tokenizer, prompt, max_new_tokens)
    int8_speed = _benchmark_generation(int8, tokenizer, prompt, max_new_tokens)
    ternary_speed = _benchmark_generation(ternary, tokenizer, prompt, max_new_tokens)
    int8_quality = _quality_against_original(original, int8, tokenizer, prompts)
    ternary_quality = _quality_against_original(original, ternary, tokenizer, prompts)

    original_size = _directory_size(Path(model_path))
    return [
        {
            "name": "Original",
            "size_bytes": original_size,
            "compression_ratio": 1.0,
            "tokens_per_sec": original_speed["tokens_per_sec"],
            "quality_score": 1.0,
            "top1_match_rate": 1.0,
            "logit_cosine": 1.0,
        },
        {
            "name": "INT8",
            "size_bytes": _directory_size(Path(int8_path)),
            "compression_ratio": original_size / max(1, _directory_size(Path(int8_path))),
            "tokens_per_sec": int8_speed["tokens_per_sec"],
            **int8_quality,
        },
        {
            "name": "RuntimeTernary",
            "size_bytes": _directory_size(Path(ternary_path)),
            "compression_ratio": original_size / max(1, _directory_size(Path(ternary_path))),
            "tokens_per_sec": ternary_speed["tokens_per_sec"],
            **ternary_quality,
        },
    ]


def _ternarize_per_output(weight: torch.Tensor, threshold_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ternary signs and one scale per output feature."""

    abs_mean = weight.abs().mean(dim=0).clamp_min(1.0e-12)
    thresholds = abs_mean * float(threshold_factor)
    signs = torch.zeros_like(weight, dtype=torch.int8)
    signs = torch.where(weight > thresholds, torch.ones_like(signs), signs)
    signs = torch.where(weight < -thresholds, -torch.ones_like(signs), signs)
    selected = signs != 0
    selected_abs = torch.where(selected, weight.abs(), torch.zeros_like(weight))
    counts = selected.sum(dim=0).clamp_min(1)
    scales = (selected_abs.sum(dim=0) / counts).clamp_min(1.0e-12)
    return signs.cpu().contiguous(), scales.cpu().float().contiguous()


def _low_rank_correction(original: torch.Tensor, approx: torch.Tensor, rank: int) -> dict[str, torch.Tensor]:
    """Build a small low-rank correction table for a Conv1D weight."""

    max_rank = min(int(rank), min(original.shape) - 1)
    if max_rank <= 0:
        return {}
    residual = original - approx
    try:
        u, s, v = torch.pca_lowrank(residual, q=min(max_rank + 2, min(residual.shape)), center=False, niter=1)
    except RuntimeError:
        return {}
    return {
        "u": u[:, :max_rank].contiguous(),
        "s": s[:max_rank].contiguous(),
        "v": v[:, :max_rank].contiguous(),
    }


def _apply_low_rank(input_tensor: torch.Tensor, correction: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Apply a low-rank correction matrix to input activations."""

    if not correction:
        return torch.zeros(input_tensor.shape[0], 0)
    return input_tensor.matmul(correction["u"]).mul(correction["s"]).matmul(correction["v"].t())


def _benchmark_generation(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> dict[str, float]:
    """Measure greedy generation tokens/sec for a loaded model."""

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        started = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - started
    new_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
    return {"new_tokens": float(new_tokens), "elapsed_sec": elapsed, "tokens_per_sec": new_tokens / max(elapsed, 1.0e-9)}


def _quality_against_original(original: Any, candidate: Any, tokenizer: Any, prompts: Sequence[str]) -> dict[str, float]:
    """Compare next-token distributions against original GPT-2."""

    cosines: list[float] = []
    top1_matches = 0
    top5_matches = 0
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            original_logits = original(**inputs).logits[:, -1, :].float()
            candidate_logits = candidate(**inputs).logits[:, -1, :].float()
            cosines.append(float(F.cosine_similarity(original_logits, candidate_logits, dim=-1).mean().item()))
            original_top1 = torch.argmax(original_logits, dim=-1)
            candidate_top1 = torch.argmax(candidate_logits, dim=-1)
            top1_matches += int(torch.equal(original_top1, candidate_top1))
            original_top5 = torch.topk(original_logits, k=5, dim=-1).indices.flatten().tolist()
            top5_matches += int(int(candidate_top1.item()) in original_top5)
    total = max(1, len(prompts))
    mean_cosine = sum(cosines) / total
    top1_rate = top1_matches / total
    top5_rate = top5_matches / total
    quality_score = max(0.0, 0.70 * mean_cosine + 0.20 * top5_rate + 0.10 * top1_rate)
    return {
        "quality_score": quality_score,
        "top1_match_rate": top1_rate,
        "top5_match_rate": top5_rate,
        "logit_cosine": mean_cosine,
    }


def _pack_2bit(values: torch.Tensor) -> torch.Tensor:
    """Pack uint values in [0, 3] into bytes."""

    flat = values.detach().cpu().to(torch.uint8).flatten()
    padded = ((int(flat.numel()) + 3) // 4) * 4
    if padded != flat.numel():
        flat = torch.cat((flat, torch.zeros(padded - flat.numel(), dtype=torch.uint8)))
    chunks = flat.reshape(-1, 4)
    return (chunks[:, 0] | (chunks[:, 1] << 2) | (chunks[:, 2] << 4) | (chunks[:, 3] << 6)).contiguous()


def _unpack_2bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack bytes into uint values."""

    data = packed.detach().cpu().to(torch.uint8).flatten()
    unpacked = torch.empty(int(data.numel()) * 4, dtype=torch.uint8)
    unpacked[0::4] = data & 0b00000011
    unpacked[1::4] = (data >> 2) & 0b00000011
    unpacked[2::4] = (data >> 4) & 0b00000011
    unpacked[3::4] = (data >> 6) & 0b00000011
    return unpacked[:count].contiguous()


def _decode_signs(packed: torch.Tensor, shape: tuple[int, ...], padded_values: int) -> torch.Tensor:
    """Decode packed ternary signs into int8 {-1, 0, +1} signs."""

    total = int(torch.tensor(shape).prod().item())
    codes = _unpack_2bit(packed, int(padded_values))[:total].to(torch.int8)
    return (codes - 1).reshape(shape).contiguous()


def _set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    """Set a nested module by dotted path."""

    parts = path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _copy_model_sidecars(source_dir: Path, target_dir: Path) -> None:
    """Copy tokenizer/config files needed to run GPT-2."""

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
    """Read and validate a runtime ternary manifest."""

    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Runtime ternary manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "aethercore_v3.runtime_ternary_gpt2":
        raise ValueError("Unsupported runtime ternary GPT-2 manifest format")
    return manifest


def _relative_l2(original: torch.Tensor, restored: torch.Tensor) -> float:
    """Return relative L2 reconstruction error."""

    diff = original - restored
    return float((diff.norm() / original.norm().clamp_min(1.0e-12)).item())


_DEFAULT_QUALITY_PROMPTS = (
    "The future of AI is",
    "Machine learning can help",
    "In a small village",
    "The most important scientific idea is",
    "Python code can",
)
