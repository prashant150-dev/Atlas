"""Runtime ternary GPT-2 conversion and benchmarking.

Run from the repo root:

    python src/compression/ternary_runtime.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compression.gpt2_packed import load_packed_gpt2_model
from src.compression.gpt2_runtime_ternary import (
    TernaryConv1D,
    TernaryEmbedding,
    TernaryLMHead,
    _DEFAULT_QUALITY_PROMPTS,
    _benchmark_generation,
    _directory_size,
    _low_rank_correction,
    _quality_against_original,
    _read_manifest,
    _relative_l2,
    _ternarize_per_output,
    compress_gpt2_runtime_ternary,
    generate_with_runtime_ternary_gpt2,
    load_runtime_ternary_gpt2_model,
)


class TernaryLinear(nn.Module):
    """Linear layer backed by int8 ternary weights and a per-output scale.

    The module stores signs as int8 values in {-1, 0, +1}. Forward uses the
    sign matrix directly for matmul and applies the layer scale afterward, so
    there is no persistent fp32 reconstruction of the weight matrix.
    """

    def __init__(
        self,
        signs: torch.Tensor,
        scale: torch.Tensor | float,
        bias: torch.Tensor | None = None,
        correction_left: torch.Tensor | None = None,
        correction_scale: torch.Tensor | None = None,
        correction_right: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if signs.ndim != 2:
            raise ValueError("signs must have shape [out_features, in_features]")
        if not torch.all((signs == -1) | (signs == 0) | (signs == 1)):
            raise ValueError("signs must contain only -1, 0, and +1")

        out_features, in_features = int(signs.shape[0]), int(signs.shape[1])
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).flatten()
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.expand(out_features).clone()
        if scale_tensor.numel() != out_features:
            raise ValueError("scale must be scalar or one value per output feature")

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", signs.detach().cpu().to(torch.int8).contiguous(), persistent=True)
        self.register_buffer("scale", scale_tensor.detach().cpu().float().contiguous(), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().cpu().float().contiguous(), persistent=True)

        if correction_left is not None and correction_scale is not None and correction_right is not None:
            self.register_buffer("correction_left", correction_left.detach().cpu().float().contiguous(), persistent=True)
            self.register_buffer("correction_scale", correction_scale.detach().cpu().float().contiguous(), persistent=True)
            self.register_buffer("correction_right", correction_right.detach().cpu().float().contiguous(), persistent=True)
            self.correction_rank = int(correction_scale.numel())
        else:
            self.register_buffer("correction_left", torch.empty(in_features, 0), persistent=True)
            self.register_buffer("correction_scale", torch.empty(0), persistent=True)
            self.register_buffer("correction_right", torch.empty(out_features, 0), persistent=True)
            self.correction_rank = 0

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        threshold_factor: float = 0.7,
        correction_rank: int = 0,
    ) -> "TernaryLinear":
        """Build a ternary replacement from a normal ``nn.Linear``."""

        weight = layer.weight.detach().cpu().float()
        signs_t, scales = _ternarize_per_output(weight.t().contiguous(), threshold_factor)
        signs = signs_t.t().contiguous()
        correction = _linear_low_rank_correction(weight, signs.float().mul(scales[:, None]), correction_rank)
        return cls(
            signs=signs,
            scale=scales,
            bias=layer.bias.detach().cpu() if layer.bias is not None else None,
            correction_left=correction.get("left"),
            correction_scale=correction.get("scale"),
            correction_right=correction.get("right"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).float()
        output = flat.matmul(self.weight.t().float()).mul(self.scale)
        if self.correction_rank:
            correction = flat.matmul(self.correction_left).mul(self.correction_scale).matmul(self.correction_right.t())
            output = output + correction
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*original_shape, self.out_features).to(dtype=x.dtype)


class TernaryGPT2Converter:
    """Convert local GPT-2 into the runtime ternary checkpoint format."""

    def __init__(
        self,
        model_path: str | Path = "models/gpt2",
        output_dir: str | Path = "experiments/gpt2_ternary",
        threshold_factor: float = 0.7,
        correction_rank: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.threshold_factor = float(threshold_factor)
        self.correction_rank = int(correction_rank)

    def convert(self) -> Any:
        """Compress GPT-2 and save it under ``experiments/gpt2_ternary``."""

        stats = compress_gpt2_runtime_ternary(
            model_path=self.model_path,
            output_dir=self.output_dir,
            threshold_factor=self.threshold_factor,
            correction_rank=self.correction_rank,
        )
        self._trim_large_sidecars()
        stats.compressed_bytes = _directory_size(self.output_dir)
        stats.compression_ratio = stats.original_bytes / max(1, stats.compressed_bytes)
        self._write_summary(stats)
        return stats

    def load(self) -> Any:
        """Load the converted runtime ternary GPT-2 model."""

        return load_runtime_ternary_gpt2_model(self.output_dir)

    def replace_linear_layers(self, module: nn.Module) -> nn.Module:
        """Replace every ``nn.Linear`` child in an arbitrary module tree."""

        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    TernaryLinear.from_linear(
                        child,
                        threshold_factor=self.threshold_factor,
                        correction_rank=self.correction_rank,
                    ),
                )
            else:
                self.replace_linear_layers(child)
        return module

    def _write_summary(self, stats: Any) -> None:
        manifest_path = self.output_dir / "manifest.json"
        summary = {
            "target_size_mb": 30,
            "actual_size_mb": _bytes_to_mb(_directory_size(self.output_dir)),
            "runtime_modules": {
                "linear": "TernaryLinear",
                "gpt2_conv1d": "TernaryConv1D",
                "embedding": "TernaryEmbedding",
                "lm_head": "TernaryLMHead",
            },
            "stats": stats.to_dict() if hasattr(stats, "to_dict") else stats,
        }
        (self.output_dir / "ternary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime_entrypoint"] = "src/compression/ternary_runtime.py"
            manifest["tokenizer_fallback"] = str(self.model_path)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _trim_large_sidecars(self) -> None:
        tokenizer_json = self.output_dir / "tokenizer.json"
        if tokenizer_json.exists() and (self.model_path / "tokenizer.json").exists():
            tokenizer_json.unlink()


@dataclass(slots=True)
class BenchmarkRow:
    name: str
    disk_mb: float
    ram_mb: float
    tokens_per_sec: float
    quality_score: float
    top1_match_rate: float
    logit_cosine: float
    sample_text: str


class TernaryBenchmark:
    """Compare original, INT8, and runtime ternary GPT-2."""

    def __init__(
        self,
        model_path: str | Path = "models/gpt2",
        int8_path: str | Path = "experiments/gpt2_int8",
        ternary_path: str | Path = "experiments/gpt2_ternary",
        prompt: str = "The future of AI is",
        max_new_tokens: int = 40,
    ) -> None:
        self.model_path = Path(model_path)
        self.int8_path = Path(int8_path)
        self.ternary_path = Path(ternary_path)
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)

    def run(self) -> list[BenchmarkRow]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        original = self._load_with_ram(lambda: AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True))
        int8 = self._load_with_ram(lambda: load_packed_gpt2_model(self.int8_path))
        ternary = self._load_with_ram(lambda: load_runtime_ternary_gpt2_model(self.ternary_path))

        quality_prompts = list(_DEFAULT_QUALITY_PROMPTS)
        int8_quality = _quality_against_original(original.model, int8.model, tokenizer, quality_prompts)
        ternary_quality = _quality_against_original(original.model, ternary.model, tokenizer, quality_prompts)

        rows = [
            self._row("Original", self.model_path, original, tokenizer, {"quality_score": 1.0, "top1_match_rate": 1.0, "logit_cosine": 1.0}),
            self._row("INT8", self.int8_path, int8, tokenizer, int8_quality),
            self._row("Ternary", self.ternary_path, ternary, tokenizer, ternary_quality),
        ]
        self._write_results(rows)
        return rows

    def _load_with_ram(self, loader: Any) -> Any:
        gc.collect()
        before = _rss_mb()
        model = loader()
        model.eval()
        gc.collect()
        after = _rss_mb()
        return type("LoadedModel", (), {"model": model, "ram_mb": max(0.0, after - before)})()

    def _row(self, name: str, folder: Path, loaded: Any, tokenizer: Any, quality: dict[str, float]) -> BenchmarkRow:
        speed = _benchmark_generation(loaded.model, tokenizer, self.prompt, self.max_new_tokens)
        sample_text = _generate_sample(loaded.model, tokenizer, self.prompt, min(40, self.max_new_tokens))
        return BenchmarkRow(
            name=name,
            disk_mb=_bytes_to_mb(_directory_size(folder)),
            ram_mb=float(loaded.ram_mb),
            tokens_per_sec=float(speed["tokens_per_sec"]),
            quality_score=float(quality.get("quality_score", 0.0)),
            top1_match_rate=float(quality.get("top1_match_rate", 0.0)),
            logit_cosine=float(quality.get("logit_cosine", 0.0)),
            sample_text=sample_text,
        )

    def _write_results(self, rows: list[BenchmarkRow]) -> None:
        self.ternary_path.mkdir(parents=True, exist_ok=True)
        payload = [asdict(row) for row in rows]
        (self.ternary_path / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def print(rows: list[BenchmarkRow]) -> None:
        print("\nAetherCore v3 Ternary Runtime Benchmark")
        print("-" * 92)
        print(f"{'Model':<10} {'Disk MB':>10} {'RAM MB':>10} {'tok/sec':>10} {'Quality':>10} {'Top1':>8} {'Cosine':>8}")
        print("-" * 92)
        for row in rows:
            print(
                f"{row.name:<10} {row.disk_mb:>10.2f} {row.ram_mb:>10.2f} {row.tokens_per_sec:>10.2f} "
                f"{row.quality_score:>10.3f} {row.top1_match_rate:>8.3f} {row.logit_cosine:>8.3f}"
            )
        print("-" * 92)
        for row in rows:
            print(f"\n[{row.name} sample]\n{row.sample_text}")


def main() -> None:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    converter = TernaryGPT2Converter()
    if not (converter.output_dir / "manifest.json").exists():
        print("Converting GPT-2 to runtime ternary...")
        stats = converter.convert()
        print(f"Saved {converter.output_dir} ({_bytes_to_mb(stats.compressed_bytes):.2f} MB, {stats.compression_ratio:.2f}x)")
    else:
        converter._trim_large_sidecars()
        manifest = _read_manifest(converter.output_dir)
        print(f"Using existing ternary checkpoint: {converter.output_dir}")
        print(f"Ternary checkpoint size: {_bytes_to_mb(_directory_size(converter.output_dir)):.2f} MB")
        print(f"Tensors ternary: {manifest.get('stats', {}).get('tensors_ternary', 'unknown')}")

    rows = TernaryBenchmark().run()
    TernaryBenchmark.print(rows)
    print(f"\nBenchmark saved to {converter.output_dir / 'benchmark.json'}")


def _linear_low_rank_correction(original: torch.Tensor, approx: torch.Tensor, rank: int) -> dict[str, torch.Tensor]:
    max_rank = min(int(rank), min(original.shape) - 1)
    if max_rank <= 0:
        return {}
    residual = original - approx
    try:
        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    except RuntimeError:
        return {}
    return {
        "left": vh[:max_rank, :].t().contiguous(),
        "scale": s[:max_rank].contiguous(),
        "right": u[:, :max_rank].contiguous(),
    }


def _generate_sample(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return _windows_rss_mb()


def _windows_rss_mb() -> float:
    if os.name != "nt":
        return 0.0
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return 0.0
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


__all__ = [
    "TernaryLinear",
    "TernaryConv1D",
    "TernaryEmbedding",
    "TernaryLMHead",
    "TernaryGPT2Converter",
    "TernaryBenchmark",
    "compress_gpt2_runtime_ternary",
    "generate_with_runtime_ternary_gpt2",
    "load_runtime_ternary_gpt2_model",
]


if __name__ == "__main__":
    main()
