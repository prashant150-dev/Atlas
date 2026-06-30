"""AetherCore v3 local demo entry point.

The demo ties together the implemented compression, routing, memory, quality,
math, code, prompt, and inference components. It uses a tiny local PyTorch model
for compression benchmarking so it runs offline on CPU.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.compression import GodCompressionEngine, QualityMetrics
from src.core import AetherCoreV3


@dataclass(frozen=True, slots=True)
class DemoReport:
    """Summary printed by the main demo."""

    original_bytes: int
    compressed_bytes_estimated: int
    compression_ratio_estimated: float
    packed_bytes_estimated: int
    packed_ratio_estimated: float
    packed_quality: dict[str, Any]
    estimate_70b: dict[str, float]
    quality: dict[str, Any]
    original_runtime_ms: float
    compressed_runtime_ms: float
    inference_response: str
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


class TinyDemoModel(nn.Module):
    """Small CPU model used for local compression verification."""

    def __init__(self) -> None:
        """Create deterministic linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""

        return self.fc2(torch.tanh(self.fc1(x)))


def build_demo_model() -> TinyDemoModel:
    """Build a deterministic tiny model."""

    torch.manual_seed(101)
    model = TinyDemoModel()
    model.eval()
    return model


def benchmark_linear(weight: torch.Tensor, reconstructed: torch.Tensor, iterations: int = 200) -> tuple[float, float]:
    """Benchmark original and reconstructed linear weights."""

    x = torch.randn(32, weight.shape[1])
    started = time.perf_counter()
    for _ in range(iterations):
        F.linear(x, weight)
    original_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    for _ in range(iterations):
        F.linear(x, reconstructed)
    compressed_ms = (time.perf_counter() - started) * 1000.0
    return original_ms, compressed_ms


def run_demo(prompt: str, output_dir: str | Path, max_tokens: int) -> DemoReport:
    """Run compression and inference demo."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = build_demo_model()
    compression = GodCompressionEngine()
    state = model.state_dict()
    stats = compression.compress_model(state, output_path / "compressed_model")

    weight = state["fc1.weight"].detach().cpu()
    compressed_layer = compression.compress_layer(weight, name="fc1.weight.demo")
    reconstructed = compression.decompress_layer(compressed_layer)
    quality: QualityMetrics = compression.benchmark(weight, reconstructed)
    packed_layer = compression.compress_layer_packed(weight, name="fc1.weight.demo.packed")
    packed_reconstructed = compression.decompress_layer(packed_layer)
    packed_quality: QualityMetrics = compression.benchmark(weight, packed_reconstructed)
    packed_bits = int(packed_layer.metadata["estimated_bits"]["total_bits"])
    packed_bytes = (packed_bits + 7) // 8
    packed_ratio = (weight.numel() * weight.element_size()) / max(1, packed_bytes)
    estimate_70b = compression.packed_compressor.estimate_model_size(70_000_000_000)
    original_ms, compressed_ms = benchmark_linear(weight, reconstructed)

    engine = AetherCoreV3(
        model_path=output_path / "compressed_model",
        config={
            "context_dir": str(output_path / "context_kv"),
            "knowledge_base_path": str(output_path / "knowledge_base.jsonl"),
        },
    )
    response = engine.generate(prompt, max_tokens=max_tokens)

    return DemoReport(
        original_bytes=stats.original_bytes,
        compressed_bytes_estimated=stats.estimated_compressed_bytes,
        compression_ratio_estimated=stats.compression_ratio_estimated,
        packed_bytes_estimated=packed_bytes,
        packed_ratio_estimated=packed_ratio,
        packed_quality=asdict(packed_quality),
        estimate_70b=estimate_70b,
        quality=asdict(quality),
        original_runtime_ms=original_ms,
        compressed_runtime_ms=compressed_ms,
        inference_response=response,
        output_dir=str(output_path),
    )


def run_aether_prompt(prompt: str, output_dir: str | Path, max_tokens: int) -> str:
    """Run the local AetherCore scaffold without the compression demo."""

    output_path = Path(output_dir)
    engine = AetherCoreV3(
        config={
            "context_dir": str(output_path / "aether_context_kv"),
            "knowledge_base_path": str(output_path / "aether_knowledge_base.jsonl"),
        },
    )
    return engine.generate(prompt, max_tokens=max_tokens)


def run_gpt2_prompt(
    prompt: str,
    model_path: str | Path = "models/gpt2",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
) -> str:
    """Run real local GPT-2 generation from a Hugging Face checkpoint."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
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


def run_compressed_gpt2_prompt(
    prompt: str,
    compressed_path: str | Path = "experiments/gpt2_packed",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
) -> str:
    """Run generation from an AetherCore packed GPT-2 folder."""

    from src.compression.gpt2_packed import generate_with_packed_gpt2

    return generate_with_packed_gpt2(
        prompt,
        compressed_dir=compressed_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def run_gpt2_speed(
    model_path: str | Path,
    compressed_path: str | Path,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, dict[str, float]]:
    """Benchmark original and packed GPT-2 generation speed."""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.compression.gpt2_packed import benchmark_generation, load_packed_gpt2_model

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    original = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    packed = load_packed_gpt2_model(compressed_path)
    return {
        "original": benchmark_generation(original, tokenizer, prompt, max_new_tokens=max_new_tokens),
        "packed": benchmark_generation(packed, tokenizer, prompt, max_new_tokens=max_new_tokens),
    }


def interactive_chat(engine: AetherCoreV3) -> None:
    """Run a small interactive chat loop."""

    print("AetherCore v3 chat. Type /exit to stop.")
    messages: list[dict[str, str]] = []
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("bye")
            return
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        response = engine.chat(messages[-6:])
        messages.append({"role": "assistant", "content": response})
        print(f"aether> {response}")


def interactive_gpt2_chat(model_path: str | Path, max_new_tokens: int, temperature: float) -> None:
    """Run a tiny interactive loop against real local GPT-2."""

    print("GPT-2 local chat. Type /exit to stop.")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("bye")
            return
        if not user_input:
            continue
        response = run_gpt2_prompt(user_input, model_path=model_path, max_new_tokens=max_new_tokens, temperature=temperature)
        print(f"gpt2> {response}")


def interactive_compressed_gpt2_chat(compressed_path: str | Path, max_new_tokens: int, temperature: float) -> None:
    """Run a tiny interactive loop against packed GPT-2."""

    print("Packed GPT-2 local chat. Type /exit to stop.")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("bye")
            return
        if not user_input:
            continue
        response = run_compressed_gpt2_prompt(user_input, compressed_path, max_new_tokens=max_new_tokens, temperature=temperature)
        print(f"packed-gpt2> {response}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the AetherCore v3 local demo.")
    parser.add_argument(
        "--mode",
        choices=("demo", "aether", "gpt2", "gpt2-compress", "gpt2-packed", "gpt2-compare", "gpt2-speed", "both"),
        default="demo",
        help="Choose original demo, AetherCore scaffold, original GPT-2, packed GPT-2 compression/inference, or comparisons.",
    )
    parser.add_argument("--prompt", default="solve x^2 - 4 = 0", help="Prompt for the demo generation.")
    parser.add_argument("--aether-prompt", default=None, help="Prompt for the AetherCore scaffold mode.")
    parser.add_argument("--gpt2-prompt", default=None, help="Prompt for real GPT-2 mode.")
    parser.add_argument("--gpt2-model-path", default="models/gpt2", help="Local GPT-2 Hugging Face model directory.")
    parser.add_argument("--compressed-gpt2-path", default="experiments/gpt2_packed", help="AetherCore packed GPT-2 directory.")
    parser.add_argument("--gpt2-new-tokens", type=int, default=60, help="Maximum new tokens for GPT-2 generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="GPT-2 sampling temperature.")
    parser.add_argument("--block-size", type=int, default=256, help="Packed ternary block size for GPT-2 compression.")
    parser.add_argument("--threshold-factor", type=float, default=0.7, help="Packed ternary threshold factor.")
    parser.add_argument("--gpt2-compression", choices=("int8", "ternary"), default="int8", help="Compression format for GPT-2 packed folders.")
    parser.add_argument("--output-dir", default="experiments/main_demo", help="Directory for demo artifacts.")
    parser.add_argument("--max-tokens", type=int, default=120, help="Approximate max response tokens.")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat after the demo.")
    return parser.parse_args()


def main() -> int:
    """Run the AetherCore v3 local demo."""

    args = parse_args()
    aether_prompt = args.aether_prompt or args.prompt
    gpt2_prompt = args.gpt2_prompt or args.prompt

    if args.mode == "aether":
        print("AetherCore scaffold response")
        print(run_aether_prompt(aether_prompt, args.output_dir, args.max_tokens))
        if args.chat:
            engine = AetherCoreV3(
                config={
                    "context_dir": str(Path(args.output_dir) / "chat_context_kv"),
                    "knowledge_base_path": str(Path(args.output_dir) / "chat_knowledge_base.jsonl"),
                },
            )
            interactive_chat(engine)
        return 0

    if args.mode == "gpt2":
        print("GPT-2 local response")
        print(run_gpt2_prompt(gpt2_prompt, args.gpt2_model_path, args.gpt2_new_tokens, args.temperature))
        if args.chat:
            interactive_gpt2_chat(args.gpt2_model_path, args.gpt2_new_tokens, args.temperature)
        return 0

    if args.mode == "gpt2-compress":
        from src.compression.gpt2_packed import compress_gpt2_packed

        stats = compress_gpt2_packed(
            model_path=args.gpt2_model_path,
            output_dir=args.compressed_gpt2_path,
            block_size=args.block_size,
            threshold_factor=args.threshold_factor,
            compression=args.gpt2_compression,
        )
        print("Packed GPT-2 compression complete")
        for key, value in stats.to_dict().items():
            if key != "details":
                print(f"  {key}: {value}")
        print(f"  details: {stats.details}")
        return 0

    if args.mode == "gpt2-packed":
        print("Packed GPT-2 response")
        print(run_compressed_gpt2_prompt(gpt2_prompt, args.compressed_gpt2_path, args.gpt2_new_tokens, args.temperature))
        if args.chat:
            interactive_compressed_gpt2_chat(args.compressed_gpt2_path, args.gpt2_new_tokens, args.temperature)
        return 0

    if args.mode == "gpt2-compare":
        print("Original GPT-2 response")
        print(run_gpt2_prompt(gpt2_prompt, args.gpt2_model_path, args.gpt2_new_tokens, args.temperature))
        print()
        print("Packed GPT-2 response")
        print(run_compressed_gpt2_prompt(gpt2_prompt, args.compressed_gpt2_path, args.gpt2_new_tokens, args.temperature))
        return 0

    if args.mode == "gpt2-speed":
        results = run_gpt2_speed(args.gpt2_model_path, args.compressed_gpt2_path, gpt2_prompt, args.gpt2_new_tokens)
        print("GPT-2 speed benchmark")
        for name, result in results.items():
            print(f"  {name}: {result['tokens_per_sec']:.3f} tokens/sec ({int(result['new_tokens'])} tokens in {result['elapsed_sec']:.4f} sec)")
        return 0

    if args.mode == "both":
        print("AetherCore scaffold response")
        print(run_aether_prompt(aether_prompt, args.output_dir, args.max_tokens))
        print()
        print("GPT-2 local response")
        print(run_gpt2_prompt(gpt2_prompt, args.gpt2_model_path, args.gpt2_new_tokens, args.temperature))
        return 0

    report = run_demo(args.prompt, args.output_dir, args.max_tokens)
    print("AetherCore v3 local demo")
    print(f"  output dir: {report.output_dir}")
    print(f"  original bytes: {report.original_bytes}")
    print(f"  estimated compressed bytes: {report.compressed_bytes_estimated}")
    print(f"  estimated compression ratio: {report.compression_ratio_estimated:.3f}x")
    print(f"  packed layer bytes estimated: {report.packed_bytes_estimated}")
    print(f"  packed layer ratio estimated: {report.packed_ratio_estimated:.3f}x")
    print(f"  packed quality score: {report.packed_quality['quality_score']:.4f}")
    print(f"  70B FP16 estimate: {report.estimate_70b['fp16_gb']:.2f} GB")
    print(f"  70B packed ternary estimate: {report.estimate_70b['packed_gb']:.2f} GB")
    print(f"  70B packed ratio estimate: {report.estimate_70b['compression_ratio']:.2f}x")
    print(f"  quality score: {report.quality['quality_score']:.4f}")
    print(f"  relative L2 error: {report.quality['relative_l2_error']:.4f}")
    print(f"  original benchmark: {report.original_runtime_ms:.3f} ms")
    print(f"  reconstructed benchmark: {report.compressed_runtime_ms:.3f} ms")
    print("  inference response:")
    print(report.inference_response)

    if args.chat:
        engine = AetherCoreV3(
            model_path=Path(args.output_dir) / "compressed_model",
            config={
                "context_dir": str(Path(args.output_dir) / "chat_context_kv"),
                "knowledge_base_path": str(Path(args.output_dir) / "chat_knowledge_base.jsonl"),
            },
        )
        interactive_chat(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
