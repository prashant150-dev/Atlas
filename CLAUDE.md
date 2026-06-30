# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AetherCore v3 is a local, CPU-only prototype for AI weight compression and inference. `promt.md` is the original aspirational spec (run "400B params on 1GB RAM"); the actual code is an **honest local scaffold** that does not claim unverifiable miracles. When the spec and the code conflict, follow the code. Compression results are reported as *estimated* vs *actual serialized* bytes, and the inference engine is a deterministic rule-based router over real sub-engines (sympy math, sandboxed Python, a byte-level token generator), not a trained large model.

The one place real model inference happens is the GPT-2 path: a local Hugging Face GPT-2 checkpoint under `models/gpt2/` is compressed (int8 or packed/runtime ternary) and reloaded into a normal Transformers model for genuine generation comparisons.

## Environment

- Python 3.12, `torch==2.11.0+cpu`, `transformers` 5.x, plus `sympy` and `safetensors`. CPU only — never assume CUDA.
- No `requirements.txt`, no test runner, no lint config, no CI. There is no `pip install -e`; the project runs from the repo root.
- Imports use absolute `src.*` paths. `src/core/inference_engine.py` inserts the repo root onto `sys.path`, but in general **run commands from `C:\aethercore_v3`** so `src` resolves.

## Running things

```bash
# Compression + scaffold inference demo (tiny local model, fully offline)
python main.py --mode demo --prompt "solve x^2 - 4 = 0"

# Rule-based scaffold engine only (math/code/knowledge routing), optional chat
python main.py --mode aether --aether-prompt "write a factorial function" --chat

# Real local GPT-2 (requires models/gpt2/ checkpoint present)
python main.py --mode gpt2 --gpt2-prompt "Hello"

# Compress GPT-2 -> packed folder, then run / compare / benchmark it
python main.py --mode gpt2-compress --gpt2-compression int8 --compressed-gpt2-path experiments/gpt2_packed
python main.py --mode gpt2-packed  --compressed-gpt2-path experiments/gpt2_packed
python main.py --mode gpt2-compare --compressed-gpt2-path experiments/gpt2_packed
python main.py --mode gpt2-speed   --compressed-gpt2-path experiments/gpt2_packed
```

`--mode` options: `demo`, `aether`, `gpt2`, `gpt2-compress`, `gpt2-packed`, `gpt2-compare`, `gpt2-speed`, `both`. All artifacts are written under `experiments/` (e.g. `experiments/main_demo/`, `experiments/gpt2_packed/`).

## Testing

There is no pytest suite. **Every implementation module has a `__main__` self-test** that doubles as its test. Run a module directly to exercise it:

```bash
python -m src.compression.engine
python -m src.core.inference_engine
python -m src.router.beast_router
python -m src.math_engine.symbolic
python -m src.code_engine.executor
# ...same pattern for memory, quality, prompt, compression.* modules
```

Run all self-tests at once:

```bash
for m in src.compression.engine src.compression.ternary_runtime src.compression.healing_qat \
         src.compression.rate_distortion_limit src.core.expert src.core.inference_engine \
         src.math_engine.symbolic src.code_engine.executor src.memory.god_manager \
         src.memory.infinite_context src.quality.hallucination_killer src.prompt.refiner \
         src.router.beast_router src.architecture.aethernet src.architecture.experiment; \
    do python -m "$m" || echo "FAILED: $m"; done
```

Self-tests assert on exact expected outputs and raise `RuntimeError` on mismatch (e.g. the math self-test requires `[-2, 2]` in the solution, the inference self-test checks for specific substrings). **When you change behavior, update the corresponding `_self_test()`** — it is the contract.

## Architecture

`main.py` (entry/orchestration) wires two largely independent stacks: the **compression** stack and the **inference scaffold** stack.

### Inference scaffold — `src/core/inference_engine.py`
`AetherCoreV3` is the top-level engine. `generate()` runs a `PipelineOrchestrator` (a `ThreadPoolExecutor` with 4 logical workers: route, refine-prompt, update-KV-memory, prefetch) and then dispatches by domain in `_generate_with_routing`:
- MATH → `BeastMathEngine` (`src/math_engine/symbolic.py`, sympy-backed, exact)
- CODE → `BeastCodeEngine` (`src/code_engine/executor.py`, generates + sandbox-executes + self-heals Python)
- otherwise → local-fact lookup / `KnowledgeBoundaryEnforcer` fallback ("I don't know based on the local knowledge base")

Supporting subsystems, each its own package with an orchestrator class:
- `src/router/beast_router.py` — `BeastRouter` = `DomainFlashClassifier` (keyword lookup, not a NN) + `ExpertPredictor` + `PrecisionDecider`, returns a `RoutingDecision`.
- `src/quality/hallucination_killer.py` — `HallucinationKiller` = `UncertaintyDetector` + `FactVerifier` (local JSONL knowledge base) + `KnowledgeBoundaryEnforcer`. Gates tokens and enforces honest uncertainty.
- `src/memory/infinite_context.py` — `HierarchicalKVCache` (tiered working/short/long-term KV with `ImportanceScorer`), `MemoryConsolidator`.
- `src/memory/god_manager.py` — `ExpertMemoryManager`, `AsyncSSDLoader`, `HotColdBalancer` (hot/warm/cold expert tiering).
- `src/prompt/refiner.py` — `AutoPromptRefiner` = `IntentDetector` + `PromptEnhancer` + `QueryDecomposer` + `ResponseValidator`.
- `src/core/expert.py` — `TernaryExpert` (ternary {-1,0,+1} weights + tiny FP16 correction table) and `ExpertPool`.

### Compression stack — `src/compression/`
- `engine.py` is the core. `GodCompressionEngine` orchestrates `TernaryQuantizer`, `DynamicBitAllocator`, `DeltaCompressor`, `SemanticDeduplicator`, `CorrectionTableExtractor`, and `PackedTernaryCompressor`. Round-trips tensors via `CompressedLayer` (`to_dict`/`from_dict`, torch.save-friendly) and reports `QualityMetrics` / `CompressionStats` that distinguish **estimated** from **actual serialized** sizes.
- `gpt2_packed.py` — `compress_gpt2_packed()` / `load_packed_gpt2_model()` / `generate_with_packed_gpt2()`: GPT-2-specific, stores weights as int8 or packed ternary in safetensors + a JSON manifest, reloads into a real Transformers model.
- `gpt2_runtime_ternary.py` — alternative runtime-ternary GPT-2 (`TernaryConv1D`/`TernaryEmbedding`/`TernaryLMHead` modules), with `compare_gpt2_variants()`.
- `ternary_runtime.py` — supporting ternary-linear + low-rank-correction runtime layers.
- `rate_distortion_limit.py` — the **Day-1 floor**: computes the architecture-independent rate-distortion bound from real GPT-2 weights (entropy ≈ 2.04 bits/weight; 100× ⇒ ≥80% signal loss). Used to keep claims honest — no method may cross this.
- `healing_qat.py` — the **Day-2 lever**: distils a frozen FP GPT-2 teacher into a ternary student via straight-through-estimator QAT ("preserve behaviour, not weights"), recovering naive ternary from ~3% to ~28% top-1 in 30 steps.

### Native ternary/sparse architecture — `src/architecture/`
The **Day-3 co-design** track: design models natively ternary + sparse-MoE rather than compressing a dense FP model after the fact. `aethernet.py` defines `AetherNet`/`AetherNetConfig` (the native model), `DenseFP` (FP baseline), `PostHocTernary` (the baseline ternarized with no retraining), and `BitAccount` (counts stored *and* active bits/weight, overhead included). `experiment.py` runs all three on synthetic tasks and writes `projects/day3_aethernet/results.json`. Thesis: on the harder task the natively-trained AetherNet nearly matches FP while post-hoc ternary collapses.

### Research track — `projects/`
The active work is an evidence-driven compression-limits study, not feature work on the scaffold. **`projects/ROADMAP.md` is the source of truth** for direction (written bilingually Hindi/English). Core thesis: exact-weight compression has a hard rate-distortion floor (Day-1, *proven*), but **changing the goal from "match weights" to "preserve behaviour"** reopens the frontier via QAT/healing, sparsity (MoE), and native low-bit co-design. Foundation results are checked in:
- `projects/day1_compression_limit/` — the rate-distortion floor (D1).
- `projects/day2_healing_qat/` — healing recovers ternary collapse (D2).
- `projects/day3_aethernet/` — native ternary+sparse vs post-hoc (D3).

Guardrails from the roadmap apply to any new experiment: **measure don't assume** (`_self_test()` is the contract), always keep a naive baseline and only claim a "win" when you beat it at equal size, count *all* overhead (scales/correction/seed) in bits/weight, and don't re-attack the proven-impossible floor.

> Root scripts `claude_guardian.py` and `auto_continue.py` are agent-automation harnesses (subprocess/GUI driving), **not** part of AetherCore — ignore them for product work.

### Package import convention
Every `src/*/__init__.py` defines `__all__` and uses a lazy `__getattr__` to import the submodule only on first attribute access (keeps `import src.x` cheap and avoids eager torch/transformers loading). When adding a public class, **add it to both `__all__` and the `__getattr__` dispatch** in that package's `__init__.py`.

## Conventions

- Dataclasses are `@dataclass(frozen=True, slots=True)` (or `slots=True` when mutable) with a `to_dict()` returning JSON-friendly dicts. Results are structured dataclasses, not loose dicts.
- Full type hints and docstrings on every public class/method; explicit input validation that raises `TypeError`/`ValueError`.
- `from __future__ import annotations` at the top of modules.
- Generated/experiment output goes under `experiments/` (much of it is git-staged scratch — `experiments/_tmp/`, `_code_sandbox/`, selftest dirs; avoid adding to it). Persistent inputs go under `models/` and `data/`.
