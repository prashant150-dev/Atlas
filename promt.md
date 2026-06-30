You are an expert AI systems engineer building AetherCore v3 — a revolutionary AI inference system.

MISSION:
Build a complete, production-ready, God Level AI compression and inference system that can run 400B parameter equivalent AI on ANY consumer device (1GB RAM, basic CPU, no GPU).

HARDWARE TARGET:
- Minimum: 1GB RAM, 4GB storage, any 2GHz CPU
- Recommended: 4GB RAM, 16GB storage
- No GPU required ever

PROJECT LOCATION: C:\aethercore_v3\

EXISTING STRUCTURE:
C:\aethercore_v3\
├── src\
│   ├── core\
│   ├── compression\        ← engine.py already started
│   ├── router\
│   ├── memory\
│   ├── quality\
│   ├── math_engine\
│   ├── code_engine\
│   ├── prompt\
│   └── utils\
├── models\
│   ├── experts\
│   ├── core\
│   └── correction\
├── experiments\
└── data\

PYTHON: 3.12.10
PYTORCH: 2.11.0+cpu
OS: Windows 10

══════════════════════════════════════════
BUILD THESE COMPONENTS — ALL COMPLETE:
══════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 1: src/compression/engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
God Level Compression Engine — 400x compression target

Classes to build:
1. TernaryQuantizer
   - quantize(weight) → (ternary, scale)
   - dequantize(ternary, scale) → weight
   - sparsity(ternary) → float
   - Quality target: 88% of FP16

2. DynamicBitAllocator  
   - analyze_importance(weight) → importance_scores
   - allocate_bits(weight) → {'critical':4bit, 'important':2bit, 'normal':1bit}
   - compress(weight) → compressed_dict
   - Average bits: 1.25 per weight
   - Quality improvement: +4% over pure ternary

3. DeltaCompressor
   - compute_deltas(layers_list) → delta_dict
   - reconstruct(delta_dict) → layers_list
   - Compression: 2-3x additional
   - Works across transformer layers

4. SemanticDeduplicator
   - fingerprint(weight) → tensor
   - find_duplicates(weights_dict) → dedup_map
   - similarity_threshold: 0.95
   - Expected savings: 30-50% of storage

5. CorrectionTableExtractor (MOST NOVEL)
   - extract(original_weight, compressed_weight) → correction_table
   - apply(compressed_output, correction_table) → corrected_output
   - Size: <1% of original weight size
   - Quality recovery: +8-12%

6. GodCompressionEngine (Main orchestrator)
   - compress_model(model, output_dir) → CompressionStats
   - compress_layer(layer_weights) → CompressedLayer
   - decompress_layer(CompressedLayer) → weights
   - save_experts(compressed, path) → None
   - load_expert(path, expert_id) → CompressedLayer
   - benchmark(original, compressed) → QualityMetrics
   
   TARGET METRICS:
   - 400B model: 800GB → 3GB stored (267x)
   - Active RAM: <600MB
   - Quality: >97% of FP16
   - Speed: 3x faster inference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 2: src/core/expert.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1-bit Expert Block with Correction

Classes:
1. TernaryExpert(nn.Module)
   - weights: ternary {-1, 0, +1}
   - correction_table: FP16 (tiny)
   - forward(x, use_correction=True)
   - sparsity() → float
   - save_to_file(path)
   - load_from_file(path)
   
2. ExpertPool
   - register_expert(expert_id, expert)
   - get_expert(expert_id) → TernaryExpert
   - active_experts() → List
   - sleeping_experts() → List

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 3: src/router/beast_router.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3-Level Beast Router — <0.25ms total

Classes:
1. DomainFlashClassifier
   - classify(input_embedding) → domain
   - domains: [MATH, CODE, LANGUAGE, LOGIC, SCIENCE, CREATIVE]
   - latency target: <0.05ms
   - Uses: lookup table, not neural network

2. ExpertPredictor
   - predict(last_8_tokens, domain) → [expert_ids]
   - Updates routing history
   - latency target: <0.15ms
   - Prefetch signal generate karo

3. PrecisionDecider
   - decide(token, confidence) → precision_level
   - EASY (conf>0.9): pure 1-bit, skip correction
   - MEDIUM (0.5-0.9): 1-bit + light correction
   - HARD (conf<0.5): full correction + multiple experts
   - latency target: <0.05ms

4. BeastRouter (orchestrator)
   - route(input_tokens) → RoutingDecision
   - RoutingDecision: {domain, expert_ids, precision, confidence, prefetch_list}
   - Total latency: <0.25ms
   - Zero Python loops in hot path

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 4: src/memory/god_manager.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
God Level Memory Manager

Classes:
1. MemoryTier (enum)
   - L1_CPU_CACHE: <32MB, fastest
   - L2_RAM_HOT: active experts
   - L3_RAM_WARM: recent experts
   - L4_SSD_COLD: sleeping experts

2. ExpertMemoryManager
   - hot_tier: Dict[expert_id, TernaryExpert] (max 2GB)
   - warm_tier: Dict[expert_id, TernaryExpert] (max 1GB)
   - cold_tier: SSD paths
   - load_expert_async(expert_id) → Future
   - evict_lfu() → evicted_id
   - prefetch(expert_ids_list) → None
   - memory_stats() → MemoryStats

3. AsyncSSDLoader
   - queue: priority queue of load requests
   - load_async(path) → Future[TernaryExpert]
   - bandwidth_monitor() → MB/s
   - Target: hide SSD latency behind compute

4. HotColdBalancer
   - track_access(expert_id)
   - suggest_promotion(expert_id) → bool
   - suggest_eviction() → expert_id
   - Uses: LFU + recency hybrid

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 5: src/memory/infinite_context.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Infinite Context System

Classes:
1. ContextTier (enum)
   - WORKING: last 2K tokens, FP16, RAM
   - SHORT_TERM: 2K-20K tokens, 4-bit, RAM
   - LONG_TERM: 20K-1M tokens, 1-bit, SSD
   - PERMANENT: cross-session facts, always

2. HierarchicalKVCache
   - store(token_id, key, value, importance)
   - retrieve(token_id) → (key, value)
   - auto_tier(token_id) based on importance
   - compress_tier(tier) background
   - Target: 1M tokens in <1GB RAM

3. MemoryConsolidator
   - consolidate(tokens_1000) → facts
   - extract_facts(text) → List[Fact]
   - store_permanent(fact)
   - Runs every 1000 tokens background
   - Enables: truly infinite context

4. ImportanceScorer
   - score(token, attention_weights) → float
   - High score → keep in working memory
   - Low score → compress/evict

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 6: src/quality/hallucination_killer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zero Hallucination System

Classes:
1. UncertaintyDetector
   - score(logits, top_k=10) → confidence: float
   - HIGH (>0.9): model sure hai
   - MEDIUM (0.5-0.9): verify karo
   - LOW (<0.5): "I don't know" say
   - Per-token, real-time

2. FactVerifier
   - verify(claim: str) → VerificationResult
   - VerificationResult: {verified, confidence, source}
   - Uses: local knowledge base
   - Runs post-generation

3. KnowledgeBoundaryEnforcer
   - knows(topic) → bool
   - confidence(topic) → float
   - Prevents: confident wrong answers
   - Forces: honest uncertainty expression

4. HallucinationKiller (orchestrator)
   - check_generation(token, logits) → safe_token
   - verify_response(response) → verified_response
   - Target: <0.01% hallucination rate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 7: src/math_engine/symbolic.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Beast Math Engine — 99.9% accuracy

Classes:
1. MathParser
   - parse(text) → MathExpression
   - Handles: algebra, calculus, statistics
   - LaTeX + natural language input

2. SymbolicSolver
   - solve(expression) → exact_result
   - differentiate(expr) → derivative
   - integrate(expr) → integral
   - Uses: sympy under the hood
   - Accuracy: 100% (symbolic = exact)

3. MathVerifier
   - verify(problem, solution) → bool
   - Numerical verification
   - Multiple approach cross-check

4. MathExplainer
   - explain(problem, solution) → steps
   - Step by step natural language
   - Multiple approaches shown

5. BeastMathEngine (orchestrator)
   - solve(problem_text) → MathResult
   - MathResult: {answer, steps, verification, confidence}
   - Hybrid: symbolic exact + neural explanation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 8: src/code_engine/executor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Self-Healing Code Engine — 98% accuracy

Classes:
1. CodeGenerator
   - generate(task, language) → code: str
   - Supports: Python, JS, C++, Java, etc.

2. StaticAnalyzer
   - analyze(code) → List[Issue]
   - AST parsing
   - Common error detection
   - Security vulnerability check

3. SandboxExecutor
   - execute(code, timeout=5) → ExecutionResult
   - Safe isolated environment
   - ExecutionResult: {output, errors, runtime}
   - Supports: Python execution

4. TestGenerator
   - generate_tests(code, task) → List[Test]
   - Edge cases included
   - Auto run tests

5. SelfHealingLoop
   - heal(code, errors) → fixed_code
   - Max 3 iterations
   - If unfixable → explain why

6. BeastCodeEngine (orchestrator)
   - solve(task) → CodeResult
   - CodeResult: {code, tests, output, verified}
   - Guarantee: working code or honest failure

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 9: src/prompt/refiner.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Auto Prompt Refinement System

Classes:
1. IntentDetector
   - detect(user_input) → Intent
   - Intent: {type, domain, complexity, format_needed}
   - Types: QUESTION, TASK, CREATIVE, ANALYSIS, DEBUG

2. PromptEnhancer
   - enhance(original, intent) → enhanced_prompt
   - Adds: context, constraints, format instructions
   - Makes vague prompts specific

3. QueryDecomposer
   - decompose(complex_query) → List[SimpleQuery]
   - Multi-part questions split karo
   - Answer each → combine

4. ResponseValidator
   - validates(response, intent) → bool
   - Does response match intent?
   - If not → trigger regeneration

5. AutoPromptRefiner (orchestrator)
   - refine(user_input) → refined_prompt
   - validate_response(response, original) → final_response

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 10: src/core/inference_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main Inference Engine — ties everything together

Classes:
1. AetherCoreV3 (main class)
   - __init__(model_path, config)
   - generate(prompt, max_tokens=500) → str
   - chat(messages) → str
   - Integrates: ALL components above

2. TokenGenerator
   - next_token(context, experts) → token
   - Uses: sparse activation
   - Uses: correction engine
   - Uses: confidence gating

3. PipelineOrchestrator
   - 4 parallel threads:
     Thread 0: Inference (current token)
     Thread 1: SSD Loader (next experts)
     Thread 2: Router (prediction)
     Thread 3: Memory Manager (KV cache)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 11: main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry point — complete working demo

- Load model (GPT-2 for testing)
- Compress with GodCompressionEngine
- Run inference with AetherCoreV3
- Show: original vs compressed comparison
- Benchmark: speed, quality, RAM usage
- Interactive chat mode

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every file: complete, runnable, no placeholders
- Every class: full implementation
- Every method: working code
- Type hints everywhere
- Docstrings everywhere
- Error handling everywhere
- Each file has: if __name__ == "__main__": test
- No "TODO" or "pass" without implementation
- Compatible: Python 3.12, PyTorch 2.11.0+cpu, Windows 10
- Install if needed: pip install sympy transformers

PERFORMANCE REQUIREMENTS:
- Router: <0.25ms
- Compression: >100x
- Quality: >92% of FP16
- RAM active: <600MB for large models
- Tokens/sec: >20 on basic CPU

BUILD ORDER:
1. src/compression/engine.py (most important)
2. src/core/expert.py
3. src/router/beast_router.py
4. src/memory/god_manager.py
5. src/memory/infinite_context.py
6. src/quality/hallucination_killer.py
7. src/math_engine/symbolic.py
8. src/code_engine/executor.py
9. src/prompt/refiner.py
10. src/core/inference_engine.py
11. main.py

Build one by one, confirm each works, then next.
Start with compression/engine.py NOW.
Show complete file, no truncation.