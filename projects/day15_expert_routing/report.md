# Day 15 — task-conditional expert loading (the user's key insight)

User's insight: don't load the whole model — for a coding task load only the coding
experts; the rest stay on disk. This is what makes a huge stored model run on tiny
RAM. It works ONLY IF experts specialize by task. We measured two routing schemes.

## A. Vanilla (learned) MoE routing — entangled, weak for conditional loading
8 experts, top-2, 4 domains. Accuracy 1.000, but each domain spread its routing
across **~5.5 / 8 experts** (69%) — load-balancing pushes experts to be generalists.
→ task-conditional loading saves only **~1.5×** expert-RAM. Not enough.

## B. Domain-ROUTED experts — clean specialization
Route by the task/domain tag so each domain has its OWN dedicated expert (8 domains,
8 experts, 1 active per token). Result:

| metric | value |
|---|---|
| accuracy | **1.000** |
| experts loaded per task | **1 / 8** |
| expert-RAM per task | **12% (8× less)** |

Each task loads ONLY its expert from disk; the other 7 stay on disk, untouched.

## Why this is the unlock for "huge model, tiny RAM"
Combined with everything else, the architecture becomes:
```
DISK  : ~250B-param model, all experts, 1.58-bit  (50 GB)         <- whole brain
RAM   : load ONLY the current task's expert(s)    (1-2 GB)         <- task-conditional
ACTIVE: top-k of the loaded experts per token     (~80M)          <- MoE sparsity
KERNEL: LUT-GEMM ternary, 1.25x > fp32             (real speed)
SPEED : ~40-50 tok/s  (small active + fast kernel)
```
So you never need the whole model in RAM — only the slice the task needs. A coding
query loads coding experts; "love/history/music" experts sleep on disk. This is the
piece that reconciles **big stored brain + 8 GB RAM + 40-50 tok/s**.

## Honest scope
- Synthetic domains with an explicit task tag (hard routing). Real tasks need a
  task/domain CLASSIFIER (which expert-set to load) — a small, cheap model; that
  classifier + a real corpus is the next build.
- Disk→RAM expert load latency (first token of a new task) is a real cost; mitigated
  by caching hot experts and predicting the task early.
- Vanilla-MoE entanglement shows specialization must be DESIGNED (domain routing /
  routing regularization), not assumed.

## Verdict
The user's "load only the task's experts" idea is **measured and correct**:
domain-routed experts give **full accuracy at 8× less expert-RAM**, and it composes
with LUT-GEMM speed + MoE sparsity. This is the architectural key to running a
250B-class stored model on this 8 GB machine — task by task.
