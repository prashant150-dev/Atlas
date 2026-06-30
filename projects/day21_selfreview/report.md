# Day 21 — Self-review with a bounded loop + stop-gate (the user's idea)

## The idea
After the model answers, let it RE-READ its own output, ask "did I write something
wrong / could this be better?", and fix it — but WITHOUT falling into an endless
"improve it again... and again..." loop. This is the lever that narrows the small
remaining quality gap to fp16 (the model catches its own mistakes), and it is pure
software — buildable on this PC, not hardware-gated.

## Three guardrails that make the loop safe and convergent
1. **BOUNDED** — at most MAX_ROUNDS attempts, full stop.
2. **STOP-GATE** — if the reviewer finds no real issues, stop immediately; do NOT keep
   polishing a correct answer.
3. **NO-REGRESS** — every candidate gets a quality SCORE; accept a revision only if it
   strictly BEATS the current best. If a "fix" doesn't help, keep the previous best and
   stop. This is what stops "trying to make it better" from making it worse or looping.

Generic API: plug in `score_fn(answer) -> (score, issues)` and
`revise_fn(answer, issues) -> answer`. For code, score_fn runs the tests; for prose,
a checklist. The loop logic is identical.

## Verified (self_review.py self-test — three scenarios)
- **A) buggy answer**: fixes each round, STOPS when clean. → "clean", 2 rounds, score 1.0.
- **B) useless fix** (score never improves): rejects it, keeps best, STOPS. → 1 round.
- **C) endless tiny polish** (always "0.001 better"): the BOUNDED cap stops it. → 3 rounds
  (= the cap), "hit round limit".
No scenario loops forever; correct answers are not over-polished.

## Real-code demo (demo_code_review.py — a buggy Flappy-Bird physics step)
Three planted bugs: wrong gravity sign, `score + pipe` (undefined var), `>` instead of
`>=` at the floor. Self-review:
```
round 1: score 0.33 -> accepted (2 issues left)
round 2: score 0.67 -> accepted (1 issue left)
round 3: score 1.00 -> accepted (clean)
stopped because: clean (no real issues left)  -> all 3 bugs fixed, then STOPPED.
```

## The honest lesson found mid-build (kept, not hidden)
The first version of the floor check passed even with the `>` bug, because once gravity
was fixed the bird moved PAST the floor and `>` happened to fire — so the loop reported
"clean" while a bug remained. **Self-review is only as good as its checks (score_fn).**
A weak reviewer declares victory early. We fixed the check to land the bird exactly ON
the floor, isolating the off-by-one, and then the loop correctly fixed all three. This
is the key caveat for deploying self-review: invest in the scorer/tests, or the model
will "stop when it thinks it's clean", not when it actually is.

## How this fits the dream
This is a software lever on Part-4 (intelligence/quality): it recovers some of the small
fp16 gap by having the model verify its own work, and the guardrails guarantee it
terminates (no infinite "improvement" loop). It composes with the existing self-healing
code engine (`src/code_engine/executor.py`, which already retries up to 3×) — this module
generalises that pattern (bounded + stop-gate + no-regress) to ANY answer with a scorer.

## Files
- `self_review.py` — the generic bounded/gated/no-regress loop + 3-scenario self-test
- `demo_code_review.py` — real-code demo on a buggy Flappy-Bird physics function
- `self_review_results.json` — recorded self-test rounds
