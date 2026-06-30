"""Demo: self-review on REAL Python code — fix bugs, then STOP (no endless polishing).

Uses the generic self_review() loop with a code-specific scorer:
  score_fn  = run the code + its checks; score = fraction of checks passing; issues =
              the actual error messages / failed assertions.
  revise_fn = a tiny rule-based "fixer" standing in for the model's revision step
              (real system: the model rewrites given the issues).

The point is the LOOP BEHAVIOUR, identical to what a real model would drive:
re-read -> find concrete problems -> fix -> re-check -> stop when clean or no gain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_review import self_review  # type: ignore

# A buggy "game logic" snippet (Flappy-Bird-ish physics): three planted bugs —
#   1. gravity has the wrong sign (bird flies up instead of falling)
#   2. score increments by 'pipe' (a typo) instead of 1
#   3. collision uses '>' where it must be '>=' (off-by-one at the floor)
BUGGY = """
def step(bird_y, vel, gravity, pipe_passed, score, floor):
    vel = vel - gravity          # BUG1: should be vel + gravity (down is positive)
    bird_y = bird_y + vel
    if pipe_passed:
        score = score + pipe      # BUG2: 'pipe' undefined; should be score + 1
    crashed = bird_y > floor      # BUG3: should be >= floor
    return bird_y, vel, score, crashed
"""

# The three fixes the reviewer can apply, in the order it would find them.
_FIXES = [
    ("vel = vel - gravity", "vel = vel + gravity"),
    ("score = score + pipe", "score = score + 1"),
    ("crashed = bird_y > floor", "crashed = bird_y >= floor"),
]


def _checks(code: str):
    """exec the code and run behavioural checks; return (score, issues)."""
    issues = []
    ns = {}
    try:
        exec(code, ns)
        step = ns["step"]
    except Exception as e:  # noqa: BLE001
        return 0.0, [f"code does not even import/parse: {e!r}"]

    # check 1: with positive gravity the bird must move DOWN (y increases) when falling
    try:
        y2, v2, _s, _c = step(bird_y=10.0, vel=0.0, gravity=1.0,
                              pipe_passed=False, score=0, floor=100.0)
        if not (y2 > 10.0):
            issues.append("physics: bird does not fall under gravity (gravity sign wrong)")
    except Exception as e:  # noqa: BLE001
        issues.append(f"crash in physics step: {e!r}")

    # check 2: passing a pipe must add exactly 1 to score
    try:
        _y, _v, s2, _c = step(bird_y=10.0, vel=0.0, gravity=1.0,
                             pipe_passed=True, score=5, floor=100.0)
        if s2 != 6:
            issues.append("scoring: passing a pipe did not add exactly 1")
    except Exception as e:  # noqa: BLE001
        issues.append(f"scoring crashed (undefined variable?): {e!r}")

    # check 3: landing EXACTLY on the floor must count as a crash (>= vs >).
    # Isolate the off-by-one: choose vel so the bird ends exactly AT floor (not past it),
    # otherwise '+gravity' would push it past and mask the '>' bug.
    try:
        # bird_y=100, vel=-1, gravity=1 -> vel becomes 0 -> bird_y stays 100 == floor
        _y, _v, _s, c3 = step(bird_y=100.0, vel=-1.0, gravity=1.0,
                             pipe_passed=False, score=0, floor=100.0)
        if not c3:
            issues.append("collision: landing exactly on the floor is not detected (off-by-one)")
    except Exception as e:  # noqa: BLE001
        issues.append(f"collision check crashed: {e!r}")

    total = 3
    score = (total - len(issues)) / total
    return score, issues


def _revise(code: str, issues: list[str]) -> str:
    """apply the first pending fix (stand-in for the model rewriting given the issues)."""
    for bad, good in _FIXES:
        if bad in code:
            return code.replace(bad, good)
    return code


def main():
    print("Task: review a buggy Flappy-Bird physics function and fix it.\n", flush=True)
    s0, iss0 = _checks(BUGGY)
    print(f"initial code: score {s0:.2f} (3 checks), problems found:", flush=True)
    for i in iss0:
        print(f"   - {i}", flush=True)

    res = self_review(BUGGY, _checks, _revise, max_rounds=5)
    print(f"\nself-review ran {len(res.rounds)} round(s):", flush=True)
    for r in res.rounds:
        print(f"   round {r.n}: score {r.score:.2f} -> {r.action}"
              + (f" (issues left: {len(r.issues)})" if r.issues else " (clean)"), flush=True)
    print(f"\nstopped because: {res.stopped_because}", flush=True)
    print(f"final score: {res.best_score:.2f}  "
          f"({'all checks pass' if res.best_score == 1.0 else 'still imperfect'})", flush=True)
    print("\nfinal code:\n" + res.final_answer, flush=True)
    print("Note: once all 3 checks passed, the loop STOPPED — it did not keep "
          "'improving' a correct function.", flush=True)


if __name__ == "__main__":
    main()
