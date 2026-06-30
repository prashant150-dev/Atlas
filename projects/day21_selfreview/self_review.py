"""Part-4 add-on: SELF-REVIEW with a bounded loop + stop-gate (anti-infinite-loop).

The idea (user's): after the model answers, let it RE-READ its own output, ask
"did I write anything wrong / could this be better?", and fix it — BUT without falling
into an endless "improve it again... again..." loop.

Three guardrails make the loop safe and convergent:
  1. BOUNDED      : at most MAX_ROUNDS attempts, full stop.
  2. STOP-GATE    : if the reviewer says "good, no real issues" -> stop immediately
                    (do NOT keep polishing something already correct).
  3. NO-REGRESS   : each candidate gets a quality SCORE; we only accept a revision that
                    BEATS the current best. If a "fix" doesn't improve the score (or the
                    score stops moving), we keep the previous best and stop. This is what
                    prevents "trying to make it better" from making it worse or looping.

This module is generic: you plug in a `score_fn(answer) -> (score, issues)` that judges an
answer and lists concrete problems, and a `revise_fn(answer, issues) -> answer` that
produces a fix. For code, score_fn = run-the-tests; for prose, score_fn = a checklist.
The loop logic is identical and is what we test here.

Run:  python projects/day21_selfreview/self_review.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

OUT = Path(__file__).resolve().parent / "self_review_results.json"

MAX_ROUNDS = 3
MIN_GAIN = 1e-9          # a revision must improve the score by at least this to be kept


@dataclass(frozen=True, slots=True)
class Round:
    """one review/revise step, recorded for transparency."""
    n: int
    score: float
    issues: list[str]
    action: str          # "accepted", "rejected-no-gain", "stop-gate-clean", "stop-bounded"

    def to_dict(self) -> dict:
        return {"round": self.n, "score": round(self.score, 4),
                "issues": list(self.issues), "action": self.action}


@dataclass(slots=True)
class SelfReviewResult:
    final_answer: object
    best_score: float
    rounds: list[Round] = field(default_factory=list)
    stopped_because: str = ""

    def to_dict(self) -> dict:
        return {"best_score": round(self.best_score, 4),
                "stopped_because": self.stopped_because,
                "n_rounds": len(self.rounds),
                "rounds": [r.to_dict() for r in self.rounds]}


def self_review(
    answer: object,
    score_fn: Callable[[object], "tuple[float, list[str]]"],
    revise_fn: Callable[[object, list[str]], object],
    max_rounds: int = MAX_ROUNDS,
    clean_threshold: float = 1.0,
) -> SelfReviewResult:
    """Re-read, judge, fix — bounded, gated, and non-regressing.

    score_fn(answer)  -> (score in [0,1] higher=better, list of concrete issue strings)
    revise_fn(ans, issues) -> a new answer attempting to fix those issues.
    """
    score, issues = score_fn(answer)
    best_answer, best_score = answer, score
    res = SelfReviewResult(final_answer=best_answer, best_score=best_score)

    for n in range(1, max_rounds + 1):
        # STOP-GATE: already clean -> don't polish further.
        if best_score >= clean_threshold or not issues:
            res.rounds.append(Round(n, best_score, issues, "stop-gate-clean"))
            res.stopped_because = "clean (no real issues left)"
            break

        candidate = revise_fn(best_answer, issues)
        cand_score, cand_issues = score_fn(candidate)

        # NO-REGRESS: only accept a strict improvement; else keep best and stop.
        if cand_score > best_score + MIN_GAIN:
            best_answer, best_score, issues = candidate, cand_score, cand_issues
            res.rounds.append(Round(n, cand_score, cand_issues, "accepted"))
            # post-accept STOP-GATE: if the fix made it clean, stop now (don't spend
            # another round polishing a correct answer).
            if best_score >= clean_threshold or not issues:
                res.stopped_because = "clean (no real issues left)"
                break
        else:
            res.rounds.append(Round(n, cand_score, cand_issues, "rejected-no-gain"))
            res.stopped_because = "no further improvement (kept previous best)"
            break
    else:
        res.stopped_because = "hit round limit (bounded)"

    res.final_answer = best_answer
    res.best_score = best_score
    return res


# ----------------------------------------------------------------------------
# Self-test: three scenarios that exercise each guardrail.
def _self_test() -> dict:
    log = {}

    # Scenario A: buggy code that the reviewer fixes in 2 rounds, then is clean.
    # answer = a number of remaining bugs; score = 1 - bugs/3; revise removes one bug.
    def score_bugs(ans):
        bugs = ans
        issues = [f"bug #{i+1}" for i in range(bugs)]
        return 1.0 - bugs / 3.0, issues

    def fix_one(ans, issues):
        return max(0, ans - 1)

    a = self_review(2, score_bugs, fix_one, max_rounds=4)   # start with 2 bugs, cap 4
    assert a.best_score == 1.0, a.to_dict()
    assert a.stopped_because.startswith("clean"), a.to_dict()
    assert len(a.rounds) == 2, a.to_dict()     # 2 fixes -> clean, stops BEFORE the cap
    log["A_fixes_then_stops_clean"] = a.to_dict()

    # Scenario B: a "fix" that does NOT help -> must reject and stop (anti-loop).
    def score_flat(ans):
        return 0.5, ["style nitpick"]          # score never improves
    def revise_noop(ans, issues):
        return ans
    b = self_review("draft", score_flat, revise_noop)
    assert b.stopped_because.startswith("no further"), b.to_dict()
    assert len(b.rounds) == 1, b.to_dict()     # tried once, saw no gain, stopped
    log["B_rejects_useless_change"] = b.to_dict()

    # Scenario C: endless tiny "improvements" -> bounded cap stops it (anti-loop).
    state = {"v": 0.0}
    def score_creep(ans):
        return state["v"], ["could be slightly better"]
    def revise_creep(ans, issues):
        state["v"] += 0.001                     # always "a little better", never done
        return ans
    c = self_review("x", score_creep, revise_creep, max_rounds=3, clean_threshold=1.0)
    assert len(c.rounds) == 3, c.to_dict()      # exactly the bound, no more
    assert c.stopped_because == "hit round limit (bounded)", c.to_dict()
    log["C_bounded_stops_endless_polish"] = c.to_dict()

    return log


def main():
    log = _self_test()
    print("Self-review loop — three guardrails verified:\n", flush=True)
    print("A) buggy answer: fixes each round, STOPS when clean (stop-gate)", flush=True)
    print(f"   -> {log['A_fixes_then_stops_clean']['stopped_because']}, "
          f"{log['A_fixes_then_stops_clean']['n_rounds']} rounds, "
          f"score {log['A_fixes_then_stops_clean']['best_score']}", flush=True)
    print("B) useless 'fix': rejects it, keeps best, STOPS (no-regress)", flush=True)
    print(f"   -> {log['B_rejects_useless_change']['stopped_because']}, "
          f"{log['B_rejects_useless_change']['n_rounds']} round", flush=True)
    print("C) endless tiny polish: BOUNDED cap stops the loop", flush=True)
    print(f"   -> {log['C_bounded_stops_endless_polish']['stopped_because']}, "
          f"{log['C_bounded_stops_endless_polish']['n_rounds']} rounds (= the cap)", flush=True)
    print("\nNo scenario loops forever; correct answers are not over-polished.", flush=True)
    OUT.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
