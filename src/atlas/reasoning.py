"""Reasoning tier (T6 + T7) — make a small model reason by thinking longer.

Two levers, no extra parameters and no GPU:

* **Test-time compute / self-consistency (T6):** for a checkable question, sample
  several chain-of-thought completions and majority-vote the final number. More
  inference compute buys reasoning the single greedy pass misses (measured on
  Qwen-1.5B: 0.38 direct -> 0.88 self-consistency).
* **Per-step reliability (T7):** a flat N-step chain succeeds with probability
  ``p**N`` (errors compound); verifying each step lifts that to ``~0.99**N``.
  ``chain_success_probability`` quantifies the gap that motivates step-verification.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

_NUM_RE = re.compile(r"-?\d+\.?\d*")

# cue words that mark a quantitative word-problem worth extra inference compute
_REASON_CUES = (
    "how many", "how much", "how far", "how long", "how old", "total", "remain",
    "left", "each", "per", "altogether", "in all", "combined", "average", "sum of",
    "difference", "twice", "half", "percent", "%",
)

_COT_SUFFIX = " Let's think step by step, then end with 'The answer is <number>'."


def extract_final_number(text: str) -> float | int | None:
    """Return the last number mentioned in ``text`` (CoT answers end with it)."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    nums = _NUM_RE.findall(text.replace(",", ""))
    if not nums:
        return None
    value = float(nums[-1])
    return int(value) if value == int(value) else value


def looks_like_reasoning(prompt: str) -> bool:
    """True when a prompt is a multi-step quantitative word problem."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    p = prompt.lower()
    has_numbers = len(_NUM_RE.findall(p)) >= 1
    has_cue = any(cue in p for cue in _REASON_CUES)
    return has_numbers and has_cue


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Outcome of a self-consistency reasoning run."""

    answer: str
    numeric: float | int | None
    confidence: float
    k: int
    votes: dict[str, int]
    samples: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


def self_consistency(model: Any, question: str, k: int = 3, temperature: float = 0.8,
                     max_new_tokens: int = 200) -> ReasoningResult:
    """Sample ``k`` chain-of-thought answers and majority-vote the final number.

    Args:
        model: A ``BaseModel`` (must honor ``sample=True`` to give diverse chains).
        question: The word problem.
        k: Number of sampled chains.
        temperature: Sampling temperature for diversity.
        max_new_tokens: Generation cap per chain.

    Returns:
        ``ReasoningResult`` with the modal answer and a vote-share confidence.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    prompt = question.rstrip() + _COT_SUFFIX
    samples: list[str] = []
    numbers: list[float | int] = []
    for i in range(k):
        text = model.generate(prompt, max_new_tokens=max_new_tokens, sample=True,
                              temperature=temperature, seed=i)
        samples.append(text)
        value = extract_final_number(text)
        if value is not None:
            numbers.append(value)

    if not numbers:
        return ReasoningResult("", None, 0.0, k, {}, samples)
    counts = Counter(numbers)
    modal, votes = counts.most_common(1)[0]
    confidence = votes / len(numbers)
    vote_map = {str(key): int(val) for key, val in counts.items()}
    answer = str(int(modal) if isinstance(modal, float) and modal == int(modal) else modal)
    return ReasoningResult(answer, modal, round(confidence, 3), k, vote_map, samples)


def chain_success_probability(p_step: float, n_steps: int, verified_p: float = 0.99) -> dict[str, float]:
    """Quantify why per-step verification beats one long unchecked chain (T7).

    Returns the success probability of a flat ``n_steps`` chain at ``p_step`` versus
    the same length with each step verified to ``verified_p``.
    """

    if not 0.0 <= p_step <= 1.0 or not 0.0 <= verified_p <= 1.0:
        raise ValueError("probabilities must be in [0, 1]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    flat = p_step ** n_steps
    verified = verified_p ** n_steps
    return {"flat": round(flat, 4), "verified": round(verified, 4),
            "lift": round(verified - flat, 4)}


class _FakeModel:
    """Test model returning canned CoT chains (3 right, 1 wrong) for offline voting."""

    def __init__(self) -> None:
        self._chains = [
            "3 boxes * 12 = 36 apples; 36 - 17 = 19. The answer is 19.",
            "12*3 is 36, minus 17 leaves 19. The answer is 19.",
            "I think it is around 20. The answer is 20.",          # wrong outlier
            "36 total, sell 17, so 19 remain. The answer is 19.",
        ]

    def generate(self, prompt: str, max_new_tokens: int | None = None, sample: bool = False,
                 temperature: float = 0.8, seed: int | None = None) -> str:
        """Return a canned chain selected by ``seed`` (deterministic for tests)."""

        return self._chains[(seed or 0) % len(self._chains)]


def _self_test() -> None:
    """Check number extraction, voting, and the depth-principle math (offline)."""

    if extract_final_number("blah blah The answer is 19.") != 19:
        raise RuntimeError("extract_final_number failed")
    if not looks_like_reasoning("A shop has 36 apples, sells 17. How many remain?"):
        raise RuntimeError("should detect a word problem")
    if looks_like_reasoning("Tell me a story about a dragon"):
        raise RuntimeError("should not flag a creative prompt")

    res = self_consistency(_FakeModel(), "A shop has 3 boxes of 12 apples, sells 17. How many remain?", k=4)
    if res.numeric != 19 or res.confidence < 0.5:
        raise RuntimeError(f"self-consistency vote wrong: {res.to_dict()}")

    depth = chain_success_probability(0.85, 20)
    if not (depth["flat"] < 0.1 < depth["verified"]):
        raise RuntimeError(f"depth principle wrong: {depth}")

    print("ATLAS reasoning self-test (offline)")
    print(f"  self-consistency: answer={res.answer} votes={res.votes} conf={res.confidence}")
    print(f"  depth (20 steps): flat={depth['flat']} verified={depth['verified']} (lift {depth['lift']})")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
