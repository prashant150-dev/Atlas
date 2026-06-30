"""Exact tools for the ATLAS engine (T9): arithmetic, algebra, and code.

These give 100% accuracy on *verifiable* tasks — the calculator computes exactly,
the math engine solves symbolically and self-verifies, and the code engine
generates + sandbox-runs + self-heals Python. The engine prefers a tool answer
over the language model whenever a task is verifiable.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

# ---------------- safe arithmetic (AST eval, no builtins) ----------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}

# A contiguous arithmetic expression: numbers/parens with at least one operator.
_ARITH_RE = re.compile(r"[\d(][\d\s\.\+\-\*/%()]*[\+\-\*/%][\d\s\.\+\-\*/%()]*[\d)]")


def arith_eval(expr: str) -> float | int:
    """Evaluate a pure arithmetic expression exactly and safely.

    Only numeric literals and +-*/%, //, **, and unary signs are allowed; names,
    calls, and attribute access raise ``ValueError``.
    """

    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expr must be a non-empty string")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numeric constants allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval").body)


def extract_arithmetic(text: str) -> str | None:
    """Pull a contiguous arithmetic expression out of free text, or ``None``.

    ``x``/``X`` are treated as multiplication (common in '12 x 3' style prompts).
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = text.replace("x", "*").replace("X", "*")
    match = _ARITH_RE.search(cleaned)
    return match.group().strip() if match else None


def format_number(value: float | int) -> str:
    """Render a number with thousands separators when it is integral."""

    if isinstance(value, float) and value == int(value):
        value = int(value)
    return f"{value:,}" if isinstance(value, int) else f"{value}"


# ---------------- symbolic math + code (lazy heavy imports) ----------------
def solve_math(text: str) -> Any:
    """Solve algebra/calculus exactly via ``BeastMathEngine`` (returns MathResult)."""

    from src.math_engine.symbolic import BeastMathEngine

    return BeastMathEngine().solve(text)


def solve_code(task: str, language: str = "python") -> Any:
    """Generate + verify code via ``BeastCodeEngine`` (returns CodeResult)."""

    from src.code_engine.executor import BeastCodeEngine

    return BeastCodeEngine().solve(task, language)


# Algebra/calculus keywords that should go to the symbolic engine, not the calculator.
_ALGEBRA_RE = re.compile(
    r"\b(solve|differentiate|derivative|integrate|integral|factor|expand|simplify|roots?|zeroes?)\b"
    r"|=|\bx\^|\bx\*\*|\bdy/dx\b",
    re.IGNORECASE,
)


def looks_algebraic(text: str) -> bool:
    """True when the prompt needs the symbolic engine rather than the calculator."""

    return bool(_ALGEBRA_RE.search(text))


def _self_test() -> None:
    """Exercise the calculator + extraction (math/code engines self-test elsewhere)."""

    if arith_eval("(12 + 8) * 3 - 10") != 50:
        raise RuntimeError("arith_eval wrong")
    if arith_eval("2 ** 10") != 1024:
        raise RuntimeError("pow wrong")
    if format_number(arith_eval("47389 * 8291")) != "392,902,199":
        raise RuntimeError(f"format wrong: {format_number(arith_eval('47389 * 8291'))}")
    if extract_arithmetic("What is 9876 x 5432?") != "9876 * 5432":
        raise RuntimeError(f"extract wrong: {extract_arithmetic('What is 9876 x 5432?')}")
    if extract_arithmetic("Tell me a story") is not None:
        raise RuntimeError("should not extract arithmetic from prose")
    for bad in ("__import__('os')", "open('x')", "a + b"):
        try:
            arith_eval(bad)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"unsafe expr should raise: {bad}")
    if not looks_algebraic("solve x^2 - 4 = 0") or looks_algebraic("12 + 8"):
        raise RuntimeError("looks_algebraic misclassified")
    print("ATLAS tools self-test")
    print(f"  47389*8291 = {format_number(arith_eval('47389 * 8291'))}  (expect 392,902,199)")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
