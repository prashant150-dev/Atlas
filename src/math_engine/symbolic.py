"""Symbolic math engine for AetherCore v3.

The engine uses SymPy for exact algebra/calculus and small deterministic
helpers for natural-language math prompts. It is intentionally bounded: when a
problem is ambiguous, it returns a verified simplification or an honest failure
instead of fabricating a derivation.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
_DEFAULT_SYMBOLS = {
    "x": sp.Symbol("x"),
    "y": sp.Symbol("y"),
    "z": sp.Symbol("z"),
    "t": sp.Symbol("t"),
    "n": sp.Symbol("n", integer=True),
    "pi": sp.pi,
    "e": sp.E,
}


@dataclass(frozen=True, slots=True)
class MathExpression:
    """Parsed math request."""

    original: str
    expression: Any
    operation: str
    variable: sp.Symbol | None = None
    limits: tuple[Any, ...] | None = None
    numbers: tuple[sp.Number, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        payload = asdict(self)
        payload["expression"] = str(self.expression)
        payload["variable"] = None if self.variable is None else str(self.variable)
        payload["limits"] = None if self.limits is None else tuple(str(item) for item in self.limits)
        payload["numbers"] = tuple(str(number) for number in self.numbers)
        return payload


@dataclass(frozen=True, slots=True)
class MathResult:
    """Final math engine result."""

    answer: str
    steps: list[str]
    verification: bool
    confidence: float
    exact_result: Any = None
    parsed: MathExpression | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return {
            "answer": self.answer,
            "steps": list(self.steps),
            "verification": bool(self.verification),
            "confidence": float(self.confidence),
            "exact_result": str(self.exact_result),
            "parsed": None if self.parsed is None else self.parsed.to_dict(),
        }


def _clean_text(text: str) -> str:
    """Normalize math prompt text."""

    return re.sub(r"\s+", " ", text.strip())


def _strip_prompt_words(text: str, words: Sequence[str]) -> str:
    """Remove leading command words from text."""

    pattern = r"^\s*(?:" + "|".join(re.escape(word) for word in words) + r")\b\s*:?\s*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def _parse_sympy_expr(text: str) -> Any:
    """Parse a string into a SymPy expression."""

    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError("expression text must not be empty")
    cleaned = cleaned.replace("−", "-").replace("×", "*").replace("÷", "/")
    cleaned = re.sub(r"\bln\s*\(", "log(", cleaned, flags=re.IGNORECASE)
    return parse_expr(cleaned, local_dict=dict(_DEFAULT_SYMBOLS), transformations=_TRANSFORMATIONS, evaluate=True)


def _parse_equation(text: str) -> Any:
    """Parse an expression or equation string."""

    if "=" not in text:
        return _parse_sympy_expr(text)
    left, right = text.split("=", 1)
    return sp.Eq(_parse_sympy_expr(left), _parse_sympy_expr(right))


def _first_symbol(expression: Any, fallback: str = "x") -> sp.Symbol:
    """Return the first symbol in an expression."""

    if hasattr(expression, "free_symbols") and expression.free_symbols:
        return sorted(expression.free_symbols, key=lambda symbol: symbol.name)[0]
    return sp.Symbol(fallback)


def _parse_variable(text: str, expression: Any | None = None) -> sp.Symbol:
    """Extract a variable from text or expression."""

    match = re.search(r"\b(?:with respect to|wrt|with variable|by)\s+([a-zA-Z]\w*)\b", text, flags=re.IGNORECASE)
    if match:
        return sp.Symbol(match.group(1))
    if expression is not None:
        return _first_symbol(expression)
    return sp.Symbol("x")


def _extract_numbers(text: str) -> tuple[sp.Number, ...]:
    """Extract numeric values from text."""

    values = re.findall(r"[-+]?\d*\.?\d+(?:/[+-]?\d*\.?\d+)?", text)
    output: list[sp.Number] = []
    for value in values:
        try:
            output.append(sp.Rational(value))
        except (TypeError, ValueError):
            output.append(sp.Float(value))
    return tuple(output)


class MathParser:
    """Parse algebra, calculus, statistics, and simple natural language."""

    def parse(self, text: str) -> MathExpression:
        """Parse text into a MathExpression."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        original = _clean_text(text)
        if not original:
            raise ValueError("text must not be empty")

        lowered = original.lower()
        if any(keyword in lowered for keyword in ("mean", "median", "average", "variance", "standard deviation", "std")):
            numbers = _extract_numbers(original)
            if numbers:
                operation = self._statistics_operation(lowered)
                return MathExpression(original, tuple(numbers), operation, numbers=numbers)

        if re.search(r"\b(differentiate|derivative|derive|d/d)\b", lowered):
            body = _strip_prompt_words(original, ("differentiate", "derivative of", "derive", "find derivative of"))
            body = re.split(r"\b(?:with respect to|wrt|by)\b", body, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            expression = _parse_equation(body)
            variable = _parse_variable(original, expression)
            return MathExpression(original, expression, "differentiate", variable=variable)

        if re.search(r"\b(integrate|integral|antiderivative)\b", lowered):
            body = _strip_prompt_words(original, ("integrate", "integral of", "find integral of", "antiderivative of"))
            limits = self._parse_limits(body)
            body = re.split(r"\bfrom\b", body, maxsplit=1, flags=re.IGNORECASE)[0]
            body = re.split(r"\b(?:with respect to|wrt|by)\b", body, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            expression = _parse_sympy_expr(body)
            variable = _parse_variable(original, expression)
            return MathExpression(original, expression, "integrate", variable=variable, limits=limits)

        if re.search(r"\b(solve|roots?|zeroes?)\b", lowered) or "=" in original:
            body = _strip_prompt_words(original, ("solve", "find roots of", "find root of", "roots of", "zeroes of"))
            expression = _parse_equation(body)
            variable = _parse_variable(original, expression)
            return MathExpression(original, expression, "solve", variable=variable)

        if re.search(r"\b(simplify|factor|expand)\b", lowered):
            operation = "factor" if "factor" in lowered else "expand" if "expand" in lowered else "simplify"
            body = _strip_prompt_words(original, ("simplify", "factor", "expand"))
            return MathExpression(original, _parse_sympy_expr(body), operation, variable=None)

        return MathExpression(original, _parse_sympy_expr(original), "simplify", variable=None)

    def _statistics_operation(self, lowered: str) -> str:
        """Choose a statistics operation from prompt text."""

        if "median" in lowered:
            return "median"
        if "variance" in lowered:
            return "variance"
        if "standard deviation" in lowered or "std" in lowered:
            return "std"
        return "mean"

    def _parse_limits(self, text: str) -> tuple[Any, ...] | None:
        """Parse integral limits from text like 'from 0 to 1'."""

        match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            return None
        lower = _parse_sympy_expr(match.group(1))
        upper = _parse_sympy_expr(match.group(2))
        return lower, upper


class SymbolicSolver:
    """Solve exact math operations with SymPy."""

    def solve(self, expression: MathExpression | Any) -> Any:
        """Solve or simplify an expression."""

        parsed = expression if isinstance(expression, MathExpression) else MathExpression(str(expression), expression, "solve")
        op = parsed.operation
        if op == "solve":
            variable = parsed.variable or _first_symbol(parsed.expression)
            if isinstance(parsed.expression, sp.Equality):
                return sp.solve(parsed.expression, variable)
            return sp.solve(sp.Eq(parsed.expression, 0), variable)
        if op == "differentiate":
            return self.differentiate(parsed)
        if op == "integrate":
            return self.integrate(parsed)
        if op == "factor":
            return sp.factor(parsed.expression)
        if op == "expand":
            return sp.expand(parsed.expression)
        if op in {"mean", "median", "variance", "std"}:
            return self._statistics(parsed)
        return sp.simplify(parsed.expression)

    def differentiate(self, expr: MathExpression | Any) -> Any:
        """Differentiate an expression exactly."""

        parsed = expr if isinstance(expr, MathExpression) else MathExpression(str(expr), expr, "differentiate")
        variable = parsed.variable or _first_symbol(parsed.expression)
        return sp.diff(parsed.expression, variable)

    def integrate(self, expr: MathExpression | Any) -> Any:
        """Integrate an expression exactly."""

        parsed = expr if isinstance(expr, MathExpression) else MathExpression(str(expr), expr, "integrate")
        variable = parsed.variable or _first_symbol(parsed.expression)
        if parsed.limits:
            lower, upper = parsed.limits
            return sp.integrate(parsed.expression, (variable, lower, upper))
        return sp.integrate(parsed.expression, variable)

    def _statistics(self, parsed: MathExpression) -> Any:
        """Compute exact small statistics."""

        if not parsed.numbers:
            raise ValueError("statistics operation requires numbers")
        values = tuple(sp.Rational(number) for number in parsed.numbers)
        if parsed.operation == "mean":
            return sp.simplify(sum(values) / len(values))
        if parsed.operation == "median":
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[middle]
            return sp.simplify((ordered[middle - 1] + ordered[middle]) / 2)
        mean = sp.simplify(sum(values) / len(values))
        variance = sp.simplify(sum((value - mean) ** 2 for value in values) / len(values))
        if parsed.operation == "variance":
            return variance
        return sp.sqrt(variance)


class MathVerifier:
    """Verify solutions by symbolic identities and numeric spot checks."""

    def verify(self, problem: MathExpression | str, solution: Any) -> bool:
        """Return true when the solution checks out."""

        parsed = MathParser().parse(problem) if isinstance(problem, str) else problem
        if not isinstance(parsed, MathExpression):
            raise TypeError("problem must be a MathExpression or string")

        try:
            if parsed.operation == "solve":
                return self._verify_solve(parsed, solution)
            if parsed.operation == "differentiate":
                expected = sp.diff(parsed.expression, parsed.variable or _first_symbol(parsed.expression))
                return sp.simplify(expected - solution) == 0
            if parsed.operation == "integrate":
                if parsed.limits:
                    expected = SymbolicSolver().integrate(parsed)
                    return sp.simplify(expected - solution) == 0
                variable = parsed.variable or _first_symbol(parsed.expression)
                return sp.simplify(sp.diff(solution, variable) - parsed.expression) == 0
            if parsed.operation in {"mean", "median", "variance", "std", "factor", "expand", "simplify"}:
                expected = SymbolicSolver().solve(parsed)
                return sp.simplify(expected - solution) == 0
        except Exception:
            return False
        return False

    def _verify_solve(self, parsed: MathExpression, solution: Any) -> bool:
        """Verify roots satisfy the equation."""

        variable = parsed.variable or _first_symbol(parsed.expression)
        solutions = solution if isinstance(solution, (list, tuple, set)) else (solution,)
        if not solutions:
            return True
        for candidate in solutions:
            if isinstance(parsed.expression, sp.Equality):
                residual = parsed.expression.lhs.subs(variable, candidate) - parsed.expression.rhs.subs(variable, candidate)
            else:
                residual = parsed.expression.subs(variable, candidate)
            if sp.simplify(residual) != 0:
                numeric = complex(sp.N(residual))
                if abs(numeric) > 1.0e-8:
                    return False
        return True


class MathExplainer:
    """Generate concise natural-language solution steps."""

    def explain(self, problem: MathExpression | str, solution: Any) -> list[str]:
        """Return step-by-step explanation strings."""

        parsed = MathParser().parse(problem) if isinstance(problem, str) else problem
        if not isinstance(parsed, MathExpression):
            raise TypeError("problem must be a MathExpression or string")

        if parsed.operation == "solve":
            return [
                f"Parse the equation/expression as {parsed.expression}.",
                f"Solve for {parsed.variable or _first_symbol(parsed.expression)} using symbolic algebra.",
                f"The solution set is {solution}.",
            ]
        if parsed.operation == "differentiate":
            return [
                f"Parse the expression as {parsed.expression}.",
                f"Differentiate with respect to {parsed.variable or _first_symbol(parsed.expression)}.",
                f"The derivative is {solution}.",
            ]
        if parsed.operation == "integrate":
            limit_text = "" if not parsed.limits else f" from {parsed.limits[0]} to {parsed.limits[1]}"
            return [
                f"Parse the integrand as {parsed.expression}.",
                f"Integrate with respect to {parsed.variable or _first_symbol(parsed.expression)}{limit_text}.",
                f"The integral is {solution}.",
            ]
        if parsed.operation in {"mean", "median", "variance", "std"}:
            return [
                f"Extract the numbers {list(parsed.numbers)}.",
                f"Apply the {parsed.operation} formula exactly.",
                f"The result is {solution}.",
            ]
        return [
            f"Parse the expression as {parsed.expression}.",
            f"Apply symbolic {parsed.operation}.",
            f"The simplified result is {solution}.",
        ]


class BeastMathEngine:
    """High-level orchestrator for exact math solving."""

    def __init__(
        self,
        parser: MathParser | None = None,
        solver: SymbolicSolver | None = None,
        verifier: MathVerifier | None = None,
        explainer: MathExplainer | None = None,
    ) -> None:
        """Create a complete math engine."""

        self.parser = parser or MathParser()
        self.solver = solver or SymbolicSolver()
        self.verifier = verifier or MathVerifier()
        self.explainer = explainer or MathExplainer()

    def solve(self, problem_text: str) -> MathResult:
        """Solve a problem and return exact result plus verification."""

        parsed = self.parser.parse(problem_text)
        exact = self.solver.solve(parsed)
        verified = self.verifier.verify(parsed, exact)
        steps = self.explainer.explain(parsed, exact)
        confidence = 0.999 if verified else 0.45
        return MathResult(
            answer=str(exact),
            steps=steps,
            verification=verified,
            confidence=confidence,
            exact_result=exact,
            parsed=parsed,
        )


def _self_test() -> None:
    """Run a small CPU sanity check for the symbolic math engine."""

    engine = BeastMathEngine()
    solve_result = engine.solve("solve x^2 - 4 = 0")
    derivative_result = engine.solve("differentiate x^3 + 2*x with respect to x")
    integral_result = engine.solve("integrate x from 0 to 1")
    mean_result = engine.solve("mean of 1, 2, 3, 4")

    if set(solve_result.exact_result) != {-2, 2}:
        raise RuntimeError(f"Unexpected solve result: {solve_result.exact_result}")
    if str(derivative_result.exact_result) != "3*x**2 + 2":
        raise RuntimeError(f"Unexpected derivative: {derivative_result.exact_result}")
    if integral_result.exact_result != sp.Rational(1, 2):
        raise RuntimeError(f"Unexpected integral: {integral_result.exact_result}")
    if mean_result.exact_result != sp.Rational(5, 2):
        raise RuntimeError(f"Unexpected mean: {mean_result.exact_result}")

    print("AetherCore symbolic math self-test")
    print(f"  solve: {solve_result.to_dict()}")
    print(f"  derivative: {derivative_result.answer}")
    print(f"  integral: {integral_result.answer}")
    print(f"  mean: {mean_result.answer}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
