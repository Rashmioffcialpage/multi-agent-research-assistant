"""A safe arithmetic calculator tool.

Deliberately not `eval()` — that would let a malicious or hallucinated
tool call execute arbitrary Python. This parses the expression into an AST
and only evaluates a fixed whitelist of numeric operators, so it can
never do anything beyond arithmetic.
"""

from __future__ import annotations

import ast
import operator

_OPERATORS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Evaluates a plain arithmetic expression, e.g. "2 * (3 + 4) / 7"."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"invalid expression: {expression!r}") from e
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as e:
        raise CalculatorError("division by zero") from e
