# src/math_tools.py
from sympy import sympify, Eq
from sympy.parsing.sympy_parser import parse_expr
from typing import Tuple

def simplify_expression(expr_str: str) -> str:
    try:
        expr = parse_expr(expr_str)
        return str(expr.simplify())
    except Exception:
        return expr_str

def verify_numeric_answer(expr_str: str, expected_value: float, tol: float = 1e-6) -> bool:
    try:
        expr = parse_expr(expr_str)
        val = float(expr.evalf())
        return abs(val - expected_value) <= tol
    except Exception:
        return False

def extract_numeric_from_text(text: str) -> Tuple[bool, float]:
    # naive: find the last number in text
    import re
    nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', text)
    if not nums:
        return False, 0.0
    return True, float(nums[-1])
