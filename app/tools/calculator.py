from langchain_core.tools import tool
from sympy import sympify
import re

@tool
def calculator(expression: str) -> dict:
    """Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), powers (^), 
    and common math functions (sqrt, abs, sin, cos, tan, log, ln).
    
    Args:
        expression: The mathematical expression to evaluate, e.g. "2 + 3 * 4" or "sqrt(144)"
    """
    try:
        # Basic sanitization: only allow alphanumeric, operators, and basic parens
        cleaned = re.sub(r'[^a-zA-Z0-9\+\-\*\/\^\(\)\.\,]', '', expression)
        # Parse and evaluate using sympy
        result = sympify(cleaned).evalf()
        return {
            "expression": expression,
            "result": float(result)
        }
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}
