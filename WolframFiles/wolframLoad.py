import re
from collections import Counter

function_map = {
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Exp": "exp",
    "Log": "log",
    "Sqrt": "sqrt",
    "Abs": "abs",
}

# Convert power expressions into named variables like x2 for pow(x, 2)
def simplify_powers(expr):
    pow_pattern = re.compile(r'pow\((\w+),\s*([0-9]+)\)')
    powers = {}
    def repl(match):
        base, exp = match.group(1), match.group(2)
        var = f"{base}{exp}"
        powers[var] = f"pow({base}, {exp})"
        return var
    expr = pow_pattern.sub(repl, expr)
    return expr, powers

# Recursively convert functions from Wolfram to C++
def convert_functions(expr):
    func_pattern = re.compile(r'([A-Za-z]+)\[([^\[\]]+?)\]')
    while True:
        match = func_pattern.search(expr)
        if not match:
            break
        full, func, arg = match.group(0), match.group(1), match.group(2)
        cpp_func = function_map.get(func, func.lower())
        replaced = f"{cpp_func}({convert_functions(arg)})"
        expr = expr.replace(full, replaced)
    return expr

# Replace ^ with pow()
def replace_pow(expr):
    return re.sub(r'(\w+|\))\^(\w+|\()', r'pow(\1, \2)', expr)

# Extract all compound subexpressions (excluding single functions and powers)
def extract_large_subexpressions(expr):
    subexprs = []

    def helper(e):
        stack = []
        i = 0
        while i < len(e):
            if e[i] == '(':
                start = i
                depth = 1
                i += 1
                while i < len(e) and depth > 0:
                    if e[i] == '(':
                        depth += 1
                    elif e[i] == ')':
                        depth -= 1
                    i += 1
                end = i
                sub = e[start:end]
                if not re.match(r'\(\w+\)', sub) and len(sub) > 5:
                    subexprs.append(sub)
                    helper(sub[1:-1])
            else:
                i += 1

    helper(expr)
    return subexprs

# Final C++ code generator
def generate_cpp_code(wolfram_expr, result_var="result"):
    expr = replace_pow(wolfram_expr)
    expr = convert_functions(expr)
    expr, pow_vars = simplify_powers(expr)

    # Find reusable large subexpressions (not single functions or x2)
    subexprs = extract_large_subexpressions(expr)
    freq = Counter(subexprs)
    reused = {k: v for k, v in freq.items() if v > 1}
    sorted_subs = sorted(reused.items(), key=lambda x: (-len(x[0]), -x[1]))

    temp_vars = {}
    counter = 1
    for sub, _ in sorted_subs:
        var = f"temp{counter}"
        if sub not in temp_vars:
            expr = expr.replace(sub, var)
            temp_vars[sub] = var
            counter += 1

    # Output code
    lines = ["#include <cmath>", "", "double compute(double x) {"]
    
    for var, code in pow_vars.items():
        lines.append(f"    double {var} = {code};")
    for sub, var in temp_vars.items():
        lines.append(f"    double {var} = {sub};")
    lines.append(f"    double {result_var} = {expr};")
    lines.append(f"    return {result_var};")
    lines.append("}")
    return "\n".join(lines)

# Example usage
if __name__ == "__main__":
    wolfram_equation = "{{(-210*x^3*y*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (90*x*y*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*x*y*Cos[w])/(x^2 + y^2 + z^2)^(5/2) + (105*x^4*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*x^2*y^2*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (75*x^2*Sin[w])/(x^2 + y^2 + z^2)^(7/2) + (15*y^2*Sin[w])/(x^2 + y^2 + z^2)^(7/2) + (6*Sin[w])/(x^2 + y^2 + z^2)^(5/2) + (w^4*Sin[w])/Sqrt[x^2 + y^2 + z^2] - w^2*((3*z^2*Sin[w])/(x^2 + y^2 + z^2)^(5/2) - Sin[w]/(x^2 + y^2 + z^2)^(3/2)) - w^2*((-6*x^2*Sin[w])/(x^2 + y^2 + z^2)^(5/2) + (2*Sin[w])/(x^2 + y^2 + z^2)^(3/2)), (-210*x^2*y^2*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (30*x^2*Cos[w])/(x^2 + y^2 + z^2)^(7/2) + (30*y^2*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - (w^4*Cos[w])/Sqrt[x^2 + y^2 + z^2] + w^2*((3*z^2*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - Cos[w]/(x^2 + y^2 + z^2)^(3/2)) + (105*x^3*y*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*x*y^3*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - w^2*((3*x^2*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - Cos[w]/(x^2 + y^2 + z^2)^(3/2) - (3*x*y*Sin[w])/(x^2 + y^2 + z^2)^(5/2)) - w^2*((3*y^2*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - Cos[w]/(x^2 + y^2 + z^2)^(3/2) + (3*x*y*Sin[w])/(x^2 + y^2 + z^2)^(5/2)), 0. - (210*x^2*y*z*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (30*y*z*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*y*z*Cos[w])/(x^2 + y^2 + z^2)^(5/2) + (105*x^3*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*x*y^2*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (30*x*z*Sin[w])/(x^2 + y^2 + z^2)^(7/2) + (6*w^2*x*z*Sin[w])/(x^2 + y^2 + z^2)^(5/2)}, {(-210*x^2*y^2*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (30*x^2*Cos[w])/(x^2 + y^2 + z^2)^(7/2) + (30*y^2*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - (w^4*Cos[w])/Sqrt[x^2 + y^2 + z^2] + w^2*((3*z^2*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - Cos[w]/(x^2 + y^2 + z^2)^(3/2)) + (105*x^3*y*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*x*y^3*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - w^2*((3*x^2*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - Cos[w]/(x^2 + y^2 + z^2)^(3/2) - (3*x*y*Sin[w])/(x^2 + y^2 + z^2)^(5/2)) - w^2*((3*y^2*Cos[w])/(x^2 + y^2 + z^2)^(5/2) - Cos[w]/(x^2 + y^2 + z^2)^(3/2) + (3*x*y*Sin[w])/(x^2 + y^2 + z^2)^(5/2)), (-210*x*y^3*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (90*x*y*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*x*y*Cos[w])/(x^2 + y^2 + z^2)^(5/2) + (105*x^2*y^2*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*y^4*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (15*x^2*Sin[w])/(x^2 + y^2 + z^2)^(7/2) + (75*y^2*Sin[w])/(x^2 + y^2 + z^2)^(7/2) - (6*Sin[w])/(x^2 + y^2 + z^2)^(5/2) - (w^4*Sin[w])/Sqrt[x^2 + y^2 + z^2] - w^2*((6*y^2*Sin[w])/(x^2 + y^2 + z^2)^(5/2) - (2*Sin[w])/(x^2 + y^2 + z^2)^(3/2)) - w^2*((-3*z^2*Sin[w])/(x^2 + y^2 + z^2)^(5/2) + Sin[w]/(x^2 + y^2 + z^2)^(3/2)), 0. - (210*x*y^2*z*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (30*x*z*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*x*z*Cos[w])/(x^2 + y^2 + z^2)^(5/2) + (105*x^2*y*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*y^3*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) + (30*y*z*Sin[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*y*z*Sin[w])/(x^2 + y^2 + z^2)^(5/2)}, {0. - (210*x^2*y*z*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (30*y*z*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*y*z*Cos[w])/(x^2 + y^2 + z^2)^(5/2) + (105*x^3*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*x*y^2*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (30*x*z*Sin[w])/(x^2 + y^2 + z^2)^(7/2) + (6*w^2*x*z*Sin[w])/(x^2 + y^2 + z^2)^(5/2), 0. - (210*x*y^2*z*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (30*x*z*Cos[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*x*z*Cos[w])/(x^2 + y^2 + z^2)^(5/2) + (105*x^2*y*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*y^3*z*Sin[w])/(x^2 + y^2 + z^2)^(9/2) + (30*y*z*Sin[w])/(x^2 + y^2 + z^2)^(7/2) - (6*w^2*y*z*Sin[w])/(x^2 + y^2 + z^2)^(5/2), 0. - (210*x*y*z^2*Cos[w])/(x^2 + y^2 + z^2)^(9/2) + (30*x*y*Cos[w])/(x^2 + y^2 + z^2)^(7/2) + (6*w^2*x*y*Cos[w])/(x^2 + y^2 + z^2)^(5/2) + (105*x^2*z^2*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (105*y^2*z^2*Sin[w])/(x^2 + y^2 + z^2)^(9/2) - (15*x^2*Sin[w])/(x^2 + y^2 + z^2)^(7/2) + (15*y^2*Sin[w])/(x^2 + y^2 + z^2)^(7/2) - w^2*((3*x^2*Sin[w])/(x^2 + y^2 + z^2)^(5/2) - Sin[w]/(x^2 + y^2 + z^2)^(3/2)) - w^2*((-3*y^2*Sin[w])/(x^2 + y^2 + z^2)^(5/2) + Sin[w]/(x^2 + y^2 + z^2)^(3/2))}}"
    cpp_code = generate_cpp_code(wolfram_equation)
    print(cpp_code)

