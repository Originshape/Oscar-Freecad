"""Derived-parameter formulas of the FeaturePlan schema (private to oscar.plan).

``{"expr": "..."}`` parameters: a tiny arithmetic grammar (numbers, parameter
names, ``+ - * / ^``, parentheses, ``sqrt``, ``abs``, ``pi``), parsed once,
dependency-ordered, cycle-checked and evaluated without ``eval``. ``schema``
re-exports every public name (``schema.parse_expr``, ``schema.ExprError``,
``schema.parameter_values`` ...); import them from there.
"""

import functools
import math
import re
from typing import Any


def _num(value: Any) -> bool:
    """A finite real number (see ``schema._num``)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


# ---------------------------------------------------------------------------
# derived parameters: {"expr": "..."} formulas over other parameters


class ExprError(ValueError):
    """A derived-parameter formula that cannot be parsed or evaluated."""


_TOKEN = re.compile(
    r"\s*(?:(?P<num>[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?|\.[0-9]+(?:[eE][+-]?[0-9]+)?)"   # ASCII digits only: \d accepts Arabic-Indic ones FreeCAD cannot parse
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/^()]))")

_FUNCTIONS = {"sqrt": math.sqrt, "abs": abs}
_CONSTANTS = {"pi": math.pi}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise ExprError(f"unexpected character {text[pos:].lstrip()[:1]!r}"
                            f" (allowed: numbers, parameter names, + - * / ^ "
                            f"( ), sqrt(), abs(), pi)")
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


@functools.lru_cache(maxsize=512)
def parse_expr(text: str) -> tuple[Any, frozenset]:
    """Parse a derived-parameter formula → ``(tree, referenced_names)``.

    Grammar: numbers, parameter names, ``+ - * / ^``, parentheses,
    ``sqrt(x)``, ``abs(x)`` and ``pi``. A unary minus directly in front of
    a power (``-a^2``) is rejected as ambiguous (FreeCAD and Python bind it
    differently); write ``-(a^2)`` or ``(-a)^2``. Raises ExprError."""
    if not isinstance(text, str) or not text.strip():
        raise ExprError("must be a non-empty formula string")
    tokens = _tokenize(text)
    pos = 0
    names: set[str] = set()

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def take():
        nonlocal pos
        tok = peek()
        pos += 1
        return tok

    def expect(op):
        kind, val = take()
        if not (kind == "op" and val == op):
            raise ExprError(f"expected {op!r}")

    def p_expr():
        node = p_term()
        while peek() in (("op", "+"), ("op", "-")):
            _k, op = take()
            node = (op, node, p_term())
        return node

    def p_term():
        node = p_unary()
        while peek() in (("op", "*"), ("op", "/")):
            _k, op = take()
            node = (op, node, p_unary())
        return node

    def p_unary():
        if peek() in (("op", "-"), ("op", "+")):
            _k, op = take()
            operand = p_unary()
            if isinstance(operand, tuple) and operand[0] == "^":
                raise ExprError("a sign in front of '^' is ambiguous; write "
                                "-(a^2) or (-a)^2")
            return ("neg", operand) if op == "-" else operand
        return p_power()

    def p_power():
        base = p_atom()
        if peek() == ("op", "^"):
            take()
            exponent = p_unary()
            return ("^", base, exponent)
        return base

    def p_atom():
        kind, val = take()
        if kind == "num":
            return ("num", float(val))
        if kind == "name":
            if peek() == ("op", "("):
                if val not in _FUNCTIONS:
                    raise ExprError(f"unknown function {val!r} (known: "
                                    f"{', '.join(sorted(_FUNCTIONS))})")
                take()
                arg = p_expr()
                expect(")")
                return ("fn", val, arg)
            if val in _CONSTANTS:
                return ("num", _CONSTANTS[val])
            names.add(val)
            return ("name", val)
        if kind == "op" and val == "(":
            node = p_expr()
            expect(")")
            return node
        raise ExprError("incomplete formula" if kind is None
                        else f"unexpected {val!r}")

    tree = p_expr()
    if pos != len(tokens):
        raise ExprError(f"unexpected {tokens[pos][1]!r} (two values need an "
                        f"operator between them)")
    return tree, frozenset(names)


def _eval_tree(node, values: dict[str, float]) -> float:
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "name":
        return values[node[1]]
    if kind == "neg":
        return -_eval_tree(node[1], values)
    if kind == "fn":
        arg = _eval_tree(node[2], values)
        if node[1] == "sqrt" and arg < 0:
            raise ExprError("sqrt of a negative number")
        return _FUNCTIONS[node[1]](arg)
    a = _eval_tree(node[1], values)
    b = _eval_tree(node[2], values)
    if kind == "+":
        return a + b
    if kind == "-":
        return a - b
    if kind == "*":
        return a * b
    if kind == "/":
        if b == 0:
            raise ExprError("divides by zero")
        return a / b
    try:
        out = math.pow(a, b)
    except (ValueError, OverflowError) as e:
        raise ExprError(f"power {a:g}^{b:g} is undefined ({e})") from None
    return out


def _eval_params(params: dict) -> tuple[dict[str, float], list[str], list[str]]:
    """Numeric value of every parameter → ``(values, problems, order)``.

    ``values`` holds plain numbers and every derived parameter that
    evaluates; ``problems`` lists unparsable formulas, unknown references,
    cycles and arithmetic errors, each naming the parameter; ``order`` is
    the dependency order of the derived parameters (sources first)."""
    values: dict[str, float] = {}
    problems: list[str] = []
    order: list[str] = []
    derived: dict[str, tuple[Any, frozenset]] = {}
    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("expr") is not None:
            try:
                derived[name] = parse_expr(spec["expr"])
            except ExprError as e:
                problems.append(f"parameter {name!r}: expr {e}")
        elif _num(spec.get("value")):
            values[name] = float(spec["value"])
    state: dict[str, int] = {}   # 1 = being visited, 2 = done (value or failed)

    def visit(name: str, stack: list[str]) -> bool:
        if name in values:
            return True
        if name not in derived:
            return False
        if state.get(name) == 2:
            return name in values
        if state.get(name) == 1:
            cycle = stack[stack.index(name):] + [name]
            problems.append("derived parameters form a cycle: "
                            + " -> ".join(cycle) + "; break it by making one "
                            "of them a plain value")
            state[name] = 2
            return False
        state[name] = 1
        tree, deps = derived[name]
        ok = True
        for dep in sorted(deps):
            if dep not in params:
                problems.append(f"parameter {name!r}: expr references "
                                f"unknown parameter {dep!r}")
                ok = False
            elif not visit(dep, stack + [name]):
                ok = False
        state[name] = 2
        if not ok:
            return False
        try:
            value = _eval_tree(tree, values)
        except ExprError as e:
            problems.append(f"parameter {name!r}: expr {e}")
            return False
        if not math.isfinite(value):
            problems.append(f"parameter {name!r}: expr is not a finite number")
            return False
        values[name] = value
        order.append(name)
        return True

    for name in derived:
        visit(name, [])
    return values, problems, order


def parameter_values(plan: dict) -> dict[str, float]:
    """Numeric value of every parameter of ``plan`` (derived parameters
    evaluated in dependency order). Parameters without a usable number are
    absent; ``validate`` reports why."""
    params = plan.get("parameters") if isinstance(plan, dict) else None
    if not isinstance(params, dict):
        return {}
    return _eval_params(params)[0]


def derived_order(plan: dict) -> list[str]:
    """Names of the derived (``expr``) parameters, dependencies first. A
    cycle or unknown reference leaves the offending names out."""
    params = plan.get("parameters") if isinstance(plan, dict) else None
    if not isinstance(params, dict):
        return []
    return _eval_params(params)[2]


def is_derived(spec: Any) -> bool:
    """True for a parameter spec that carries an ``expr`` formula."""
    return isinstance(spec, dict) and spec.get("expr") is not None
