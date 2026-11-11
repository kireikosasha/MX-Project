#!/usr/bin/env python3
"""
The exact part for arithmetic: every way to combine the numbers of a problem, computed exactly.

An expression is a tree: ("n", i) is the i-th number of the text, (op, left, right) an operation.
For k numbers all expressions with up to two operations over different numbers are listed once per
meaning: (a + b) - c and a - (c - b) are the same expression, the list keeps one of them. "The same" is
decided by computing both on probe values with exact fractions, not by rewriting rules.

The network never computes: it only says which of the listed expressions the text means.
"""
from fractions import Fraction
from functools import lru_cache

OPS = "+-*/"
PROBES = [[Fraction(p) for p in row] for row in (
    (1009, 2003, 3011, 4001, 5003, 6007, 7001, 8009, 9001, 10007),
    (7919, 104729, 1299709, 15485863, 179424673, 2038074743, 541, 3571, 27457, 224737))]


def value(t, nums):
    """the exact value of the expression, None when it divides by zero"""
    if t[0] == "n":
        return nums[t[1]]
    a, b = value(t[1], nums), value(t[2], nums)
    if a is None or b is None:
        return None
    if t[0] == "+":
        return a + b
    if t[0] == "-":
        return a - b
    if t[0] == "*":
        return a * b
    return a / b if b != 0 else None


def meaning(t):
    """what tells expressions apart: the values on the probes"""
    return tuple(value(t, p) for p in PROBES)


def leaves(t):
    return [t[1]] if t[0] == "n" else leaves(t[1]) + leaves(t[2])


def n_ops(t):
    return 0 if t[0] == "n" else 1 + n_ops(t[1]) + n_ops(t[2])


def _one(a, b):
    x, y = ("n", a), ("n", b)
    return [("+", x, y), ("-", x, y), ("-", y, x), ("*", x, y), ("/", x, y), ("/", y, x)]


@lru_cache(maxsize=None)
def expressions(k, max_ops=2):
    """all different expressions over k numbers: first with one operation, then with two"""
    out, seen = [], set()

    def add(t):
        m = meaning(t)
        if m not in seen:
            seen.add(m)
            out.append(t)

    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    for a, b in pairs:
        for t in _one(a, b):
            add(t)
    if max_ops >= 2:
        for a, b in pairs:
            for inner in _one(a, b):
                for c in range(k):
                    if c in (a, b):
                        continue
                    x = ("n", c)
                    for t in (("+", inner, x), ("-", inner, x), ("-", x, inner), ("*", inner, x), ("/", inner, x), ("/", x, inner)):
                        add(t)
    return tuple(out)


def from_prefix(tokens):
    """['-', 'number1', 'number0'] -> tree; None when it has something that is not a number of the text"""
    pos = [0]

    def read():
        if pos[0] >= len(tokens):
            return None
        tok = tokens[pos[0]]
        pos[0] += 1
        if tok in OPS:
            a = read()
            b = read()
            return None if a is None or b is None else (tok, a, b)
        if tok.startswith("number"):
            return ("n", int(tok[6:]))
        return None

    t = read()
    return t if t is not None and pos[0] == len(tokens) else None


def find(t, k, max_ops=2):
    """the place of an expression in expressions(k), None when it is not there"""
    if t is None or t[0] == "n" or n_ops(t) > max_ops or len(set(leaves(t))) != len(leaves(t)) or max(leaves(t)) >= k:
        return None
    m = meaning(t)
    for i, c in enumerate(expressions(k, max_ops)):
        if meaning(c) == m:
            return i
    return None


def show(t, nums=None):
    if t[0] == "n":
        if nums is None:
            return f"n{t[1]}"
        v = nums[t[1]]
        return str(v.numerator) if v.denominator == 1 else str(float(v))
    return f"({show(t[1], nums)} {t[0]} {show(t[2], nums)})"


def facts(t, nums, v):
    """exact facts about a computed expression that the network may look at (it cannot compute them itself)"""
    used = [nums[i] for i in leaves(t)]
    inner = [value(s, nums) for s in (t[1], t[2]) if s[0] != "n"]
    return [
        1.0 if v.denominator == 1 else 0.0,                                  # a whole number
        1.0 if v < 1 else 0.0,
        1.0 if v >= max(used) else 0.0,                                      # not less than everything it is made of
        1.0 if v <= min(used) else 0.0,
        1.0 if all(u.denominator == 1 for u in used) else 0.0,               # made of whole numbers
        1.0 if inner and inner[0] is not None and inner[0].denominator == 1 else 0.0,
        1.0 if inner and inner[0] is not None and inner[0] > 0 else 0.0,
        1.0 if inner else 0.0,                                               # two operations
    ]


N_FACTS = 8


if __name__ == "__main__":
    for k in range(2, 8):
        print(k, "numbers:", len(expressions(k, 1)), "expressions with one operation,", len(expressions(k, 2)), "with up to two")
    nums = [Fraction(7), Fraction(12), Fraction(15)]
    t = from_prefix("- + number0 number1 number2".split())
    print(show(t, nums), "=", value(t, nums), "; place", find(t, 3), "of", len(expressions(3)), "->", show(expressions(3)[find(t, 3)]))
