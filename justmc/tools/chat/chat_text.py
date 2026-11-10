#!/usr/bin/env python3
"""
Numbers and step-by-step calculation of the big Millennium chat model (shared by the data builder, the
reference implementation and the export; demo_reader.jc does the same in JustMC).

The model never sees digits it has to compute with:
  - the numbers of a message become slot tokens [N1] .. [N6] in the order they appear;
  - to calculate, the reply starts with steps in prefix form:   [*] [N1] 64 [R1]  [+] [R1] [N2] [R2]
      operation token, two operands (a slot, an earlier result, or a number that is one vocabulary token),
      then the outcome token, which is not chosen by the model: the code computes the step and feeds
      [R1] .. [R4] (the result of an arithmetic step), or [T] / [F] for a comparison, [F] for an impossible
      step (division by zero);
  - in the text of the reply [N1] / [R2] are written as the numbers they stand for.
So the arithmetic is exact, and what the model learns is which steps a question needs.
"""
import math
import re

N_SLOTS = 6
R_SLOTS = 4
OPS = ["+", "-", "*", "/", "%", "^", ">", "<", "="]
COMPARE = {">", "<", "="}
OP_TOKENS = [f"[{o}]" for o in OPS]
SLOT_TOKENS = [f"[N{i + 1}]" for i in range(N_SLOTS)]
RES_TOKENS = [f"[R{i + 1}]" for i in range(R_SLOTS)]
SPECIAL = ["[U]", "[B]", "[E]", "[W]"] + OP_TOKENS + SLOT_TOKENS + RES_TOKENS + ["[T]", "[F]"]
BIG = 1e12                     # results beyond it are "impossible"

# the same three replacements as ch_prepare in demo_reader.jc
_TIMES = re.compile(r"(?<=[0-9])[xXхХ](?=[0-9])")
_DEC = re.compile(r"(?<![0-9.,])([0-9]{1,12})[.,]([0-9]{1,6})(?![.,]?[0-9])")
NUM_RE = re.compile(r"[0-9]{1,12}(?:d[0-9]{1,6})?")


def prepare(t):
    """multiplication signs -> *, 5x3 -> 5 * 3, a decimal number becomes one word: 2,5 -> 2d5"""
    t = t.replace("×", "*").replace("÷", "/")
    t = _TIMES.sub(" * ", t)
    return _DEC.sub(r"\1d\2", t)


def number(word):
    """the value of a word of the prepared text, or None"""
    if not NUM_RE.fullmatch(word):
        return None
    return float(word.replace("d", "."))


def fmt(v):
    """a number as the reply shows it: at most 2 decimals, no trailing zeros"""
    if v != v or abs(v) >= BIG:
        return "∞"
    q = math.floor(round(abs(v) * 100, 9) + 0.5)
    ip, fp = int(q // 100), int(q % 100)
    s = str(ip)
    if fp:
        s += "." + (str(fp // 10) if fp % 10 == 0 else f"{fp:02d}")
    return ("-" if v < 0 and (ip or fp) else "") + s


def calc(op, a, b):
    """-> ("R", value) for arithmetic, ("T",) / ("F",) for a comparison, ("F",) for an impossible step"""
    if op == ">":
        return ("T",) if a > b else ("F",)
    if op == "<":
        return ("T",) if a < b else ("F",)
    if op == "=":
        return ("T",) if abs(a - b) < 5e-10 else ("F",)
    if op == "+":
        r = a + b
    elif op == "-":
        r = a - b
    elif op == "*":
        r = a * b
    elif op == "/":
        if b == 0:
            return ("F",)
        r = a / b
    elif op == "%":
        if b == 0:
            return ("F",)
        r = a - b * math.floor(round(a / b, 9))
    elif op == "^":
        if abs(b) > 64 or (a == 0 and b < 0) or (a < 0 and b != math.floor(b)):
            return ("F",)
        if a != 0 and round(b * math.log10(abs(a)), 9) > 12.5:          # too big to compute
            return ("F",)
        try:
            r = math.pow(a, b)
        except (OverflowError, ValueError):
            return ("F",)
    else:
        raise ValueError(op)
    r = round(r, 9)
    if r != r or abs(r) >= BIG:
        return ("F",)
    return ("R", r)
