"""Synthetic scope-tracking task given as records.

Each episode is a log of scope openings, closings and variable bindings.
Queries ask about the variables visible at that point of the log:
  xup v?  -> the value bound to v in the innermost scope that binds it
  sco v?  -> the label of that innermost scope
  hob v?  -> yes/no, whether v is bound in any open scope
Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)
"""

import random
import sys

# record operations (at most 6): kag = bind, oru = open scope, ceq = close scope
# query operations (at most 3): xup = value, sco = scope, hob = is bound
VARS = ["v%d" % i for i in range(1, 9)]
VALS = ["k%d" % i for i in range(1, 9)]
LABELS = ["s%d" % i for i in range(1, 9)]


def lookup(stack, var):
    """Return (scope label, value) of the innermost binding of var, or (None, None)."""
    for label, table in reversed(stack):
        if var in table:
            return label, table[var]
    return None, None


def visible_vars(stack):
    seen = []
    for _, table in stack:
        for var in table:
            if var not in seen:
                seen.append(var)
    return seen


def record(rng, stack, max_depth):
    used = [label for label, _ in stack]
    free = [label for label in LABELS if label not in used]
    can_open = bool(free) and len(stack) < max_depth
    r = rng.random()
    if not stack or (can_open and r < 0.3):
        label = rng.choice(free)
        stack.append([label, {}])
        return "oru %s." % label
    if r < 0.45:
        label, _ = stack.pop()
        return "ceq %s." % label
    var = rng.choice(VARS)
    val = rng.choice(VALS)
    stack[-1][1][var] = val
    return "kag %s %s." % (var, val)


def query(rng, stack):
    vis = visible_vars(stack)
    hidden = [var for var in VARS if var not in vis]
    r = rng.random()
    if vis and r < 0.4:
        op = "xup"
    elif vis and r < 0.7:
        op = "sco"
    else:
        op = "hob"
    if op == "hob":
        if vis and (not hidden or rng.random() < 0.5):
            var = rng.choice(vis)
        else:
            var = rng.choice(hidden or VARS)
        label, _ = lookup(stack, var)
        ans = "yes" if label is not None else "no"
    else:
        var = rng.choice(vis)
        label, val = lookup(stack, var)
        ans = val if op == "xup" else label
    return "%s %s?\t%s" % (op, var, ans)


def make_episode(rng, size):
    max_depth = min(len(LABELS), 2 + size)
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    total = n_rec + n_q
    q_pos = set(rng.sample(range(1, total), n_q))  # position 0 is always a record
    stack = []
    lines = []
    for pos in range(total):
        if pos in q_pos:
            lines.append(query(rng, stack))
        else:
            lines.append(record(rng, stack, max_depth))
    return lines


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            lines = make_episode(rng, size)
            key = "\n".join(lines)
            if key not in seen:
                break
        seen.add(key)
        for i, line in enumerate(lines, 1):
            print("%d %s" % (i, line))


if __name__ == "__main__":
    main(sys.argv)
