#!/usr/bin/env python3
"""Synthetic record-ledger task: items change hands between holders (exchange).

Usage: python gen.py N SEED [SIZE]   (SIZE in 1, 3, 10; default 1)

Record operations (2-4 letter codes, not English words):
  ka i h    item i comes into the possession of holder h
  xe i h    item i passes from its current holder to holder h
  sw a b    items a and b exchange their holders
  ud i      item i leaves circulation (no holder afterwards)
Query operations:
  fx i?     which holder holds item i now
  lx h?     which items holder h holds now (sorted, comma-separated)
  hs i h?   does holder h hold item i now (yes / no)
Entity codes: holders h1..h8, items i1..i8.
"""
import random
import sys

HOLDERS = ["h%d" % d for d in range(1, 9)]
ITEMS = ["i%d" % d for d in range(1, 9)]
SIZES = (1, 3, 10)


def apply_op(holder, op):
    kind = op[0]
    if kind == "ka" or kind == "xe":
        holder[op[1]] = op[2]
    elif kind == "sw":
        a, b = op[1], op[2]
        holder[a], holder[b] = holder[b], holder[a]
    else:  # ud
        holder[op[1]] = None


def replay(ops, k):
    holder = {}
    for op in ops[:k]:
        apply_op(holder, op)
    return holder


def touched(op):
    return op[1:3] if op[0] == "sw" else op[1:2]


def record_text(op):
    if op[0] == "ud":
        return "ud %s." % op[1]
    return "%s %s %s." % (op[0], op[1], op[2])


def gen_records(rng, holders, items, n_rec):
    holder = {}
    ops = []
    while len(ops) < n_rec:
        new = [i for i in items if i not in holder]
        held = [i for i in items if holder.get(i) is not None]
        pairs = [(a, b) for a in held for b in held
                 if a < b and holder[a] != holder[b]]
        options = []
        if new:
            options.append(("ka", 3))
        if held:
            options.append(("xe", 4))
        if pairs:
            options.append(("sw", 2))
        if len(held) > 2:  # keep at least two held items so chains never stall
            options.append(("ud", 1))
        kind = rng.choices([o for o, _ in options], [w for _, w in options])[0]
        if kind == "ka":
            op = ("ka", rng.choice(new), rng.choice(holders))
        elif kind == "xe":
            i = rng.choice(held)
            op = ("xe", i, rng.choice([h for h in holders if h != holder[i]]))
        elif kind == "sw":
            a, b = rng.choice(pairs)
            op = ("sw", a, b)
        else:
            op = ("ud", rng.choice(held))
        apply_op(holder, op)
        ops.append(op)
    return ops


def gen_queries(rng, ops, holders, items, n_q):
    """Return [(k, query_text, answer)] where k = number of records above it."""
    n = len(ops)
    found = []
    seen = set()
    attempts = 0
    while len(found) < n_q and attempts < 300:
        attempts += 1
        k = rng.randint(2, n)
        state = replay(ops, k)
        kind = rng.choice(["fx", "lx", "hs"])
        if kind == "fx":
            cands = [i for i in items if state.get(i) is not None]
            if not cands:
                continue
            i = rng.choice(cands)
            text, ans, rel = "fx %s" % i, state[i], {i}
        elif kind == "lx":
            h = rng.choice(holders)
            mine = sorted(i for i in items if state.get(i) == h)
            if not mine:
                continue
            text, ans, rel = "lx %s" % h, ",".join(mine), set(mine)
        else:
            cands = [i for i in items if i in state]
            if not cands:
                continue
            i = rng.choice(cands)
            h = rng.choice(holders)
            text = "hs %s %s" % (i, h)
            ans = "yes" if state[i] == h else "no"
            rel = {i}
        # the answer must rest on at least two records that touch the relevant items
        support = sum(1 for op in ops[:k] if rel & set(touched(op)))
        if support < 2:
            continue
        if (k, text) in seen:
            continue
        seen.add((k, text))
        found.append((k, text, ans))
    return found


def make_episode(rng, size):
    n_rec = rng.randint(3, 12) * size
    n_hold = rng.randint(2, min(8, 4 + size // 2))
    n_item = rng.randint(2, min(8, 4 + size // 2))
    holders = sorted(rng.sample(HOLDERS, n_hold))
    items = sorted(rng.sample(ITEMS, n_item))
    ops = gen_records(rng, holders, items, n_rec)
    qs = gen_queries(rng, ops, holders, items, rng.randint(1, 5))
    if not qs:
        return None
    at = {}
    for k, text, ans in qs:
        at.setdefault(k, []).append((text, ans))
    lines = []
    for k in range(0, n_rec + 1):
        if k > 0:
            lines.append(record_text(ops[k - 1]))
        for text, ans in at.get(k, []):
            lines.append("%s?\t%s" % (text, ans))
    return lines


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    rows = []
    for _ in range(n):
        while True:
            lines = make_episode(rng, size)
            if lines is None:
                continue
            key = "\n".join(lines)
            if key not in seen:
                seen.add(key)
                break
        rows.extend("%d %s" % (num, line) for num, line in enumerate(lines, 1))
    if rows:
        sys.stdout.write("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
