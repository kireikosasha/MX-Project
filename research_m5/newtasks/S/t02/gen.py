#!/usr/bin/env python3
"""Synthetic record-log task: items change hands between holders (exchange).

Each episode is a log of records (acquire, pass, swap, discard) followed by
queries about who holds what at that point. Answers come from a simulation.

Usage: python gen.py N SEED [SIZE]     SIZE is 1 (default), 3 or 10.
"""
import random
import sys

REC_OPS = ("dax", "mev", "swp", "zoq")   # acquire, pass on, swap, discard
QRY_OPS = ("hol", "kep", "ynq")          # holder of item, items of holder, holds?
HOLDERS = ["b%d" % i for i in range(1, 9)]
ITEMS = ["k%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)


def choose_record(rng, holders, items, holder_of):
    """Apply one random legal record to the state; return (text, holders, items)."""
    free = [i for i in items if holder_of[i] is None]
    held = [i for i in items if holder_of[i] is not None]
    pairs = [(i, j) for i in held for j in held
             if i < j and holder_of[i] != holder_of[j]]
    weights = {"dax": 3 if free else 0,
               "mev": 4 if held else 0,
               "zoq": 1 if held else 0,
               "swp": 2 if pairs else 0}
    ops = [op for op in REC_OPS if weights[op] > 0]
    op = rng.choices(ops, weights=[weights[o] for o in ops])[0]

    if op == "dax":            # holder h receives a free item i
        i = rng.choice(free)
        h = rng.choice(holders)
        holder_of[i] = h
        return "dax %s %s." % (h, i), [h], [i]
    if op == "mev":            # item i passes from its holder a to holder b
        i = rng.choice(held)
        a = holder_of[i]
        b = rng.choice([h for h in holders if h != a])
        holder_of[i] = b
        return "mev %s %s %s." % (a, i, b), [a, b], [i]
    if op == "zoq":            # holder a discards item i (item becomes free)
        i = rng.choice(held)
        a = holder_of[i]
        holder_of[i] = None
        return "zoq %s %s." % (a, i), [a], [i]
    # swp: holder a gives item i to b and receives item j from b
    i, j = rng.choice(pairs)
    a, b = holder_of[i], holder_of[j]
    holder_of[i], holder_of[j] = b, a
    return "swp %s %s %s %s." % (a, i, b, j), [a, b], [i, j]


def choose_query(rng, holders, items, holder_of, mh, mi, asked):
    """Pick a query answerable from the state; return (text, answer) or None."""
    cands = []
    for i in items:                       # who holds item i now
        if holder_of[i] is not None and mi[i] >= 2:
            cands.append(("hol", [i], holder_of[i]))
    for h in holders:                     # what does holder h hold now
        held = [i for i in items if holder_of[i] == h]
        if held and mh[h] >= 2:
            cands.append(("kep", [h], ",".join(sorted(held))))
    for h in holders:                     # does holder h hold item i now
        for i in items:
            if mi[i] >= 2 and mh[h] >= 2:
                cands.append(("ynq", [h, i], "yes" if holder_of[i] == h else "no"))
    cands = [c for c in cands if (c[0], tuple(c[1])) not in asked]
    if not cands:
        return None
    op, ents, ans = rng.choice(cands)
    asked.add((op, tuple(ents)))
    return "%s %s?" % (op, " ".join(ents)), ans


def try_episode(rng, size):
    """Return a list of (text, answer-or-None) rows, or None if it failed."""
    holders = rng.sample(HOLDERS, rng.randint(3, 5))
    items = rng.sample(ITEMS, rng.randint(2, 4))
    holder_of = {i: None for i in items}
    mh = {h: 0 for h in holders}          # how many records mention each holder
    mi = {i: 0 for i in items}            # how many records mention each item
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    asked = set()
    rows = []
    qdone = 0
    for r in range(n_rec):
        text, hs, its = choose_record(rng, holders, items, holder_of)
        for h in hs:
            mh[h] += 1
        for i in its:
            mi[i] += 1
        rows.append((text, None))
        if qdone < n_q and r >= 1 and rng.random() < 0.3:
            q = choose_query(rng, holders, items, holder_of, mh, mi, asked)
            if q is not None:
                rows.append(q)
                qdone += 1
    while qdone < n_q:
        q = choose_query(rng, holders, items, holder_of, mh, mi, asked)
        if q is None:
            return None
        rows.append(q)
        qdone += 1
    return rows


def gen_episode(rng, size):
    while True:
        rows = try_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            rows = gen_episode(rng, size)
            key = "\n".join(t + ("\t" + a if a is not None else "") for t, a in rows)
            if key not in seen:
                break
        seen.add(key)
        for k, (text, ans) in enumerate(rows, 1):
            line = "%d %s" % (k, text)
            if ans is not None:
                line += "\t" + ans
            print(line)


if __name__ == "__main__":
    main(sys.argv)
