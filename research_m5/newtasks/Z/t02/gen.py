#!/usr/bin/env python3
"""Synthetic record-log task: weighted vote tallies.

Records cast, withdraw, swap and re-weight votes of voters for candidates.
Queries ask for a weighted count, the candidate a voter currently backs,
or the leader(s) among a given set of candidates. Answers come from a
simulation of the records above each query.

Usage: python gen.py N SEED SIZE   (SIZE in 1, 3, 10; default 1)
"""
import random
import sys

VOTERS = ["v%d" % i for i in range(1, 9)]
CANDIDATES = ["c%d" % i for i in range(1, 5)]


def simulate(records):
    vote = {}    # voter -> candidate currently backed
    weight = {}  # voter -> weight (default 1)
    for op, args in records:
        if op == "cst":
            vote[args[0]] = args[1]
        elif op == "wdr":
            vote.pop(args[0], None)
        elif op == "swp":
            a, b = args
            va = vote.pop(a, None)
            vb = vote.pop(b, None)
            if va is not None:
                vote[b] = va
            if vb is not None:
                vote[a] = vb
        elif op == "wgt":
            weight[args[0]] = args[1]
    return vote, weight


def tally(vote, weight, cand):
    return sum(weight.get(v, 1) for v, c in vote.items() if c == cand)


def answer(records, query):
    """Return the answer string, or None when it is undefined."""
    vote, weight = simulate(records)
    qop, args = query
    if qop == "cnt":
        return str(tally(vote, weight, args[0]))
    if qop == "hld":
        return vote.get(args[0])
    counts = {c: tally(vote, weight, c) for c in args}
    top = max(counts.values())
    if top == 0:
        return None
    return ",".join(c for c in args if counts[c] == top)


def support(records, query, full):
    """Number of earlier records whose removal changes the answer."""
    n = 0
    for i in range(len(records)):
        if answer(records[:i] + records[i + 1:], query) != full:
            n += 1
    return n


def make_records(rng, size):
    n = rng.randint(3 * size, 12 * size)
    max_weight = 3 * size
    recs = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            recs.append(("cst", (rng.choice(VOTERS), rng.choice(CANDIDATES))))
        elif r < 0.60:
            recs.append(("wdr", (rng.choice(VOTERS),)))
        elif r < 0.80:
            a, b = rng.sample(VOTERS, 2)
            recs.append(("swp", (a, b)))
        else:
            recs.append(("wgt", (rng.choice(VOTERS), rng.randint(1, max_weight))))
    return recs


def make_queries(rng, records, want):
    chosen = []
    seen = set()
    for _ in range(400):
        if len(chosen) >= want:
            break
        p = rng.randint(2, len(records))  # query sits after record p
        kind = rng.randrange(3)
        if kind == 0:
            q = ("cnt", (rng.choice(CANDIDATES),))
        elif kind == 1:
            q = ("hld", (rng.choice(VOTERS),))
        else:
            q = ("bst", tuple(sorted(rng.sample(CANDIDATES, rng.randint(2, 4)))))
        if (p, q) in seen:
            continue
        prefix = records[:p]
        full = answer(prefix, q)
        if full is None or support(prefix, q, full) < 2:
            continue
        seen.add((p, q))
        chosen.append((p, q, full))
    return chosen


def render(records, queries):
    lines = []
    for k in range(1, len(records) + 1):
        op, args = records[k - 1]
        lines.append(" ".join([op] + [str(a) for a in args]) + ".")
        for p, q, full in queries:
            if p == k:
                lines.append(" ".join([q[0]] + list(q[1])) + "?\t" + full)
    return lines


def make_episode(rng, size):
    while True:
        recs = make_records(rng, size)
        qs = make_queries(rng, recs, rng.randint(1, 5))
        if qs:
            return render(recs, qs)


def main(argv):
    count = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    for _ in range(count):
        while True:
            lines = make_episode(rng, size)
            key = "\n".join(lines)
            if key not in seen:
                seen.add(key)
                break
        for i, line in enumerate(lines, 1):
            print("%d %s" % (i, line))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
