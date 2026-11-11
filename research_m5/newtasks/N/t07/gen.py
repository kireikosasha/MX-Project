#!/usr/bin/env python3
"""Synthetic ledger task given as records.

Holders h1..h8 carry amounts. Records set, add, remove and move amounts.
Queries ask a holder's current amount, which holder of a listed group has
the most (unique maximum only), or whether one holder has at least as much
as another. Answers come from a simulation of the records above each query.

Usage: python gen.py N SEED [SIZE]     SIZE in {1, 3, 10}, default 1
"""
import random
import sys

HOLDER_COUNT = 8

SET_OP = "tev"
ADD_OP = "kav"
REMOVE_OP = "dul"
MOVE_OP = "pim"

AMT_Q = "amt"
MOST_Q = "wex"
GTE_Q = "gte"


def record(rng, names, bal, maxamt):
    funded = [h for h in names if bal[h] > 0]
    r = rng.random()
    if r < 0.15:
        h = rng.choice(names)
        v = rng.randint(0, maxamt)
        bal[h] = v
        return "%s %s %d." % (SET_OP, h, v)
    if r < 0.45 or not funded:
        h = rng.choice(names)
        v = rng.randint(1, maxamt)
        bal[h] += v
        return "%s %s %d." % (ADD_OP, h, v)
    if r < 0.70:
        h = rng.choice(funded)
        v = rng.randint(1, min(bal[h], maxamt))
        bal[h] -= v
        return "%s %s %d." % (REMOVE_OP, h, v)
    a = rng.choice(funded)
    b = rng.choice([h for h in names if h != a])
    v = rng.randint(1, min(bal[a], maxamt))
    bal[a] -= v
    bal[b] += v
    return "%s %s %s %d." % (MOVE_OP, a, b, v)


def query(rng, names, bal):
    for _ in range(30):
        t = rng.random()
        if t < 0.4:
            h = rng.choice(names)
            return "%s %s?" % (AMT_Q, h), str(bal[h])
        if t < 0.7:
            m = rng.randint(2, min(4, len(names)))
            group = sorted(rng.sample(names, m))
            vals = [bal[h] for h in group]
            top = max(vals)
            if vals.count(top) == 1:
                winner = group[vals.index(top)]
                return "%s %s?" % (MOST_Q, " ".join(group)), winner
        else:
            a, b = rng.sample(names, 2)
            ans = "yes" if bal[a] >= bal[b] else "no"
            return "%s %s %s?" % (GTE_Q, a, b), ans
    h = rng.choice(names)
    return "%s %s?" % (AMT_Q, h), str(bal[h])


def episode(rng, size):
    maxamt = 20 * size
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    k = rng.randint(3, 6)
    names = ["h%d" % i for i in sorted(rng.sample(range(1, HOLDER_COUNT + 1), k))]
    bal = {h: 0 for h in names}

    # queries go after at least two records, so each has something to read
    positions = sorted(rng.randint(2, n_rec) for _ in range(n_q))
    rows = []
    qi = 0
    for i in range(1, n_rec + 1):
        rows.append(record(rng, names, bal, maxamt))
        while qi < n_q and positions[qi] == i:
            text, ans = query(rng, names, bal)
            rows.append(text + "\t" + ans)
            qi += 1
    return ["%d %s" % (j + 1, row) for j, row in enumerate(rows)]


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            lines = episode(rng, size)
            key = "\n".join(lines)
            if key not in seen:
                break
        seen.add(key)
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
