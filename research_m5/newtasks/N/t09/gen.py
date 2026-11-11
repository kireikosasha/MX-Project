#!/usr/bin/env python3
"""Synthetic record-chain episodes.

Records give one relation between pairs of entities: "vek A B." makes B the
entity directly above A (each entity has at most one link upward), and
"dro A." removes A's link upward. Queries follow the relation two or more
steps: "ulo X?" the top of the chain of X, "pik X?" the entity exactly two
steps above X, "vom X Y?" whether Y is above X at distance two or more.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

RECORD_OPS = ("vek", "dro")
QUERY_OPS = ("ulo", "pik", "vom")
POOL = [f"k{i}" for i in range(1, 9)]
SIZES = (1, 3, 10)


class Retry(Exception):
    """Raised when a partial episode cannot be completed; the episode is redrawn."""


def ancestors(parent, x):
    """Entities above x, nearest first."""
    out = []
    while x in parent:
        x = parent[x]
        out.append(x)
    return out


def make_record(rng, parent, ents):
    kids = [e for e in ents if e in parent]
    if kids and rng.random() < 0.15:
        child = rng.choice(kids)
        del parent[child]
        return f"dro {child}."
    for _ in range(50):
        a = rng.choice(ents)
        b = rng.choice(ents)
        if a == b or parent.get(a) == b:
            continue
        if a in ancestors(parent, b):  # would close a cycle
            continue
        parent[a] = b
        return f"vek {a} {b}."
    if kids:
        child = rng.choice(kids)
        del parent[child]
        return f"dro {child}."
    raise Retry()


def make_query(rng, parent, ents):
    """Return (row text without number, answer) for a query whose answer
    needs at least two links, or None if no such query exists right now."""
    deep = [e for e in ents if len(ancestors(parent, e)) >= 2]
    if not deep:
        return None
    ops = list(QUERY_OPS)
    rng.shuffle(ops)
    for op in ops:
        x = rng.choice(deep)
        anc = ancestors(parent, x)
        if op == "ulo":
            return f"ulo {x}?", anc[-1]
        if op == "pik":
            return f"pik {x}?", anc[1]
        if op == "vom":
            if rng.random() < 0.5:
                y = rng.choice(anc[1:])
                ans = "yes"
            else:
                non = [e for e in ents if e != x and e not in anc]
                if not non:
                    continue
                y = rng.choice(non)
                ans = "no"
            return f"vom {x} {y}?", ans
    return None


def make_episode(rng, size):
    m = rng.randint(4, 8)
    ents = sorted(rng.sample(POOL, m))
    recs = rng.randint(3 * size, 12 * size)
    qs = rng.randint(1, 5)
    parent = {}
    rows = []
    while recs or qs:
        want_q = qs > 0 and (recs == 0 or rng.random() < qs / (qs + recs))
        if want_q:
            q = make_query(rng, parent, ents)
            if q is not None:
                text, ans = q
                rows.append(f"{text}\t{ans}")
                qs -= 1
                continue
        if recs == 0:
            raise Retry()
        rows.append(make_record(rng, parent, ents))
        recs -= 1
    return rows


def generate(n, seed, size):
    rng = random.Random(seed)
    seen = set()
    episodes = []
    tries = 0
    while len(episodes) < n:
        tries += 1
        if tries > 100000:
            raise RuntimeError("could not generate enough distinct episodes")
        try:
            rows = make_episode(rng, size)
        except Retry:
            continue
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(rows)
    return episodes


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
    out = []
    for rows in generate(n, seed, size):
        for i, row in enumerate(rows, 1):
            out.append(f"{i} {row}")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
