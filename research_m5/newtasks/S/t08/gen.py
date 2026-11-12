#!/usr/bin/env python3
"""Synthetic map task given as records: places and four fixed directions.

Usage: python gen.py N SEED [SIZE]      SIZE is 1, 3 or 10 (default 1)

Records (3 tokens each): one place lies in one of four directions of another.
    kuv p1 p2.    p1 lies north of p2
    tiz p1 p2.    p1 lies south of p2
    pem p1 p2.    p1 lies east of p2
    vob p1 p2.    p1 lies west of p2
Queries:
    qnu p1 p2?    yes/no: does p1 lie north of p2?
    qeu p1 p2?    yes/no: does p1 lie east of p2?
    qpa p1 p2?    the places on the one and only chain of records from p1 to
                  p2, in order, after p1, separated by commas
A query is answered only from the records above it, and it is placed only
where its answer follows from those records and no single record gives it.
"""
import random
import sys
from itertools import combinations
from math import comb

PLACES = [f"p{i}" for i in range(1, 9)]
Q_NORTH, Q_EAST, Q_PATH = "qnu", "qeu", "qpa"
MAX_RECORDS = 4 * comb(8, 2)  # distinct true record rows for 8 places


def record_texts(places, ys, xs):
    """Every true record row for the hidden positions: (axis, hi, lo, row).

    Each unordered pair gives two facts (north-south and east-west), and each
    fact can be written in two ways (kuv a b / tiz b a, pem a b / vob b a).
    """
    out = []
    for i, j in combinations(range(len(places)), 2):
        for axis, pos in (("y", ys), ("x", xs)):
            hi, lo = (i, j) if pos[i] > pos[j] else (j, i)
            H, L = places[hi], places[lo]
            if axis == "y":
                out.append(("y", H, L, f"kuv {H} {L}."))
                out.append(("y", H, L, f"tiz {L} {H}."))
            else:
                out.append(("x", H, L, f"pem {H} {L}."))
                out.append(("x", H, L, f"vob {L} {H}."))
    return out


def answer_direction(facts, kind, a, c):
    """yes/no from the facts (axis, hi, lo) above; None if not determined."""
    axis = "y" if kind == Q_NORTH else "x"
    for ax, h, l in facts:
        if ax == axis and {h, l} == {a, c}:
            return None  # one record gives it directly: not a multi-record case
    adj = {}
    for ax, h, l in facts:
        if ax == axis:
            adj.setdefault(h, []).append(l)

    def reach(src, dst):
        seen, stack = {src}, [src]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v == dst:
                    return True
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    if reach(a, c):
        return "yes"
    if reach(c, a):
        return "no"
    return None


def answer_path(facts, a, c):
    """The places after a on the unique simple chain of records a..c, or None."""
    nbr = {}
    for _, h, l in facts:
        nbr.setdefault(h, set()).add(l)
        nbr.setdefault(l, set()).add(h)
    found = []

    def walk(u, path):
        if len(found) > 1:
            return
        if u == c:
            found.append(list(path))
            return
        for v in sorted(nbr.get(u, ())):
            if v not in path:
                path.append(v)
                walk(v, path)
                path.pop()

    walk(a, [a])
    if len(found) != 1 or len(found[0]) < 3:  # at least two records
        return None
    return ",".join(found[0][1:])


def answer(facts, kind, a, c):
    if kind == Q_PATH:
        return answer_path(facts, a, c)
    return answer_direction(facts, kind, a, c)


def make_episode(rng, size):
    """Return the episode's rows (without numbers), or None to retry."""
    target = min(rng.randint(3, 12) * size, MAX_RECORDS)
    k_min = 3
    while 4 * comb(k_min, 2) < target:
        k_min += 1
    k = rng.randint(k_min, min(8, k_min + 2))
    places = sorted(rng.sample(PLACES, k), key=lambda s: int(s[1:]))
    ys = rng.sample(range(20), k)  # distinct north-south coordinates
    xs = rng.sample(range(20), k)  # distinct east-west coordinates
    recs = rng.sample(record_texts(places, ys, xs), target)
    facts = [r[:3] for r in recs]

    cands = []
    for kind in (Q_NORTH, Q_EAST, Q_PATH):
        for a in places:
            for c in places:
                if a == c:
                    continue
                valid = [p for p in range(1, len(recs) + 1)
                         if answer(facts[:p], kind, a, c) is not None]
                if valid:
                    cands.append((kind, a, c, valid))
    if not cands:
        return None

    rng.shuffle(cands)
    q = min(rng.randint(1, 5), len(cands))
    chosen = []
    for kind, a, c, valid in cands[:q]:
        p = rng.choice(valid)
        chosen.append((p, kind, a, c))
    chosen.sort(key=lambda t: t[0])

    rows = []
    qi = 0
    for p in range(len(recs) + 1):
        while qi < len(chosen) and chosen[qi][0] == p:
            _, kind, a, c = chosen[qi]
            ans = answer(facts[:p], kind, a, c)
            rows.append(f"{kind} {a} {c}?\t{ans}")
            qi += 1
        if p < len(recs):
            rows.append(recs[p][3])
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        print("usage: python gen.py N SEED [SIZE]", file=sys.stderr)
        return 2
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        print("SIZE must be 1, 3 or 10", file=sys.stderr)
        return 2
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        rows = make_episode(rng, size)
        if rows is None:
            continue
        text = "\n".join(f"{i} {row}" for i, row in enumerate(rows, 1))
        if text in seen:
            continue
        seen.add(text)
        episodes.append(text)
    print("\n\n".join(episodes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
