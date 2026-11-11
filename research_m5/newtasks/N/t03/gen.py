#!/usr/bin/env python3
"""Synthetic records task: comparisons on one scale.

Records state that one entity is above or below another on a single scale.
Queries ask about the order implied by several records: whether x is above y,
which of a few listed entities is the highest, which is the lowest. Every query
is asked only after the records it depends on, and only when the records so far
determine its answer.

Usage: python gen.py N SEED [SIZE]
  N     number of episodes to print
  SEED  seed for random.Random
  SIZE  1, 3 or 10 (default 1)
"""
import itertools
import random
import sys

REC_ABOVE = "zog"   # "zog a b."  : a is above b
REC_BELOW = "vem"   # "vem a b."  : a is below b (the same as b above a)
Q_PAIR = "qab"      # "qab a b?"  : is a above b (yes / no)
Q_HIGH = "mxq"      # "mxq a b c?": which listed entity is the highest
Q_LOW = "mnq"       # "mnq a b c?": which listed entity is the lowest

KIND = "k"
POOL = [KIND + str(i) for i in range(1, 9)]


def reach_matrix(m, edges):
    """edges: (x, y) pairs meaning x is above y. Returns the transitive closure."""
    r = [[False] * m for _ in range(m)]
    for x, y in edges:
        r[x][y] = True
    for k in range(m):
        for i in range(m):
            if r[i][k]:
                for j in range(m):
                    if r[k][j]:
                        r[i][j] = True
    return r


def candidates(m, names, edges, rng):
    """Queries (text, answer) that the records so far determine and that
    cannot be read from a single record."""
    direct = set(edges)
    r = reach_matrix(m, edges)
    out = []
    for a, b in itertools.permutations(range(m), 2):
        if (a, b) in direct or (b, a) in direct:
            continue
        if r[a][b] or r[b][a]:
            ans = "yes" if r[a][b] else "no"
            out.append((f"{Q_PAIR} {names[a]} {names[b]}?", ans))
    for k in (3, 4):
        for group in itertools.combinations(range(m), k):
            for w in group:
                others = [x for x in group if x != w]
                listed = list(group)
                if all(r[w][x] for x in others) and any((w, x) not in direct for x in others):
                    rng.shuffle(listed)
                    text = f"{Q_HIGH} " + " ".join(names[x] for x in listed) + "?"
                    out.append((text, names[w]))
                if all(r[x][w] for x in others) and any((x, w) not in direct for x in others):
                    rng.shuffle(listed)
                    text = f"{Q_LOW} " + " ".join(names[x] for x in listed) + "?"
                    out.append((text, names[w]))
    return out


def try_episode(rng, size):
    m = rng.randint(3, 8)
    names = rng.sample(POOL, m)
    order = list(range(m))          # order[t]: entity at position t, top first
    rng.shuffle(order)
    pos = [0] * m
    for t, e in enumerate(order):
        pos[e] = t

    # Records, all consistent with the hidden order. Half of them compare
    # neighbours in the order, so longer chains appear.
    n_rec = rng.randint(3, 12) * size
    recs = []                       # (row text, x, y) with x above y
    for _ in range(n_rec):
        if rng.random() < 0.5:
            t = rng.randrange(m - 1)
            x, y = order[t], order[t + 1]
        else:
            x, y = rng.sample(range(m), 2)
            if pos[x] > pos[y]:
                x, y = y, x
        if rng.random() < 0.5:
            text = f"{REC_ABOVE} {names[x]} {names[y]}."
        else:
            text = f"{REC_BELOW} {names[y]} {names[x]}."
        recs.append((text, x, y))

    # Queries, each placed after the prefix of records that determines it.
    n_q = rng.randint(1, 5)
    placed = []                     # (cut, query text, answer); cut = records above it
    for _ in range(n_q):
        cuts = list(range(1, n_rec + 1))
        rng.shuffle(cuts)
        for cut in cuts:
            edges = [(x, y) for _, x, y in recs[:cut]]
            cands = candidates(m, names, edges, rng)
            if cands:
                text, ans = rng.choice(cands)
                placed.append((cut, text, ans))
                break
        else:
            return None             # no determined query here: draw another episode

    by_cut = {}
    for cut, text, ans in placed:
        by_cut.setdefault(cut, []).append(f"{text}\t{ans}")
    rows = []
    for i, (text, _, _) in enumerate(recs, start=1):
        rows.append(text)
        rows.extend(by_cut.get(i, []))
    return rows


def make_episode(rng, size):
    while True:
        rows = try_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for i, row in enumerate(make_episode(rng, size), start=1):
            lines.append(f"{i} {row}")
    if lines:
        print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
