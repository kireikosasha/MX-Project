#!/usr/bin/env python3
"""Synthetic records task: comparison on a scale.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)

A hidden ranking of entities h1..h8 is drawn, and each record states a true
fact about it:
  tox a b.   a is above b
  vuz a b.   a is below b
  zeq a n.   a sits at level n (higher level = higher on the scale)
A query is answered only from the rows above it, using the transitive closure
of those facts:
  qis a b?<TAB>yes|no     is a above b
  rav a b c?<TAB>x        the highest of the listed entities
  vok a b c?<TAB>x        the lowest of the listed entities
A query is only asked when its answer follows from the records above it and
needs at least two of them (no single record states it).
"""
import itertools
import random
import sys

POOL = 8
KIND = "h"
ABOVE, BELOW, LEVEL = "tox", "vuz", "zeq"
Q_CMP, Q_TOP, Q_BOTTOM = "qis", "rav", "vok"


def closure(facts, ents):
    """Return (index map, transitive 'above' matrix) implied by the facts."""
    idx = {e: i for i, e in enumerate(ents)}
    n = len(ents)
    above = [[False] * n for _ in range(n)]
    levels = {}
    for f in facts:
        if f[0] == "gt":
            above[idx[f[1]]][idx[f[2]]] = True
        else:
            levels[idx[f[1]]] = f[2]
    for i in levels:
        for j in levels:
            if levels[i] > levels[j]:
                above[i][j] = True
    for k in range(n):
        for i in range(n):
            if above[i][k]:
                for j in range(n):
                    if above[k][j]:
                        above[i][j] = True
    return idx, above


def answer(facts, ents, query):
    """Answer a query from the facts, or None if the facts do not settle it."""
    kind, items = query
    idx, above = closure(facts, ents)
    pos = [idx[e] for e in items]
    if kind == Q_CMP:
        x, y = pos
        if above[x][y]:
            return "yes"
        if above[y][x]:
            return "no"
        return None
    if kind == Q_TOP:
        for s in pos:
            if all(above[s][t] for t in pos if t != s):
                return ents[s]
        return None
    for s in pos:
        if all(above[t][s] for t in pos if t != s):
            return ents[s]
    return None


def minimal_support_size(facts, ents, query, ans):
    """Size of an inclusion-minimal subset of facts that still gives ans."""
    keep = list(facts)
    i = 0
    while i < len(keep):
        trial = keep[:i] + keep[i + 1:]
        if answer(trial, ents, query) == ans:
            keep = trial
        else:
            i += 1
    return len(keep)


def find_query(rng, ents, facts, used):
    """Pick a settled query that needs at least two records. Returns (text, ans)."""
    cands = []
    for x, y in itertools.combinations(ents, 2):
        cands.append((Q_CMP, [x, y]))
    for k in (3, 4):
        for combo in itertools.combinations(ents, k):
            cands.append((Q_TOP, list(combo)))
            cands.append((Q_BOTTOM, list(combo)))
    rng.shuffle(cands)
    for kind, items in cands:
        items = list(items)
        rng.shuffle(items)
        key = kind + " " + " ".join(sorted(items))
        if key in used:
            continue
        q = (kind, items)
        ans = answer(facts, ents, q)
        if ans is None:
            continue
        if kind == Q_CMP and any(f[0] == "gt" and set(f[1:]) == set(items) for f in facts):
            continue  # a single record already states this pair: a lookup, not reasoning
        if minimal_support_size(facts, ents, q, ans) < 2:
            continue
        used.add(key)
        return f"{kind} {' '.join(items)}?", ans
    return None


def new_record(rng, ents, rank, level):
    """Draw one true record. Returns (row text, semantic fact)."""
    if rng.random() < 0.3:
        e = rng.choice(ents)
        return f"{LEVEL} {e} {level[e]}.", ("lvl", e, level[e])
    a, b = rng.sample(ents, 2)
    if rank[a] > rank[b]:
        a, b = b, a  # now a is above b
    if rng.random() < 0.5:
        return f"{ABOVE} {a} {b}.", ("gt", a, b)
    return f"{BELOW} {b} {a}.", ("gt", a, b)


def make_episode(rng, size):
    """One attempt at an episode. Returns a list of (row text, answer) or None."""
    m = rng.randint(3, POOL)
    codes = sorted(rng.sample(range(1, POOL + 1), m))
    ents = [f"{KIND}{c}" for c in codes]
    perm = ents[:]
    rng.shuffle(perm)  # hidden ranking, perm[0] is the top
    rank = {e: i for i, e in enumerate(perm)}
    vals = sorted(rng.sample(range(1, 30 * size + 1), m), reverse=True)
    level = {perm[i]: vals[i] for i in range(m)}

    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    facts, rows, used = [], [], set()
    left = n_q
    while True:
        k = len(facts)
        if left and k >= 2 and (k == n_rec or rng.random() < 0.4):
            found = find_query(rng, ents, facts, used)
            if found:
                rows.append(found)
                left -= 1
                continue
            if k == n_rec:
                return None
        if k < n_rec:
            text, fact = new_record(rng, ents, rank, level)
            facts.append(fact)
            rows.append((text, None))
            continue
        return rows if left == 0 else None


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    done = 0
    while done < n:
        rows = None
        while rows is None:
            rows = make_episode(rng, size)
        key = tuple(rows)
        if key in seen:
            continue
        seen.add(key)
        for i, (text, ans) in enumerate(rows, 1):
            line = f"{i} {text}"
            if ans is not None:
                line += "\t" + ans
            print(line)
        done += 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
