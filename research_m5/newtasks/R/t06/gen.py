#!/usr/bin/env python3
"""Synthetic ORDER-IN-TIME task written as records.

Items k1-k4 move through places p1-p5. The line order of the records is time.
Record ops:
  mv K P.       item K arrives at place P
  jn K K2.      item K arrives at the place where item K2 is at that moment
Query ops (answered only from the records above the query):
  bf K P?       place item K was in just before its first arrival at P
  nx K P?       place item K went to just after its first arrival at P
  fs P K1 K2?   which of the two items arrived at P first
Usage: python gen.py N SEED [SIZE]     SIZE in {1, 3, 10}, default 1
"""
import random
import sys
from itertools import combinations

ITEMS = ["k1", "k2", "k3", "k4"]
PLACES = ["p1", "p2", "p3", "p4", "p5"]


def history(records):
    """Return {item: [(place, line_index), ...]} of arrivals, in time order."""
    loc = {}
    hist = {k: [] for k in ITEMS}
    for idx, rec in enumerate(records):
        toks = rec[:-1].split()
        if toks[0] == "mv":
            item, place = toks[1], toks[2]
        else:  # jn
            item, place = toks[1], loc[toks[2]]
        loc[item] = place
        hist[item].append((place, idx))
    return hist


def answer(query, prefix):
    """Answer a query from the records in prefix (the records above it)."""
    hist = history(prefix)
    toks = query[:-1].split()
    op = toks[0]
    if op in ("bf", "nx"):
        k, p = toks[1], toks[2]
        i = [pl for pl, _ in hist[k]].index(p)  # first arrival at p
        j = i - 1 if op == "bf" else i + 1
        return hist[k][j][0]
    p, k1, k2 = toks[1], toks[2], toks[3]
    t1 = next(t for pl, t in hist[k1] if pl == p)
    t2 = next(t for pl, t in hist[k2] if pl == p)
    return k1 if t1 < t2 else k2


def gen_records(rng, n):
    loc = {}
    recs = []
    while len(recs) < n:
        placed = list(loc)
        pairs = [(a, b) for a in ITEMS for b in placed
                 if a != b and loc[b] != loc.get(a)]
        if pairs and rng.random() < 0.35:
            a, b = rng.choice(pairs)
            recs.append(f"jn {a} {b}.")
            loc[a] = loc[b]
        else:
            it = rng.choice(ITEMS)
            p = rng.choice([x for x in PLACES if x != loc.get(it)])
            recs.append(f"mv {it} {p}.")
            loc[it] = p
    return recs


def candidates(recs, rng):
    """All answerable queries as (query_text, line_index_needed)."""
    hist = history(recs)
    out = []
    for k in ITEMS:
        h = hist[k]
        for i, (p, t) in enumerate(h):
            if any(pp == p for pp, _ in h[:i]):
                continue  # only the first arrival at p counts
            if i >= 1:
                out.append((f"bf {k} {p}?", t))
            if i + 1 < len(h):
                out.append((f"nx {k} {p}?", h[i + 1][1]))
    for p in PLACES:
        firsts = {}
        for k in ITEMS:
            for pl, t in hist[k]:
                if pl == p:
                    firsts[k] = t
                    break
        for k1, k2 in combinations(sorted(firsts), 2):
            if rng.random() < 0.5:
                k1, k2 = k2, k1
            out.append((f"fs {p} {k1} {k2}?", max(firsts[k1], firsts[k2])))
    return out


def gen_episode(rng, size):
    while True:
        recs = gen_records(rng, rng.randint(3 * size, 12 * size))
        cands = candidates(recs, rng)
        if not cands:
            continue
        pending = rng.sample(cands, min(rng.randint(1, 5), len(cands)))
        rows = []
        for idx, rec in enumerate(recs):
            rows.append(rec)
            for q in list(pending):
                text, need = q
                if need == idx or (need < idx and rng.random() < 0.3):
                    pending.remove(q)
                    rows.append(f"{text}\t{answer(text, recs[:idx + 1])}")
        for text, _ in pending:
            rows.append(f"{text}\t{answer(text, recs)}")
        return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        while True:
            rows = gen_episode(rng, size)
            key = "\n".join(rows)
            if key not in seen:
                break
        seen.add(key)
        for i, row in enumerate(rows, 1):
            out.append(f"{i} {row}")
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv)
