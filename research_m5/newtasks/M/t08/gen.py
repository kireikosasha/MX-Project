#!/usr/bin/env python3
"""Synthetic record-log reasoning task: places on a map.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)

Every row is a record or a query. A walker writes a log:
  mv DIR k.     walk k steps in direction DIR (never crossing its own trail)
  mk PLACE.     mark the current position as PLACE
Direction codes: dn (north), ds (south), de (east), dw (west).

Queries (answer after one TAB):
  qd A B DIR?   yes/no: is A in direction DIR from B? (dn: A is above B, etc.)
  qr A B?       the marked places on the way from A to B, in trail order
"""
import random
import sys

DIRS = {"dn": (0, 1), "ds": (0, -1), "de": (1, 0), "dw": (-1, 0)}
DIR_CODES = ["dn", "ds", "de", "dw"]
PLACES = ["p%d" % i for i in range(1, 9)]
MAX_MARKS = 8


class Trapped(Exception):
    """The walker has no free cell to move into; the episode is redrawn."""


def gen_records(rng, size):
    total = rng.randint(3 * size, 12 * size)
    marks = rng.randint(2, min(MAX_MARKS, (total + 1) // 2))
    moves = total - marks
    # No two marks are adjacent in the log: each gap holds at most one mark.
    mark_gaps = set(rng.sample(range(moves + 1), marks))
    codes = rng.sample(PLACES, marks)
    kmax = size + 2

    recs = []
    x = y = 0
    visited = {(0, 0)}
    next_code = 0
    for gap in range(moves + 1):
        if gap in mark_gaps:
            recs.append(["mk", codes[next_code]])
            next_code += 1
        if gap < moves:
            options = []
            for d in DIR_CODES:
                dx, dy = DIRS[d]
                run = 0
                cx, cy = x, y
                while run < kmax and (cx + dx, cy + dy) not in visited:
                    cx += dx
                    cy += dy
                    run += 1
                if run:
                    options.append((d, run))
            if not options:
                raise Trapped()
            d, run = rng.choice(options)
            k = rng.randint(1, run)
            dx, dy = DIRS[d]
            for _ in range(k):
                x += dx
                y += dy
                visited.add((x, y))
            recs.append(["mv", d, str(k)])
    return recs


def simulate(recs):
    """Walk the records; return {place: (row index, (x, y))} for marked places."""
    x = y = 0
    where = {}
    for i, r in enumerate(recs):
        if r[0] == "mk":
            where[r[1]] = (i, (x, y))
        else:
            dx, dy = DIRS[r[1]]
            k = int(r[2])
            x += dx * k
            y += dy * k
    return where


def answer(prefix, q):
    where = simulate(prefix)
    if q[0] == "qd":
        _, a, b, d = q
        (_, (ax, ay)), (_, (bx, by)) = where[a], where[b]
        dx, dy = DIRS[d]
        return "yes" if (ax - bx) * dx + (ay - by) * dy > 0 else "no"
    _, a, b = q
    ia, ib = where[a][0], where[b][0]
    lo, hi = min(ia, ib), max(ia, ib)
    chain = sorted((i, p) for p, (i, _) in where.items() if lo <= i <= hi)
    names = [p for _, p in chain]
    if ia > ib:
        names.reverse()
    return ",".join(names)


def pick_queries(rng, recs):
    """Return (cut, tokens) pairs; cut = number of records above the query."""
    marks = [(i, r[1]) for i, r in enumerate(recs) if r[0] == "mk"]
    pairs = []
    for a in range(len(marks)):
        for b in range(a + 1, len(marks)):
            ia, ca = marks[a]
            ib, cb = marks[b]
            mvs = sum(1 for r in recs[ia + 1:ib] if r[0] == "mv")
            pairs.append((mvs, ia, ib, ca, cb))
    # Prefer pairs whose answer rests on at least two movement records.
    strong = [p for p in pairs if p[0] >= 2]
    pool = strong or pairs
    queries = []
    for _ in range(rng.randint(1, 5)):
        _, ia, ib, ca, cb = rng.choice(pool)
        if rng.random() < 0.5:
            ca, cb = cb, ca
        if rng.random() < 0.5:
            toks = ["qd", ca, cb, rng.choice(DIR_CODES)]
        else:
            toks = ["qr", ca, cb]
        cut = rng.randint(ib + 1, len(recs))
        queries.append((cut, toks))
    return queries


def gen_episode(rng, size):
    while True:
        try:
            recs = gen_records(rng, size)
            break
        except Trapped:
            continue
    queries = sorted(pick_queries(rng, recs), key=lambda q: q[0])
    lines = []
    qi = 0
    n = 0
    for idx in range(len(recs) + 1):
        while qi < len(queries) and queries[qi][0] == idx:
            toks = queries[qi][1]
            n += 1
            lines.append("%d %s?\t%s" % (n, " ".join(toks), answer(recs[:idx], toks)))
            qi += 1
        if idx < len(recs):
            n += 1
            lines.append("%d %s." % (n, " ".join(recs[idx])))
    return lines


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
    out = []
    for _ in range(n):
        while True:
            lines = gen_episode(rng, size)
            key = "\n".join(lines)
            if key not in seen:
                seen.add(key)
                break
        out.extend(lines)
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
