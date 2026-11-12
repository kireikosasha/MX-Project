#!/usr/bin/env python3
"""Synthetic containment task given as records.

Parts (p) and wholes (a) are linked by "vop child whole." (put child into whole)
and undone by "dro child whole." (take it out again). Queries ask about
containment at any depth: "xin X Y?" (is X inside Y), "cnt Y?" (how many parts
are inside Y), "zul X?" (the top-level wholes that X is inside).
Answers come from simulating the records above each query.

Usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1).
"""
import random
import sys

PARTS = ["p%d" % i for i in range(1, 7)]
WHOLES = ["a%d" % i for i in range(1, 5)]
ALL = PARTS + WHOLES
REC_OPS = ["vop", "dro"]
QRY_OPS = ["xin", "cnt", "zul"]


class World:
    def __init__(self):
        self.up = {}    # child -> {whole: line that created the link}
        self.down = {}  # whole -> set of direct children

    def parents(self, x):
        return self.up.get(x, {})

    def add(self, c, p, line):
        self.up.setdefault(c, {})[p] = line
        self.down.setdefault(p, set()).add(c)

    def remove(self, c, p):
        del self.up[c][p]
        if not self.up[c]:
            del self.up[c]
        self.down[p].discard(c)
        if not self.down[p]:
            del self.down[p]

    def climb(self, x):
        """Everything x is inside, at any depth: (depth per node, creating lines)."""
        depth = {x: 0}
        lines = set()
        frontier = [x]
        while frontier:
            nxt = []
            for c in sorted(frontier):
                for p in sorted(self.up.get(c, {})):
                    lines.add(self.up[c][p])
                    if p not in depth:
                        depth[p] = depth[c] + 1
                        nxt.append(p)
            frontier = nxt
        return depth, lines

    def descend(self, y):
        """Everything inside y, at any depth: (set of nodes, creating lines)."""
        seen = {y}
        lines = set()
        frontier = [y]
        while frontier:
            nxt = []
            for p in sorted(frontier):
                for c in sorted(self.down.get(p, ())):
                    lines.add(self.up[c][p])
                    if c not in seen:
                        seen.add(c)
                        nxt.append(c)
            frontier = nxt
        seen.discard(y)
        return seen, lines


def valid_adds(w):
    out = []
    for p in WHOLES:
        inside_p = w.climb(p)[0]  # p and everything p is inside
        for c in ALL:
            if c == p or p in w.parents(c):
                continue
            if c in inside_p:  # c already contains p: would create a cycle
                continue
            out.append((c, p))
    return sorted(out)


def record(w, line, rng):
    adds = valid_adds(w)
    links = sorted((c, p) for c in w.up for p in w.up[c])
    if adds and (not links or rng.random() < 0.6):
        c, p = rng.choice(adds)
        w.add(c, p, line)
        return "vop %s %s." % (c, p)
    c, p = rng.choice(links)
    w.remove(c, p)
    return "dro %s %s." % (c, p)


def make_query(w, rng):
    """Return (question text, answer, creating lines) or None if not usable."""
    op = rng.choice(QRY_OPS)
    if op == "xin":
        x = rng.choice(ALL)
        y = rng.choice(WHOLES)
        if x == y:
            return None
        depth, lines = w.climb(x)
        if len(lines) < 2:
            return None
        if y in depth:
            if depth[y] < 2:  # direct link only: a lookup, not a chain
                return None
            ans = "yes"
        else:
            ans = "no"
        return "xin %s %s?" % (x, y), ans, lines
    if op == "cnt":
        y = rng.choice(WHOLES)
        below, lines = w.descend(y)
        if len(lines) < 2:
            return None
        n = sum(1 for z in below if z in PARTS)
        return "cnt %s?" % y, str(n), lines
    # zul: top-level wholes that x is inside
    x = rng.choice(ALL)
    depth, lines = w.climb(x)
    tops = sorted(z for z in depth if z != x and not w.parents(z))
    if not tops or len(lines) < 2:
        return None
    return "zul %s?" % x, ",".join(tops), lines


def attempt(rng, size):
    w = World()
    n_rec = rng.randint(3, 12) * size
    n_qry = rng.randint(1, 5)
    thresholds = sorted(rng.randint(2, n_rec) for _ in range(n_qry))
    rows = []
    recs = 0
    qi = 0
    while recs < n_rec or qi < n_qry:
        if qi < n_qry and recs >= thresholds[qi]:
            got = None
            for _ in range(40):
                got = make_query(w, rng)
                if got:
                    break
            if got:
                text, ans, _ = got
                rows.append(text + "\t" + ans)
                qi += 1
                continue
            if recs >= n_rec:
                return None
        rows.append(record(w, len(rows) + 1, rng))
        recs += 1
    return rows


def build_episode(rng, size, seen):
    while True:
        rows = attempt(rng, size)
        if rows is not None and tuple(rows) not in seen:
            seen.add(tuple(rows))
            return rows


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n_eps = int(args[0])
    seed = int(args[1])
    size = int(args[2]) if len(args) > 2 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n_eps):
        rows = build_episode(rng, size, seen)
        for i, row in enumerate(rows, 1):
            out.append("%d %s" % (i, row))
    print("\n".join(out))


if __name__ == "__main__":
    main()
