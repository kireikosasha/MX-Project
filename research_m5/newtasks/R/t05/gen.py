#!/usr/bin/env python3
"""Synthetic record-log task: changing states.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)

Entities: items i1..i8 move between rooms r1..r4 and carry a status g1..g3.
Records (operation codes zeb, kuv, mir) change the states:
    zeb i r.   item i moves to room r
    kuv i g.   item i gets status g
    mir r g.   every item in room r at that moment gets status g
Queries (operation codes ro, ks, lu) ask the present state:
    ro i?      room of item i
    ks i?      status of item i
    lu r?      sorted items in room r (or lu g? sorted items with status g)
Answers come from a simulation of the records above each query.
"""
import random
import sys

ITEMS = ["i%d" % k for k in range(1, 9)]
ROOMS = ["r1", "r2", "r3", "r4"]
STATUS = ["g1", "g2", "g3"]


class World:
    def __init__(self):
        self.loc = {}
        self.st = {}
        self.touch = {i: 0 for i in ITEMS}

    def in_room(self, r):
        return [i for i in ITEMS if self.loc.get(i) == r]

    def with_status(self, g):
        return [i for i in ITEMS if self.st.get(i) == g]

    def apply(self, rec):
        op = rec[0]
        if op == "zeb":
            _, i, r = rec
            self.loc[i] = r
            self.touch[i] += 1
        elif op == "kuv":
            _, i, g = rec
            self.st[i] = g
            self.touch[i] += 1
        else:
            _, r, g = rec
            for i in self.in_room(r):
                self.st[i] = g
                self.touch[i] += 1


def record_candidates(w):
    out = []
    for i in ITEMS:
        for r in ROOMS:
            if w.loc.get(i) != r:
                out.append(("zeb", i, r))
        for g in STATUS:
            if w.st.get(i) != g:
                out.append(("kuv", i, g))
    for r in ROOMS:
        for g in STATUS:
            if any(w.st.get(i) != g for i in w.in_room(r)):
                out.append(("mir", r, g))
    return out


def query_candidates(w):
    """Return (op, arg, answer) for every question whose present answer is defined
    and rests on at least two records."""
    out = []
    for i in ITEMS:
        if w.touch[i] >= 2:
            if i in w.loc:
                out.append(("ro", i, w.loc[i]))
            if i in w.st:
                out.append(("ks", i, w.st[i]))
    for r in ROOMS:
        items = w.in_room(r)
        if len(items) >= 2:
            out.append(("lu", r, ",".join(items)))
    for g in STATUS:
        items = w.with_status(g)
        if len(items) >= 2:
            out.append(("lu", g, ",".join(items)))
    return out


def make_episode(rng, size):
    recs_left = rng.randint(3, 12) * size
    qs_left = rng.randint(1, 5)
    w = World()
    rows = []
    asked = set()
    while recs_left or qs_left:
        if qs_left and rng.random() * (recs_left + qs_left) < qs_left:
            cands = [c for c in query_candidates(w) if c[:2] not in asked]
            if cands:
                op, arg, ans = rng.choice(cands)
                asked.add((op, arg))
                rows.append("%s %s?\t%s" % (op, arg, ans))
                qs_left -= 1
                continue
        if not recs_left:
            return None
        cands = record_candidates(w)
        if not cands:
            return None
        rec = rng.choice(cands)
        w.apply(rec)
        rows.append(" ".join(rec) + ".")
        recs_left -= 1
    return rows


def episode_text(rows):
    return "\n".join("%d %s" % (k, row) for k, row in enumerate(rows, 1))


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            rows = None
            while rows is None:
                rows = make_episode(rng, size)
            text = episode_text(rows)
            if text not in seen:
                break
        seen.add(text)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
