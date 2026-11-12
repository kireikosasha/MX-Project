#!/usr/bin/env python3
"""Synthetic record-log task: changing states.

Usage: python gen.py N SEED [SIZE]

Entities: items k1..k8 (each has a current location box), boxes b1..b8
(each has a current status s1..s3). Records change states over time;
queries ask the present state of one entity or which entities are in a
given state. Answers come from a simulation run over the records above
each query.
"""
import random
import sys

ITEMS = ["k%d" % i for i in range(1, 9)]
BOXES = ["b%d" % i for i in range(1, 9)]
STATES = ["s1", "s2", "s3"]


class World:
    def __init__(self):
        self.loc = {}       # item -> box it is currently in
        self.status = {}    # box -> state it currently has
        self.mentions = {}  # entity code -> number of records naming it so far

    def mention(self, *codes):
        for c in codes:
            self.mentions[c] = self.mentions.get(c, 0) + 1

    def count(self, code):
        return self.mentions.get(code, 0)


def make_record(rng, w):
    """Return the tokens of one record and apply it to the world."""
    located = sorted(w.loc)
    r = rng.random()
    if r < 0.55 or (r >= 0.8 and not located):
        k = rng.choice(ITEMS)
        b = rng.choice([b for b in BOXES if w.loc.get(k) != b])
        w.loc[k] = b
        w.mention(k, b)
        return ["mv", k, b]
    if r < 0.8:
        b = rng.choice(BOXES)
        s = rng.choice([s for s in STATES if w.status.get(b) != s])
        w.status[b] = s
        w.mention(b, s)
        return ["fx", b, s]
    k = rng.choice(located)
    del w.loc[k]
    w.mention(k)
    return ["rm", k]


def candidate_queries(w):
    """All queries whose answer is defined and rests on several records.

    Each candidate is (query tokens, answer).
    """
    cands = []
    items_in = {}
    for k, b in sorted(w.loc.items()):
        items_in.setdefault(b, []).append(k)
        if w.count(k) >= 2:
            cands.append((["wh", k], b))
    boxes_with = {}
    for b, s in sorted(w.status.items()):
        boxes_with.setdefault(s, []).append(b)
        if w.count(b) >= 2:
            cands.append((["sb", b], s))
    for b, ks in sorted(items_in.items()):
        if w.count(b) >= 2:
            cands.append((["ls", b], ",".join(sorted(ks))))
    for s, bs in sorted(boxes_with.items()):
        if w.count(s) >= 2:
            cands.append((["ls", s], ",".join(sorted(bs))))
    return cands


def try_episode(rng, size):
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    total = n_rec + n_q
    qpos = set(rng.sample(range(2, total), n_q))  # at least two rows first
    w = World()
    rows = []
    asked = set()
    for i in range(total):
        if i in qpos:
            cands = [c for c in candidate_queries(w) if " ".join(c[0]) not in asked]
            if not cands:
                return None
            tokens, answer = rng.choice(cands)
            asked.add(" ".join(tokens))
            rows.append(" ".join(tokens) + "?" + "\t" + answer)
        else:
            rows.append(" ".join(make_record(rng, w)) + ".")
    return rows


def make_episode(rng, size):
    while True:
        rows = try_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    rng = random.Random(seed)
    for _ in range(n):
        for i, row in enumerate(make_episode(rng, size), 1):
            print(i, row)


if __name__ == "__main__":
    main(sys.argv)
