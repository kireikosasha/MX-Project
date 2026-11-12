#!/usr/bin/env python3
"""Synthetic group-membership task written as log-like records.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)

Entities: p1..p8 are persons, g1..g8 are groups. A person is in at most one
group at a time; a group may be empty.

Record operations:
  jn p g    person p (currently free) joins group g
  mv p g    person p (currently in some group) moves to a different group g
  lv p      person p (currently in a group) leaves that group
  mg s d    every member of group s moves into group d (s is non-empty, s != d)

Query operations (answers are computed from the simulated state):
  ql p?     which group is person p in now           -> one group code
  qm g?     who is in group g now                    -> sorted codes, comma-separated
  qn g?     how many people are in group g now       -> a number
"""
import random
import sys

PERSONS = ["p%d" % i for i in range(1, 9)]
GROUPS = ["g%d" % i for i in range(1, 9)]


class World:
    def __init__(self):
        self.group_of = {p: None for p in PERSONS}
        self.mentions = {}  # entity code -> number of records naming it

    def members(self, g):
        return [p for p in PERSONS if self.group_of[p] == g]


def random_record(rng, w):
    """Apply one random valid record to the world; return (text, entities named)."""
    free = [p for p in PERSONS if w.group_of[p] is None]
    inside = [p for p in PERSONS if w.group_of[p] is not None]
    busy = [g for g in GROUPS if w.members(g)]
    ops, weights = [], []
    if free:
        ops.append("jn")
        weights.append(3)
    if inside:
        ops.append("mv")
        weights.append(2)
        ops.append("lv")
        weights.append(2)
    if busy:
        ops.append("mg")
        weights.append(2)
    op = rng.choices(ops, weights)[0]

    if op == "jn":
        p = rng.choice(free)
        g = rng.choice(GROUPS)
        w.group_of[p] = g
        return "jn %s %s." % (p, g), [p, g]
    if op == "mv":
        p = rng.choice(inside)
        g = rng.choice([x for x in GROUPS if x != w.group_of[p]])
        w.group_of[p] = g
        return "mv %s %s." % (p, g), [p, g]
    if op == "lv":
        p = rng.choice(inside)
        w.group_of[p] = None
        return "lv %s." % p, [p]
    src = rng.choice(busy)
    dst = rng.choice([x for x in GROUPS if x != src])
    for p in w.members(src):
        w.group_of[p] = dst
    return "mg %s %s." % (src, dst), [src, dst]


def pick_query(rng, w, used):
    """Return (op, entity, answer) for a query not yet asked in this episode."""
    cands = []
    for p in PERSONS:
        if w.group_of[p] is not None:
            cands.append(("ql", p, w.group_of[p]))
    for g in GROUPS:
        mem = w.members(g)
        if mem:
            cands.append(("qm", g, ",".join(mem)))
        cands.append(("qn", g, str(len(mem))))
    cands = [c for c in cands if (c[0], c[1]) not in used]
    # Prefer entities named by at least two records with a non-empty answer,
    # so the answer needs several records and is not a trivial zero.
    strong = [c for c in cands if w.mentions.get(c[1], 0) >= 2 and c[2] != "0"]
    op, ent, ans = rng.choice(strong or cands)
    used.add((op, ent))
    return op, ent, ans


def make_episode(rng, size):
    n_rec = rng.randint(3 * size, 12 * size)
    n_qry = rng.randint(1, 5)
    total = n_rec + n_qry
    # Queries sit at random slots after at least two rows, so each has records above it.
    qslots = set(rng.sample(range(2, total), n_qry))
    w = World()
    used = set()
    rows = []
    for i in range(total):
        if i in qslots:
            op, ent, ans = pick_query(rng, w, used)
            rows.append("%s %s?\t%s" % (op, ent, ans))
        else:
            text, ents = random_record(rng, w)
            for e in ents:
                w.mentions[e] = w.mentions.get(e, 0) + 1
            rows.append(text)
    return rows


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    lines = []
    for _ in range(n):
        while True:
            rows = make_episode(rng, size)
            key = "\n".join(rows)
            if key not in seen:
                break
        seen.add(key)
        lines.extend("%d %s" % (i, r) for i, r in enumerate(rows, 1))
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv)
