#!/usr/bin/env python3
"""Synthetic record-ledger task: identity classes built by merges, plus values.

Records (operation codes mrg, vxl):
    mrg a b.   merge the classes of two entities of the same kind
    vxl e v.   set the value of entity e (a later vxl for e overrides it)

Queries (operation codes qrp, qsm, qtt):
    qrp e?     representative (smallest code) of the class of e
    qsm a b?   yes/no: are a and b in the same class
    qtt e?     sum of current values over the whole class of e

Usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1).
"""
import random
import sys

RECORD_OPS = ("mrg", "vxl")
QUERY_OPS = ("qrp", "qsm", "qtt")
KINDS = ("p", "u")
POOL = {k: [k + str(d) for d in range(1, 9)] for k in KINDS}
ALL_ENTITIES = POOL["p"] + POOL["u"]
SIZES = (1, 3, 10)


def find(parent, x):
    while parent.get(x, x) != x:
        x = parent[x]
    return x


def members(parent, x):
    root = find(parent, x)
    return [e for e in ALL_ENTITIES if find(parent, e) == root]


def attempt(rng, size):
    """Build one episode; return its rows, or None if no query needs 2+ records."""
    nrec = rng.randint(3 * size, 12 * size)
    nq = rng.randint(1, 5)
    total = nrec + nq
    qslots = set(rng.sample(range(1, total), nq))  # row 0 is always a record

    parent = {}
    vals = {}
    mentioned = set()
    recs = []      # entity tuples of earlier records, in order
    lines = []
    strong = False  # some query's answer rests on at least two records

    for i in range(total):
        n = i + 1
        if i not in qslots:
            if rng.random() < 0.6:
                kind = rng.choice(KINDS)
                a, b = rng.sample(POOL[kind], 2)
                ra, rb = find(parent, a), find(parent, b)
                if ra != rb:
                    parent[ra] = rb
                ents = (a, b)
                text = "mrg %s %s." % (a, b)
            else:
                e = rng.choice(ALL_ENTITIES)
                v = rng.randint(1, 9 * size)
                vals[e] = v
                ents = (e,)
                text = "vxl %s %d." % (e, v)
            recs.append(ents)
            mentioned.update(ents)
            lines.append("%d %s" % (n, text))
            continue

        pool = sorted(mentioned)
        qt = rng.choice(QUERY_OPS)
        same_kind = [(a, b) for a in pool for b in pool
                     if a < b and a[0] == b[0]]
        if qt == "qsm" and not same_kind:
            qt = "qrp"
        if qt == "qsm":
            a, b = rng.choice(same_kind)
            focus = (a, b)
            same = find(parent, a) == find(parent, b)
            answer = "yes" if same else "no"
            text = "qsm %s %s?" % (a, b)
        else:
            e = rng.choice(pool)
            focus = (e,)
            if qt == "qrp":
                answer = min(members(parent, e))
                text = "qrp %s?" % e
            else:
                answer = str(sum(vals.get(m, 0) for m in members(parent, e)))
                text = "qtt %s?" % e
        scope = set()
        for f in focus:
            scope.update(members(parent, f))
        support = sum(1 for ents in recs if all(x in scope for x in ents))
        if support >= 2:
            strong = True
        lines.append("%d %s\t%s" % (n, text, answer))

    return lines if strong else None


def gen_episode(rng, size):
    while True:
        rows = attempt(rng, size)
        if rows is not None:
            return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    try:
        count = int(argv[1])
        seed = int(argv[2])
        size = int(argv[3]) if len(argv) == 4 else 1
    except ValueError:
        sys.stderr.write("N, SEED and SIZE must be integers\n")
        return 2
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2

    rng = random.Random(seed)
    seen = set()
    for _ in range(count):
        while True:
            rows = gen_episode(rng, size)
            key = "\n".join(rows)
            if key not in seen:
                break
        seen.add(key)
        for row in rows:
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
