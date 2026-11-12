#!/usr/bin/env python3
"""Synthetic record-log task: comparison on one scale.

Usage: python gen.py N SEED [SIZE]

Each episode is a log of records saying which entity is above which on a
single scale, interleaved with queries. Queries are answered only from the
records above them, using the transitive closure of those records. A query is
only emitted when its answer is determined by the records above it and needs
at least two records (no single record states the answer).
"""
import random
import sys

POOL = ["k%d" % i for i in range(1, 9)]   # the only entity codes (kind k)
REC_ABOVE = "zor"   # zor a b.  -> a is above b
REC_BELOW = "qen"   # qen a b.  -> a is below b
Q_YESNO = "xo"      # xo a b?   -> yes / no (is a above b)
Q_TOP = "mx"        # mx a b c? -> the entity above all the others in the list
Q_ORDER = "ruk"     # ruk a b c? -> the listed entities from highest to lowest

# entities in the hidden order: (min, max) per SIZE
SIZE_ENTITIES = {1: (3, 5), 3: (4, 7), 10: (6, 8)}


def closure(pairs):
    """Transitive closure of a set of (above, below) pairs."""
    rel = set(pairs)
    while True:
        new = {(x, w) for (x, y) in rel for (z, w) in rel if y == z} - rel
        if not new:
            return rel
        rel |= new


def find_query(rng, direct, records, tries=60):
    """Return a query row whose answer follows from the records above, or None."""
    rel = closure(records)
    ments = sorted({e for pair in records for e in pair})
    if len(ments) < 2:
        return None
    for _ in range(tries):
        kind = rng.choice([Q_YESNO, Q_TOP, Q_ORDER])
        if kind == Q_YESNO:
            a, b = rng.sample(ments, 2)
            if (a, b) in direct or (b, a) in direct:
                continue  # a single record states it: a lookup, not reasoning
            if (a, b) in rel:
                ans = "yes"
            elif (b, a) in rel:
                ans = "no"
            else:
                continue  # not determined by the records above
            return "%s %s %s?\t%s" % (Q_YESNO, a, b, ans)

        k = rng.choice([3, 4])
        if len(ments) < k:
            continue
        group = rng.sample(ments, k)
        if kind == Q_TOP:
            tops = [w for w in group
                    if all((w, o) in rel for o in group if o != w)]
            if len(tops) != 1:
                continue
            w = tops[0]
            if all((w, o) in direct for o in group if o != w):
                continue  # every comparison is a single stated record
            return "%s %s?\t%s" % (Q_TOP, " ".join(group), w)

        # Q_ORDER: the group must be totally ordered by the records above
        if any(not ((x, y) in rel or (y, x) in rel)
               for x in group for y in group if x < y):
            continue
        if all((x, y) in direct or (y, x) in direct
               for x in group for y in group if x < y):
            continue  # every pair is a single stated record
        ordered = sorted(group, key=lambda w: -sum((w, o) in rel
                                                   for o in group if o != w))
        return "%s %s?\t%s" % (Q_ORDER, " ".join(group), ",".join(ordered))
    return None


def make_episode(rng, size):
    """One attempt at an episode; returns a list of row bodies or None."""
    lo, hi = SIZE_ENTITIES[size]
    m = rng.randint(lo, hi)
    ents = rng.sample(POOL, m)
    truth = ents[:]
    rng.shuffle(truth)                    # hidden order, truth[0] is highest

    # pairs the records may state (above, below): mostly neighbours in the
    # hidden order, so that other comparisons need chains of records
    stated = []
    for i in range(m - 1):
        if rng.random() < 0.85:
            stated.append((truth[i], truth[i + 1]))
    if not stated:
        stated.append((truth[0], truth[1]))
    for _ in range(rng.randint(0, 1 + size // 3)):
        i, j = sorted(rng.sample(range(m), 2))
        pair = (truth[i], truth[j])
        if pair not in stated:
            stated.append(pair)

    n_rec = rng.randint(3, 12) * size
    n_qry = rng.randint(1, 5)
    total = n_rec + n_qry
    slots = set(rng.sample(range(max(2, total // 4), total), n_qry))

    rows = []
    direct = set()
    records = []
    for idx in range(total):
        if idx in slots:
            q = find_query(rng, direct, records)
            if q is None:
                return None
            rows.append(q)
        else:
            above, below = rng.choice(stated)
            direct.add((above, below))
            records.append((above, below))
            if rng.random() < 0.5:
                rows.append("%s %s %s." % (REC_ABOVE, above, below))
            else:
                rows.append("%s %s %s." % (REC_BELOW, below, above))
    return rows


def episode(rng, size):
    for _ in range(1000):
        rows = make_episode(rng, size)
        if rows is not None:
            return rows
    raise RuntimeError("could not build an episode")


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZE_ENTITIES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    for _ in range(n):
        for i, row in enumerate(episode(rng, size), 1):
            print("%d %s" % (i, row))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
