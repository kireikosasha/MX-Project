#!/usr/bin/env python3
"""Synthetic records task: comparison on one scale (standard library only).

Each episode is a short log of rows. Records state that one entity is above
or below another on a single scale. Queries ask whether one entity is above
another (hy), or which entity of a listed group is the highest (tp) or the
lowest (bt). Every answer follows from the records above the query, and
most answers need a chain of two or more records.

Usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1)
"""
import random
import sys

RECORD_OPS = ("ro", "xu")        # "ro a b." a is above b ; "xu a b." a is below b
QUERY_OPS = ("hy", "tp", "bt")   # "hy a b?" is a above b ; "tp ..?" highest ; "bt ..?" lowest
ENTITY_POOL = ["k%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)
ENTITIES_PER_EPISODE = {1: (3, 5), 3: (4, 7), 10: (6, 8)}
REMOVE_PROB = 0.5
MAX_ATTEMPTS = 100000


def closure(facts):
    """Transitive closure of (above, below) pairs."""
    c = set(facts)
    for k in ENTITY_POOL:
        for i in ENTITY_POOL:
            if (i, k) in c:
                for j in ENTITY_POOL:
                    if (k, j) in c:
                        c.add((i, j))
    return c


def find_query(rng, ents, direct):
    """Return a query row whose answer follows from the records in `direct`, or None."""
    known = closure(direct)
    for _ in range(300):
        op = rng.choice(QUERY_OPS)
        if op == "hy":
            a, b = rng.sample(ents, 2)
            if (a, b) in direct or (b, a) in direct:
                continue  # answerable from a single record
            if (a, b) in known:
                ans = "yes"
            elif (b, a) in known:
                ans = "no"
            else:
                continue  # not determined by the records so far
            return "hy %s %s?\t%s" % (a, b, ans)
        group = rng.sample(ents, rng.randint(2, min(4, len(ents))))
        found = None
        for m in group:
            pairs = [(m, s) if op == "tp" else (s, m) for s in group if s != m]
            if all(p in known for p in pairs) and any(p not in direct for p in pairs):
                found = m
                break
        if found is None:
            continue
        return "%s %s?\t%s" % (op, " ".join(sorted(group)), found)
    return None


def make_episode(rng, size):
    """Return the rows of one episode, numbering not yet added."""
    lo, hi = ENTITIES_PER_EPISODE[size]
    for _ in range(MAX_ATTEMPTS):
        k = rng.randint(lo, hi)
        ents = rng.sample(ENTITY_POOL, k)
        order = ents[:]
        rng.shuffle(order)  # order[0] is the highest
        facts = [(order[i], order[j]) for i in range(k) for j in range(i + 1, k)]

        # Drop facts that the remaining ones still imply, so they need a chain.
        keep = facts[:]
        rng.shuffle(keep)
        for f in list(keep):
            if rng.random() < REMOVE_PROB:
                trial = [g for g in keep if g != f]
                if f in closure(trial):
                    keep = trial

        n = size * rng.randint(3, 12)
        if n <= len(keep):
            chosen = rng.sample(keep, n)
        else:
            chosen = keep + [rng.choice(keep) for _ in range(n - len(keep))]
        rng.shuffle(chosen)

        records = []
        for a, b in chosen:  # a is above b
            if rng.random() < 0.5:
                records.append(((a, b), "ro %s %s." % (a, b)))
            else:
                records.append(((a, b), "xu %s %s." % (b, a)))

        qn = rng.randint(1, 5)
        total = len(records) + qn
        slots = set(rng.sample(range(total), qn))
        rows, direct, it = [], set(), iter(records)
        ok = True
        for p in range(total):
            if p in slots:
                q = find_query(rng, ents, direct)
                if q is None:
                    ok = False
                    break
                rows.append(q)
            else:
                fact, text = next(it)
                direct.add(fact)
                rows.append(text)
        if ok:
            return rows
    raise RuntimeError("could not build an episode")


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n_episodes, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    for _ in range(n_episodes):
        for i, row in enumerate(make_episode(rng, size), 1):
            print("%d %s" % (i, row))


if __name__ == "__main__":
    main(sys.argv)
