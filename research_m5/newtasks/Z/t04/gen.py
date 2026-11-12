#!/usr/bin/env python3
"""Synthetic record-only reasoning task: messages, their authors, replies and reactions.

Records (operation codes zef, kov, ruk):
    zef p m.    person p wrote message m
    kov m n.    message m is a reply to message n
    ruk p m.    person p reacted to message m

Queries (operation codes tov, cnt, lik), all asked after the last record:
    tov m?      the person who wrote the message that m replies to
    cnt m?      how many reactions message m has (only asked when it is at least 2)
    lik p q?    yes if p reacted to some message written by q, else no

Usage: python gen.py N SEED SIZE      (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

SIZES = {
    # msgs: message count, pers: person count, react: reaction count,
    # total: allowed number of records in an episode
    1: {"msgs": (2, 4), "pers": (2, 4), "react": (0, 6), "total": (3, 12)},
    3: {"msgs": (4, 8), "pers": (3, 8), "react": (4, 25), "total": (9, 36)},
    10: {"msgs": (6, 8), "pers": (6, 8), "react": (20, 60), "total": (30, 79)},
}
REPLY_PROB = 0.6
MAX_TRIES = 100000


def build_records(rng, cfg):
    """Random records for one episode. Returns (records, message codes, person codes)."""
    m_count = rng.randint(*cfg["msgs"])
    p_count = rng.randint(*cfg["pers"])
    msg_ids = ["m%d" % d for d in rng.sample(range(1, 9), m_count)]
    per_ids = ["p%d" % d for d in rng.sample(range(1, 9), p_count)]

    records = []
    for m in msg_ids:
        records.append(("zef", rng.choice(per_ids), m))  # one author per message

    for i in range(1, m_count):  # a reply always points to an earlier message: no cycles
        if rng.random() < REPLY_PROB:
            records.append(("kov", msg_ids[i], msg_ids[rng.randrange(i)]))

    pairs = [(p, m) for p in per_ids for m in msg_ids]
    r_count = min(rng.randint(*cfg["react"]), len(pairs))
    for p, m in rng.sample(pairs, r_count):  # each person reacts to a message at most once
        records.append(("ruk", p, m))

    rng.shuffle(records)
    return records, msg_ids, per_ids


def simulate(records):
    """Closed-world state built from all records of the episode."""
    author, parent, reacts = {}, {}, set()
    for op, x, y in records:
        if op == "zef":
            author[y] = x
        elif op == "kov":
            parent[x] = y
        else:
            reacts.add((x, y))
    return author, parent, reacts


def candidate_queries(records, msg_ids, per_ids):
    """All queries with a defined, non-trivial answer. Each is (text, answer)."""
    author, parent, reacts = simulate(records)
    out = []
    for m in msg_ids:
        if m in parent:
            out.append(("tov %s?" % m, author[parent[m]]))
        n = sum(1 for (_, mm) in reacts if mm == m)
        if n >= 2:
            out.append(("cnt %s?" % m, str(n)))
    written_by = set(author.values())
    for p in per_ids:
        for q in per_ids:
            if q not in written_by:
                continue
            hit = any(author.get(m) == q and (p, m) in reacts for m in msg_ids)
            out.append(("lik %s %s?" % (p, q), "yes" if hit else "no"))
    return out


def record_text(rec):
    op, x, y = rec
    return "%s %s %s." % (op, x, y)


def make_episode(rng, size, seen):
    """Return the rows of one new episode (records, then queries), as text without numbers."""
    cfg = SIZES[size]
    for _ in range(MAX_TRIES):
        records, msg_ids, per_ids = build_records(rng, cfg)
        if not cfg["total"][0] <= len(records) <= cfg["total"][1]:
            continue
        cands = candidate_queries(records, msg_ids, per_ids)
        n_queries = rng.randint(1, 5)
        if len(cands) < n_queries:
            continue
        rng.shuffle(cands)
        rows = [record_text(r) for r in records]
        rows += ["%s\t%s" % (text, ans) for text, ans in cands[:n_queries]]
        key = tuple(rows)
        if key in seen:
            continue
        seen.add(key)
        return rows
    raise RuntimeError("could not build an episode for SIZE %d" % size)


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in SIZES:
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        for i, row in enumerate(make_episode(rng, size, seen), 1):
            print("%d %s" % (i, row))


if __name__ == "__main__":
    main(sys.argv)
