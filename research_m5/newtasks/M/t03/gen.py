#!/usr/bin/env python3
"""Synthetic records task: comparisons on one scale.

Usage: python gen.py N SEED [SIZE]      (SIZE is 1, 3 or 10; default 1)

Each record says that one entity is above or below another on one scale.
Queries ask whether one entity is above another (yes/no), or which of a few
listed entities is the highest or the lowest (an entity code). Every answer
is decided by the records above the query, and each query needs a chain of
records (no single record states the answer).
"""
import random
import sys

ENTS = ["k%d" % i for i in range(1, 9)]  # one entity kind, 8 codes at most
OP_ABOVE = "tavo"  # record: first entity is above the second
OP_BELOW = "dosi"  # record: first entity is below the second
Q_ABOVE = "quv"    # query: is the first entity above the second? yes/no
Q_TOP = "zeb"      # query: which of the listed entities is the highest?
Q_LOW = "rhun"     # query: which of the listed entities is the lowest?
NENT = len(ENTS)


def add_fact(M, direct, hi, lo):
    """Add 'hi is above lo' to the transitive closure M (M[a][b]: a above b)."""
    direct.add((hi, lo))
    preds = [x for x in range(NENT) if x == hi or M[x][hi]]
    succs = [y for y in range(NENT) if y == lo or M[lo][y]]
    for x in preds:
        for y in succs:
            M[x][y] = True


def earliest(facts, kind, args):
    """Return (k, answer) for the first prefix of k records that decides the
    query and needs more than one record, or None.  facts are (hi, lo) index
    pairs in record order; args are entity indices."""
    M = [[False] * NENT for _ in range(NENT)]
    direct = set()
    for k, (hi, lo) in enumerate(facts, 1):
        add_fact(M, direct, hi, lo)
        if kind == "quv":
            a, b = args
            if M[a][b] and (a, b) not in direct:
                return k, "yes"
            if M[b][a] and (b, a) not in direct:
                return k, "no"
        else:
            for s in args:
                others = [t for t in args if t != s]
                if kind == "top" and all(M[s][t] for t in others):
                    if any((s, t) not in direct for t in others):
                        return k, ENTS[s]
                if kind == "low" and all(M[t][s] for t in others):
                    if any((t, s) not in direct for t in others):
                        return k, ENTS[s]
    return None


def pick_queries(rng, facts, q):
    """Choose q queries (kind, args, answer, position) decided by the records.
    Position p means the query is placed after p records (p >= decision point)."""
    ents = sorted({x for f in facts for x in f})
    if len(ents) < 2:
        return None
    chosen, used = [], set()
    for _ in range(400):
        if len(chosen) == q:
            break
        if len(ents) >= 3:
            kind = rng.choice(["quv", "quv", "top", "low"])
        else:
            kind = "quv"
        if kind == "quv":
            args = tuple(rng.sample(ents, 2))
        else:
            n = rng.choice([3, 4]) if len(ents) >= 4 else 3
            args = tuple(rng.sample(ents, n))
        key = (kind, frozenset(args))
        if key in used:
            continue
        res = earliest(facts, kind, args)
        if res is None:
            continue
        k, ans = res
        used.add(key)
        p = rng.randint(k, len(facts))
        chosen.append((p, kind, args, ans))
    if len(chosen) < q:
        return None
    return chosen


def query_row(kind, args, ans):
    op = {"quv": Q_ABOVE, "top": Q_TOP, "low": Q_LOW}[kind]
    names = " ".join(ENTS[x] for x in args)
    return "%s %s?\t%s" % (op, names, ans)


def make_episode(rng, size, seen):
    """Build one episode as a list of row texts (without numbers)."""
    while True:
        m = rng.randint(3, min(NENT, 4 + size // 2))
        pool = rng.sample(range(NENT), m)
        order = pool[:]
        rng.shuffle(order)
        rank = {e: i for i, e in enumerate(order)}  # hidden order, 0 = highest
        pairs = [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]]
        d = rng.randint(m - 1, len(pairs))
        used_pairs = rng.sample(pairs, d)
        n_rec = rng.randint(3, 12) * size

        facts, rec_rows = [], []
        for _ in range(n_rec):
            a, b = rng.choice(used_pairs)
            hi, lo = (a, b) if rank[a] < rank[b] else (b, a)
            facts.append((hi, lo))
            if rng.random() < 0.5:
                rec_rows.append("%s %s %s." % (OP_ABOVE, ENTS[hi], ENTS[lo]))
            else:
                rec_rows.append("%s %s %s." % (OP_BELOW, ENTS[lo], ENTS[hi]))

        q = rng.randint(1, 5)
        chosen = pick_queries(rng, facts, q)
        if chosen is None:
            continue

        # Check every answer against the hidden order.
        for _, kind, args, ans in chosen:
            if kind == "quv":
                a, b = args
                truth = "yes" if rank[a] < rank[b] else "no"
            elif kind == "top":
                truth = ENTS[min(args, key=lambda x: rank[x])]
            else:
                truth = ENTS[max(args, key=lambda x: rank[x])]
            assert ans == truth, (kind, args, ans, truth)

        placed = {}
        for p, kind, args, ans in chosen:
            placed.setdefault(p, []).append(query_row(kind, args, ans))
        rows = []
        for i in range(n_rec + 1):
            rows.extend(placed.get(i, []))
            if i < n_rec:
                rows.append(rec_rows[i])

        sig = tuple(rows)
        if sig in seen:
            continue
        seen.add(sig)
        return rows


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
    for _ in range(n):
        rows = make_episode(rng, size, seen)
        for i, text in enumerate(rows, 1):
            print("%d %s" % (i, text))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
