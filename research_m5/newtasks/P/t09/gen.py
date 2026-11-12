#!/usr/bin/env python3
"""Synthetic record-chain reasoning task generator.

Each episode is a list of rows. Records set or drop the single upward link
of an entity (each entity has at most one boss). Queries ask about the chain
above an entity: two steps up, the top of the chain, or the chain depth.
Queries are only asked about entities whose chain has at least two links at
the moment of the query, so the answer needs several records.

Usage: python gen.py N SEED [SIZE]     SIZE in {1, 3, 10}, default 1
"""
import random
import sys

ENTS = ["u%d" % i for i in range(1, 9)]   # 8 entity codes, one kind
REC_OPS = ["hvo", "dro"]                  # hvo: set boss, dro: drop boss
QRY_OPS = ["twa", "xtp", "nkd"]           # twa: two steps up, xtp: top, nkd: depth
DROP_PROB = 0.2


def ancestors(boss, x):
    """Chain above x, nearest first. Finite because the links never form a cycle."""
    chain = []
    cur = boss.get(x)
    while cur is not None:
        chain.append(cur)
        cur = boss.get(cur)
    return chain


def make_record(rng, boss, p_root):
    if boss and rng.random() < DROP_PROB:
        x = rng.choice(sorted(boss))
        del boss[x]
        return "dro %s." % x
    roots = [e for e in ENTS if e not in boss]
    for _ in range(100):
        x = rng.choice(ENTS)
        if roots and rng.random() < p_root:
            y = rng.choice(roots)
        else:
            y = rng.choice(ENTS)
        # reject self links, cycles, and no-change links
        if y == x or x in ancestors(boss, y) or boss.get(x) == y:
            continue
        boss[x] = y
        return "hvo %s %s." % (x, y)
    if boss:
        x = rng.choice(sorted(boss))
        del boss[x]
        return "dro %s." % x
    return None


def try_episode(rng, size):
    boss = {}
    records = rng.randint(3 * size, 12 * size)
    queries = rng.randint(1, 5)
    p_root = min(0.9, 0.3 + 0.1 * size)   # larger sizes extend chains more often
    rows = []
    while records or queries:
        cands = [x for x in ENTS if len(ancestors(boss, x)) >= 2]
        can_q = bool(cands)
        if queries and (records == 0 or (can_q and rng.random() < queries / (queries + records))):
            if not can_q:
                return None
            x = rng.choice(cands)
            op = rng.choice(QRY_OPS)
            chain = ancestors(boss, x)
            if op == "twa":
                ans = chain[1]
            elif op == "xtp":
                ans = chain[-1]
            else:
                ans = str(len(chain))
            rows.append("%s %s?\t%s" % (op, x, ans))
            queries -= 1
        else:
            rec = make_record(rng, boss, p_root)
            if rec is None:
                return None
            rows.append(rec)
            records -= 1
    return rows


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 3):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        sys.exit(2)
    n = int(args[0])
    seed = int(args[1])
    size = int(args[2]) if len(args) == 3 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        sys.exit(2)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        rows = None
        while rows is None:
            rows = try_episode(rng, size)
        for i, row in enumerate(rows, 1):
            out.append("%d %s" % (i, row))
    print("\n".join(out))


if __name__ == "__main__":
    main()
