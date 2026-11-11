#!/usr/bin/env python3
"""Synthetic record-log reasoning task: chains of one relation.

Records give one relation between pairs of entities:
  zor X Y.  Y becomes the direct superior of X (X's single upward link).
  kel X.    X loses its superior.
Every entity has at most one superior, so the links form chains.

Queries follow the relation two or more steps and are answered by simulation
from the records above them:
  ruv X?    the top of the chain of X (the entity with no superior)
  gox X?    the entity two steps above X
  vul Y X?  yes / no: is Y above X (Y reached from X through the chain)

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10, default 1)
"""
import random
import sys

POOL = ["p%d" % i for i in range(1, 9)]  # 8 entity codes of one kind
MAX_DEPTH = {1: 3, 3: 5, 10: 7}          # longest allowed chain per size


def ancestors(parent, x):
    """Entities above x, nearest first, up to the top of its chain."""
    chain = []
    cur = parent[x]
    while cur is not None:
        chain.append(cur)
        cur = parent[cur]
    return chain


def max_chain(parent):
    return max(len(ancestors(parent, e)) for e in POOL)


def can_link(parent, x, y, max_depth):
    """True if making y the superior of x creates no cycle and keeps chains short."""
    if x == y:
        return False
    cur = y
    while cur is not None:
        if cur == x:
            return False
        cur = parent[cur]
    trial = dict(parent)
    trial[x] = y
    return max_chain(trial) <= max_depth


def query_options(parent, mentioned):
    """All answerable queries (grouped by op) that need two or more links."""
    names = sorted(mentioned)
    opts = {"ruv": [], "gox": [], "vul": []}
    for x in names:
        chain = ancestors(parent, x)
        if len(chain) >= 2:
            opts["ruv"].append(([x], chain[-1]))
            opts["gox"].append(([x], chain[1]))
    for x in names:
        chain = ancestors(parent, x)
        for y in names:
            if y == x:
                continue
            if y not in chain:
                opts["vul"].append(([y, x], "no"))
            elif chain.index(y) >= 1:  # direct superior (index 0) is a one-record lookup
                opts["vul"].append(([y, x], "yes"))
    return {k: v for k, v in opts.items() if v}


def make_record(rng, parent, max_depth):
    linked = [e for e in POOL if parent[e] is not None]
    if linked and rng.random() < 0.2:
        x = rng.choice(linked)
        parent[x] = None
        return "kel", [x]
    for _ in range(300):
        x = rng.choice(POOL)
        y = rng.choice(POOL)
        if parent[x] != y and can_link(parent, x, y, max_depth):
            parent[x] = y
            return "zor", [x, y]
    return None


def try_episode(rng, size):
    max_depth = MAX_DEPTH[size]
    records = rng.randint(3, 12) * size
    queries = rng.randint(1, 5)
    parent = {e: None for e in POOL}
    mentioned = set()
    rows = []
    while records or queries:
        if queries and mentioned and (not records or rng.random() < 0.3):
            opts = query_options(parent, mentioned)
            if opts:
                op = rng.choice(sorted(opts))
                args, answer = rng.choice(opts[op])
                rows.append("%d %s %s?\t%s" % (len(rows) + 1, op, " ".join(args), answer))
                queries -= 1
                continue
        if not records:
            return None
        made = make_record(rng, parent, max_depth)
        if made is None:
            return None
        op, args = made
        mentioned.update(args)
        rows.append("%d %s %s." % (len(rows) + 1, op, " ".join(args)))
        records -= 1
    return "\n".join(rows) + "\n"


def make_episode(rng, size, seen):
    while True:
        text = try_episode(rng, size)
        if text is not None and text not in seen:
            seen.add(text)
            return text


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    count = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in MAX_DEPTH:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    chunks = [make_episode(rng, size, seen) for _ in range(count)]
    sys.stdout.write("".join(chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
