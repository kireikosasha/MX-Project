#!/usr/bin/env python3
"""Synthetic record-log task: identity merging with links and unlinks.

Each episode is a log of identity records:
  qo x.        introduce identity x (order of introduction matters)
  zep x y.     declare identities x and y to be linked (same person)
  dak x y.     remove the direct link between x and y
Queries are answered by simulating the log:
  pok x?       the canonical identity of x's group (earliest introduced)
  vor x y?     yes / no: are x and y in the same group
  lum x?       all identities of x's group, sorted, comma-separated
Usage: python gen.py N SEED [SIZE]
"""
import argparse
import random

IDS = ["a%d" % i for i in range(1, 9)]
REC_OPS = ["qo", "zep", "dak"]
QRY_OPS = ["pok", "lum", "vor"]


class World:
    def __init__(self):
        self.order = []     # identities in order of introduction
        self.links = set()  # frozensets of two identities

    def group(self, x):
        seen = {x}
        stack = [x]
        while stack:
            cur = stack.pop()
            for link in self.links:
                if cur in link:
                    (other,) = link - {cur}
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        return seen

    def canonical(self, group):
        return min(group, key=self.order.index)


def make_record(rng, w):
    free = [x for x in IDS if x not in w.order]
    pairs = [(a, b) for i, a in enumerate(w.order) for b in w.order[i + 1:]
             if frozenset((a, b)) not in w.links]
    options, weights = [], []
    if free:
        options.append("qo"); weights.append(3)
    if pairs:
        options.append("zep"); weights.append(4)
    if w.links:
        options.append("dak"); weights.append(2)
    op = rng.choices(options, weights=weights)[0]
    if op == "qo":
        x = rng.choice(free)
        w.order.append(x)
        return "qo %s." % x
    if op == "zep":
        a, b = rng.choice(pairs)
        w.links.add(frozenset((a, b)))
        return "zep %s %s." % (a, b)
    link = rng.choice(sorted(w.links, key=lambda l: tuple(sorted(l))))
    w.links.remove(link)
    a, b = sorted(link)
    return "dak %s %s." % (a, b)


def make_query(rng, w):
    cands = [x for x in w.order if len(w.group(x)) >= 2]
    if not cands:
        return None
    x = rng.choice(cands)
    g = w.group(x)
    op = rng.choice(QRY_OPS)
    if op == "pok":
        return "pok %s?\t%s" % (x, w.canonical(g))
    if op == "lum":
        return "lum %s?\t%s" % (x, ",".join(sorted(g)))
    want_yes = rng.random() < 0.5
    if want_yes:
        y = rng.choice(sorted(g - {x}))
        ans = "yes"
    else:
        outside = [z for z in w.order if z not in g]
        if outside:
            y = rng.choice(outside)
            ans = "no"
        else:
            y = rng.choice(sorted(g - {x}))
            ans = "yes"
    return "vor %s %s?\t%s" % (x, y, ans)


def make_episode(rng, size):
    while True:
        w = World()
        n_rec = rng.randint(3, 12) * size
        n_qry = rng.randint(1, 5)
        rows = []
        failed = False
        while n_rec or n_qry:
            if n_qry and (n_rec == 0 or rng.random() < n_qry / (n_rec + n_qry)):
                q = make_query(rng, w)
                if q is not None:
                    rows.append(q)
                    n_qry -= 1
                    continue
                if n_rec == 0:
                    failed = True
                    break
            rows.append(make_record(rng, w))
            n_rec -= 1
        if not failed:
            return rows


def main():
    ap = argparse.ArgumentParser(description="Generate record-log reasoning episodes.")
    ap.add_argument("n", type=int, help="number of episodes")
    ap.add_argument("seed", type=int, help="random seed")
    ap.add_argument("size", type=int, nargs="?", default=1, help="episode size (1, 3 or 10)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    for _ in range(args.n):
        while True:
            rows = make_episode(rng, args.size)
            key = "\n".join(rows)
            if key not in seen:
                break
        seen.add(key)
        for i, row in enumerate(rows, 1):
            print("%d %s" % (i, row))


if __name__ == "__main__":
    main()
