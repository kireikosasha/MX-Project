#!/usr/bin/env python3
"""Synthetic record-reasoning task: name aliases and tags.

Records:
  lnk a b.   names a and b become aliases (same class; symmetric, transitive)
  tg a t.    the alias class of a gets tag t (a class holds at most one tag)
Queries:
  xeq a b?   yes/no: are a and b in the same alias class
  xtg a?     the tag of the alias class of a (entity code of kind t)
  xsz a?     the number of names in the alias class of a

Usage: python gen.py N SEED [SIZE]     SIZE in {1, 3, 10}, default 1
"""
import random
import sys
from itertools import combinations

NAMES = ["n%d" % i for i in range(1, 9)]
TAGS = ["t%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)


class World:
    """Simulation state: name -> class label, class label -> tag."""

    def __init__(self):
        self.lab = {n: n for n in NAMES}
        self.tag = {}

    def same(self, a, b):
        return self.lab[a] == self.lab[b]

    def members(self, a):
        return [n for n in NAMES if self.lab[n] == self.lab[a]]

    def tag_of(self, a):
        return self.tag.get(self.lab[a])

    def link(self, a, b):
        la, lb = self.lab[a], self.lab[b]
        if la == lb:
            return
        for n in NAMES:
            if self.lab[n] == lb:
                self.lab[n] = la
        if lb in self.tag:
            self.tag[la] = self.tag.pop(lb)

    def set_tag(self, a, t):
        self.tag[self.lab[a]] = t


def link_ok(w, a, b):
    ta, tb = w.tag_of(a), w.tag_of(b)
    return not (ta and tb and ta != tb)


def make_record(w, rng):
    pairs = list(combinations(NAMES, 2))
    prod = [("lnk", a, b) for a, b in pairs
            if not w.same(a, b) and link_ok(w, a, b)]
    prod += [("tg", n, t) for n in NAMES if w.tag_of(n) is None for t in TAGS]
    noop = [("lnk", a, b) for a, b in pairs if w.same(a, b)]
    noop += [("tg", n, w.tag_of(n)) for n in NAMES if w.tag_of(n) is not None]
    if prod and (not noop or rng.random() < 0.75):
        kind, x, y = rng.choice(prod)
    else:
        kind, x, y = rng.choice(noop)
    if kind == "lnk":
        w.link(x, y)
        return "lnk %s %s." % (x, y)
    w.set_tag(x, y)
    return "tg %s %s." % (x, y)


def make_query(w, rng):
    tagged = [n for n in NAMES if w.tag_of(n) and len(w.members(n)) >= 2]
    sized = [n for n in NAMES if len(w.members(n)) >= 2]
    opts = ["xeq"]
    if tagged:
        opts.append("xtg")
    if sized:
        opts.append("xsz")
    op = rng.choice(opts)
    if op == "xtg":
        n = rng.choice(tagged)
        return "xtg %s?" % n, w.tag_of(n)
    if op == "xsz":
        n = rng.choice(sized)
        return "xsz %s?" % n, str(len(w.members(n)))
    same_pairs = [(a, b) for a, b in combinations(NAMES, 2) if w.same(a, b)]
    diff_pairs = [(a, b) for a, b in combinations(NAMES, 2) if not w.same(a, b)]
    if same_pairs and (not diff_pairs or rng.random() < 0.5):
        a, b = rng.choice(same_pairs)
        ans = "yes"
    else:
        a, b = rng.choice(diff_pairs)
        ans = "no"
    a, b = rng.sample((a, b), 2)
    return "xeq %s %s?" % (a, b), ans


def episode(rng, size):
    """Return the list of rows (text without the leading number) for one episode."""
    w = World()
    n_rec = rng.randint(3 * size, 12 * size)
    qs_left = rng.randint(1, 5)
    recs = 0
    rows = []
    while recs < n_rec or qs_left > 0:
        rem = (n_rec - recs) + qs_left
        want_query = qs_left > 0 and recs >= 2 and (
            recs >= n_rec or rng.random() < qs_left / rem)
        if want_query:
            q, ans = make_query(w, rng)
            rows.append(q + "\t" + ans)
            qs_left -= 1
        else:
            rows.append(make_record(w, rng))
            recs += 1
    return rows


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n_eps = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    lines = []
    for _ in range(n_eps):
        for i, row in enumerate(episode(rng, size), 1):
            lines.append("%d %s" % (i, row))
    sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
