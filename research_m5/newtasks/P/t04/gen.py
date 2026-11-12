#!/usr/bin/env python3
"""Synthetic records task: kinds, sub-kinds and properties with inheritance.

Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)

Record rows (a = individual, k = kind, p = property):
  kex a k     individual a is of kind k
  ruv k1 k2   kind k1 is a sub-kind of kind k2 (each kind has at most one parent)
  zog k p     kind k has property p; so do all sub-kinds of k
  miq a p     individual a has property p directly
  lun k N     kind k has limit N

Query rows (answered from the records above only, closed world):
  wex a p?    does a have property p, directly or by inheritance?  -> yes / no
  pif a?      all properties of a, sorted, comma-separated
  yuk a?      largest limit among all kinds of a (its own kinds and their ancestors)
"""
import random
import sys

NENT = 8
REC_WEIGHT = {"kex": 3, "ruv": 2, "zog": 2, "miq": 1, "lun": 1}


def ind(i):
    return "a%d" % i


def kin(i):
    return "k%d" % i


def prp(i):
    return "p%d" % i


class World:
    def __init__(self):
        self.kinds_of = {ind(i): set() for i in range(NENT)}
        self.aprops = {ind(i): set() for i in range(NENT)}
        self.parent = {}
        self.kprops = {kin(i): set() for i in range(NENT)}
        self.limit = {}

    def ancestors(self, k):
        out = []
        while k in self.parent:
            k = self.parent[k]
            out.append(k)
        return out

    def height(self, k):
        kids = [c for c, p in self.parent.items() if p == k]
        return 0 if not kids else 1 + max(self.height(c) for c in kids)

    def closure(self, a):
        ks = set()
        for k in self.kinds_of[a]:
            ks.add(k)
            ks.update(self.ancestors(k))
        return ks

    def props(self, a):
        ps = set(self.aprops[a])
        for k in self.closure(a):
            ps |= self.kprops[k]
        return ps


def pick_record(w, rng, size):
    maxd = min(7, 1 + size)      # longer sub-kind chains at bigger sizes
    maxn = 10 * size - 1         # bigger numbers at bigger sizes
    cands = {t: [] for t in REC_WEIGHT}
    for i in range(NENT):
        a = ind(i)
        for j in range(NENT):
            k = kin(j)
            if k not in w.kinds_of[a]:
                cands["kex"].append((a, k))
            p = prp(j)
            if p not in w.aprops[a]:
                cands["miq"].append((a, p))
            if p not in w.kprops[k]:
                cands["zog"].append((k, p))
    for x in w.kprops:
        if x in w.parent:
            continue
        for y in w.kprops:
            if y == x or x in w.ancestors(y):
                continue
            if len(w.ancestors(y)) + 1 + w.height(x) <= maxd:
                cands["ruv"].append((x, y))
    for k in w.kprops:
        if k not in w.limit:
            cands["lun"].append((k, None))

    types = [t for t in REC_WEIGHT if cands[t]]
    if not types:
        return None
    t = rng.choices(types, weights=[REC_WEIGHT[u] for u in types])[0]
    x, y = rng.choice(cands[t])
    if t == "kex":
        w.kinds_of[x].add(y)
        return "kex %s %s." % (x, y)
    if t == "ruv":
        w.parent[x] = y
        return "ruv %s %s." % (x, y)
    if t == "zog":
        w.kprops[x].add(y)
        return "zog %s %s." % (x, y)
    if t == "miq":
        w.aprops[x].add(y)
        return "miq %s %s." % (x, y)
    n = rng.randint(1, maxn)
    w.limit[x] = n
    return "lun %s %d." % (x, n)


def pick_query(w, rng, asked):
    cands = {"wex_yes": [], "wex_no": [], "pif": [], "yuk": []}
    for i in range(NENT):
        a = ind(i)
        if not w.kinds_of[a]:
            continue
        props = w.props(a)
        for j in range(NENT):
            p = prp(j)
            if p not in w.aprops[a]:  # only inherited facts need reasoning
                if p in props:
                    cands["wex_yes"].append("wex %s %s?\tyes" % (a, p))
                else:
                    cands["wex_no"].append("wex %s %s?\tno" % (a, p))
        if any(p not in w.aprops[a] for p in props):
            cands["pif"].append("pif %s?\t%s" % (a, ",".join(sorted(props))))
        lims = [w.limit[k] for k in w.closure(a) if k in w.limit]
        if lims:
            cands["yuk"].append("yuk %s?\t%d" % (a, max(lims)))
    for t in cands:
        cands[t] = [c for c in cands[t] if c.split("\t")[0] not in asked]
    types = [t for t in cands if cands[t]]
    if not types:
        return None
    line = rng.choice(cands[rng.choice(types)])
    asked.add(line.split("\t")[0])
    return line


def make_episode(rng, size):
    w = World()
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    lines = []
    asked = set()
    nr = nq = 0
    while nr < n_rec or nq < n_q:
        rq, rr = n_q - nq, n_rec - nr
        if rq > 0 and (rr <= 0 or rng.random() < rq / (rq + rr)):
            q = pick_query(w, rng, asked)
            if q is not None:
                lines.append(q)
                nq += 1
                continue
        r = pick_record(w, rng, size)
        if r is None:
            return None
        lines.append(r)
        nr += 1
    return lines


def main(argv):
    n = int(argv[0]) if len(argv) > 0 else 1
    seed = int(argv[1]) if len(argv) > 1 else 0
    size = int(argv[2]) if len(argv) > 2 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        lines = None
        for _attempt in range(10000):
            cand = make_episode(rng, size)
            if cand is not None and tuple(cand) not in seen:
                lines = cand
                seen.add(tuple(cand))
                break
        if lines is None:
            sys.exit("could not generate a new episode")
        for i, line in enumerate(lines, 1):
            out.append("%d %s" % (i, line))
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
