#!/usr/bin/env python3
"""Synthetic record-log reasoning task: kinds, sub-kinds and properties.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)

Record rows (facts that are stated):
  qa i k.     individual i is of kind k
  zo k1 k2.   kind k1 is a sub-kind of kind k2
  hv k p.     kind k has property p

Query rows (answered only from the rows above them in the same episode):
  ym A B?     yes/no: is A in B by inheritance (A an individual: B is one of
              its kinds, directly or through sub-kinds; A a kind: B is a
              proper ancestor of A)
  gp X P?     yes/no: does X (individual or kind) have property P, inherited
              from its kinds and their ancestors
  vx i?       the sorted, comma-separated list of all kinds of individual i
"""
import random
import sys

GAP = {1: 3, 3: 5, 10: 7}   # largest index distance allowed in a sub-kind link
N_ENT = 8                   # entity codes per kind: i1..i8, k1..k8, p1..p8


class State:
    """Facts stated so far in one episode; answers are derived from these."""

    def __init__(self):
        self.direct_kind = {}   # individual -> kinds stated by qa
        self.parents = {}       # kind -> parent kinds stated by zo
        self.kind_prop = {}     # kind -> properties stated by hv
        self.kinds_seen = set()

    def add(self, rec):
        op, a, b = rec
        if op == "qa":
            self.direct_kind.setdefault(a, set()).add(b)
            self.kinds_seen.add(b)
        elif op == "zo":
            self.parents.setdefault(a, set()).add(b)
            self.kinds_seen.update((a, b))
        else:
            self.kind_prop.setdefault(a, set()).add(b)
            self.kinds_seen.add(a)

    def ancestors(self, k):
        seen = set()
        stack = list(self.parents.get(k, ()))
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(self.parents.get(x, ()))
        return seen

    def kinds_of(self, ind):
        out = set()
        for k in self.direct_kind.get(ind, ()):
            out.add(k)
            out |= self.ancestors(k)
        return out

    def has_prop(self, x, p):
        if x[0] == "i":
            ks = self.kinds_of(x)
        else:
            ks = {x} | self.ancestors(x)
        return any(p in self.kind_prop.get(k, ()) for k in ks)


def rand_record(rng, gap):
    r = rng.random()
    if r < 0.4:
        return ("qa", "i%d" % rng.randint(1, N_ENT), "k%d" % rng.randint(1, N_ENT))
    if r < 0.6:
        lo = rng.randint(1, N_ENT - 1)
        hi = rng.randint(lo + 1, min(N_ENT, lo + gap))
        return ("zo", "k%d" % lo, "k%d" % hi)   # lower index is always the sub-kind: no cycles
    return ("hv", "k%d" % rng.randint(1, N_ENT), "p%d" % rng.randint(1, N_ENT))


def record_text(rec):
    return "%s %s %s." % rec


def make_query(rng, st, asked):
    """Return one query row whose answer is derived from st, or None."""
    inds = sorted(st.direct_kind)
    kinds = sorted(st.kinds_seen)
    props = ["p%d" % i for i in range(1, N_ENT + 1)]
    want_yes = rng.random() < 0.5
    ops = ["ym", "gp", "vx"]
    rng.shuffle(ops)
    for op in ops:
        yes, no = [], []
        if op == "ym":
            for a in inds + kinds:
                for b in kinds:
                    if a == b:
                        continue
                    if a[0] == "i":
                        ans = b in st.kinds_of(a)
                        trivial = ans and b in st.direct_kind.get(a, ())
                    else:
                        ans = b in st.ancestors(a)
                        trivial = ans and b in st.parents.get(a, ())
                    if trivial:
                        continue
                    (yes if ans else no).append((a, b))
            pool = yes if want_yes else no
            rest = no if want_yes else yes
            cands = pool or rest
            if not cands:
                continue
            a, b = rng.choice(cands)
            ans = b in (st.kinds_of(a) if a[0] == "i" else st.ancestors(a))
            text = "ym %s %s?" % (a, b)
        elif op == "gp":
            for x in inds + kinds:
                for p in props:
                    ans = st.has_prop(x, p)
                    if ans and x[0] == "k" and p in st.kind_prop.get(x, ()):
                        continue
                    (yes if ans else no).append((x, p))
            pool = yes if want_yes else no
            rest = no if want_yes else yes
            cands = pool or rest
            if not cands:
                continue
            x, p = rng.choice(cands)
            ans = st.has_prop(x, p)
            text = "gp %s %s?" % (x, p)
        else:
            # vx: the list must contain at least one inherited kind
            cands = []
            for x in inds:
                ks = st.kinds_of(x)
                if ks and not ks <= st.direct_kind[x]:
                    cands.append(x)
            if not cands:
                continue
            x = rng.choice(cands)
            text = "vx %s?" % x
            ans = None
        if text in asked:
            continue
        asked.add(text)
        if op == "vx":
            answer = ",".join(sorted(st.kinds_of(x)))
        else:
            answer = "yes" if ans else "no"
        return text + "\t" + answer
    return None


def make_episode(rng, size):
    gap = GAP[size]
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    st = State()
    rows = []
    seen = set()
    asked = set()
    made = 0
    tries = 0
    while made < n_rec or n_q > 0:
        tries += 1
        if tries > 100000:
            return None
        if n_q > 0 and made >= 2 and (made >= n_rec or rng.random() < 0.3):
            q = make_query(rng, st, asked)
            if q is not None:
                rows.append(q)
                n_q -= 1
                continue
        if made < n_rec:
            rec = rand_record(rng, gap)
            if rec in seen:
                continue
            seen.add(rec)
            st.add(rec)
            rows.append(record_text(rec))
            made += 1
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in GAP:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    used = set()
    out = []
    for _ in range(n):
        while True:
            rows = make_episode(rng, size)
            if rows is None:
                continue
            key = "\n".join(rows)
            if key in used:
                continue
            used.add(key)
            break
        for idx, row in enumerate(rows, 1):
            out.append("%d %s\n" % (idx, row))
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
