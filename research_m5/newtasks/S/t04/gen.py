#!/usr/bin/env python3
"""Synthetic records task: kinds, sub-kinds and properties with inheritance.

Usage: python gen.py N SEED [SIZE]      (SIZE is 1, 3 or 10; default 1)

Entity codes (8 of each kind, always available at every SIZE):
  i1..i8 individuals, c1..c8 kinds, p1..p8 properties.

Record operations:
  bev i c.     individual i is of kind c
  lod c1 c2.   kind c1 is a sub-kind of kind c2
  hax c p.     kind c has property p

Query operations (answered by inheritance through sub-kinds):
  qkn i?       all kinds of i, sorted, comma separated
  qhp i p?     yes or no: does i have property p
  gip i?       all properties of i, sorted, comma separated
"""
import random
import sys

POOL = 8
BEV, LOD, HAX = "bev", "lod", "hax"
QKN, QHP, GIP = "qkn", "qhp", "gip"
SIZES = (1, 3, 10)


def ind(n):
    return f"i{n}"


def kind(n):
    return f"c{n}"


def prop(n):
    return f"p{n}"


class World:
    def __init__(self):
        self.types = {}    # individual -> set of direct kinds
        self.parents = {}  # kind -> set of direct parent kinds
        self.props = {}    # kind -> set of properties of that kind

    def up(self, k):
        """The kind k together with all of its ancestors."""
        seen, stack = set(), [k]
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(self.parents.get(x, ()))
        return seen

    def kinds(self, i):
        """All kinds of individual i, including inherited ones."""
        out = set()
        for k in self.types.get(i, ()):
            out |= self.up(k)
        return out

    def props_of(self, i):
        """All properties of individual i, inherited through its kinds."""
        out = set()
        for k in self.kinds(i):
            out |= self.props.get(k, set())
        return out


def random_record(rng, w):
    """Return a record string, or None if the drawn record is not usable."""
    r = rng.random()
    if r < 0.4:
        i, k = ind(rng.randint(1, POOL)), kind(rng.randint(1, POOL))
        s = w.types.setdefault(i, set())
        if k in s:
            return None
        s.add(k)
        return f"{BEV} {i} {k}."
    if r < 0.7:
        a, b = kind(rng.randint(1, POOL)), kind(rng.randint(1, POOL))
        if a in w.up(b) or b in w.parents.get(a, ()):
            return None  # would create a cycle or a duplicate
        w.parents.setdefault(a, set()).add(b)
        return f"{LOD} {a} {b}."
    c, p = kind(rng.randint(1, POOL)), prop(rng.randint(1, POOL))
    s = w.props.setdefault(c, set())
    if p in s:
        return None
    s.add(p)
    return f"{HAX} {c} {p}."


def random_query(rng, w):
    """Return (query text, answer) for a valid query, or None."""
    known = [i for i in sorted(w.types) if w.types[i]]
    if not known:
        return None
    op = rng.choice([QKN, QHP, GIP])
    i = rng.choice(known)
    if op == QKN:
        ks = w.kinds(i)
        if len(ks) == len(w.types[i]):
            return None  # require at least one inherited kind
        return f"{QKN} {i}?", ",".join(sorted(ks))
    if op == QHP:
        p = prop(rng.randint(1, POOL))
        return f"{QHP} {i} {p}?", "yes" if p in w.props_of(i) else "no"
    ps = w.props_of(i)
    if not ps:
        return None
    return f"{GIP} {i}?", ",".join(sorted(ps))


def episode(rng, size):
    """Return a list of (text, answer) rows; answer is None for records."""
    while True:
        w = World()
        n_rec = rng.randint(3 * size, 12 * size)
        n_q = rng.randint(1, 5)
        rows = []
        steps = 0
        while (n_rec or n_q) and steps < 100000:
            steps += 1
            if n_q and rows and (not n_rec or rng.random() < 0.3):
                q = random_query(rng, w)
                if q is not None:
                    rows.append(q)
                    n_q -= 1
                    continue
                if not n_rec:
                    break
            if n_rec:
                r = random_record(rng, w)
                if r is not None:
                    rows.append((r, None))
                    n_rec -= 1
        if not n_rec and not n_q:
            return rows


def main(argv):
    if len(argv) < 3:
        print("usage: python gen.py N SEED [SIZE]", file=sys.stderr)
        return 2
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in SIZES:
        print("SIZE must be 1, 3 or 10", file=sys.stderr)
        return 2
    rng = random.Random(seed)
    seen = set()
    lines = []
    for _ in range(n):
        while True:
            rows = episode(rng, size)
            key = tuple(rows)
            if key not in seen:
                break
        seen.add(key)
        for num, (text, ans) in enumerate(rows, 1):
            if ans is None:
                lines.append(f"{num} {text}")
            else:
                lines.append(f"{num} {text}\t{ans}")
    sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
