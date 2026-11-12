#!/usr/bin/env python3
"""Synthetic records task about kinds, sub-kinds and properties.

Record rows state facts: which kind an individual is of, that one kind is a
sub-kind of another, and which property a kind has. Query rows ask which kinds
or which properties an individual has, or whether it has a given property.
Those answers follow by inheritance along sub-kind links.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

# Record operations
OP_KIND = "ziq"    # ziq x k.     individual x is of kind k
OP_SUB = "bav"     # bav k1 k2.   kind k1 is a sub-kind of kind k2
OP_PROP = "lum"    # lum k p.     kind k has property p

# Query operations
Q_KINDS = "qok"    # qok x?       all kinds of x, sorted, comma-separated
Q_HAS = "mez"      # mez x p?     does x have property p: yes / no
Q_PROPS = "tuv"    # tuv x?       all properties of x, sorted, comma-separated

SIZES = (1, 3, 10)


class World:
    """Facts stated so far in one episode, with inheritance computed on demand."""

    def __init__(self):
        self.facts = []      # (op, a, b) in the order they were stated
        self.seen = set()

    def add(self, fact):
        self.facts.append(fact)
        self.seen.add(fact)

    def ancestors(self, k):
        """Kinds reachable from k through sub-kind links, including k itself."""
        out = {k}
        frontier = [k]
        while frontier:
            c = frontier.pop()
            for op, a, b in self.facts:
                if op == OP_SUB and a == c and b not in out:
                    out.add(b)
                    frontier.append(b)
        return out

    def inherit(self, x):
        """Return (direct kinds, all kinds, sub-kind links used, property links used)."""
        direct = {a_k for op, a, a_k in self.facts if op == OP_KIND and a == x}
        kinds = set()
        for d in direct:
            kinds |= self.ancestors(d)
        sub_used = [(a, b) for op, a, b in self.facts if op == OP_SUB and a in kinds]
        prop_used = [(a, b) for op, a, b in self.facts if op == OP_PROP and a in kinds]
        return direct, kinds, sub_used, prop_used


def query_candidates(w, individuals, properties):
    """All queries that the current state answers and that need several records."""
    cands = []
    for x in individuals:
        direct, kinds, sub_used, prop_used = w.inherit(x)
        if not direct:
            continue
        props = sorted({p for _, p in prop_used})
        if len(direct) + len(sub_used) >= 2:
            cands.append((Q_KINDS, x, None, ",".join(sorted(kinds))))
        if len(direct) + len(sub_used) + len(prop_used) >= 2:
            if props:
                cands.append((Q_PROPS, x, None, ",".join(props)))
            for p in properties:
                cands.append((Q_HAS, x, p, "yes" if p in props else "no"))
    return cands


def propose_fact(rng, w, individuals, kinds, properties):
    r = rng.random()
    if r < 0.35:
        return (OP_KIND, rng.choice(individuals), rng.choice(kinds))
    if r < 0.65:
        child, parent = rng.choice(kinds), rng.choice(kinds)
        if child == parent or child in w.ancestors(parent):
            return None  # would create a cycle or a self-link
        return (OP_SUB, child, parent)
    return (OP_PROP, rng.choice(kinds), rng.choice(properties))


def add_record(rng, w, individuals, kinds, properties):
    """Add one new fact; repeat a fact only when no new one turns up quickly."""
    fallback = None
    for _ in range(40):
        fact = propose_fact(rng, w, individuals, kinds, properties)
        if fact is None:
            continue
        if fact not in w.seen:
            w.add(fact)
            return fact
        if fallback is None:
            fallback = fact
    if fallback is not None:
        w.add(fallback)
    return fallback


def build_episode(rng, size):
    """Return a list of row bodies (without numbers), or None if this try failed."""
    individuals = sorted("x%d" % i for i in rng.sample(range(1, 9), rng.randint(4, 8)))
    kinds = sorted("k%d" % i for i in rng.sample(range(1, 9), rng.randint(4, 8)))
    properties = sorted("p%d" % i for i in rng.sample(range(1, 9), rng.randint(3, 8)))
    target_rec = rng.randint(3 * size, 12 * size)
    target_q = rng.randint(1, 5)

    w = World()
    rows = []
    asked = set()
    n_rec = n_q = steps = 0
    while n_rec < target_rec or n_q < target_q:
        steps += 1
        if steps > 4000:
            return None
        if n_q < target_q and n_rec >= 2 and (n_rec >= target_rec or rng.random() < 0.3):
            cands = [c for c in query_candidates(w, individuals, properties)
                     if (c[0], c[1], c[2]) not in asked]
            if cands:
                op, x, p, ans = rng.choice(cands)
                asked.add((op, x, p))
                words = [op, x] if p is None else [op, x, p]
                rows.append(" ".join(words) + "?\t" + ans)
                n_q += 1
                continue
            if n_rec >= target_rec:
                return None
        if n_rec < target_rec:
            fact = add_record(rng, w, individuals, kinds, properties)
            if fact is not None:
                rows.append("%s %s %s." % fact)
                n_rec += 1
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.exit("SIZE must be 1, 3 or 10")

    rng = random.Random(seed)
    episodes_seen = set()
    out = []
    for _ in range(n):
        while True:
            rows = build_episode(rng, size)
            if rows is None:
                continue
            key = "\n".join(rows)
            if key not in episodes_seen:
                episodes_seen.add(key)
                break
        for i, row in enumerate(rows, 1):
            out.append("%d %s" % (i, row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv)
