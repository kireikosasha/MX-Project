#!/usr/bin/env python3
"""Synthetic records task: kinds, sub-kinds and properties with inheritance.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)

Each episode is a numbered list of rows. Records say which kind an individual
belongs to, that one kind is a sub-kind of another, and which property a kind
has. Queries ask for the kinds or properties an individual has by inheritance,
or whether it has a given property. Answers come from a simulation over the
records above each query.
"""
import random
import sys

SIZES = (1, 3, 10)

OP_KIND = "zib"   # zib a1 c2.   individual a1 is of kind c2
OP_SUB = "gor"    # gor c2 c1.   kind c2 is a sub-kind of kind c1
OP_PROP = "wuk"   # wuk c1 p3.   kind c1 has property p3

Q_KINDS = "rux"   # rux a1?      all kinds of a1, by inheritance (list)
Q_PROPS = "pim"   # pim a1?      all properties of a1, by inheritance (list)
Q_HAS = "vek"     # vek a1 p3?   does a1 have p3, by inheritance (yes/no)

RECORD_WEIGHTS = {OP_KIND: 4, OP_SUB: 2, OP_PROP: 3}


class Kb:
    """Facts stated so far in one episode, each tied to its record index."""

    def __init__(self):
        self.kind_of = {}   # individual -> {kind: record index}
        self.parent = {}    # kind -> {parent kind: record index}
        self.prop_of = {}   # kind -> {property: record index}
        self.texts = []     # distinct records, in order

    def ancestors(self, kind):
        seen = set()
        stack = [kind]
        while stack:
            for p in self.parent.get(stack.pop(), {}):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def height(self, kind, memo):
        """Longest chain of sub-kind links from kind up to a root."""
        if kind not in memo:
            memo[kind] = max([1 + self.height(p, memo)
                              for p in self.parent.get(kind, {})] or [0])
        return memo[kind]

    def inherited_kinds(self, ind):
        """Each kind of ind, direct or inherited, mapped to the records it rests on."""
        found = {}
        queue = []
        for k in sorted(self.kind_of.get(ind, {})):
            found[k] = {self.kind_of[ind][k]}
            queue.append(k)
        i = 0
        while i < len(queue):
            k = queue[i]
            i += 1
            for p in sorted(self.parent.get(k, {})):
                if p not in found:
                    found[p] = found[k] | {self.parent[k][p]}
                    queue.append(p)
        return found

    def inherited_props(self, ind):
        """Each property of ind, inherited from its kinds, mapped to the records it rests on."""
        kinds = self.inherited_kinds(ind)
        found = {}
        for k in sorted(kinds):
            for p in sorted(self.prop_of.get(k, {})):
                support = kinds[k] | {self.prop_of[k][p]}
                if p not in found or len(support) < len(found[p]):
                    found[p] = support
        return found

    def query_options(self, props):
        """Every (query text, answer) pair that is valid in the current state."""
        options = []
        for ind in sorted(self.kind_of):
            kinds = self.inherited_kinds(ind)
            kind_support = set().union(*kinds.values())
            if len(kind_support) >= 2:
                options.append((f"{Q_KINDS} {ind}?", ",".join(sorted(kinds))))
            has = self.inherited_props(ind)
            if has and len(set().union(*has.values())) >= 2:
                options.append((f"{Q_PROPS} {ind}?", ",".join(sorted(has))))
            for p in props:
                if p in has:
                    options.append((f"{Q_HAS} {ind} {p}?", "yes"))
                elif len(kind_support) >= 2:
                    options.append((f"{Q_HAS} {ind} {p}?", "no"))
        return options

    def fits_depth(self, c, d, knds, limit):
        """True if adding 'c is a sub-kind of d' keeps every chain within limit."""
        self.parent.setdefault(c, {})[d] = -1
        try:
            memo = {}
            return max(self.height(k, memo) for k in knds) <= limit
        finally:
            del self.parent[c][d]

    def record_options(self, op, inds, knds, prps, depth_limit):
        """Every new (text, fact) record of this operation that may be added now."""
        options = []
        if op == OP_KIND:
            for a in inds:
                for c in knds:
                    if c not in self.kind_of.get(a, {}):
                        options.append((f"{op} {a} {c}.", ("kind", a, c)))
        elif op == OP_SUB:
            for c in knds:
                for d in knds:
                    if c == d or d in self.parent.get(c, {}) or c in self.ancestors(d):
                        continue
                    if self.fits_depth(c, d, knds, depth_limit):
                        options.append((f"{op} {c} {d}.", ("sub", c, d)))
        else:
            for c in knds:
                for p in prps:
                    if p not in self.prop_of.get(c, {}):
                        options.append((f"{op} {c} {p}.", ("prop", c, p)))
        return options

    def add(self, fact, text):
        idx = len(self.texts)
        self.texts.append(text)
        table, x, y = fact
        if table == "kind":
            self.kind_of.setdefault(x, {})[y] = idx
        elif table == "sub":
            self.parent.setdefault(x, {})[y] = idx
        else:
            self.prop_of.setdefault(x, {})[y] = idx

    def add_random_record(self, rng, inds, knds, prps, depth_limit):
        ops = sorted(RECORD_WEIGHTS)
        first = rng.choices(ops, weights=[RECORD_WEIGHTS[o] for o in ops])[0]
        for op in [first] + [o for o in ops if o != first]:
            options = self.record_options(op, inds, knds, prps, depth_limit)
            if options:
                text, fact = rng.choice(options)
                self.add(fact, text)
                return text
        # Every possible fact is already stated: repeat an earlier record.
        return rng.choice(self.texts)


def make_episode(rng, size):
    """Return the rows of one episode (without numbers), or None if it failed."""
    depth_limit = min(7, size + 2)
    inds = ["a%d" % i for i in sorted(rng.sample(range(1, 9), rng.randint(6, 8)))]
    knds = ["c%d" % i for i in sorted(rng.sample(range(1, 9), rng.randint(6, 8)))]
    props = ["p%d" % i for i in sorted(rng.sample(range(1, 9), rng.randint(5, 8)))]
    records = rng.randint(3, 12) * size
    queries = rng.randint(1, 5)

    kb = Kb()
    rows = []
    asked = set()
    while records or queries:
        if queries and (not records or rng.random() < queries / (queries + records)):
            options = [o for o in kb.query_options(props) if o[0] not in asked]
            if options:
                # Pick the query operation first so that list, yes/no and
                # property questions are all common.
                ops = sorted({o[0].split(" ")[0] for o in options})
                chosen = rng.choice(ops)
                text, answer = rng.choice([o for o in options if o[0].startswith(chosen + " ")])
                asked.add(text)
                rows.append(f"{text}\t{answer}")
                queries -= 1
                continue
            if not records:
                return None
        rows.append(kb.add_random_record(rng, inds, knds, props, depth_limit))
        records -= 1
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.exit("SIZE must be 1, 3 or 10")

    rng = random.Random(seed)
    seen = set()
    lines = []
    for _ in range(n):
        while True:
            rows = make_episode(rng, size)
            if rows is None:
                continue
            key = "\n".join(rows)
            if key not in seen:
                break
        seen.add(key)
        lines.extend(f"{i} {row}" for i, row in enumerate(rows, 1))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv)
