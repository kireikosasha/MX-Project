#!/usr/bin/env python3
"""Synthetic record-log reasoning task: alias links with removal and queries.

Records are link events between entity codes n1..n8:
  ali a b.   add a link between a and b
  drp a b.   remove the direct link between a and b
  unl a.     remove every direct link touching a
Queries ask about the connected groups formed by the links so far:
  eqv a b?   yes/no: are a and b connected (directly or through others)?
  grp a?     sorted, comma-separated codes of a's connected group
  sz a?      number of codes in a's connected group

Usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1).
"""
import random
import sys

RECORD_OPS = ["ali", "drp", "unl"]
QUERY_OPS = ["eqv", "grp", "sz"]
ENTITIES = ["n%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)


class State:
    def __init__(self):
        self.edges = set()  # set of frozensets of two entity codes

    def apply(self, op, args):
        if op == "ali":
            self.edges.add(frozenset(args))
        elif op == "drp":
            self.edges.discard(frozenset(args))
        elif op == "unl":
            a = args[0]
            self.edges = {ed for ed in self.edges if a not in ed}

    def neighbors(self, e):
        out = set()
        for ed in self.edges:
            if e in ed:
                out |= set(ed) - {e}
        return out

    def component(self, e):
        seen = {e}
        stack = [e]
        while stack:
            x = stack.pop()
            for y in self.neighbors(x):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen

    def comp_edges(self, e):
        comp = self.component(e)
        return sum(1 for ed in self.edges if ed <= comp)


def make_record(rng, st, ents):
    r = rng.random()
    if r < 0.15 and st.edges:
        op = "drp"
        args = list(rng.choice(sorted(tuple(sorted(e)) for e in st.edges)))
    elif r < 0.25:
        op = "unl"
        args = [rng.choice(ents)]
    else:
        op = "ali"
        args = rng.sample(ents, 2)
    return op, args


def make_query(rng, st, ents):
    # The queried entity must sit in a group with at least two links, so the
    # answer needs several records, not a single lookup.
    cands = [e for e in ents if st.comp_edges(e) >= 2]
    if not cands:
        return None
    a = rng.choice(cands)
    op = rng.choice(QUERY_OPS)
    if op == "eqv":
        b = rng.choice([e for e in ents if e != a])
        ans = "yes" if b in st.component(a) else "no"
        return op, [a, b], ans
    comp = st.component(a)
    if op == "grp":
        return op, [a], ",".join(sorted(comp))
    return op, [a], str(len(comp))


def gen_episode(rng, size):
    ents = sorted(rng.sample(ENTITIES, rng.randint(4, 8)))
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    st = State()
    items = []  # (kind, op, args, answer)
    emitted = 0
    qdone = 0
    while emitted < n_rec or qdone < n_q:
        place_query = qdone < n_q and emitted >= 1 and (
            emitted >= n_rec or rng.random() < (n_q - qdone) / (n_rec - emitted + 1))
        if place_query:
            q = make_query(rng, st, ents)
            if q is not None:
                items.append(("q",) + q)
                qdone += 1
                continue
            if emitted >= n_rec:
                return None
        op, args = make_record(rng, st, ents)
        st.apply(op, args)
        items.append(("r", op, args, None))
        emitted += 1
    return items


def render(items):
    lines = []
    for i, it in enumerate(items, 1):
        body = " ".join([it[1]] + it[2])
        if it[0] == "r":
            lines.append("%d %s." % (i, body))
        else:
            lines.append("%d %s?\t%s" % (i, body, it[3]))
    return lines


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        while True:
            items = gen_episode(rng, size)
            if items is None:
                continue
            lines = render(items)
            text = "\n".join(lines)
            if text not in seen:
                break
        seen.add(text)
        out.extend(lines)
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
