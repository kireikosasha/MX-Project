#!/usr/bin/env python3
"""Generate synthetic kinds-and-properties reading-comprehension stories.

Usage: python gen.py N SEED

Individuals belong to kinds; kinds form a small hierarchy and carry
(possibly negated) properties that their members inherit. Questions ask
whether an individual has a property, whether it belongs to a kind it
reaches only through the hierarchy, or which things have a property.
Answers come from a forward-chaining simulation over the statements
above each question, and a question is asked only when its answer is
determined by those statements.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli", "Fay",
         "Gus", "Hal", "Ivy", "Jon", "Kit", "Lou"]
KINDS = ["cat", "dog", "mammal", "bird", "fish", "creature"]
EDGES = [("cat", "mammal"), ("dog", "mammal"), ("mammal", "creature"),
         ("bird", "creature"), ("fish", "creature")]
PROPS = ["warm", "fast", "loud"]


class World:
    def __init__(self):
        self.members = []  # (name, kind)
        self.edges = []    # (kind, parent kind)
        self.props = []    # (kind, prop, value)

    def ancestors(self, kind):
        seen = {kind}
        todo = [kind]
        while todo:
            k = todo.pop()
            for a, b in self.edges:
                if a == k and b not in seen:
                    seen.add(b)
                    todo.append(b)
        return seen

    def kinds_of(self, name):
        out = set()
        for n, k in self.members:
            if n == name:
                out |= self.ancestors(k)
        return out

    def names(self):
        return sorted({n for n, _ in self.members})

    def value(self, name, prop):
        """True or False if derivable from the statements, else None."""
        ks = self.kinds_of(name)
        pos = any(p == prop and v and k in ks for k, p, v in self.props)
        neg = any(p == prop and not v and k in ks for k, p, v in self.props)
        if pos and neg:
            return "conflict"
        if pos:
            return True
        if neg:
            return False
        return None

    def consistent(self):
        return all(self.value(n, p) != "conflict"
                   for n in self.names() for p in PROPS)


def try_add(world, store, fact):
    if fact in store:
        return False
    store.append(fact)
    if world.consistent():
        return True
    store.pop()
    return False


def make_statement(rng, world, names):
    for _ in range(60):
        r = rng.random()
        if r < 0.4:
            n, k = rng.choice(names), rng.choice(KINDS)
            if try_add(world, world.members, (n, k)):
                return f"{n} is a {k}."
        elif r < 0.75:
            k, p, v = rng.choice(KINDS), rng.choice(PROPS), rng.random() < 0.5
            if try_add(world, world.props, (k, p, v)):
                return f"Every {k} is {p}." if v else f"Every {k} is not {p}."
        else:
            e = rng.choice(EDGES)
            if try_add(world, world.edges, e):
                return f"Every {e[0]} is a {e[1]}."
    return None


def make_question(rng, world, asked):
    names = world.names()
    if not names:
        return None
    opts = []
    for n in names:
        direct = {k for m, k in world.members if m == n}
        for p in PROPS:
            v = world.value(n, p)
            if v is not None:
                opts.append((f"Is {n} {p}?", "yes" if v else "no"))
        # kinds reached only through the hierarchy (not stated directly)
        for k in sorted(world.kinds_of(n) - direct):
            opts.append((f"Is {n} a {k}?", "yes"))
    for p in PROPS:
        vals = [world.value(n, p) for n in names]
        if all(v is not None for v in vals) and any(vals):
            ans = ",".join(sorted(n.lower() for n, v in zip(names, vals) if v))
            opts.append((f"Which things are {p}?", ans))
    opts = [o for o in opts if o[0] not in asked]
    if not opts:
        return None
    text, ans = rng.choice(opts)
    asked.add(text)
    return text, ans


def try_story(rng):
    world = World()
    names = rng.sample(NAMES, rng.randint(2, 4))
    target = rng.randint(3, 12)
    want_q = rng.randint(1, 5)
    lines = []  # (text, answer or None for statements)
    asked = set()
    nst = 0
    qs = 0
    while nst < target:
        if nst >= 2 and qs < want_q and rng.random() < 0.4:
            q = make_question(rng, world, asked)
            if q:
                lines.append(q)
                qs += 1
                continue
        s = make_statement(rng, world, names)
        if s is None:
            return None
        lines.append((s, None))
        nst += 1
    tries = 0
    while qs < want_q and tries < 20:
        tries += 1
        q = make_question(rng, world, asked)
        if q:
            lines.append(q)
            qs += 1
    if qs == 0:
        return None
    return lines


def build_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(build_story(rng), 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv)
