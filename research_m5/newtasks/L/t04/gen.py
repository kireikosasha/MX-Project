#!/usr/bin/env python3
"""Synthetic kinds-and-properties reading comprehension generator.

Usage: python gen.py N SEED

World: individuals (names) each get exactly one stated kind. Kinds form a
fixed tree (cat, dog -> mammal -> creature; bird, fish -> creature). Stories
state memberships, kind-tree edges, kind-level property rules ("Every K is P")
and individual properties ("X is P"). Questions are answered by forward
chaining over the statements above them in the same story. Under this closed
world, a fact that cannot be derived from those statements is answered "no".
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cal", "Dee", "Eve", "Gus", "Hal", "Ivy"]
KINDS = ["cat", "dog", "mammal", "bird", "fish", "creature"]
PARENT = {"cat": "mammal", "dog": "mammal", "mammal": "creature",
          "bird": "creature", "fish": "creature"}
PROPS = ["warm", "fast", "wet", "tall"]


class World:
    def __init__(self):
        self.kind = {}   # name -> (kind, line of membership statement)
        self.edge = {}   # kind -> (parent, line of edge statement)
        self.rule = {}   # (kind, prop) -> line of rule statement
        self.prop = {}   # name -> {prop: line of property statement}


def chain(w, name):
    """Kinds of an individual, from its own kind upward, with proof sets."""
    k, line = w.kind[name]
    out = [(k, {line})]
    while k in w.edge:
        parent, eline = w.edge[k]
        proof = out[-1][1] | {eline}
        k = parent
        out.append((k, proof))
    return out


def has_kind(w, name, kind):
    for k, proof in chain(w, name):
        if k == kind:
            return proof
    return None


def has_prop(w, name, prop):
    if prop in w.prop.get(name, {}):
        return {w.prop[name][prop]}
    for k, proof in chain(w, name):
        if (k, prop) in w.rule:
            return proof | {w.rule[(k, prop)]}
    return None


def options(w, first):
    if first:
        return ["member"]
    opts = []
    if any(n not in w.kind for n in NAMES):
        opts.append("member")
    if any(k not in w.edge for k in PARENT):
        opts.append("edge")
    if any((k, p) not in w.rule for k in KINDS for p in PROPS):
        opts.append("rule")
    if any(p not in w.prop.get(n, {}) for n in w.kind for p in PROPS):
        opts.append("prop")
    return opts


def add_statement(rng, w, line, first):
    opt = rng.choice(options(w, first))
    if opt == "member":
        name = rng.choice([n for n in NAMES if n not in w.kind])
        kind = rng.choice(KINDS)
        w.kind[name] = (kind, line)
        return f"{name} is a {kind}."
    if opt == "edge":
        kind = rng.choice([k for k in PARENT if k not in w.edge])
        parent = PARENT[kind]
        w.edge[kind] = (parent, line)
        return f"Every {kind} is a {parent}."
    if opt == "rule":
        kind, prop = rng.choice([(k, p) for k in KINDS for p in PROPS
                                 if (k, p) not in w.rule])
        w.rule[(kind, prop)] = line
        return f"Every {kind} is {prop}."
    name, prop = rng.choice([(n, p) for n in w.kind for p in PROPS
                             if p not in w.prop.get(n, {})])
    w.prop.setdefault(name, {})[prop] = line
    return f"{name} is {prop}."


def make_question(rng, w):
    things = sorted(w.kind)
    for _ in range(60):
        t = rng.choice(("yn_kind", "yn_prop", "which", "count"))
        if t == "yn_kind":
            name = rng.choice(things)
            kind = rng.choice(KINDS)
            proof = has_kind(w, name, kind)
            if proof is None:
                return f"Is {name} a {kind}?", "no"
            if len(proof) >= 2:
                return f"Is {name} a {kind}?", "yes"
        elif t == "yn_prop":
            name = rng.choice(things)
            prop = rng.choice(PROPS)
            proof = has_prop(w, name, prop)
            if proof is None:
                return f"Is {name} {prop}?", "no"
            if len(proof) >= 2:
                return f"Is {name} {prop}?", "yes"
        elif t == "which":
            prop = rng.choice(PROPS)
            names = [n for n in things if has_prop(w, n, prop) is not None]
            if names:
                return (f"Which things are {prop}?",
                        ",".join(n.lower() for n in names))
        else:
            prop = rng.choice(PROPS)
            cnt = sum(1 for n in things if has_prop(w, n, prop) is not None)
            return f"How many things are {prop}?", str(cnt)
    prop = rng.choice(PROPS)
    cnt = sum(1 for n in things if has_prop(w, n, prop) is not None)
    return f"How many things are {prop}?", str(cnt)


def make_story(rng):
    n = rng.randint(3, 12)          # statements
    q = rng.randint(1, 5)           # questions
    total = n + q
    qslots = set(rng.sample(range(1, total), q))   # slot 0 is a statement
    w = World()
    lines = []
    for slot in range(total):
        num = slot + 1
        if slot in qslots:
            text, ans = make_question(rng, w)
            lines.append(f"{num} {text}\t{ans}")
        else:
            text = add_statement(rng, w, num, first=(slot == 0))
            lines.append(f"{num} {text}")
    return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
