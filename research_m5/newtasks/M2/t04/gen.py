#!/usr/bin/env python3
"""Garden-world reading comprehension: kinds and properties.

Named plants belong to a kind, each kind belongs to a group, and kinds or
groups can have properties that their members inherit. Questions ask about a
member's property, its group, or its classes. A question is asked only when
the statements above it determine the answer, and the answer is taken from a
simulation of the full world.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cole", "Dana", "Eli", "Fay"]
PARENT = {"rose": "flower", "tulip": "flower", "pine": "plant", "fern": "plant"}
KINDS = list(PARENT)
GROUPS = ["flower", "plant"]
PROPS = ["tall", "soft", "wet", "sweet"]


def cap(s):
    return s[0].upper() + s[1:]


def sentence(f):
    tag = f[0]
    if tag == "kind":       # ("kind", name, kind)
        return cap(f"{f[1]} is a {f[2]}.")
    if tag == "sub":        # ("sub", kind, group)
        return cap(f"every {f[1]} is a {f[2]}.")
    if tag == "rule":       # ("rule", class, prop, value)
        return cap(f"every {f[1]} is {'' if f[3] else 'not '}{f[2]}.")
    # ("ind", name, prop, value)
    return cap(f"{f[1]} is {'' if f[3] else 'not '}{f[2]}.")


def kind_of(facts, n):
    for f in facts:
        if f[0] == "kind" and f[1] == n:
            return f[2]
    return None


def parent_of(facts, k):
    for f in facts:
        if f[0] == "sub" and f[1] == k:
            return f[2]
    return None


def prop_value(facts, n, p):
    """Return (value, direct) for name n and property p, or (None, False)."""
    for f in facts:
        if f[0] == "ind" and f[1] == n and f[2] == p:
            return f[3], True
    k = kind_of(facts, n)
    if k is None:
        return None, False
    classes = {k}
    g = parent_of(facts, k)
    if g is not None:
        classes.add(g)
    vals = {f[3] for f in facts if f[0] == "rule" and f[1] in classes and f[2] == p}
    if len(vals) != 1:
        return None, False
    return vals.pop(), False


def answers(facts, names, include_direct):
    """All questions the facts settle, as (question, answer) pairs."""
    out = []
    for n in names:
        for p in PROPS:
            v, direct = prop_value(facts, n, p)
            if v is not None and (include_direct or not direct):
                out.append((cap(f"is {n} {p}?"), "yes" if v else "no"))
        k = kind_of(facts, n)
        g = parent_of(facts, k) if k is not None else None
        if g is not None:
            for big in GROUPS:
                out.append((cap(f"is {n} a {big}?"), "yes" if big == g else "no"))
            out.append((cap(f"what is {n}?"), ",".join(sorted([k, g]))))
    return out


def make_world(rng):
    while True:
        names = rng.sample(NAMES, rng.randint(2, 4))
        kind = {n: rng.choice(KINDS) for n in names}
        classes = sorted(set(kind.values()) | {PARENT[k] for k in kind.values()})
        rule = {}
        for x in classes:
            for p in PROPS:
                r = rng.random()
                if r < 0.35:
                    rule[(x, p)] = True
                elif r < 0.6:
                    rule[(x, p)] = False
        ind = {}
        for n in names:
            for p in PROPS:
                r = rng.random()
                if r < 0.15:
                    ind[(n, p)] = True
                elif r < 0.25:
                    ind[(n, p)] = False
        # No individual may get both values of a property from its sources.
        consistent = True
        for n in names:
            for p in PROPS:
                vals = set()
                if (n, p) in ind:
                    vals.add(ind[(n, p)])
                for x in (kind[n], PARENT[kind[n]]):
                    if (x, p) in rule:
                        vals.add(rule[(x, p)])
                if len(vals) > 1:
                    consistent = False
        if consistent:
            return names, kind, rule, ind


def make_story(rng):
    names, kind, rule, ind = make_world(rng)
    world = [("kind", n, kind[n]) for n in names]
    world += [("sub", k, PARENT[k]) for k in KINDS]
    world += [("rule", x, p, v) for (x, p), v in rule.items()]
    world += [("ind", n, p, v) for (n, p), v in ind.items()]
    truth = dict(answers(world, names, True))

    m = rng.randint(3, min(12, len(world)))
    chosen = rng.sample(world, m)
    target = rng.randint(1, 5)

    prefix, lines, asked = [], [], set()

    def add_question():
        cands = [c for c in answers(prefix, names, False) if c[0] not in asked]
        if not cands:
            return False
        q, a = rng.choice(cands)
        assert truth[q] == a, (q, a, truth[q])  # prefix must agree with world
        asked.add(q)
        lines.append(("q", q, a))
        return True

    for i, f in enumerate(chosen):
        prefix.append(f)
        lines.append(("s", sentence(f)))
        if len(asked) < target and i >= 1 and rng.random() < 0.5:
            add_question()
    while len(asked) < target and add_question():
        pass

    if not asked:
        return None
    out = []
    for num, item in enumerate(lines, 1):
        if item[0] == "s":
            out.append(f"{num} {item[1]}")
        else:
            out.append(f"{num} {item[1]}\t{item[2]}")
    return out


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(count):
        story = None
        while story is None:
            story = make_story(rng)
        for line in story:
            print(line)


if __name__ == "__main__":
    main()
