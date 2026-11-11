#!/usr/bin/env python3
"""Synthetic reading task: kinds and their properties.

Individuals (names) belong to kinds. Kinds form a fixed tree
(puppy < dog < animal, kitten < cat < animal, bird < animal), and kinds carry
properties (colour, speed, size). A story states some memberships, some
kind-to-kind links and some kind-level properties. Questions ask about a
property of an individual that was told only about its kind, so the answer
has to be derived through the kind chain.

The world is closed: a property that cannot be derived from the statements
above the question is false.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cy", "Dan", "Eva", "Fay", "Gus", "Hal"]
KINDS = ["puppy", "dog", "kitten", "cat", "bird", "animal"]
PARENT = {"puppy": "dog", "dog": "animal", "kitten": "cat",
          "cat": "animal", "bird": "animal"}
COLORS = ["brown", "gray", "white", "black"]
PAIRS = [("fast", "slow"), ("big", "small")]
ADJS = COLORS + [a for pair in PAIRS for a in pair]


def art(word):
    return "an" if word[0] in "aeiou" else "a"


def ancestors(kind, edges):
    """Kind followed by its stated ancestors (only stated links count)."""
    chain = [kind]
    k = kind
    while k in edges:
        k = PARENT[k]
        chain.append(k)
    return chain


def facts_of(name, members, edges, props):
    """Kinds and properties that the story entails for one individual."""
    kinds = set()
    for n, k in members:
        if n == name:
            kinds.update(ancestors(k, edges))
    adjs = {a for k, a in props if k in kinds}
    return kinds, adjs


def candidates(members, edges, props, asked):
    """All questions that can be answered from the statements so far."""
    named = sorted({n for n, _ in members})
    info = {n: facts_of(n, members, edges, props) for n in named}
    cands = []
    for n in named:
        _, adjs = info[n]
        colours = [c for c in COLORS if c in adjs]
        if len(colours) == 1:
            cands.append(("What color is %s?" % n, colours[0]))
        for pair in PAIRS:
            if not adjs & set(pair):
                continue  # nothing about this pair can be derived
            for a in pair:
                answer = "yes" if a in adjs else "no"
                cands.append(("Is %s %s?" % (n, a), answer))
    for k in KINDS:
        listed = [n for n in named if k in info[n][0]]
        if listed and not any((n, k) in members for n in listed):
            answer = ",".join(n.lower() for n in listed)
            cands.append(("Who is %s %s?" % (art(k), k), answer))
    return [c for c in cands if c[0] not in asked]


def make_statement(rng, pool, members, edges, props):
    for _ in range(50):
        r = rng.random()
        if r < 0.40:
            n = rng.choice(pool)
            k = rng.choice(KINDS)
            if (n, k) in members:
                continue
            members.append((n, k))
            return "%s is %s %s." % (n, art(k), k)
        if r < 0.65:
            open_kinds = [k for k in KINDS if k in PARENT and k not in edges]
            if not open_kinds:
                continue
            k = rng.choice(open_kinds)
            edges.add(k)
            return "Every %s is %s %s." % (k, art(PARENT[k]), PARENT[k])
        k = rng.choice(KINDS)
        a = rng.choice(ADJS)
        if (k, a) in props:
            continue
        props.add((k, a))
        return "Every %s is %s." % (k, a)
    return None


def try_story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    pool = rng.sample(NAMES, rng.randint(2, 4))
    members, edges, props = [], set(), set()
    asked = set()
    lines = []
    st = q = 0
    while st < n_st or q < n_q:
        if q < n_q and (st >= n_st or (st >= 2 and rng.random() < 0.5)):
            cands = candidates(members, edges, props, asked)
            if cands:
                text, answer = rng.choice(cands)
                asked.add(text)
                lines.append((text, answer))
                q += 1
                continue
            if st >= n_st:
                return None
        s = make_statement(rng, pool, members, edges, props)
        if s is None:
            return None
        lines.append((s, None))
        st += 1
    return lines


def make_story(rng):
    for _ in range(100000):
        lines = try_story(rng)
        if lines is not None:
            return lines
    raise RuntimeError("could not build a story")


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, answer) in enumerate(make_story(rng), 1):
            if answer is None:
                out.append("%d %s" % (i, text))
            else:
                out.append("%d %s\t%s" % (i, text, answer))
    print("\n".join(out))


if __name__ == "__main__":
    main()
