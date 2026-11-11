#!/usr/bin/env python3
"""Generate synthetic chain-of-relation reading comprehension stories.

Usage: python gen.py N SEED

Each story says who is the parent (or boss) of whom, one link at a time.
Questions need two or more links to answer: a grandparent, the top of a
chain, or a yes/no question about a grandparent. Answers come from a
simulation of the stated facts, so each answer follows from the
statements above its question only.
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cara", "Dan", "Eve", "Finn", "Gus", "Hana",
         "Ivan", "Joy", "Kim", "Leo", "Mia", "Noah"]
RELATIONS = ["parent", "boss"]


class World:
    """Stated facts. Each person has at most one stated superior, no cycles."""

    def __init__(self, rel):
        self.rel = rel
        self.up = {}  # child -> its one stated superior

    def ancestors(self, x):
        out = []
        while x in self.up:
            x = self.up[x]
            out.append(x)
        return out


def new_fact(world, rng, names):
    """Add one link child -> superior and return its sentence (or None)."""
    cands = []
    for b in names:
        if b in world.up:
            continue
        for a in names:
            if a != b and b not in world.ancestors(a):
                cands.append((a, b))
    if not cands:
        return None
    a, b = rng.choice(cands)
    world.up[b] = a
    rel = world.rel
    main = f"{a} is the {rel} of {b}."
    if rel == "parent":
        alt = f"{b} has the parent {a}."
    else:
        alt = f"{b} reports to {a}."
    return main if rng.random() < 0.5 else alt


def new_question(world, rng, names, asked):
    """Pick a question that the stated facts fully determine, or None."""
    rel = world.rel
    cands = []
    for x in names:
        anc = world.ancestors(x)
        if len(anc) < 2:
            continue  # need a chain of at least two links
        gp = anc[1]
        top = anc[-1]
        cands.append((f"Who is the {rel} of the {rel} of {x}?", gp))
        cands.append((f"Who is the top of the {rel} chain of {x}?", top))
        cands.append((f"Is {gp} the {rel} of the {rel} of {x}?", "yes"))
        wrong = [n for n in names if n not in (gp, x)]
        if wrong:
            guess = rng.choice(wrong)
            cands.append((f"Is {guess} the {rel} of the {rel} of {x}?", "no"))
    cands = [c for c in cands if c[0] not in asked]
    if not cands:
        return None
    text, ans = rng.choice(cands)
    asked.add(text)
    return text, ans


def attempt(rng):
    rel = rng.choice(RELATIONS)
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    names = rng.sample(NAMES, rng.randint(n_st + 1, len(NAMES)))
    world = World(rel)
    asked = set()
    lines = []  # (text, answer); answer is None for statements
    nq = 0
    nst = 0
    while nst < n_st:
        fact = new_fact(world, rng, names)
        if fact is None:
            return None
        lines.append((fact, None))
        nst += 1
        if nq < n_q and rng.random() < 0.4:
            q = new_question(world, rng, names, asked)
            if q is not None:
                lines.append(q)
                nq += 1
    while nq < n_q:
        q = new_question(world, rng, names, asked)
        if q is None:
            break
        lines.append(q)
        nq += 1
    if nq == 0:
        return None
    return lines


def make_story(rng):
    while True:
        lines = attempt(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
