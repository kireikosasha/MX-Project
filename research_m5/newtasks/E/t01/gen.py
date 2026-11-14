#!/usr/bin/env python3
"""Synthetic containment-tracking reading task.

Usage: python gen.py N SEED
Prints N stories. Entities are people, containers (box, bag) and objects
(key, ball, coin) that move between places. A person can take, carry, put
into a container, or leave things. Questions are answered by simulating the
world, so every answer follows from the statements above the question.
"""
import random
import sys

NAMES = ["ann", "ben", "cal", "dan"]
PLACES = ["hall", "room", "park", "shed"]
CONTAINERS = ["box", "bag"]
OBJECTS = ["key", "ball", "coin"]


def cap(w):
    return w[0].upper() + w[1:]


def root(parent, places, e):
    """Place where e is located (through people and containers), or None."""
    cur = e
    while True:
        p = parent[cur]
        if p is None:
            return None
        if p in places:
            return p
        cur = p


def holder(parent, persons, places, e):
    """Person who (directly or through containers) carries e, else nobody."""
    cur = e
    while True:
        p = parent[cur]
        if p in places:
            return "nobody"
        if p in persons:
            return p
        cur = p


def inside(parent, places, x, y):
    """True if container y is above x in the containment chain."""
    cur = parent[x]
    while cur is not None and cur not in places:
        if cur == y:
            return True
        cur = parent[cur]
    return False


def make_statement(rng, parent, persons, items, containers, places):
    acts = []  # (sentence, entity, new parent)
    for P in persons:
        cur = parent[P]
        for pl in places:
            if pl != cur:
                acts.append((f"{cap(P)} went to the {pl}.", P, pl))
        if cur is None:
            continue
        for X in items:
            if parent[X] == cur:
                acts.append((f"{cap(P)} took the {X}.", X, P))
            if parent[X] == P:
                acts.append((f"{cap(P)} left the {X}.", X, cur))
                for C in containers:
                    if C != X and (parent[C] == P or parent[C] == cur):
                        acts.append((f"{cap(P)} put the {X} in the {C}.", X, C))
    for X in items:
        if parent[X] is None:
            for pl in places:
                acts.append((f"The {X} is in the {pl}.", X, pl))
    if not acts:
        return None
    text, ent, newp = rng.choice(acts)
    parent[ent] = newp
    return text


def make_question(rng, parent, persons, items, containers, places):
    cands = []
    for X in items:
        if parent[X] is None or parent[X] in places:
            continue  # need at least one hop of containment or holding
        top = root(parent, places, X)
        if top is None:
            continue
        cands.append((f"Where is the {X}?", top))
        cands.append((f"Who has the {X}?", holder(parent, persons, places, X)))
        for Y in containers + places:
            if Y == X:
                continue
            if Y in places:
                ans = top == Y
            else:
                ans = inside(parent, places, X, Y)
            cands.append((f"Is the {X} in the {Y}?", "yes" if ans else "no"))
    for C in containers:
        kids = sorted(x for x in items if parent[x] == C)
        if kids:
            cands.append((f"What is in the {C}?", ",".join(kids)))
    if not cands:
        return None
    return rng.choice(cands)


def try_story(rng):
    persons = rng.sample(NAMES, rng.randint(2, 4))
    containers = rng.sample(CONTAINERS, rng.randint(1, 2))
    objects = rng.sample(OBJECTS, rng.randint(2, 3))
    places = rng.sample(PLACES, rng.randint(2, 4))
    items = containers + objects
    parent = {e: None for e in persons + items}

    ns = rng.randint(3, 12)
    nq = rng.randint(1, 5)
    total = ns + nq
    qpos = set(rng.sample(range(1, total), nq))  # first line is a statement

    lines = []
    for i in range(total):
        if i in qpos:
            q = make_question(rng, parent, persons, items, containers, places)
            if q is None:
                return None
            lines.append(q)
        else:
            s = make_statement(rng, parent, persons, items, containers, places)
            if s is None:
                return None
            lines.append((s, None))
    return lines


def make_story(rng):
    for _ in range(100000):
        lines = try_story(rng)
        if lines is not None:
            return lines
    raise RuntimeError("could not generate a story")


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
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    main()
