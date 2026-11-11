#!/usr/bin/env python3
"""Synthetic map-reasoning stories: places lie north, south, east or west of each other.

Usage: python gen.py N SEED   (prints N stories; all randomness from random.Random(SEED))

Each story has a hidden map: every place gets a north-south rank and an east-west rank.
Statements reveal some of these facts, phrased in one of two ways. Questions ask either
"What is <dir> of the X?" or "Is the X <dir> of the Y?". Answers are computed from the
transitive closure of the facts stated above the question, and a question is only
asked when that closure determines the answer completely.
"""
import random
import sys

PLACES = ["park", "farm", "lake", "mill", "fort", "town", "cave", "port", "camp", "pier"]
DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
AXIS = {"north": "ns", "south": "ns", "east": "ew", "west": "ew"}
UP = {"ns": "north", "ew": "east"}
MAX_STMTS = 12


def fact_text(rng, a, d, b):
    """Render the fact 'a is d of b' in one of two fixed patterns."""
    if rng.random() < 0.5:
        return "The %s is %s of the %s." % (a, d, b)
    return "The %s is %s of the %s." % (b, OPP[d], a)


def closure(rel):
    """Transitive closure of a set of (greater, lesser) pairs."""
    rel = set(rel)
    while True:
        new = {(a, d) for (a, b) in rel for (c, d) in rel if b == c and a != d} - rel
        if not new:
            return rel
        rel |= new


def holds(rel, a, d, b):
    """True if 'a is d of b' according to a set of (greater, lesser) pairs for the axis."""
    if d in ("north", "east"):
        return (a, b) in rel
    return (b, a) in rel


def make_question(rng, facts, truth):
    """Pick a question whose answer the facts determine and that needs chaining.

    facts: list of (a, direction, b) stated above the question.
    truth(a, d, b): the hidden-world relation, used only as a self-check.
    Returns (question_text, answer) or None.
    """
    places = sorted({x for (a, _, b) in facts for x in (a, b)})
    direct = {"ns": set(), "ew": set()}
    for a, d, b in facts:
        direct[AXIS[d]].add((a, b) if d in ("north", "east") else (b, a))
    clo = {ax: closure(direct[ax]) for ax in direct}

    def comparable(ax, x, y):
        return (x, y) in clo[ax] or (y, x) in clo[ax]

    def stated(ax, x, y):
        return (x, y) in direct[ax] or (y, x) in direct[ax]

    cands = []
    # "What is <dir> of the X?" -- every other place must be comparable with X on that axis.
    for x in places:
        others = [y for y in places if y != x]
        for d in DIRS:
            ax = AXIS[d]
            if not others or not all(comparable(ax, x, y) for y in others):
                continue
            ans = sorted(y for y in others if holds(clo[ax], y, d, x))
            if not ans or all(stated(ax, y, x) for y in ans):
                continue
            assert ans == sorted(y for y in others if truth(y, d, x))
            cands.append(("What is %s of the %s?" % (d, x), ",".join(ans)))
    # "Is the X <dir> of the Y?" -- the pair must be comparable and not stated directly.
    for x in places:
        for y in places:
            if x == y:
                continue
            for d in DIRS:
                ax = AXIS[d]
                if not comparable(ax, x, y) or stated(ax, x, y):
                    continue
                yes = holds(clo[ax], x, d, y)
                assert yes == truth(x, d, y)
                cands.append(("Is the %s %s of the %s?" % (x, d, y), "yes" if yes else "no"))
    if not cands:
        return None
    return rng.choice(cands)


def gen_story_once(rng):
    """Return the lines of one story (statements and questions in order), or None."""
    k = rng.randint(3, 6)
    places = rng.sample(PLACES, k)
    pos = {}
    for ax in ("ns", "ew"):
        order = places[:]
        rng.shuffle(order)
        pos[ax] = {p: i for i, p in enumerate(order)}

    def truth(a, d, b):
        p = pos[AXIS[d]]
        return p[a] > p[b] if d in ("north", "east") else p[a] < p[b]

    def true_fact(a, b, ax):
        if pos[ax][a] < pos[ax][b]:
            a, b = b, a
        return (a, UP[ax], b)

    # Mandatory facts: consecutive places on each axis, which fix the whole order on that axis.
    mandatory, keys = [], set()
    for ax in ("ns", "ew"):
        order = sorted(places, key=lambda p: pos[ax][p])
        for i in range(k - 1):
            mandatory.append(true_fact(order[i], order[i + 1], ax))
            keys.add((ax, frozenset((order[i], order[i + 1]))))
    extras = []
    for ax in ("ns", "ew"):
        for i, a in enumerate(places):
            for b in places[i + 1:]:
                if (ax, frozenset((a, b))) not in keys:
                    extras.append(true_fact(a, b, ax))

    room = MAX_STMTS - len(mandatory)
    chosen = mandatory + rng.sample(extras, rng.randint(0, min(room, len(extras))))
    rng.shuffle(chosen)
    texts = [fact_text(rng, a, d, b) for (a, d, b) in chosen]
    n = len(chosen)

    # Questions: position t means the question follows the first t statements.
    nq = rng.randint(1, 5)
    qs, seen = {}, set()
    for _ in range(80):
        if len(seen) >= nq:
            break
        t = rng.randint(1, n)
        cand = make_question(rng, chosen[:t], truth)
        if cand is None or cand[0] in seen:
            continue
        seen.add(cand[0])
        qs.setdefault(t, []).append(cand)
    if not seen:
        return None

    lines = []
    for i in range(n + 1):
        for text, ans in qs.get(i, []):
            lines.append("%s\t%s" % (text, ans))
        if i < n:
            lines.append(texts[i])
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    count, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        story = None
        while story is None:
            story = gen_story_once(rng)
        out.extend("%d %s" % (i, line) for i, line in enumerate(story, 1))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
