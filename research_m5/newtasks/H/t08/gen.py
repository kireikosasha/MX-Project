"""Synthetic places-on-a-map reading task.

Places lie north, south, east or west of each other. Statements give some
of these relations; questions ask what lies in a direction from a place, or
which way one place is from another. Answers are computed from a hidden
layout of places on a small grid, and each question is kept only when the
statements above it determine a single answer by chaining at least two
statements.

Usage: python gen.py N SEED
"""
import random
import sys

PLACES = ["park", "school", "shop", "lake", "farm", "mill", "bridge", "tower",
          "market", "castle", "harbor", "hill", "garden", "station", "field",
          "quarry"]
DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
FACT_PATTERNS = ["The {a} is {d} of the {b}.", "The {a} lies {d} of the {b}."]


def true_dirs(pos, a, b):
    """Directions d such that place a is d of place b in the hidden layout."""
    ax, ay = pos[a]
    bx, by = pos[b]
    out = set()
    if ay > by:
        out.add("north")
    if ay < by:
        out.add("south")
    if ax > bx:
        out.add("east")
    if ax < bx:
        out.add("west")
    return out


def closure(facts):
    """rel[d] is the set of pairs (x, y) such that x is d of y, derived from
    the statements (with inverses and transitivity within one direction)."""
    rel = {d: set() for d in DIRS}
    for a, b, d in facts:
        rel[d].add((a, b))
        rel[OPP[d]].add((b, a))
    for d in DIRS:
        r = rel[d]
        while True:
            new = {(x, z) for (x, y) in r for (y2, z) in r
                   if y == y2 and x != z} - r
            if not new:
                break
            r |= new
    return rel


def make_fact(rng, names, pos, facts):
    stated = {frozenset((a, b)) for a, b, _ in facts}
    cands = []
    for a in names:
        for b in names:
            if a != b and frozenset((a, b)) not in stated:
                for d in DIRS:
                    if d in true_dirs(pos, a, b):
                        cands.append((a, b, d))
    if not cands:
        return None
    return rng.choice(cands)


def make_question(rng, names, pos, facts, asked_texts):
    stated = {frozenset((a, b)) for a, b, _ in facts}
    cl = closure(facts)
    cands = []
    # Kind 1: "What is <dir> of the <place>?" -> one place.
    for p in names:
        for d in DIRS:
            truth = [x for x in names if x != p and d in true_dirs(pos, x, p)]
            if len(truth) != 1:
                continue
            x = truth[0]
            deduced = {u for (u, v) in cl[d] if v == p}
            if deduced == {x} and frozenset((x, p)) not in stated:
                cands.append((f"What is {d} of the {p}?", x))
    # Kind 2: "Which way is the <a> from the <b>?" -> one direction.
    for a in names:
        for b in names:
            if a == b or frozenset((a, b)) in stated:
                continue
            truth = [d for d in DIRS if d in true_dirs(pos, a, b)]
            if len(truth) != 1:
                continue
            d = truth[0]
            deduced = {e for e in DIRS if (a, b) in cl[e]}
            if deduced == {d}:
                cands.append((f"Which way is the {a} from the {b}?", d))
    cands = [c for c in cands if c[0] not in asked_texts]
    if not cands:
        return None
    return rng.choice(cands)


def try_story(rng):
    n_places = rng.randint(4, 6)
    names = rng.sample(PLACES, n_places)
    cells = [(x, y) for x in range(4) for y in range(4)]
    pos = dict(zip(names, rng.sample(cells, n_places)))
    max_facts = min(12, n_places * (n_places - 1) // 2)
    n_facts = rng.randint(3, max_facts)
    target_q = rng.randint(1, 5)

    facts = []
    lines = []          # (text, answer or None)
    asked_texts = set()
    asked = 0
    while True:
        can_fact = len(facts) < n_facts
        can_ask = asked < target_q and len(facts) >= 2
        if can_ask and (not can_fact or rng.random() < 0.4):
            q = make_question(rng, names, pos, facts, asked_texts)
            if q is not None:
                lines.append(q)
                asked_texts.add(q[0])
                asked += 1
                continue
            if not can_fact:
                break
        if not can_fact:
            break
        f = make_fact(rng, names, pos, facts)
        if f is None:
            return None
        facts.append(f)
        a, b, d = f
        lines.append((rng.choice(FACT_PATTERNS).format(a=a, d=d, b=b), None))
    if asked == 0:
        return None
    return lines


def make_story(rng):
    while True:
        story = try_story(rng)
        if story is not None:
            return story


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for i, (text, answer) in enumerate(make_story(rng), 1):
            if answer is None:
                print(f"{i} {text}")
            else:
                print(f"{i} {text}\t{answer}")


if __name__ == "__main__":
    main()
