#!/usr/bin/env python3
"""Generate synthetic map-direction stories (north / south / east / west).

Usage: python gen.py N SEED

A story has a hidden map of 3-7 places at random positions. Statements say
that one place is north, south, east or west of another (not necessarily next
to it). They are read off the hidden positions, so they never contradict each
other. Questions ask what lies in a direction from a place, whether one place
is in a direction from another, or which direction one place is from another.

Answers follow from the statements above the question alone, using
transitivity (A north of B and B north of C give A north of C) and the fact
that north/south and east/west are inverses. The story is read as complete:
a relation holds exactly when it follows from the statements. A question is
only asked when every part of its answer needs at least two chained
statements, so no answer is a single stated sentence.
"""
import random
import sys

PLACES = ["park", "lake", "farm", "town", "hill", "camp",
          "cave", "mill", "bay", "port", "fort", "wood"]
DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def closure(n, edges):
    """Transitive closure of (x, y) pairs over range(n)."""
    reach = [[False] * n for _ in range(n)]
    for x, y in edges:
        reach[x][y] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True
    return reach


def norm(d, a, b):
    """'a is d of b' as (axis, x, y), meaning x is N or E of y."""
    if d == "north":
        return ("N", a, b)
    if d == "south":
        return ("N", b, a)
    if d == "east":
        return ("E", a, b)
    return ("E", b, a)  # west


def closures(p, direct):
    return {ax: closure(p, [(x, y) for a, x, y in direct if a == ax])
            for ax in "NE"}


def entailed(CL, d, a, b):
    ax, x, y = norm(d, a, b)
    return CL[ax][x][y]


def is_direct(direct, d, a, b):
    return norm(d, a, b) in direct


def chained(CL, direct, d, a, b):
    """True when 'a is d of b' follows from the story but is not a single statement."""
    return entailed(CL, d, a, b) and not is_direct(direct, d, a, b)


def statement_text(rng, names, fact):
    ax, x, y = fact
    words = ("north", "south") if ax == "N" else ("east", "west")
    if rng.random() < 0.5:
        subj, obj, word = x, y, words[0]
    else:
        subj, obj, word = y, x, words[1]
    if rng.random() < 0.5:
        return f"The {names[subj]} is {word} of the {names[obj]}."
    return f"The {names[subj]} is to the {word} of the {names[obj]}."


def question_candidates(p, names, direct, CL):
    out = []
    # What is <d> of the A?  -> every place lying <d> of A (list if several)
    for d in DIRS:
        for a in range(p):
            subs = [x for x in range(p) if x != a and entailed(CL, d, x, a)]
            if subs and all(chained(CL, direct, d, x, a) for x in subs):
                ans = ",".join(sorted(names[x] for x in subs))
                out.append((f"What is {d} of the {names[a]}?", ans))
    for a in range(p):
        for b in range(p):
            if a == b:
                continue
            # Which direction is the A from the B?  -> entailed direction(s)
            ds = [d for d in DIRS if entailed(CL, d, a, b)]
            if ds and all(chained(CL, direct, d, a, b) for d in ds):
                out.append((f"Which direction is the {names[a]} from the {names[b]}?",
                            ",".join(sorted(ds))))
            # Is the A <d> of the B?  -> yes if entailed, no if the opposite is
            for d in DIRS:
                if chained(CL, direct, d, a, b):
                    out.append((f"Is the {names[a]} {d} of the {names[b]}?", "yes"))
                elif chained(CL, direct, OPP[d], a, b):
                    out.append((f"Is the {names[a]} {d} of the {names[b]}?", "no"))
    return out


def try_story(rng):
    p = rng.randint(3, 7)
    names = rng.sample(PLACES, p)
    ys = rng.sample(range(100), p)  # hidden north-south positions
    xs = rng.sample(range(100), p)  # hidden east-west positions
    facts = []  # normalized true facts: ("N", x, y) = x north of y, etc.
    for x in range(p):
        for y in range(p):
            if x != y:
                if ys[x] > ys[y]:
                    facts.append(("N", x, y))
                if xs[x] > xs[y]:
                    facts.append(("E", x, y))
    n_stmt = rng.randint(3, min(12, len(facts)))
    n_q = rng.randint(1, 5)

    items = []  # (text, answer or None for statements)
    direct = set()
    n_s = n_qd = 0
    asked = set()
    while n_s < n_stmt or n_qd < n_q:
        CL = closures(p, direct)
        cands = []
        if n_qd < n_q and n_s > 0:
            cands = [c for c in question_candidates(p, names, direct, CL)
                     if c[0] not in asked]
        if cands and (n_s == n_stmt or rng.random() < 0.5):
            text, ans = rng.choice(cands)
            asked.add(text)
            items.append((text, ans))
            n_qd += 1
        elif n_s < n_stmt:
            free = [f for f in facts if f not in direct]
            fresh = [f for f in free if not CL[f[0]][f[1]][f[2]]]
            fact = rng.choice(fresh or free)
            direct.add(fact)
            items.append((statement_text(rng, names, fact), None))
            n_s += 1
        else:
            return None
    return items


def make_story(rng):
    while True:
        items = try_story(rng)
        if items is not None:
            return items


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for k, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                lines.append(f"{k} {text}")
            else:
                lines.append(f"{k} {text}\t{ans}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
