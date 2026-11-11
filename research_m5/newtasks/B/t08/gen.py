#!/usr/bin/env python3
"""Synthetic map task: places lie north, south, east or west of each other.

Usage: python gen.py N SEED

Prints N stories. Each line is "k sentence." (statement) or
"k question?<TAB>answer" (question). Numbers restart at 1 in every story.
A hidden world places a few places on a small grid; statements are true facts
of that world. Questions are only emitted when their answer follows from the
statements above them (closure under transitivity of each direction), and the
answer must need at least two statements to derive.
"""
import random
import sys

PLACES = ["park", "school", "market", "lake", "farm", "tower", "bridge", "garden"]
DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
GRID = [(x, y) for x in range(3) for y in range(3)]


def holds(p, q, d):
    """True if a place at grid cell p lies in direction d of a place at cell q."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    if d == "north":
        return dy > 0
    if d == "south":
        return dy < 0
    if d == "east":
        return dx > 0
    return dx < 0  # west


class Known:
    """Facts stated so far and their closure under transitivity."""

    def __init__(self):
        self.direct = set()  # (a, d, b) for every stated fact, both orientations
        self.rel = {d: set() for d in DIRS}  # (a, b) in rel[d]: a lies d of b

    def add(self, a, d, b):
        self.direct.add((a, d, b))
        self.direct.add((b, OPP[d], a))
        self.rel[d].add((a, b))
        self.rel[OPP[d]].add((b, a))
        for d2 in DIRS:
            r = self.rel[d2]
            changed = True
            while changed:
                changed = False
                new = set()
                for (x, y) in r:
                    for (y2, z) in r:
                        if y == y2 and x != z and (x, z) not in r:
                            new.add((x, z))
                if new:
                    r |= new
                    changed = True

    def pos(self, a, d, b):
        """a lies d of b follows from the statements."""
        return (a, b) in self.rel[d]

    def neg(self, a, d, b):
        """a does not lie d of b follows (b lies d of a)."""
        return (b, a) in self.rel[d]

    def known(self, a, d, b):
        return self.pos(a, d, b) or self.neg(a, d, b)


def yes_no_texts(a, d, b):
    return [f"is the {a} {d} of the {b}?", f"does the {a} lie {d} of the {b}?"]


def question_candidates(k, names, asked):
    """(text, answer) pairs that the statements so far fully determine."""
    out = []
    for x in names:
        others = [z for z in names if z != x]
        for d in DIRS:
            if not all(k.known(z, d, x) for z in others):
                continue  # the full set of places in that direction must be settled
            ys = sorted(y for y in others if k.pos(y, d, x))
            if not ys:
                continue
            if all((y, d, x) in k.direct for y in ys):
                continue  # every answer is a single lookup
            text = f"what is {d} of the {x}?"
            if text not in asked:
                out.append((text, ",".join(ys)))
    for a in names:
        for b in names:
            if a == b:
                continue
            for d in DIRS:
                if k.pos(a, d, b) and (a, d, b) not in k.direct:
                    ans = "yes"
                elif k.neg(a, d, b) and (b, d, a) not in k.direct:
                    ans = "no"
                else:
                    continue
                for text in yes_no_texts(a, d, b):
                    if text not in asked:
                        out.append((text, ans))
    return out


def make_story(rng):
    """Return a list of (text, answer_or_None) entries for one story."""
    while True:
        names = rng.sample(PLACES, rng.randint(3, 5))
        cells = rng.sample(GRID, len(names))
        place = dict(zip(names, cells))
        facts = [
            (a, d, b)
            for a in names
            for b in names
            if a != b
            for d in DIRS
            if holds(place[a], place[b], d)
        ]
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        k = Known()
        entries = []
        asked = set()
        s = q = 0
        failed = False
        while s < n_stmt or q < n_q:
            must_q = s == n_stmt
            if q < n_q and (must_q or rng.random() < 0.35):
                cands = question_candidates(k, names, asked)
                if cands:
                    text, ans = rng.choice(cands)
                    asked.add(text)
                    entries.append((text, ans))
                    q += 1
                    continue
            if must_q:
                failed = True
                break
            cands = [f for f in facts if f not in k.direct]
            if not cands:
                failed = True
                break
            fresh = [f for f in cands if not k.pos(*f)]
            pool = fresh if fresh and rng.random() < 0.8 else cands
            a, d, b = rng.choice(pool)
            k.add(a, d, b)
            if rng.random() < 0.5:
                sent = f"the {a} is {d} of the {b}."
            else:
                sent = f"the {a} lies {d} of the {b}."
            entries.append((sent, None))
            s += 1
        if not failed:
            return entries


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), start=1):
            if ans is None:
                lines.append(f"{i} {text}")
            else:
                lines.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
