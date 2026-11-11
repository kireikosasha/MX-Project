#!/usr/bin/env python3
"""Synthetic container-tracking reading comprehension task.

Usage: python gen.py N SEED

Stories describe people putting things into containers (box, bag, tin,
crate), taking them out, moving containers between places (desk, shelf,
floor, table). "In" is transitive: an item is in a container when it is
inside it at any depth. Every answer is computed from the simulated state
at the moment the question is asked.
"""
import random
import sys

CONTAINERS = ["box", "bag", "tin", "crate"]
THINGS = ["ball", "key", "apple", "coin"]
PLACES = ["desk", "shelf", "floor", "table"]
PEOPLE = ["Ann", "Ben", "Kim"]


class World:
    def __init__(self):
        # item -> ("place", place name) or ("item", container name)
        self.parent = {}

    def introduced(self, x):
        return x in self.parent

    def root_place(self, x):
        p = self.parent[x]
        while p[0] == "item":
            p = self.parent[p[1]]
        return p[1]

    def inside(self, x, c):
        """True if x is somewhere inside container c (any depth)."""
        p = self.parent.get(x)
        while p is not None and p[0] == "item":
            if p[1] == c:
                return True
            p = self.parent.get(p[1])
        return False

    def things_inside(self, c):
        found, stack = [], [c]
        while stack:
            cur = stack.pop()
            for x, p in self.parent.items():
                if p == ("item", cur):
                    if x in THINGS:
                        found.append(x)
                    else:
                        stack.append(x)
        return sorted(found)


def statement(rng, w, conts, things):
    items = conts + things
    intro = [x for x in items if w.introduced(x)]
    unintro = [x for x in items if not w.introduced(x)]
    intro_conts = [c for c in conts if w.introduced(c)]
    puts = [(x, c) for x in intro for c in intro_conts
            if c != x and not w.inside(c, x) and w.parent[x] != ("item", c)]
    takes = [x for x in intro if w.parent[x][0] == "item"]
    moves = [c for c in intro_conts if w.parent[c][0] == "place"]

    actions = []
    if unintro:
        actions.append("intro")
    if puts:
        actions.append("put")
    if takes:
        actions.append("take")
    if moves:
        actions.append("move")
    if not actions:
        return None

    act = rng.choice(actions)
    person = rng.choice(PEOPLE)

    if act == "intro":
        x = rng.choice(unintro)
        options = [("place", p) for p in PLACES]
        options += [("item", c) for c in intro_conts]
        kind, dest = rng.choice(options)
        if kind == "place":
            w.parent[x] = ("place", dest)
            return "The %s is on the %s." % (x, dest)
        w.parent[x] = ("item", dest)
        return "The %s is in the %s." % (x, dest)

    if act == "put":
        x, c = rng.choice(puts)
        w.parent[x] = ("item", c)
        return "%s put the %s in the %s." % (person, x, c)

    if act == "take":
        x = rng.choice(takes)
        old = w.parent[x][1]
        dest = rng.choice(PLACES)
        w.parent[x] = ("place", dest)
        return "%s took the %s out of the %s and put it on the %s." % (
            person, x, old, dest)

    c = rng.choice(moves)
    cur = w.parent[c][1]
    dest = rng.choice([p for p in PLACES if p != cur])
    w.parent[c] = ("place", dest)
    return "%s moved the %s to the %s." % (person, c, dest)


def question(rng, w, conts, things):
    items = conts + things
    intro = [x for x in items if w.introduced(x)]
    intro_conts = [c for c in conts if w.introduced(c)]
    pairs = [(x, c) for x in intro for c in intro_conts if x != c]

    kinds = ["where"]
    if pairs:
        kinds.append("isin")
    if intro_conts:
        kinds += ["whatin", "count"]
    kind = rng.choice(kinds)

    if kind == "where":
        x = rng.choice(intro)
        pat = rng.choice(["Where is the %s?", "Where is the %s now?"])
        return pat % x, w.root_place(x)

    if kind == "isin":
        x, c = rng.choice(pairs)
        pat = "Is the %s in the %s?" % (x, c)
        return pat, ("yes" if w.inside(x, c) else "no")

    c = rng.choice(intro_conts)
    if kind == "whatin":
        found = w.things_inside(c)
        return "What is in the %s?" % c, (",".join(found) if found else "nothing")

    return "How many things are in the %s?" % c, str(len(w.things_inside(c)))


def try_story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_st + n_q
    if n_q > total - 1:
        return None
    # questions never in position 1, so at least one statement is above each
    qslots = set(rng.sample(range(2, total + 1), n_q))

    conts = rng.sample(CONTAINERS, rng.randint(2, 3))
    things = rng.sample(THINGS, rng.randint(2, 4))
    w = World()
    lines = []
    for i in range(1, total + 1):
        if i in qslots:
            text, ans = question(rng, w, conts, things)
            lines.append("%d %s\t%s" % (i, text, ans))
        else:
            s = statement(rng, w, conts, things)
            if s is None:
                return None
            lines.append("%d %s" % (i, s))
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
