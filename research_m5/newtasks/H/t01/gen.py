#!/usr/bin/env python3
"""Synthetic container-reasoning stories.

Usage: python gen.py N SEED

Things and containers sit inside containers or places; places are the
top level and never move. Statements put an object into a location
("is in") or move an object, and its contents move with it ("is moved to").
Questions ask for the place at the top of a chain, whether an object is
inside a location, what is inside a location, or how many things are inside.
"""
import random
import sys

THINGS = ["key", "coin", "apple", "ball", "book"]
CONTAINERS = ["box", "bag", "chest"]
PLACES = ["hall", "kitchen", "garden"]


class World:
    def __init__(self, things, containers, places):
        self.things = things
        self.containers = containers
        self.places = places
        self.movables = things + containers
        self.locations = containers + places
        self.parent = {m: None for m in self.movables}  # places have no entry

    def chain(self, x):
        """Locations above x, nearest first."""
        out = []
        p = self.parent.get(x)
        while p is not None:
            out.append(p)
            p = self.parent.get(p)
        return out

    def contents(self, y):
        """All movables inside location y, directly or indirectly."""
        return [m for m in self.movables if y in self.chain(m)]

    def valid(self, x, y):
        """Can x be put inside y without a cycle?"""
        return y != x and x not in self.chain(y)


def place_statement(w, rng):
    unplaced = [m for m in w.movables if w.parent[m] is None]
    placed = [m for m in w.movables if w.parent[m] is not None]
    if unplaced and (not placed or rng.random() < 0.6):
        x = rng.choice(unplaced)
        y = rng.choice([y for y in w.locations if w.valid(x, y)])
        w.parent[x] = y
        return f"The {x} is in the {y}."
    x = rng.choice(placed)
    pool = [y for y in w.locations if w.valid(x, y) and y != w.parent[x]]
    y = rng.choice(pool)
    w.parent[x] = y
    return f"The {x} is moved to the {y}."


def question(w, rng):
    placed = [m for m in w.movables if w.parent[m] is not None]
    # "Which place" needs a chain of at least two links ending at a place.
    where_ok = [m for m in placed
                if len(w.chain(m)) >= 2 and w.chain(m)[-1] in w.places]
    options = []
    if where_ok:
        options.append("where")
    if placed:
        options.append("is")
    options += ["which", "how"]
    kind = rng.choice(options)

    if kind == "where":
        x = rng.choice(where_ok)
        return f"Which place is the {x} in?", w.chain(x)[-1]

    if kind == "is":
        x = rng.choice(placed)
        locs = [y for y in w.locations if y != x]
        yes_pool = [y for y in locs if y in w.chain(x)]
        no_pool = [y for y in locs if y not in w.chain(x)]
        if no_pool and (not yes_pool or rng.random() < 0.5):
            y = rng.choice(no_pool)
            ans = "no"
        else:
            y = rng.choice(yes_pool)
            ans = "yes"
        return f"Is the {x} in the {y}?", ans

    y = rng.choice(w.locations)
    if kind == "which":
        names = sorted(w.contents(y))
        return f"Which things are in the {y}?", ",".join(names) or "none"

    # kind == "how"
    return f"How many things are in the {y}?", str(len(w.contents(y)))


def story(rng):
    things = rng.sample(THINGS, rng.randint(2, 4))
    containers = rng.sample(CONTAINERS, rng.randint(1, 3))
    places = rng.sample(PLACES, rng.randint(2, 3))
    w = World(things, containers, places)

    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    st_left, q_left = n_st, n_q
    lines = []
    for i in range(n_st + n_q):
        if i == 0:
            is_q = False  # nothing is placed yet, so a question cannot be asked
        elif st_left == 0:
            is_q = True
        elif q_left == 0:
            is_q = False
        else:
            is_q = rng.random() < q_left / (st_left + q_left)
        if is_q:
            text, ans = question(w, rng)
            lines.append(f"{i + 1} {text}\t{ans}")
            q_left -= 1
        else:
            lines.append(f"{i + 1} {place_statement(w, rng)}")
            st_left -= 1
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
