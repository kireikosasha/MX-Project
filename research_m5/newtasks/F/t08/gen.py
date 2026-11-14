"""Synthetic map-direction reading task.

Places lie on a small grid. Stories state some true directional facts
("The park is north of the school."). Questions ask what lies in a direction
from a place, whether one place is in a direction from another, or in which
direction one place is from another. Answers are computed by closing the
stated facts under transitivity and inversion, and a question is only asked
when the stated facts alone do not already give the answer.

Usage: python gen.py N SEED
"""
import random
import sys

PLACES = ["park", "school", "river", "bridge", "farm",
          "market", "lake", "hill", "tower", "village"]
DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
GRID = 4


def build(facts, k):
    """Direct north/east matrices from facts (a, d, b): a is d of b."""
    north = [[False] * k for _ in range(k)]
    east = [[False] * k for _ in range(k)]
    for a, d, b in facts:
        if d == "north":
            north[a][b] = True
        elif d == "south":
            north[b][a] = True
        elif d == "east":
            east[a][b] = True
        else:
            east[b][a] = True
    return north, east


def close(m, k):
    """Transitive closure (Warshall)."""
    c = [row[:] for row in m]
    for x in range(k):
        for i in range(k):
            if c[i][x]:
                for j in range(k):
                    if c[x][j]:
                        c[i][j] = True
    return c


def closed(mats, k):
    return close(mats[0], k), close(mats[1], k)


def holds(mats, d, a, b):
    north, east = mats
    if d == "north":
        return north[a][b]
    if d == "south":
        return north[b][a]
    if d == "east":
        return east[a][b]
    return east[b][a]


def decided(mats, d, a, b):
    """'yes' / 'no' if the mats decide 'a is d of b', else None."""
    if holds(mats, d, a, b):
        return "yes"
    if holds(mats, OPP[d], a, b):
        return "no"
    return None


def make_question(rng, facts, names, k):
    direct = build(facts, k)
    full = closed(direct, k)
    for _ in range(40):
        kind = rng.randrange(3)
        a, b = rng.sample(range(k), 2)
        d = rng.choice(DIRS)
        if kind == 0:
            found = sorted(names[x] for x in range(k)
                           if x != a and holds(full, d, x, a))
            base = sorted(names[x] for x in range(k)
                          if x != a and holds(direct, d, x, a))
            if found and found != base:
                return (f"What is {d} of the {names[a]}?\t"
                        + ",".join(found))
        elif kind == 1:
            c = decided(full, d, a, b)
            if c is not None and c != decided(direct, d, a, b):
                return f"Is the {names[a]} {d} of the {names[b]}?\t{c}"
        else:
            ds = sorted(x for x in DIRS if holds(full, x, a, b))
            base = sorted(x for x in DIRS if holds(direct, x, a, b))
            if ds and ds != base:
                return (f"In which direction is the {names[a]} "
                        f"from the {names[b]}?\t" + ",".join(ds))
    return None


def random_fact(rng, cells, k):
    a, b = rng.sample(range(k), 2)
    (xa, ya), (xb, yb) = cells[a], cells[b]
    opts = []
    if xa > xb:
        opts.append("east")
    if xa < xb:
        opts.append("west")
    if ya > yb:
        opts.append("north")
    if ya < yb:
        opts.append("south")
    return (a, rng.choice(opts), b)


def statement(rng, fact, names):
    a, d, b = fact
    frame = rng.choice(["The {} is {} of the {}.",
                        "The {} lies {} of the {}."])
    return frame.format(names[a], d, names[b])


def make_story(rng):
    while True:
        k = rng.randint(4, 6)
        names = [PLACES[i] for i in rng.sample(range(len(PLACES)), k)]
        cells = rng.sample([(x, y) for x in range(GRID)
                            for y in range(GRID)], k)
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        total = n_stmt + n_q
        slots = set(rng.sample(range(2, total), n_q))
        lines, facts = [], []
        ok = True
        for i in range(total):
            if i in slots:
                q = make_question(rng, facts, names, k)
                if q is None:
                    ok = False
                    break
                lines.append(q)
            else:
                f = random_fact(rng, cells, k)
                facts.append(f)
                lines.append(statement(rng, f, names))
        if ok:
            return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(count):
        for i, line in enumerate(make_story(rng), 1):
            print(f"{i} {line}")


if __name__ == "__main__":
    main()
