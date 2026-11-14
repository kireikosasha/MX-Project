"""Generate synthetic container/location reading-comprehension stories.

Usage: python gen.py N SEED

Things sit inside containers, containers sit inside places (or inside other
containers). Statements move things and containers around. Questions ask where
something is, whether it is inside a given place or container, how many things
are inside a place, or what a container holds. Answers come from simulating the
world state at the point where the question is asked.
"""
import random
import sys

PLACES = ["kitchen", "garden", "hall", "attic"]
THINGS = ["key", "ball", "coin", "book"]
CONTAINERS = ["box", "bag", "chest"]
NAMES = ["ann", "ben", "cal"]


class World:
    def __init__(self):
        # entity -> the place or container it is directly inside
        self.parent = {}

    def known_containers(self):
        return [c for c in CONTAINERS if c in self.parent]

    def ancestors(self, x):
        """Chain of containers/places holding x, nearest first, top place last."""
        out = []
        p = self.parent.get(x)
        while p is not None:
            out.append(p)
            p = self.parent.get(p)
        return out


def make_statement(world, rng):
    known = list(world.parent)
    kc = world.known_containers()
    opts = {"intro": [], "to": [], "in": [], "out": []}

    # introduce a not yet mentioned thing or container
    for x in THINGS + CONTAINERS:
        if x not in world.parent:
            for y in PLACES + kc:
                opts["intro"].append((x, y))

    for x in known:
        for p in PLACES:
            if world.parent[x] != p:
                opts["to"].append((x, p))
        for c in kc:
            if c != x and c != world.parent[x] and x not in world.ancestors(c):
                opts["in"].append((x, c))
        if world.parent[x] in CONTAINERS:
            opts["out"].append((x, world.parent[x]))

    kinds = [k for k in ("intro", "to", "in", "out") if opts[k]]
    kind = rng.choice(kinds)
    x, y = rng.choice(opts[kind])
    n = rng.choice(NAMES).capitalize()

    if kind == "intro":
        world.parent[x] = y
        return f"The {x} is in the {y}."
    if kind == "to":
        world.parent[x] = y
        return f"{n} takes the {x} to the {y}."
    if kind == "in":
        world.parent[x] = y
        return f"{n} puts the {x} in the {y}."
    # "out": x leaves container y and ends up where y is
    world.parent[x] = world.parent[y]
    return f"{n} takes the {x} out of the {y}."


def make_question(world, rng):
    known = list(world.parent)
    kc = world.known_containers()
    opts = {"where": [], "isin": [], "count": [], "whatin": []}

    for x in known:
        opts["where"].append(x)
        for y in PLACES + kc:
            if y != x:
                opts["isin"].append((x, y))
    opts["count"] = list(PLACES)
    for c in kc:
        children = sorted(x for x in known if world.parent[x] == c)
        if children:
            opts["whatin"].append((c, children))

    kinds = [k for k in ("where", "isin", "count", "whatin") if opts[k]]
    kind = rng.choice(kinds)

    if kind == "where":
        x = rng.choice(opts[kind])
        return f"Where is the {x}?", world.ancestors(x)[-1]
    if kind == "isin":
        x, y = rng.choice(opts[kind])
        ans = "yes" if y in world.ancestors(x) else "no"
        return f"Is the {x} in the {y}?", ans
    if kind == "count":
        p = rng.choice(opts[kind])
        n = sum(1 for x in THINGS if x in world.parent and p in world.ancestors(x))
        return f"How many things are in the {p}?", str(n)
    c, children = rng.choice(opts[kind])
    return f"What is in the {c}?", ",".join(children)


def make_story(rng):
    world = World()
    s_left = rng.randint(3, 12)
    q_left = rng.randint(1, 5)
    lines = []
    while s_left or q_left:
        can_ask = q_left > 0 and len(world.parent) > 0
        if can_ask and (s_left == 0 or rng.random() < 0.35):
            text, ans = make_question(world, rng)
            lines.append(text + "\t" + ans)
            q_left -= 1
        else:
            lines.append(make_statement(world, rng))
            s_left -= 1
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
