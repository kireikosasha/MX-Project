#!/usr/bin/env python3
"""Synthetic containment-reasoning stories.

Things and containers can be in rooms, in containers, or held by people.
People move between rooms. Questions ask where a thing is, whether it is
inside something, who holds it, or what/how many items are inside something.

Usage: python gen.py N SEED
"""
import random
import sys

PEOPLE = ["ann", "ben", "cara", "dan"]
OBJECTS = ["key", "apple", "ball", "book"]
CONTAINERS = ["box", "bag", "chest"]
ROOMS = ["kitchen", "garden", "hall", "office"]
MAX_TRIES = 5000


class World:
    """parent[e] is the thing e is directly in, held by, or placed in.
    People point to rooms. Objects and containers point to a room,
    a container, or a person (who holds them)."""

    def __init__(self):
        self.parent = {}

    def path(self, e):
        """Chain of parents from e up to a room, or None if not fully known."""
        out = []
        cur = e
        while cur not in ROOMS:
            if cur not in self.parent or len(out) > 10:
                return None
            cur = self.parent[cur]
            out.append(cur)
        return out

    def room(self, e):
        if e in ROOMS:
            return e
        p = self.path(e)
        return p[-1] if p else None

    def holder(self, e):
        for a in self.path(e) or []:
            if a in PEOPLE:
                return a
        return None

    def inside(self, e, y):
        p = self.path(e)
        return p is not None and y in p

    def cycle(self, x, y):
        """True if setting parent[x] = y would create a loop."""
        cur, steps = y, 0
        while cur is not None and steps < 20:
            if cur == x:
                return True
            cur = self.parent.get(cur)
            steps += 1
        return False


def statement(w, rng, people, objs, conts, rooms):
    items = objs + conts
    places = rooms + conts
    opts = []
    for p in people:
        for r in rooms:
            if w.parent.get(p) != r:
                opts.append(("move", p, r))
    for x in items:
        for y in places:
            if y != x and w.parent.get(x) != y and not w.cycle(x, y):
                opts.append(("fact", x, y))
    for p in people:
        rp = w.room(p)
        if rp is None:
            continue
        for x in items:
            if w.room(x) == rp and w.holder(x) is None:
                opts.append(("take", p, x))
            if w.holder(x) == p:
                for y in places:
                    if y == x or w.parent.get(x) == y or w.cycle(x, y):
                        continue
                    if y in ROOMS and y != rp:
                        continue
                    if y in CONTAINERS and w.room(y) != rp:
                        continue
                    opts.append(("put", p, x, y))
    if not opts:
        return None
    o = rng.choice(opts)
    if o[0] == "move":
        _, p, r = o
        w.parent[p] = r
        return f"{p} went to the {r}."
    if o[0] == "fact":
        _, x, y = o
        w.parent[x] = y
        return f"The {x} is in the {y}."
    if o[0] == "take":
        _, p, x = o
        w.parent[x] = p
        return f"{p} took the {x}."
    _, p, x, y = o
    w.parent[x] = y
    return f"{p} put the {x} in the {y}."


def question(w, rng, items, places):
    """Return (question text, answer) computed from the current world, or None."""
    opts = []
    all_known = all(w.path(i) is not None for i in items)
    for x in items:
        p = w.path(x)
        if p is None or len(p) < 2:  # answer must need at least two facts
            continue
        opts.append(("where", x))
        if w.holder(x):
            opts.append(("who", x))
        for y in places:
            if y != x:
                opts.append(("is", x, y))
    if all_known:  # list/count need every item's location known
        for y in places:
            inside = [i for i in items if w.inside(i, y)]
            if inside and any(w.path(i)[0] != y for i in inside):
                opts.append(("what", y))
                opts.append(("count", y))
    if not opts:
        return None
    o = rng.choice(opts)
    k = o[0]
    if k == "where":
        x = o[1]
        return f"Where is the {x}?", w.room(x)
    if k == "is":
        x, y = o[1], o[2]
        return f"Is the {x} in the {y}?", ("yes" if w.inside(x, y) else "no")
    if k == "who":
        x = o[1]
        return f"Who has the {x}?", w.holder(x)
    y = o[1]
    names = sorted(i for i in items if w.inside(i, y))
    if k == "what":
        return f"What is in the {y}?", ",".join(names)
    return f"How many items are in the {y}?", str(len(names))


def try_story(rng):
    people = rng.sample(PEOPLE, rng.randint(2, 3))
    objs = rng.sample(OBJECTS, rng.randint(1, 3))
    conts = rng.sample(CONTAINERS, rng.randint(1, 2))
    rooms = rng.sample(ROOMS, rng.randint(2, 3))
    items = objs + conts
    places = rooms + conts
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    slots = sorted(rng.randint(2, n_st) for _ in range(n_q))

    w = World()
    lines = []  # (text, answer or None)
    qi = 0
    for k in range(1, n_st + 1):
        s = statement(w, rng, people, objs, conts, rooms)
        if s is None:
            return None
        lines.append((s, None))
        while qi < len(slots) and slots[qi] == k:
            q = question(w, rng, items, places)
            if q is None:
                return None
            lines.append(q)
            qi += 1
    return lines


def cap(s):
    return s[0].upper() + s[1:]


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        for _ in range(MAX_TRIES):
            story = try_story(rng)
            if story:
                break
        if not story:
            raise SystemExit("could not build a story")
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                out.append(f"{i} {cap(text)}")
            else:
                out.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
