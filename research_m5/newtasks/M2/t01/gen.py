#!/usr/bin/env python3
"""Synthetic containment-tracking reading-comprehension stories.

Usage: python gen.py N SEED
Prints N stories; all randomness comes from random.Random(SEED).

World: persons, objects, containers and places. A place directly holds
persons, objects and containers. An object or container can be inside a
container (possibly nested); a person holds what is inside it. Every placed
entity has a chain of containment that ends in a place. Statements change the
state; questions are answered from the state after the statements above them.
The statements of a story fully define its state (closed world), and questions
are only asked about placed entities, so every answer is determined.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cal", "Dee", "Eli", "Fay"]
OBJECTS = ["key", "ball", "apple", "coin", "hat"]
CONTAINERS = ["box", "bag", "chest"]
PLACES = ["room", "hall", "garden"]

PERSON, OBJ, CONT, PLACE = "person", "object", "container", "place"


class World:
    def __init__(self, names, objs, conts, places):
        self.kind = {}
        for x in names:
            self.kind[x] = PERSON
        for x in objs:
            self.kind[x] = OBJ
        for x in conts:
            self.kind[x] = CONT
        for x in places:
            self.kind[x] = PLACE
        self.parent = {}  # entity -> the entity it is directly inside
        self.used = set()  # places that have appeared in a statement

    def of(self, kind):
        return [x for x in self.kind if self.kind[x] == kind]

    def root(self, e):
        """Place that e is in, through any containers or person; None if unplaced."""
        while self.kind[e] != PLACE:
            e = self.parent.get(e)
            if e is None:
                return None
        return e

    def ancestors(self, e):
        out = []
        p = self.parent.get(e)
        while p is not None:
            out.append(p)
            p = self.parent.get(p)
        return out

    def apply(self, op):
        e, target = op
        self.parent[e] = target
        if self.kind[target] == PLACE:
            self.used.add(target)

    def statements(self):
        """All valid statements now, as (sentence, operation) pairs."""
        opts = []
        persons = self.of(PERSON)
        objs = self.of(OBJ)
        conts = self.of(CONT)
        places = self.of(PLACE)
        things = objs + conts

        # A/B: a person goes to a place (carried things go too).
        for n in persons:
            for p in places:
                if self.parent.get(n) != p:
                    opts.append((f"{n} went to the {p}.", (n, p)))
                    opts.append((f"{n} is in the {p}.", (n, p)))

        # C: a person takes a thing from the same place.
        for n in persons:
            if self.root(n) is None:
                continue
            for t in things:
                if self.root(t) == self.root(n) and self.parent.get(t) != n:
                    opts.append((f"{n} took the {t}.", (t, n)))

        # D: a person puts a thing they hold into a container in the same place.
        for n in persons:
            if self.root(n) is None:
                continue
            for t in things:
                if self.parent.get(t) != n:
                    continue
                for c in conts:
                    if c != t and self.root(c) == self.root(n) and t not in self.ancestors(c):
                        opts.append((f"{n} put the {t} in the {c}.", (t, c)))

        # E: a thing or container is put directly in a placed container or place.
        for t in things:
            for c in conts + places:
                if c == t or self.parent.get(t) == c:
                    continue
                if self.root(c) is None or t in self.ancestors(c):
                    continue
                opts.append((f"The {t} is in the {c}.", (t, c)))

        if not opts:
            raise RuntimeError("no valid statement")
        return opts

    def questions(self):
        """All answerable questions now, as (question text, answer) pairs."""
        qs = []
        for e in self.of(PERSON) + self.of(OBJ) + self.of(CONT):
            r = self.root(e)
            if r is None:
                continue
            if self.kind[e] == PERSON:
                qs.append((f"Where is {e}?", r))
            else:
                qs.append((f"Where is the {e}?", r))

        things = self.of(OBJ) + self.of(CONT)
        holders = [c for c in self.of(CONT) if self.root(c) is not None]
        holders += [p for p in self.of(PLACE) if p in self.used]

        # What is directly inside a container or place.
        for c in holders:
            kids = sorted(x.lower() for x in self.parent if self.parent[x] == c)
            qs.append((f"What is in the {c}?", ",".join(kids) or "nothing"))

        # Is a thing inside a container or place (at any depth)?
        for c in holders:
            for e in things:
                if e == c or self.root(e) is None:
                    continue
                ans = "yes" if c in self.ancestors(e) else "no"
                qs.append((f"Is the {e} in the {c}?", ans))

        # How many things are inside a place (at any depth)?
        for p in self.of(PLACE):
            if p in self.used:
                count = sum(1 for o in self.of(OBJ) if self.root(o) == p)
                qs.append((f"How many things are in the {p}?", str(count)))
        return qs


def make_story(rng):
    names = rng.sample(NAMES, rng.randint(2, 4))
    objs = rng.sample(OBJECTS, rng.randint(2, 4))
    conts = rng.sample(CONTAINERS, rng.randint(1, 3))
    places = rng.sample(PLACES, rng.randint(2, 3))
    world = World(names, objs, conts, places)

    k = rng.randint(3, 12)  # statements
    q = rng.randint(1, 5)  # questions
    layout = ["S"] * k + ["Q"] * q
    rng.shuffle(layout)
    first = layout.index("S")  # a question needs at least one statement above it
    layout[0], layout[first] = layout[first], layout[0]

    lines = []
    for slot in layout:
        num = len(lines) + 1
        if slot == "S":
            text, op = rng.choice(world.statements())
            world.apply(op)
            lines.append(f"{num} {text}")
        else:
            text, answer = rng.choice(world.questions())
            lines.append(f"{num} {text}\t{answer}")
    return lines


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n, seed = int(argv[1]), int(argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main(sys.argv)
