#!/usr/bin/env python3
"""Synthetic container-tracking reading task.

Usage: python gen.py N SEED

Stories describe people who move between rooms, take things, and put things
into boxes, bags and chests. Containers can be moved too, and can sit inside
other containers. Questions ask where a thing is, whether it is in a given
place, who holds it, or what a container holds. All answers come from a
simulation of the world, applied statement by statement.
"""
import random
import sys

NAMES = ["ann", "bob", "eve", "dan"]
ROOMS = ["hall", "kitchen", "garden", "office"]
COLORS = ["red", "blue", "green"]
CTYPES = ["box", "bag", "chest"]
THINGS = ["key", "coin", "book", "pen", "ball"]


def cap(text):
    return text[0].upper() + text[1:]


class World:
    def __init__(self, rng):
        self.rng = rng
        self.kind = {}     # name -> person / room / container / thing
        self.parent = {}   # placed entity -> the entity it is directly in
        self.order = []    # all entities, in a fixed order
        for p in rng.sample(NAMES, rng.randint(2, 3)):
            self._add(p, "person")
        for r in rng.sample(ROOMS, rng.randint(2, 4)):
            self._add(r, "room")
        combos = [f"{c} {t}" for c in COLORS for t in CTYPES]
        for name in rng.sample(combos, rng.randint(1, 3)):
            self._add(name, "container")
        for th in rng.sample(THINGS, rng.randint(2, 4)):
            self._add(th, "thing")

    def _add(self, name, kind):
        self.kind[name] = kind
        self.order.append(name)

    def of(self, kind):
        return [e for e in self.order if self.kind[e] == kind]

    def ancestors(self, e):
        out = []
        while e in self.parent:
            e = self.parent[e]
            out.append(e)
        return out

    def root(self, e):
        anc = self.ancestors(e)
        return anc[-1] if anc else e

    def mentioned(self, room):
        return any(self.parent.get(e) == room for e in self.order)

    def options(self):
        rooms = self.of("room")
        placed_containers = [c for c in self.of("container") if c in self.parent]
        opts = {"place": [], "went": [], "took": [], "put": []}

        # Place an unplaced entity: a person in a room, a thing or container
        # in a room or in a container that is already placed.
        for e in self.order:
            if e in self.parent or self.kind[e] == "room":
                continue
            if self.kind[e] == "person":
                opts["place"] += [(e, r) for r in rooms]
            else:
                targets = rooms + [c for c in placed_containers if c != e]
                opts["place"] += [(e, t) for t in targets]

        for p in self.of("person"):
            if p not in self.parent:
                continue
            here = self.parent[p]
            opts["went"] += [(p, r) for r in rooms if r != here]
            # Take a thing (or container) lying in the same room, not already
            # held by this person.
            for e in self.order:
                if self.kind[e] in ("person", "room") or e not in self.parent:
                    continue
                if p in self.ancestors(e) or self.root(e) != here:
                    continue
                opts["took"].append((p, e))
            # Put a held thing or container into this room or a container here.
            for e in self.order:
                if self.kind[e] in ("person", "room") or self.parent.get(e) != p:
                    continue
                targets = [here] + [
                    c for c in placed_containers
                    if c != e and self.root(c) == here and e not in self.ancestors(c)
                ]
                opts["put"] += [(p, e, t) for t in targets]
        return opts

    def statement(self):
        opts = self.options()
        weights = {"place": 3, "went": 2, "took": 2, "put": 2}
        kinds = [k for k in weights if opts[k]]
        kind = self.rng.choices(kinds, [weights[k] for k in kinds])[0]
        args = self.rng.choice(opts[kind])
        if kind == "place":
            e, t = args
            self.parent[e] = t
            if self.kind[e] == "person":
                return f"{cap(e)} is in the {t}."
            return f"The {e} is in the {t}."
        if kind == "went":
            p, r = args
            self.parent[p] = r
            return f"{cap(p)} went to the {r}."
        if kind == "took":
            p, e = args
            self.parent[e] = p
            return f"{cap(p)} took the {e}."
        p, e, t = args
        self.parent[e] = t
        return f"{cap(p)} put the {e} in the {t}."

    def question(self, asked):
        cands = []
        # Only things needing at least two links to reach a room are asked
        # about, so the answer depends on several statements.
        for x in self.order:
            if x not in self.parent or self.kind[x] == "person":
                continue
            anc = self.ancestors(x)
            if len(anc) < 2:
                continue
            cands.append((f"where is the {x}?", anc[-1]))
            people = [a for a in anc if self.kind[a] == "person"]
            if people:
                cands.append((f"who has the {x}?", people[0]))
            for y in self.order:
                k = self.kind[y]
                if y == x or k not in ("room", "container"):
                    continue
                if k == "container" and y not in self.parent:
                    continue
                if k == "room" and not self.mentioned(y):
                    continue
                if x in self.ancestors(y):
                    continue
                answer = "yes" if y in anc else "no"
                cands.append((f"is the {x} in the {y}?", answer))
        for c in self.order:
            if self.kind[c] != "container" or c not in self.parent:
                continue
            kids = [e for e in self.order if self.parent.get(e) == c]
            if kids and all(self.kind[k] == "thing" for k in kids):
                cands.append((f"what is in the {c}?", ",".join(sorted(kids))))
        cands = [q for q in cands if q[0] not in asked]
        if not cands:
            return None
        text, answer = self.rng.choice(cands)
        asked.add(text)
        return cap(text), answer


def make_story(rng):
    while True:
        world = World(rng)
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        slots = set(rng.sample(range(1, n_stmt + 1), min(n_q, n_stmt)))
        lines, asked, n_asked = [], set(), 0
        for s in range(1, n_stmt + 1):
            lines.append(world.statement())
            if s in slots:
                q = world.question(asked)
                if q:
                    lines.append(f"{q[0]}\t{q[1]}")
                    n_asked += 1
        if n_asked >= 1:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
