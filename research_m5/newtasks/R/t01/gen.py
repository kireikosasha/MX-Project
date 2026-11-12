#!/usr/bin/env python3
"""Synthetic containment records.

Rooms hold boxes; boxes and rooms hold items; boxes may also hold boxes.
Moving a box carries everything inside it. Queries ask where a thing is
now, how many things are inside something now, or whether a thing is
inside something now.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)
"""
import random
import sys

# Record operations (3):
#   pt k c   put item k into box or room c
#   mv b c   move box b into room or box c (its contents move with it)
#   ej k     take item k out of its box and place it in the box's container
# Query operations (3):
#   rm x?    which room x is in now (answer: room code)
#   nm x?    how many things are inside x now, at any depth (answer: number)
#   cn x y?  is x inside y now (answer: yes / no)


class World:
    def __init__(self, rooms, boxes, items):
        self.rooms = rooms
        self.boxes = boxes
        self.items = items
        self.parent = {}  # entity -> its container; absent means not placed yet

    def ancestors(self, x):
        out = []
        while x in self.parent:
            x = self.parent[x]
            out.append(x)
        return out

    def records(self):
        acts = []
        holders = self.boxes + self.rooms
        for k in self.items:
            for c in holders:
                if self.parent.get(k) != c:
                    acts.append(("pt", k, c))
        for b in self.boxes:
            for c in self.rooms + self.boxes:
                if c != b and self.parent.get(b) != c and b not in self.ancestors(c):
                    acts.append(("mv", b, c))
        for k in self.items:
            box = self.parent.get(k)
            if box in self.boxes and box in self.parent:
                acts.append(("ej", k, None))
        return acts

    def apply(self, act):
        op, a, c = act
        if op == "ej":
            self.parent[a] = self.parent[self.parent[a]]
        else:
            self.parent[a] = c

    @staticmethod
    def text(act):
        op, a, c = act
        if op == "ej":
            return f"ej {a}."
        return f"{op} {a} {c}."

    def queries(self):
        qs = []
        things = self.items + self.boxes
        for x in things:
            anc = self.ancestors(x)
            if len(anc) >= 2 and anc[-1] in self.rooms:
                qs.append((f"rm {x}?", anc[-1]))
        for x in self.boxes + self.rooms:
            n = sum(1 for y in things if x in self.ancestors(y))
            if n >= 2:
                qs.append((f"nm {x}?", str(n)))
        for x in things:
            anc = self.ancestors(x)
            if anc:
                for y in self.boxes + self.rooms:
                    if y != x:
                        qs.append((f"cn {x} {y}?", "yes" if y in anc else "no"))
        return qs


def make_episode(rng, size, seen):
    while True:
        rooms = [f"r{i}" for i in sorted(rng.sample(range(1, 9), rng.randint(2, 3)))]
        boxes = [f"b{i}" for i in sorted(rng.sample(range(1, 9), rng.randint(2, min(8, 2 + size))))]
        items = [f"k{i}" for i in sorted(rng.sample(range(1, 9), rng.randint(2, min(8, 2 + 2 * size))))]
        w = World(rooms, boxes, items)
        rec_left = rng.randint(3 * size, 12 * size)
        q_left = rng.randint(1, 5)
        rows = []
        while rec_left or q_left:
            if q_left and rows and (not rec_left or rng.random() < q_left / (rec_left + q_left)):
                qs = w.queries()
                if qs:
                    text, ans = rng.choice(qs)
                    rows.append(f"{text}\t{ans}")
                    q_left -= 1
                    continue
            acts = w.records() if rec_left else []
            if not acts:
                break
            act = rng.choice(acts)
            rows.append(World.text(act))
            w.apply(act)
            rec_left -= 1
        if rec_left or q_left:
            continue
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        for i, row in enumerate(make_episode(rng, size, seen), 1):
            out.append(f"{i} {row}")
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
