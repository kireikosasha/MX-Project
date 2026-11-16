#!/usr/bin/env python3
"""Generate synthetic containment-reasoning stories.

Usage: python gen.py N SEED

Persons carry items and boxes; boxes and items lie in rooms or boxes.
"reside in" is transitive: an entity resides in a container or room if it
sits there directly or through other containers. Every answer is computed by
simulating the statements above the question.
"""
import random
import sys

PEOPLE = ["ada", "ben", "cora", "dan"]
ROOMS = ["hall", "cellar", "garden", "attic"]
BOXES = ["box", "bag", "chest"]
ITEMS = ["key", "coin", "gem"]


def cap(word):
    return word[0].upper() + word[1:]


class World:
    def __init__(self, people, rooms, boxes, items):
        self.people = people
        self.rooms = rooms
        self.boxes = boxes
        self.items = items
        self.parent = {}  # entity -> room, box or person it is attached to
        self.src = {}     # entity -> index of the statement that set that link

    def set_link(self, entity, parent, idx):
        self.parent[entity] = parent
        self.src[entity] = idx

    def trace(self, x):
        """Return (room, ancestors, source statement indices) for entity x."""
        srcs = set()
        ancestors = []
        cur = x
        while cur not in self.rooms:
            srcs.add(self.src[cur])
            cur = self.parent[cur]
            ancestors.append(cur)
        return cur, ancestors, srcs

    def legal_actions(self):
        groups = {"stroll": [], "clasp": [], "release": [], "tuck": [], "fish": []}
        par = self.parent
        for p in self.people:
            for r in self.rooms:
                if r != par[p]:
                    groups["stroll"].append((p, r))
            for x in self.items + self.boxes:
                if par[x] == par[p]:
                    groups["clasp"].append((p, x))
                if par[x] == p:
                    groups["release"].append((p, x))
            for x in self.items:
                if par[x] == p:
                    for c in self.boxes:
                        if par[c] == p or par[c] == par[p]:
                            groups["tuck"].append((p, x, c))
                elif par[x] in self.boxes:
                    c = par[x]
                    if par[c] == p or par[c] == par[p]:
                        groups["fish"].append((p, x, c))
        return {k: v for k, v in groups.items() if v}


def make_question(w, rng):
    """Return (question text, answer) or None if no well-supported question found."""
    containers = w.boxes + w.rooms
    holdable = w.items + w.boxes
    for _ in range(40):
        kind = rng.randrange(3)
        if kind == 0:
            x = rng.choice(holdable)
            room, _, srcs = w.trace(x)
            if len(srcs) >= 2:
                return (f"Which room encloses the {x}?", room)
        elif kind == 1:
            x = rng.choice(holdable)
            y = rng.choice(containers)
            if x == y:
                continue
            _, anc, srcs = w.trace(x)
            if len(srcs) >= 2:
                return (f"Does the {x} reside in the {y}?", "yes" if y in anc else "no")
        else:
            y = rng.choice(containers)
            found = [t for t in w.items if y in w.trace(t)[1]]
            if not found:
                continue
            srcs = set()
            for t in found:
                srcs |= w.trace(t)[2]
            if len(srcs) >= 2:
                return (f"Which things reside in the {y}?", ",".join(sorted(found)))
    return None


def build_story(rng):
    """Return a list of (text, answer-or-None) lines, or None if this attempt failed."""
    total = rng.randint(3, 12)
    n_questions = rng.randint(1, 5)
    rooms = rng.sample(ROOMS, rng.randint(2, 4))
    people = rng.sample(PEOPLE, rng.randint(1, 3))
    boxes = rng.sample(BOXES, rng.randint(1, 2))
    items = rng.sample(ITEMS, rng.randint(1, 3))
    if len(people) + len(boxes) + len(items) > total:
        return None

    w = World(people, rooms, boxes, items)
    lines = []
    nst = 0

    def add_fact(text, entity, parent):
        nonlocal nst
        w.set_link(entity, parent, nst)
        lines.append((text, None))
        nst += 1

    # Initial placements: persons, then boxes, then items.
    for p in people:
        r = rng.choice(rooms)
        add_fact(f"{cap(p)} stands in the {r}.", p, r)
    for b in boxes:
        if rng.random() < 0.7:
            r = rng.choice(rooms)
            add_fact(f"The {b} rests in the {r}.", b, r)
        else:
            p = rng.choice(people)
            add_fact(f"{cap(p)} clasps the {b}.", b, p)
    for x in items:
        roll = rng.random()
        if roll < 0.4:
            r = rng.choice(rooms)
            add_fact(f"The {x} rests in the {r}.", x, r)
        elif roll < 0.8:
            b = rng.choice(boxes)
            add_fact(f"The {x} rests in the {b}.", x, b)
        else:
            p = rng.choice(people)
            add_fact(f"{cap(p)} clasps the {x}.", x, p)

    dyn_left = total - len(lines)
    q_left = n_questions
    while True:
        if q_left and (dyn_left == 0 or rng.random() < 0.35):
            q = make_question(w, rng)
            if q is not None:
                lines.append(q)
                q_left -= 1
                continue
        if dyn_left == 0:
            break
        groups = w.legal_actions()
        kind = rng.choice(sorted(groups))
        opt = rng.choice(groups[kind])
        if kind == "stroll":
            p, r = opt
            add_fact(f"{cap(p)} strolls to the {r}.", p, r)
        elif kind == "clasp":
            p, x = opt
            add_fact(f"{cap(p)} clasps the {x}.", x, p)
        elif kind == "release":
            p, x = opt
            add_fact(f"{cap(p)} releases the {x}.", x, w.parent[p])
        elif kind == "tuck":
            p, x, c = opt
            add_fact(f"{cap(p)} tucks the {x} in the {c}.", x, c)
        else:
            p, x, c = opt
            add_fact(f"{cap(p)} fishes the {x} from the {c}.", x, p)
        dyn_left -= 1

    if q_left:
        return None
    return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(count):
        story = None
        while story is None:
            story = build_story(rng)
        for number, (text, answer) in enumerate(story, 1):
            if answer is None:
                print(f"{number} {text}")
            else:
                print(f"{number} {text}\t{answer}")


if __name__ == "__main__":
    main()
