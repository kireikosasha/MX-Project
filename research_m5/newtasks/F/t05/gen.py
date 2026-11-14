#!/usr/bin/env python3
"""Synthetic state-tracking reading comprehension stories.

Usage: python gen.py N SEED

People move between rooms, take and drop objects, and open and close room
doors. Each story first states the initial state, then mixes events and
questions. Questions are answered by simulating the statements above them.
"""
import random
import sys

PEOPLE = ["ann", "ben", "cal", "dee"]
ROOMS = ["kitchen", "garden", "hall", "shed"]
OBJECTS = ["ball", "key", "book", "cup"]


def cap(word):
    return word[0].upper() + word[1:]


def make_story(rng):
    people = rng.sample(PEOPLE, rng.choice([2, 3]))
    objs = rng.sample(OBJECTS, rng.randint(1, 3))
    doored = rng.sample(ROOMS, rng.randint(1, 2))

    loc = {p: rng.choice(ROOMS) for p in people}   # person -> room
    held = {}    # object -> holder name, or None when on the floor
    floor = {}   # object -> room, valid while the object is on the floor
    door = {}    # room -> "open" / "closed", only for rooms with a stated door

    setup = []
    for p in people:
        setup.append(f"{cap(p)} is in the {loc[p]}.")
    for o in objs:
        if rng.random() < 0.5:
            h = rng.choice(people)
            held[o] = h
            setup.append(f"{cap(h)} has the {o}.")
        else:
            held[o] = None
            floor[o] = rng.choice(ROOMS)
            setup.append(f"The {o} is in the {floor[o]}.")
    for r in doored:
        door[r] = rng.choice(["open", "closed"])
        setup.append(f"The {r} door is {door[r]}.")

    def where(o):
        return loc[held[o]] if held[o] is not None else floor[o]

    def options():
        opts = []
        for p in people:
            for r in ROOMS:
                if r != loc[p]:
                    opts.append(("move", p, r))
        for o in objs:
            for p in people:
                if held[o] is None and floor[o] == loc[p]:
                    opts.append(("take", p, o))
                if held[o] == p:
                    opts.append(("drop", p, o))
        for r in doored:
            for p in people:
                if loc[p] == r:
                    opts.append(("close" if door[r] == "open" else "open", p, r))
        return opts

    def act():
        kind, p, x = rng.choice(options())
        if kind == "move":
            loc[p] = x
            return f"{cap(p)} went to the {x}."
        if kind == "take":
            held[x] = p
            return f"{cap(p)} took the {x}."
        if kind == "drop":
            held[x] = None
            floor[x] = loc[p]
            return f"{cap(p)} dropped the {x}."
        if kind == "open":
            door[x] = "open"
            return f"{cap(p)} opened the {x} door."
        door[x] = "closed"
        return f"{cap(p)} closed the {x} door."

    def ask():
        qs = []
        for p in people:
            qs.append((f"Where is {cap(p)}", loc[p]))
            for r in ROOMS:
                qs.append((f"Is {cap(p)} in the {r}", "yes" if loc[p] == r else "no"))
        for o in objs:
            qs.append((f"Where is the {o}", where(o)))
            qs.append((f"Who has the {o}", held[o] or "nobody"))
        for r in doored:
            qs.append((f"Is the {r} door open", "yes" if door[r] == "open" else "no"))
        for r in ROOMS:
            names = sorted(p for p in people if loc[p] == r)
            qs.append((f"Who is in the {r}", ",".join(names) or "nobody"))
        q, a = rng.choice(qs)
        return f"{q}?\t{a}"

    max_events = min(12 - len(setup), 7)
    n_events = rng.randint(1, max_events)
    n_q = rng.randint(1, 5)
    total = n_events + n_q
    q_slots = set(rng.sample(range(total), n_q))
    body = [ask() if i in q_slots else act() for i in range(total)]

    lines = setup + body
    return [f"{i} {t}" for i, t in enumerate(lines, 1)]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
