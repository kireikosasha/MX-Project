#!/usr/bin/env python3
"""Synthetic group-membership task: people join, leave and move between groups.

Usage: python gen.py N SEED
Prints N stories. Numbering restarts at 1 for each story.
"""
import random
import sys

NAMES = ["ann", "ben", "cal", "dora", "eli", "fay", "gus", "hana"]
GROUPS = ["red", "blue", "green", "gold"]


def statement(rng, people, groups, state):
    """Pick a legal event, apply it to state, and return the sentence."""
    out = [p for p in people if state[p] is None]
    inside = [p for p in people if state[p] is not None]
    kinds = []
    if out:
        kinds.append("join")
    if inside:
        kinds += ["leave", "move"]
    kind = rng.choice(kinds)
    if kind == "join":
        p = rng.choice(out)
        g = rng.choice(groups)
        state[p] = g
        return f"{p} joins the {g} group."
    p = rng.choice(inside)
    cur = state[p]
    if kind == "leave":
        state[p] = None
        return f"{p} leaves the {cur} group."
    new = rng.choice([g for g in groups if g != cur])
    state[p] = new
    return f"{p} moves from the {cur} group to the {new} group."


def question(rng, people, groups, state):
    """Return (question text without '?', answer) from the current state."""
    kind = rng.randrange(4)
    if kind == 0:
        p = rng.choice(people)
        g = state[p]
        return f"which group is {p} in", (g if g else "none")
    if kind == 1:
        g = rng.choice(groups)
        members = sorted(p for p in people if state[p] == g)
        return f"who is in the {g} group", (",".join(members) if members else "none")
    if kind == 2:
        g = rng.choice(groups)
        n = sum(1 for p in people if state[p] == g)
        return f"how many are in the {g} group", str(n)
    p = rng.choice(people)
    g = rng.choice(groups)
    return f"is {p} in the {g} group", ("yes" if state[p] == g else "no")


def make_story(rng):
    people = rng.sample(NAMES, rng.randint(2, 5))
    groups = rng.sample(GROUPS, rng.randint(2, 4))
    state = {p: None for p in people}
    slots = ["S"] * rng.randint(3, 12) + ["Q"] * rng.randint(1, 5)
    while True:
        rng.shuffle(slots)
        if slots[0] == "S":  # every question has at least one statement above it
            break
    lines = []
    for i, slot in enumerate(slots, 1):
        if slot == "S":
            lines.append(f"{i} {statement(rng, people, groups, state)}")
        else:
            q, a = question(rng, people, groups, state)
            lines.append(f"{i} {q}?\t{a}")
    return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(2)
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
