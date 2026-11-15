#!/usr/bin/env python3
"""Synthetic state-tracking reading comprehension task.

Things (doors, jars, devices) have states that actions change over time.
Devices can be wired together; wired devices always share one state, so an
action on one device changes every device it is connected to (directly or
through other wires). Questions ask about the state of things now.

Usage: python gen.py N SEED
"""
import random
import sys
from itertools import combinations

NAMES = ["ann", "bob", "cal", "dee", "eli"]

# thing -> its two possible states
THINGS = {
    "door": ("open", "closed"),
    "jar": ("full", "empty"),
    "lamp": ("on", "off"),
    "fan": ("on", "off"),
    "heater": ("on", "off"),
}
DEVICES = ("lamp", "fan", "heater")
ALL_STATES = ("open", "closed", "on", "off", "full", "empty")

# state -> action sentence (the actor changes the thing into that state)
ACTION = {
    "open": "{a} opens the {t}.",
    "closed": "{a} closes the {t}.",
    "full": "{a} fills the {t}.",
    "empty": "{a} empties the {t}.",
    "on": "{a} turns the {t} on.",
    "off": "{a} turns the {t} off.",
}


def component(start, wires):
    """All things connected to `start` through wires (including itself)."""
    seen = {start}
    grew = True
    while grew:
        grew = False
        for w in wires:
            if seen & w and not w <= seen:
                seen |= w
                grew = True
    return seen


def make_question(rng, things, st):
    kind = rng.choice(["is", "which", "how"])
    if kind == "is":
        t = rng.choice(things)
        s = rng.choice(THINGS[t])
        return f"Is the {t} {s}?", ("yes" if st[t] == s else "no")
    s = rng.choice(ALL_STATES)
    if kind == "which":
        hits = sorted(t for t in things if st[t] == s)
        if hits:
            return f"Which things are {s}?", ",".join(hits)
    count = sum(1 for t in things if st[t] == s)
    return f"How many things are {s}?", str(count)


def make_story(rng):
    k = rng.randint(2, 4)
    things = rng.sample(list(THINGS), k)
    state = {t: rng.choice(THINGS[t]) for t in things}

    lines = []  # statement texts, in order
    snaps = []  # state of every thing after each statement
    for t in things:
        lines.append(f"The {t} is {state[t]}.")
        snaps.append(dict(state))

    wires = set()
    n_events = rng.randint(max(1, 3 - k), 12 - k)
    for _ in range(n_events):
        devs = [t for t in things if t in DEVICES]
        # wire two devices only if they are in the same state and not yet connected
        pairs = [(a, b) for a, b in combinations(devs, 2)
                 if state[a] == state[b] and b not in component(a, wires)]
        if pairs and rng.random() < 0.3:
            a, b = rng.choice(pairs)
            wires.add(frozenset((a, b)))
            lines.append(f"The {a} is wired to the {b}.")
        else:
            t = rng.choice(things)
            s = rng.choice(THINGS[t])
            actor = rng.choice(NAMES).capitalize()
            for x in component(t, wires):
                state[x] = s
            lines.append(ACTION[s].format(a=actor, t=t))
        snaps.append(dict(state))

    # questions go after at least k statements, in the middle or at the end
    total = len(lines)
    gaps = sorted(rng.randint(k, total) for _ in range(rng.randint(1, 5)))

    out = []
    num = 0
    qi = 0
    for p in range(total + 1):
        while qi < len(gaps) and gaps[qi] == p:
            q, ans = make_question(rng, things, snaps[p - 1])
            num += 1
            out.append(f"{num} {q}\t{ans}")
            qi += 1
        if p < total:
            num += 1
            out.append(f"{num} {lines[p]}")
    return out


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
