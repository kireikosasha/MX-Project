#!/usr/bin/env python3
"""Synthetic possession-and-exchange reading task generator.

Usage: python gen.py N SEED

Each story is a short sequence of statements about people passing, dropping,
and swapping things. Questions are interleaved and answered by simulating the
statements above them. Standard library only; all randomness comes from
random.Random(SEED).
"""
import random
import sys

NAMES = ["ann", "bob", "cal", "dee", "eli", "fay"]
THINGS = ["ball", "hat", "book", "key", "cup", "coin"]


class World:
    def __init__(self, names, things):
        self.names = names
        self.things = things
        self.holder = {t: None for t in things}  # thing -> person or None

    def held_by(self, person):
        return sorted(t for t in self.things if self.holder[t] == person)


def make_statement(rng, w):
    """Pick a valid action, apply it to the world, return (text, things_used)."""
    while True:
        kind = rng.choice(["took", "gave", "dropped", "swapped"])
        if kind == "took":
            p = rng.choice(w.names)
            t = rng.choice(w.things)
            if w.holder[t] == p:
                continue
            w.holder[t] = p
            return f"{p.capitalize()} took the {t}.", [t]
        if kind == "gave":
            p, q = rng.sample(w.names, 2)
            held = w.held_by(p)
            if not held:
                continue
            t = rng.choice(held)
            w.holder[t] = q
            return f"{p.capitalize()} gave the {t} to {q}.", [t]
        if kind == "dropped":
            p = rng.choice(w.names)
            held = w.held_by(p)
            if not held:
                continue
            t = rng.choice(held)
            w.holder[t] = None
            return f"{p.capitalize()} dropped the {t}.", [t]
        # swapped: p gives their thing a to q, q gives their thing b to p
        p, q = rng.sample(w.names, 2)
        hp = w.held_by(p)
        hq = w.held_by(q)
        if not hp or not hq:
            continue
        a = rng.choice(hp)
        b = rng.choice(hq)
        w.holder[a] = q
        w.holder[b] = p
        return f"{p.capitalize()} swapped the {a} with {q} for the {b}.", [a, b]


def make_question(rng, w, used):
    """Ask a question about the current world state, return (text, answer)."""
    t = rng.choice(used) if used else rng.choice(w.things)
    p = rng.choice(w.names)
    kind = rng.randrange(3)
    if kind == 0:
        holder = w.holder[t]
        return f"Who has the {t}?", (holder if holder is not None else "nobody")
    if kind == 1:
        return f"Does {p} have the {t}?", ("yes" if w.holder[t] == p else "no")
    held = w.held_by(p)
    return f"What does {p} have?", (",".join(held) if held else "nothing")


def make_story(rng):
    names = rng.sample(NAMES, rng.randint(3, len(NAMES)))
    things = rng.sample(THINGS, rng.randint(3, len(THINGS)))
    w = World(names, things)

    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    # A question placed after k statements (k >= 2) is answered from those k.
    slots = [rng.randint(2, n_stmt) for _ in range(n_q)]

    used = []
    body = []  # list of (text) without numbers
    for k in range(n_stmt + 1):
        for _ in range(slots.count(k)):
            q, a = make_question(rng, w, used)
            body.append(f"{q}\t{a}")
        if k < n_stmt:
            s, things_used = make_statement(rng, w)
            used.extend(x for x in things_used if x not in used)
            body.append(s)
    return [f"{i} {line}" for i, line in enumerate(body, start=1)]


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
