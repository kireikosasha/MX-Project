"""Synthetic exchange-of-things reading task.

Usage: python gen.py N SEED
Prints N stories. Each statement says who has what, who gives a thing to whom,
or who swaps a thing with whom. Questions ask who has a thing, whether a person
has a thing, what a person has, or how many things a person has. Answers are
computed by simulating the world.
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eve"]
THINGS = ["hat", "cup", "ball", "book", "pen"]


class World:
    def __init__(self):
        self.held = {}                          # thing -> holder (each thing has at most one holder)
        self.touches = {t: 0 for t in THINGS}   # number of statements that changed each thing
        self.events = {n: 0 for n in NAMES}     # number of statements involving each person

    def holdings(self, person):
        return sorted(t for t, p in self.held.items() if p == person)


def statement(rng, w):
    opts = []
    for t in THINGS:
        if t not in w.held:
            for n in NAMES:
                opts.append(("has", n, t))
        else:
            p = w.held[t]
            for m in NAMES:
                if m != p:
                    opts.append(("gives", p, t, m))
                    for u in w.holdings(m):
                        opts.append(("swaps", p, t, u, m))
    op = rng.choice(opts)
    kind = op[0]
    if kind == "has":
        _, n, t = op
        w.held[t] = n
        w.touches[t] += 1
        w.events[n] += 1
        return f"{n} has a {t}."
    if kind == "gives":
        _, p, t, m = op
        w.held[t] = m
        w.touches[t] += 1
        w.events[p] += 1
        w.events[m] += 1
        return f"{p} gives the {t} to {m}."
    _, p, t, u, m = op
    w.held[t] = m
    w.held[u] = p
    w.touches[t] += 1
    w.touches[u] += 1
    w.events[p] += 1
    w.events[m] += 1
    return f"{p} swaps the {t} for the {u} with {m}."


def question(rng, w):
    # Only ask about facts that needed at least two statements to settle.
    opts = []
    for t in THINGS:
        if w.touches[t] >= 2:
            opts.append(("who", t))
            for n in NAMES:
                opts.append(("does", n, t))
    for n in NAMES:
        if w.events[n] >= 2:
            opts.append(("what", n))
            opts.append(("how", n))
    if not opts:
        return None
    op = rng.choice(opts)
    if op[0] == "who":
        t = op[1]
        ans = w.held.get(t, "nobody")
        text = f"Who has the {t}?"
    elif op[0] == "does":
        _, n, t = op
        ans = "yes" if w.held.get(t) == n else "no"
        text = f"Does {n} have the {t}?"
    elif op[0] == "what":
        n = op[1]
        hs = w.holdings(n)
        ans = ",".join(hs) if hs else "nothing"
        text = f"What does {n} have?"
    else:
        n = op[1]
        ans = str(len(w.holdings(n)))
        text = f"How many things does {n} have?"
    return f"{text}\t{ans}"


def make_story(rng):
    while True:
        w = World()
        n = rng.randint(3, 12)
        q = rng.randint(1, 5)
        total = n + q
        qpos = set(rng.sample(range(1, total), q))
        lines = []
        for i in range(total):
            item = question(rng, w) if i in qpos else statement(rng, w)
            if item is None:
                break
            lines.append(item)
        else:
            return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        lines = make_story(rng)
        for i, line in enumerate(lines, 1):
            out.append(f"{i} {line[0].upper()}{line[1:]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
