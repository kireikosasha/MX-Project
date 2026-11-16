"""Synthetic reading-comprehension task: exchange of things.

People hold physical objects. Statements describe giving, passing, taking,
swapping, putting down and picking up objects. Questions ask who has an
object, what a person has, whether a person has an object, or how many
things a person has. All answers come from simulating the statements above
each question.

Usage: python gen.py N SEED
"""

import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli", "Fay"]
OBJECTS = ["ball", "book", "cup", "hat", "key"]


def candidate_actions(holder):
    """Return every valid action given the current holder of each object."""
    cands = []
    for a in NAMES:
        for o in OBJECTS:
            h = holder[o]
            if h == a:
                for b in NAMES:
                    if b != a:
                        cands.append(("give", a, o, b))
                cands.append(("down", a, o))
                for o2 in OBJECTS:
                    if holder[o2] not in (None, a):
                        cands.append(("swap", a, o, o2, holder[o2]))
            elif h is None:
                cands.append(("up", a, o))
            else:
                cands.append(("take", a, o, h))
    return cands


def render(rng, c):
    kind = c[0]
    if kind == "give":
        _, a, o, b = c
        verb = rng.choice(["gives", "passes"])
        return f"{a} {verb} the {o} to {b}."
    if kind == "take":
        _, a, o, b = c
        return f"{a} takes the {o} from {b}."
    if kind == "swap":
        _, a, o, o2, b = c
        return f"{a} swaps the {o} for the {o2} with {b}."
    if kind == "down":
        _, a, o = c
        return f"{a} puts the {o} down."
    _, a, o = c
    return f"{a} picks up the {o}."


def apply(holder, c):
    kind = c[0]
    if kind == "give":
        holder[c[2]] = c[3]
    elif kind == "take":
        holder[c[2]] = c[1]
    elif kind == "swap":
        _, a, o, o2, b = c
        holder[o] = b
        holder[o2] = a
    elif kind == "down":
        holder[c[2]] = None
    else:
        holder[c[2]] = c[1]


def make_question(rng, holder):
    kind = rng.randrange(4)
    if kind == 0:
        o = rng.choice(OBJECTS)
        h = holder[o]
        return f"Who has the {o}?", (h.lower() if h else "nobody")
    if kind == 1:
        a = rng.choice(NAMES)
        held = [o for o in OBJECTS if holder[o] == a]
        ans = ",".join(held) if held else "nothing"
        return f"What does {a} have?", ans
    if kind == 2:
        a = rng.choice(NAMES)
        held = [o for o in OBJECTS if holder[o] == a]
        if held and rng.random() < 0.5:
            o = rng.choice(held)
        else:
            o = rng.choice(OBJECTS)
        ans = "yes" if holder[o] == a else "no"
        return f"Does {a} have the {o}?", ans
    a = rng.choice(NAMES)
    count = sum(1 for o in OBJECTS if holder[o] == a)
    return f"How many things does {a} have?", str(count)


def make_story(rng):
    holder = {o: None for o in OBJECTS}
    k = rng.randint(3, 12)
    q = rng.randint(1, 5)
    slots = sorted(rng.randint(1, k) for _ in range(q))
    entries = []
    qi = 0
    for i in range(1, k + 1):
        c = rng.choice(candidate_actions(holder))
        entries.append(render(rng, c))
        apply(holder, c)
        while qi < q and slots[qi] == i:
            text, ans = make_question(rng, holder)
            entries.append(f"{text}\t{ans}")
            qi += 1
    return entries


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for idx, entry in enumerate(make_story(rng), 1):
            out.append(f"{idx} {entry}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
