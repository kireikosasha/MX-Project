#!/usr/bin/env python3
"""Generate synthetic possession-tracking stories.

People hold, give, swap and drop items. Questions ask who holds an item,
whether a person holds an item, or what a person holds. Answers are computed
by simulating the statements above each question.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli"]
ITEMS = ["ball", "cup", "hat", "lamp", "book"]


def make_statement(rng, owner):
    """Pick one random event, apply it to owner (item -> name or None)."""
    held = [i for i in ITEMS if owner[i] is not None]
    kinds = ["has", "has", "has", "swap", "swap"]
    if held:
        kinds += ["gives", "gives", "gives", "drops"]
    kind = rng.choice(kinds)

    if kind == "has":
        item = rng.choice(ITEMS)
        name = rng.choice(NAMES)
        prev = owner[item]
        text = f"{name} has the {item}."
        owner[item] = name
        if prev == name:
            items, names = set(), set()
        else:
            items = {item}
            names = {name} | ({prev} if prev is not None else set())
    elif kind == "gives":
        item = rng.choice(held)
        giver = owner[item]
        receiver = rng.choice([n for n in NAMES if n != giver])
        text = f"{giver} gives the {item} to {receiver}."
        owner[item] = receiver
        items = {item}
        names = {giver, receiver}
    elif kind == "drops":
        item = rng.choice(held)
        name = owner[item]
        text = f"{name} drops the {item}."
        owner[item] = None
        items = {item}
        names = {name}
    else:
        a, b = rng.sample(NAMES, 2)
        text = f"{a} swaps with {b}."
        items = set()
        for i in ITEMS:
            if owner[i] == a:
                owner[i] = b
                items.add(i)
            elif owner[i] == b:
                owner[i] = a
                items.add(i)
        names = {a, b} if items else set()

    return {"text": text, "items": items, "names": names}


def make_question(rng, owner, evs):
    """Return (question, answer) whose answer is backed by at least two earlier
    events that changed the relevant item or person, or None if no candidate passes."""
    for _ in range(40):
        t = rng.randrange(3)
        if t == 0:
            x = rng.choice(ITEMS)
            holder = owner[x]
            q = f"Who has the {x}?"
            a = holder.lower() if holder is not None else "nobody"
            support = sum(x in e["items"] for e in evs)
        elif t == 1:
            n = rng.choice(NAMES)
            x = rng.choice(ITEMS)
            q = f"Does {n} have the {x}?"
            a = "yes" if owner[x] == n else "no"
            support = sum(x in e["items"] for e in evs)
        else:
            n = rng.choice(NAMES)
            q = f"What does {n} have?"
            held = sorted(i for i in ITEMS if owner[i] == n)
            a = ",".join(held) if held else "nothing"
            support = sum(n in e["names"] for e in evs)
        if support >= 2:
            return q, a.lower()
    return None


def make_story(rng):
    """Return a list of (text, answer) lines; answer is None for statements."""
    while True:
        n = rng.randint(3, 12)
        q = rng.randint(1, 5)
        owner = {i: None for i in ITEMS}
        events, snaps = [], []
        for _ in range(n):
            events.append(make_statement(rng, owner))
            snaps.append(dict(owner))
        ks = sorted(rng.randint(2, n) for _ in range(q))
        asked = {}
        ok = True
        for k in ks:
            res = make_question(rng, snaps[k - 1], events[:k])
            if res is None:
                ok = False
                break
            asked.setdefault(k, []).append(res)
        if not ok:
            continue
        lines = []
        for i in range(1, n + 1):
            lines.append((events[i - 1]["text"], None))
            lines.extend(asked.get(i, []))
        return lines


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n_stories):
        story = make_story(rng)
        for num, (text, answer) in enumerate(story, 1):
            if answer is None:
                out.append(f"{num} {text}")
            else:
                out.append(f"{num} {text}\t{answer}")
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
