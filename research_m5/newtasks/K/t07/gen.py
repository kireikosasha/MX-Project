"""Synthetic quantity-tracking reading task (bAbI-style, own wording).

Usage: python gen.py N SEED

Prints N stories. People start with some items, then find, lose, give and
take items from each other. Questions ask how many items someone has,
whether one person has more or fewer than another, who has the most, or
who has none. Every answer is computed by simulating the story up to the
question.
"""
import random
import sys

NAMES = ["Mary", "John", "Sandra", "Fred", "Lucy"]
THINGS = ["apple", "coin", "ball"]


def noun(thing, n):
    return thing if n == 1 else thing + "s"


def do_event(rng, counts, thing):
    """Apply one random gain/loss/transfer to counts and return its sentence."""
    people = list(counts)
    options = []
    for a in people:
        options.append(("gain", a))
        if counts[a] > 0:
            options.append(("lose", a))
        for b in people:
            if b == a:
                continue
            if counts[a] > 0:
                options.append(("give", a, b))
            if counts[b] > 0:
                options.append(("take", a, b))
    op = rng.choice(options)
    kind = op[0]
    if kind == "gain":
        a = op[1]
        n = rng.randint(1, 3)
        counts[a] += n
        return f"{a} finds {n} {noun(thing, n)}."
    if kind == "lose":
        a = op[1]
        n = rng.randint(1, counts[a])
        counts[a] -= n
        return f"{a} loses {n} {noun(thing, n)}."
    a, b = op[1], op[2]
    if kind == "give":
        n = rng.randint(1, counts[a])
        counts[a] -= n
        counts[b] += n
        return f"{a} gives {n} {noun(thing, n)} to {b}."
    n = rng.randint(1, counts[b])
    counts[b] -= n
    counts[a] += n
    return f"{a} takes {n} {noun(thing, n)} from {b}."


def make_question(rng, state, thing):
    """Return (question text without '?', answer) computed from state."""
    plural = thing + "s"
    people = sorted(state)
    values = [state[p] for p in people]
    top = max(values)
    kinds = ["count", "none"]
    if len(people) >= 2:
        kinds.append("compare")
        if values.count(top) == 1:
            kinds.append("most")
    kind = rng.choice(kinds)

    if kind == "count":
        a = rng.choice(people)
        if rng.random() < 0.5:
            q = f"How many {plural} does {a} have"
        else:
            q = f"How many {plural} has {a}"
        return q, str(state[a])

    if kind == "compare":
        a, b = rng.sample(people, 2)
        if rng.random() < 0.5:
            q = f"Does {a} have more {plural} than {b}"
            ans = state[a] > state[b]
        else:
            q = f"Does {a} have fewer {plural} than {b}"
            ans = state[a] < state[b]
        return q, "yes" if ans else "no"

    if kind == "most":
        winner = next(p for p in people if state[p] == top)
        return f"Who has the most {plural}", winner.lower()

    zero = [p.lower() for p in people if state[p] == 0]
    return f"Who has no {plural}", ",".join(zero) if zero else "nobody"


def make_story(rng):
    thing = rng.choice(THINGS)
    people = rng.sample(NAMES, rng.randint(2, 4))
    total = rng.randint(max(3, len(people) + 1), 12)
    pending = list(people)
    rng.shuffle(pending)
    counts = {}
    texts = []
    snaps = []

    while len(texts) < total:
        if pending and (len(counts) < 2 or rng.random() < 0.4):
            p = pending.pop()
            n = rng.randint(0, 4)
            counts[p] = n
            texts.append(f"{p} has {n} {noun(thing, n)}.")
        else:
            texts.append(do_event(rng, counts, thing))
        snaps.append(dict(counts))

    nq = rng.randint(1, 5)
    qpos = [rng.randint(1, total) for _ in range(nq)]
    items = []
    for k in range(1, total + 1):
        items.append(texts[k - 1])
        for pos in qpos:
            if pos == k:
                items.append(make_question(rng, snaps[k - 1], thing))

    lines = []
    for i, item in enumerate(items, 1):
        if isinstance(item, tuple):
            lines.append(f"{i} {item[0]}?\t{item[1]}")
        else:
            lines.append(f"{i} {item}")
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
