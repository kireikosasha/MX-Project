"""Synthetic quantity-tracking reading task.

Usage: python gen.py N SEED

Each story describes people who hold counts of things. Statements either set an
initial count, add to a count, remove from a count, or move items between two
people. Questions ask for a count now, or whether a person has exactly n items.
All answers are computed by simulating the statements above each question.
"""

import random
import sys

NAMES = ["Ann", "Bob", "Cal", "Dee", "Eli", "Fay"]
# (plural, singular) pairs for the things people hold
THINGS = [("apples", "apple"), ("coins", "coin"), ("rocks", "rock"), ("pears", "pear")]


def noun(thing, n):
    plural, singular = thing
    return singular if n == 1 else plural


def gen_story(rng):
    people = rng.sample(NAMES, rng.randint(2, 4))
    things = rng.sample(THINGS, rng.randint(1, 2))
    counts = {}      # (person, thing index) -> current count
    touched = set()  # (person, thing index) pairs mentioned so far
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_stmt + n_q
    # question positions are never the first line, so every question has
    # at least one statement above it
    q_slots = set(rng.sample(range(1, total), n_q))

    lines = []
    for pos in range(total):
        if pos in q_slots:
            lines.append(make_question(rng, people, things, counts))
        else:
            lines.append(make_statement(rng, people, things, counts, touched))
    return lines


def make_statement(rng, people, things, counts, touched):
    fresh = [(p, t) for p in people for t in range(len(things))
             if (p, t) not in touched]
    holders = [(p, t) for (p, t), c in counts.items() if c >= 1]
    ops = ["gain"]
    if fresh:
        ops.append("init")
    if holders:
        ops += ["loss", "transfer"]
    op = rng.choice(ops)

    if op == "init":
        p, ti = rng.choice(fresh)
        n = rng.randint(1, 9)
        verb = rng.choice(["has", "owns"])
        counts[(p, ti)] = n
        touched.add((p, ti))
        return f"{p} {verb} {n} {noun(things[ti], n)}."

    if op == "gain":
        p = rng.choice(people)
        ti = rng.randrange(len(things))
        n = rng.randint(1, 5)
        verb = rng.choice(["gets", "finds"])
        counts[(p, ti)] = counts.get((p, ti), 0) + n
        touched.add((p, ti))
        return f"{p} {verb} {n} {noun(things[ti], n)}."

    if op == "loss":
        p, ti = rng.choice(holders)
        n = rng.randint(1, counts[(p, ti)])
        counts[(p, ti)] -= n
        touched.add((p, ti))
        return f"{p} loses {n} {noun(things[ti], n)}."

    # transfer between two different people
    p, ti = rng.choice(holders)
    n = rng.randint(1, counts[(p, ti)])
    q = rng.choice([x for x in people if x != p])
    counts[(p, ti)] -= n
    counts[(q, ti)] = counts.get((q, ti), 0) + n
    touched.add((p, ti))
    touched.add((q, ti))
    if rng.random() < 0.5:
        return f"{p} gives {n} {noun(things[ti], n)} to {q}."
    return f"{q} takes {n} {noun(things[ti], n)} from {p}."


def make_question(rng, people, things, counts):
    p = rng.choice(people)
    ti = rng.randrange(len(things))
    c = counts.get((p, ti), 0)
    if rng.random() < 0.5:
        return f"How many {things[ti][0]} does {p} have?\t{c}"
    n = max(0, c + rng.choice([-1, 0, 1]))
    answer = "yes" if n == c else "no"
    return f"Does {p} have {n} {noun(things[ti], n)}?\t{answer}"


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n_stories):
        for i, line in enumerate(gen_story(rng), start=1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
