#!/usr/bin/env python3
"""Synthetic reading-comprehension task: who holds what, with swaps.

Each person holds exactly one thing at a time. Facts either give a person
an item or say that two people who already hold something trade their
items. Questions ask what a person holds, who holds an item, or whether a
person holds an item. Answers come from simulating the facts above each
question, so the solver has to follow every trade in order.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eva", "Fred", "Gus", "Hana"]
ITEMS = ["key", "coin", "ball", "kite", "cup", "hat", "pen", "book"]


def hold_fact(rng, holder, free_names, free_items):
    person = free_names.pop()
    item = free_items.pop()
    holder[person] = item
    if rng.random() < 0.5:
        return "%s holds the %s." % (person, item)
    return "%s has the %s." % (person, item)


def swap_fact(rng, holder):
    a, b = rng.sample(sorted(holder), 2)
    holder[a], holder[b] = holder[b], holder[a]
    if rng.random() < 0.5:
        return "%s and %s swap their things." % (a, b)
    return "%s and %s trade what they hold." % (a, b)


def question(rng, holder):
    people = sorted(holder)
    kind = rng.randrange(3)
    if kind == 0:
        p = rng.choice(people)
        return "What does %s hold?" % p, holder[p]
    if kind == 1:
        item = rng.choice(sorted(holder.values()))
        owner = [p for p in people if holder[p] == item][0]
        return "Who holds the %s?" % item, owner.lower()
    p = rng.choice(people)
    others = [holder[q] for q in people if holder[q] != holder[p]]
    if others and rng.random() < 0.5:
        item = rng.choice(others)
        return "Does %s hold the %s?" % (p, item), "no"
    return "Does %s hold the %s?" % (p, holder[p]), "yes"


def make_story(rng):
    total = rng.randint(4, 12)                   # all lines, statements + questions
    nq = rng.randint(1, min(5, total - 3))       # at least 3 plain statements
    q_positions = set(rng.sample(range(1, total), nq))
    holder = {}
    free_names = list(NAMES)
    free_items = list(ITEMS)
    rng.shuffle(free_names)
    rng.shuffle(free_items)
    lines = []
    for i in range(total):
        if i in q_positions:
            text, answer = question(rng, holder)
            lines.append((text, answer))
        elif len(holder) < 2 or (rng.random() < 0.45 and free_names):
            lines.append((hold_fact(rng, holder, free_names, free_items), None))
        else:
            lines.append((swap_fact(rng, holder), None))
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for num, (text, answer) in enumerate(make_story(rng), start=1):
            if answer is None:
                out.append("%d %s" % (num, text))
            else:
                out.append("%d %s\t%s" % (num, text, answer))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
