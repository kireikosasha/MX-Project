#!/usr/bin/env python3
"""Generate synthetic rule-based reading-comprehension stories.

Usage: python gen.py N SEED

Prints N stories to standard output. Each story is a list of numbered
sentences (numbering restarts at 1 for every story). Statements end with a
full stop. Questions end with a question mark, then a TAB, then the answer.

Each story states three general rules (about dogs, about the key and the
gate, and about things a person takes), plus facts about people, objects and
actions. Every answer is computed by simulating the world, so a question is
answered by applying the stated rules to the facts stated above it.
"""
import random
import sys

PEOPLE = ["ann", "bob", "cara", "dan"]
ANIMALS = ["rex", "tom"]
PLACES = ["hall", "kitchen", "garden", "shed"]
OTHER_OBJECTS = ["box", "ball"]
RULES = [
    "Every dog is an animal.",
    "Whoever has the key can open the gate.",
    "Whatever a person takes goes wherever the person goes.",
]


def cap(word):
    return word[0].upper() + word[1:]


def make_story(rng):
    people = rng.sample(PEOPLE, rng.randint(2, 3))
    objects = ["key", rng.choice(OTHER_OBJECTS)]
    loc = {p: rng.choice(PLACES) for p in people}      # where each person is
    place = {o: rng.choice(PLACES) for o in objects}   # where an object lies while nobody holds it
    holder = {o: None for o in objects}               # person holding each object, or None
    animal = rng.choice(ANIMALS)
    has_fact = rng.random() < 0.6

    init = [f"{cap(p)} is in the {loc[p]}." for p in people]
    init += [f"The {o} is in the {place[o]}." for o in objects]
    init += RULES
    if has_fact:
        init.append(f"{cap(animal)} is a dog.")
    rng.shuffle(init)

    def where(o):
        if holder[o] is not None:
            return loc[holder[o]]
        return place[o]

    def actions():
        acts = []
        for p in people:
            for L in PLACES:
                if L != loc[p]:
                    acts.append(("goes", p, L))
        for o in objects:
            h = holder[o]
            if h is None:
                for p in people:
                    if loc[p] == place[o]:
                        acts.append(("takes", p, o))
            else:
                for q in people:
                    if q != h and loc[q] == loc[h]:
                        acts.append(("gives", h, o, q))
        return acts

    def do(a):
        if a[0] == "goes":
            _, p, L = a
            loc[p] = L
            return f"{cap(p)} goes to the {L}."
        if a[0] == "takes":
            _, p, o = a
            holder[o] = p
            return f"{cap(p)} takes the {o}."
        _, h, o, q = a
        holder[o] = q
        return f"{cap(h)} gives the {o} to {cap(q)}."

    def candidates():
        cands = []
        for o in objects:
            cands.append((f"Where is the {o}?", where(o)))
        for p in people:
            cands.append((f"Where is {cap(p)}?", loc[p]))
        for L in sorted(set(loc[p] for p in people)):
            names = sorted(p for p in people if loc[p] == L)
            cands.append((f"Who is in the {L}?", ",".join(names)))
        if holder["key"] is not None:
            for p in people:
                ans = "yes" if holder["key"] == p else "no"
                cands.append((f"Can {cap(p)} open the gate?", ans))
        if has_fact:
            cands.append((f"Is {cap(animal)} an animal?", "yes"))
        return cands

    n_act = rng.randint(1, min(6, 12 - len(init)))
    n_q = rng.randint(1, 5)
    slots = ["A"] * n_act + ["Q"] * n_q
    rng.shuffle(slots)

    entries = list(init)
    asked = set()
    for slot in slots:
        if slot == "A":
            entries.append(do(rng.choice(actions())))
        else:
            cands = candidates()
            fresh = [c for c in cands if c[0] not in asked] or cands
            text, ans = rng.choice(fresh)
            asked.add(text)
            entries.append(f"{text}\t{ans}")
    return entries


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, entry in enumerate(make_story(rng), 1):
            out.append(f"{i} {entry}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
