#!/usr/bin/env python3
"""Synthetic possession-and-exchange reading task.

Usage: python gen.py N SEED

Prints N stories. Each statement is "<n> <Sentence>." and each question is
"<n> <Question>?<TAB><answer>". Numbers restart at 1 for each story.
Answers come from simulating the world, so they are never written by hand.
"""
import random
import sys

PEOPLE = ["ann", "ben", "cal", "dee"]
THINGS = ["ball", "cup", "hat", "key", "book"]


def cap(word):
    return word[0].upper() + word[1:]


class World:
    def __init__(self, people, things):
        self.people = list(people)
        self.things = list(things)
        self.holder = {t: None for t in self.things}   # thing -> person or None
        self.person_refs = {p: 0 for p in self.people}  # statements naming person
        self.thing_refs = {t: 0 for t in self.things}   # statements naming thing

    def held(self, p):
        return sorted(t for t in self.things if self.holder[t] == p)


def statement_options(world):
    """Legal statements in the current state, grouped by kind.

    Each option is (text, changes, people_named, things_named) where
    changes maps thing -> new holder (None means nobody).
    """
    groups = {"get": [], "give": [], "trade": [], "lose": []}
    for p in world.people:
        for t in world.things:
            h = world.holder[t]
            if h != p:
                if h is None:
                    text = f"{cap(p)} got the {t}."
                    named = [p]
                else:
                    text = f"{cap(p)} got the {t} from {cap(h)}."
                    named = [p, h]
                groups["get"].append((text, {t: p}, named, [t]))
            else:
                for q in world.people:
                    if q == p:
                        continue
                    text = f"{cap(p)} gave the {t} to {cap(q)}."
                    groups["give"].append((text, {t: q}, [p, q], [t]))
                groups["lose"].append((f"{cap(p)} lost the {t}.", {t: None}, [p], [t]))
    for t1 in world.things:
        p = world.holder[t1]
        if p is None:
            continue
        for t2 in world.things:
            q = world.holder[t2]
            if q is None or q == p or t2 == t1:
                continue
            text = f"{cap(p)} traded the {t1} to {cap(q)} for the {t2}."
            groups["trade"].append((text, {t1: q, t2: p}, [p, q], [t1, t2]))
    return {k: v for k, v in groups.items() if v}


def question_options(world):
    """Questions whose answers need at least two earlier statements about the subject."""
    qs = []
    for t in world.things:
        if world.thing_refs[t] < 2:
            continue
        holder = world.holder[t]
        qs.append((f"Who has the {t}", holder if holder else "nobody"))
        for p in world.people:
            answer = "yes" if holder == p else "no"
            qs.append((f"Does {cap(p)} have the {t}", answer))
    for p in world.people:
        if world.person_refs[p] < 2:
            continue
        held = world.held(p)
        qs.append((f"What does {cap(p)} have", ",".join(held) if held else "nothing"))
        qs.append((f"How many things does {cap(p)} have", str(len(held))))
    return qs


def attempt(rng, people, things, kinds):
    world = World(people, things)
    lines = []
    for kind in kinds:
        if kind == "S":
            groups = statement_options(world)
            group = rng.choice(sorted(groups))
            text, changes, named_people, named_things = rng.choice(groups[group])
            world.holder.update(changes)
            for p in named_people:
                world.person_refs[p] += 1
            for t in set(named_things):
                world.thing_refs[t] += 1
            lines.append(text)
        else:
            qs = question_options(world)
            if not qs:
                return None
            text, answer = rng.choice(qs)
            lines.append(f"{text}?\t{answer}")
    return lines


def make_story(rng):
    while True:
        people = rng.sample(PEOPLE, rng.randint(2, len(PEOPLE)))
        things = rng.sample(THINGS, rng.randint(2, len(THINGS)))
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        kinds = ["S"] * n_stmt + ["Q"] * n_q
        rng.shuffle(kinds)
        if kinds[:2] != ["S", "S"]:
            continue
        lines = attempt(rng, people, things, kinds)
        if lines is not None:
            return lines


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 2
    n = int(argv[1])
    rng = random.Random(int(argv[2]))
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
