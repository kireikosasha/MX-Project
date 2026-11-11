#!/usr/bin/env python3
"""Synthetic container-tracking reading task.

Usage: python gen.py N SEED

Prints N short stories about people moving things into and out of containers
and carrying containers between places. Every story is produced by simulating
a small world; question answers are read from that simulation.

Line format: "<number> <sentence>". Statements end with '.'. Questions end with
'?', then a TAB, then the answer. Numbers restart at 1 for every story.
"""
import random
import sys

NAMES = ["Sam", "Mia", "Tom", "Ana"]
THINGS = ["key", "ball", "coin", "book", "apple"]
CONTAINERS = ["box", "bag", "jar", "basket"]
PLACES = ["kitchen", "garden", "hall", "office"]

MAX_STATEMENTS = 8
MAX_QUESTIONS = 4


class World:
    def __init__(self, containers, things):
        self.containers = containers
        self.things = things
        self.box_at = {}    # container -> place
        self.inside = {}    # thing -> container, or None when loose
        self.loose_at = {}  # thing -> place, used only when loose

    def place_of(self, thing):
        c = self.inside[thing]
        if c is not None:
            return self.box_at[c]
        return self.loose_at[thing]

    def contents(self, c):
        return sorted(t for t in self.things if self.inside[t] == c)


def random_action(rng, w):
    """Apply one random legal action to the world and return (sentence, None)."""
    loose = [t for t in w.things if w.inside[t] is None]
    contained = [t for t in w.things if w.inside[t] is not None]
    options = ["move"]
    if loose:
        options.append("put")
    if contained:
        options.append("take")
    op = rng.choice(options)
    actor = rng.choice(NAMES)

    if op == "put":
        t = rng.choice(loose)
        c = rng.choice(w.containers)
        w.inside[t] = c
        w.loose_at.pop(t, None)
        return f"{actor} put the {t} in the {c}.", None

    if op == "take":
        t = rng.choice(contained)
        c = w.inside[t]
        w.inside[t] = None
        w.loose_at[t] = w.box_at[c]
        return f"{actor} took the {t} out of the {c}.", None

    c = rng.choice(w.containers)
    p = rng.choice([x for x in PLACES if x != w.box_at[c]])
    w.box_at[c] = p
    return f"{actor} moved the {c} to the {p}.", None


def make_question(rng, w):
    """Return (question_text, answer, derived). derived means the answer needs
    at least two statements: a thing's container and that container's place."""
    while True:
        kind = rng.choice(["where_t", "where_c", "in_c", "in_p", "count", "list"])

        if kind == "where_t":
            t = rng.choice(w.things)
            return f"Where is the {t}?", w.place_of(t), w.inside[t] is not None

        if kind == "where_c":
            c = rng.choice(w.containers)
            return f"Where is the {c}?", w.box_at[c], False

        if kind == "in_c":
            t = rng.choice(w.things)
            c = rng.choice(w.containers)
            ans = "yes" if w.inside[t] == c else "no"
            return f"Is the {t} in the {c}?", ans, False

        if kind == "in_p":
            ent = rng.choice(w.things + w.containers)
            if ent in w.inside:
                here = w.place_of(ent)
            else:
                here = w.box_at[ent]
            p = here if rng.random() < 0.5 else rng.choice(PLACES)
            ans = "yes" if p == here else "no"
            derived = ent in w.inside and w.inside[ent] is not None
            return f"Is the {ent} in the {p}?", ans, derived

        if kind == "count":
            c = rng.choice(w.containers)
            return f"How many things are in the {c}?", str(len(w.contents(c))), False

        if kind == "list":
            c = rng.choice(w.containers)
            items = w.contents(c)
            if not items:
                continue
            return f"What is in the {c}?", ",".join(items), False


def build_story(rng):
    """Return a list of (text, answer) pairs, or None if the attempt is rejected."""
    containers = rng.sample(CONTAINERS, rng.choice([1, 2, 3]))
    things = rng.sample(THINGS, rng.choice([1, 2, 3]))
    w = World(containers, things)
    lines = []

    # Introductions: containers first, then things (loose or inside a container).
    for c in containers:
        p = rng.choice(PLACES)
        w.box_at[c] = p
        lines.append((f"The {c} is in the {p}.", None))
    for t in things:
        if rng.random() < 0.5:
            c = rng.choice(containers)
            w.inside[t] = c
            lines.append((f"The {t} is in the {c}.", None))
        else:
            p = rng.choice(PLACES)
            w.inside[t] = None
            w.loose_at[t] = p
            lines.append((f"The {t} is in the {p}.", None))

    intros = len(lines)
    total = rng.randint(intros + 2, MAX_STATEMENTS)
    n_actions = total - intros
    n_questions = rng.randint(1, MAX_QUESTIONS)
    slots = ["A"] * n_actions + ["Q"] * n_questions
    rng.shuffle(slots)

    has_derived = False
    for slot in slots:
        if slot == "A":
            lines.append(random_action(rng, w))
        else:
            text, ans, derived = make_question(rng, w)
            has_derived = has_derived or derived
            lines.append((text, ans))

    if not has_derived:
        return None
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = build_story(rng)
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
