#!/usr/bin/env python3
"""Synthetic reading-comprehension generator: changing states.

Lamps are switched on, switched off or flipped; boxes are opened or closed.
Each story first states the initial state of 2-3 objects, then applies
state-changing actions. Questions ask the current state of an object,
computed by simulating the story up to the question's position.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cal", "Dee"]
LAMP_COLORS = ["red", "green", "blue"]
BOX_SIZES = ["big", "small"]


def make_story(rng):
    objects = [("lamp", c) for c in LAMP_COLORS] + [("box", s) for s in BOX_SIZES]
    intro_objs = rng.sample(objects, rng.choice([2, 3]))
    n_intro = len(intro_objs)

    # state[(kind, name)] is True for a lamp that is on, or a box that is open.
    state = {}
    stmts = []       # statement texts, in order
    stmt_objs = []   # object each statement is about
    snapshots = [{}]  # snapshots[k] = state after the first k statements

    for key in intro_objs:
        kind, name = key
        on = rng.random() < 0.5
        state[key] = on
        if kind == "lamp":
            word = "on" if on else "off"
        else:
            word = "open" if on else "closed"
        stmts.append(f"The {name} {kind} is {word}.")
        stmt_objs.append(key)
        snapshots.append(dict(state))

    n_act = rng.randint(1, 12 - n_intro)
    for _ in range(n_act):
        key = rng.choice(intro_objs)
        kind, name = key
        actor = rng.choice(NAMES)
        if kind == "lamp":
            act = rng.choice(["on", "off", "flip"])
            if act == "flip":
                state[key] = not state[key]
                text = f"{actor} flips the {name} lamp."
            else:
                state[key] = (act == "on")
                text = f"{actor} turns the {name} lamp {act}."
        else:
            act = rng.choice(["open", "close"])
            state[key] = (act == "open")
            verb = "opens" if act == "open" else "closes"
            text = f"{actor} {verb} the {name} box."
        stmts.append(text)
        stmt_objs.append(key)
        snapshots.append(dict(state))

    s = len(stmts)
    q_count = rng.randint(1, 5)
    positions = sorted(rng.randint(n_intro, s) for _ in range(q_count))

    questions_at = {}
    for pos in positions:
        # Prefer objects whose state was changed by an action above the question.
        touched = set(stmt_objs[n_intro:pos])
        cands = [o for o in intro_objs if o in touched] or intro_objs
        key = rng.choice(cands)
        kind, name = key
        st = snapshots[pos][key]
        if rng.random() < 0.5:
            if kind == "lamp":
                q = f"Is the {name} lamp on?"
                a = "yes" if st else "no"
            else:
                q = f"Is the {name} box open?"
                a = "yes" if st else "no"
        else:
            q = f"What is the state of the {name} {kind}?"
            if kind == "lamp":
                a = "on" if st else "off"
            else:
                a = "open" if st else "closed"
        questions_at.setdefault(pos, []).append(f"{q}\t{a}")

    items = []
    for i in range(s + 1):
        items.extend(questions_at.get(i, []))
        if i < s:
            items.append(stmts[i])
    return items


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        items = make_story(rng)
        for k, item in enumerate(items, 1):
            out.append(f"{k} {item}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
