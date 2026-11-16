"""Synthetic task generator: changing states (things switched on/off, opened/closed).

Usage: python gen.py N SEED
Prints N stories. Each story is a short sequence of statements about objects
whose states are changed by actions, with questions asked about the states at
that point in the story. Answers are computed by simulating the world.
"""
import random
import sys

NAMES = ["ann", "ben", "cal", "dan", "eva", "fay"]

# Each object has two opposite states and one action verb per state.
OBJECTS = {
    "lamp": {
        "states": ("on", "off"),
        "verbs": {"on": "turns on", "off": "turns off"},
    },
    "door": {
        "states": ("open", "closed"),
        "verbs": {"open": "opens", "closed": "closes"},
    },
    "box": {
        "states": ("open", "closed"),
        "verbs": {"open": "opens", "closed": "closes"},
    },
}

ALL_STATES = ("on", "off", "open", "closed")


def make_question(rng, state):
    """Ask about the state now. `state` maps established objects to states."""
    if rng.random() < 0.5:
        obj = rng.choice(sorted(state))
        s = rng.choice(OBJECTS[obj]["states"])
        question = f"Is the {obj} {s}?"
        answer = "yes" if state[obj] == s else "no"
    else:
        s = rng.choice(ALL_STATES)
        hits = sorted(o for o in state if state[o] == s)
        question = f"Which things are {s}?"
        answer = ",".join(hits) if hits else "none"
    return question, answer


def make_story(rng):
    n = rng.randint(3, 12)
    n_questions = rng.randint(1, min(5, n))
    question_after = set(rng.sample(range(1, n + 1), n_questions))
    objs = rng.sample(sorted(OBJECTS), rng.randint(1, len(OBJECTS)))

    state = {}  # only objects whose state has been established so far
    lines = []
    for k in range(1, n + 1):
        if k == 1:
            kind = "init"
        else:
            r = rng.random()
            kind = "init" if r < 0.2 else ("act" if r < 0.8 else "wait")

        if kind == "init":
            obj = rng.choice(objs)
            s = rng.choice(OBJECTS[obj]["states"])
            text = f"The {obj} is {s}."
            state[obj] = s
        elif kind == "act":
            obj = rng.choice(objs)
            s = rng.choice(OBJECTS[obj]["states"])
            text = f"{rng.choice(NAMES)} {OBJECTS[obj]['verbs'][s]} the {obj}."
            state[obj] = s
        else:
            text = f"{rng.choice(NAMES)} waits."
        lines.append(text)

        if k in question_after:
            lines.append(make_question(rng, state))

    out = []
    idx = 0
    for item in lines:
        idx += 1
        if isinstance(item, tuple):
            question, answer = item
            out.append(f"{idx} {question}\t{answer}")
        else:
            out.append(f"{idx} {item}")
    return out


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n_stories):
        out.extend(make_story(rng))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
