import random
import sys

# Changing states: objects have a state (open/closed, on/off, full/empty).
# Actions change the state; questions ask the state now, or count things in a state.

NAMES = ["Ann", "Bob", "Dan", "Eve", "Fay", "Gus"]

# (object, possible states, [(action verb phrase, state it causes), ...])
KINDS = [
    ("box", ["open", "closed"], [("opens", "open"), ("closes", "closed")]),
    ("door", ["open", "closed"], [("opens", "open"), ("closes", "closed")]),
    ("lamp", ["on", "off"], [("turns on", "on"), ("turns off", "off")]),
    ("jar", ["full", "empty"], [("fills", "full"), ("empties", "empty")]),
]

ALL_STATES = ["open", "closed", "on", "off", "full", "empty"]


def make_story(rng):
    kinds = rng.sample(KINDS, rng.randint(2, 4))
    objs = {}
    lines = []  # statements end with '.'; questions are "question?<TAB>answer"

    # Introduce each object with its starting state (one of each kind at most,
    # so "the box" always means one thing).
    for obj, states, acts in kinds:
        state = rng.choice(states)
        objs[obj] = {"states": states, "acts": acts, "state": state, "changed": False}
        lines.append(f"The {obj} is {state}.")

    # Total statements = introductions + changes, kept within 3..12.
    n_changes = rng.randint(1, 12 - len(kinds))
    n_questions = rng.randint(1, 5)
    events = ["C"] * n_changes + ["Q"] * n_questions
    rng.shuffle(events)

    for ev in events:
        if ev == "C":
            obj = rng.choice(list(objs))
            verb, state = rng.choice(objs[obj]["acts"])
            name = rng.choice(NAMES)
            objs[obj]["state"] = state
            objs[obj]["changed"] = True
            lines.append(f"{name} {verb} the {obj}.")
        else:
            # Prefer objects that have changed, so the answer needs several sentences.
            changed = [o for o in objs if objs[o]["changed"]]
            if changed and rng.random() < 0.8:
                obj = rng.choice(changed)
            else:
                obj = rng.choice(list(objs))
            kind = rng.randint(0, 2)
            if kind == 0:
                q = rng.choice([f"What is the state of the {obj} now",
                                f"What is the state of the {obj}"])
                ans = objs[obj]["state"]
            elif kind == 1:
                s = rng.choice(objs[obj]["states"])
                q = f"Is the {obj} {s}"
                ans = "yes" if objs[obj]["state"] == s else "no"
            else:
                s = rng.choice(ALL_STATES)
                q = f"How many things are {s}"
                ans = str(sum(1 for o in objs.values() if o["state"] == s))
            lines.append(f"{q}?\t{ans}")

    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, text in enumerate(make_story(rng), 1):
            out.append(f"{i} {text}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
