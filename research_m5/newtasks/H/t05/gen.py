"""Synthetic task: changing states.

People move between rooms and carry, switch, fill, empty, open and close
objects. Each object has a state (on/off, full/empty, open/closed) and a room.
Questions ask the state of things as it is at that point in the story.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ana", "ben", "cy", "dee"]
ROOMS = ["hall", "kitchen", "garden"]
OBJECTS = {
    "lamp": {"states": ("on", "off"),
             "verb": {"on": "switches the lamp on", "off": "switches the lamp off"}},
    "jar": {"states": ("full", "empty"),
            "verb": {"full": "fills the jar", "empty": "empties the jar"}},
    "cup": {"states": ("full", "empty"),
            "verb": {"full": "fills the cup", "empty": "empties the cup"}},
    "box": {"states": ("open", "closed"),
            "verb": {"open": "opens the box", "closed": "closes the box"}},
    "door": {"states": ("open", "closed"),
             "verb": {"open": "opens the door", "closed": "closes the door"}},
}


def cap(s):
    return s[0].upper() + s[1:]


def make_story(rng):
    people = rng.sample(NAMES, rng.randint(1, 3))
    kinds = rng.sample(sorted(OBJECTS), rng.randint(1, 3))
    loc = {}      # person or object -> room
    state = {}    # object -> current state
    changes = {}  # person or object -> number of changes so far

    # Introduction: every person and object is placed and every object gets a state.
    blocks = []
    for p in people:
        loc[p] = rng.choice(ROOMS)
        changes[p] = 0
        blocks.append([f"{cap(p)} is in the {loc[p]}."])
    for k in kinds:
        loc[k] = rng.choice(ROOMS)
        state[k] = rng.choice(OBJECTS[k]["states"])
        changes[k] = 0
        blocks.append([f"The {k} is in the {loc[k]}.", f"The {k} is {state[k]}."])
    rng.shuffle(blocks)
    items = [("S", t) for b in blocks for t in b]

    # Action statements and questions, interleaved after the introduction.
    intro_n = len(items)
    total = rng.randint(max(3, intro_n), 12)
    n_act = total - intro_n
    n_q = rng.randint(1, 5)
    slots = ["S"] * n_act + ["Q"] * n_q
    rng.shuffle(slots)

    for slot in slots:
        if slot == "S":
            acts = []
            for p in people:
                for r in ROOMS:
                    if r != loc[p]:
                        acts.append(("goes", p, None, r))
                for k in kinds:
                    if loc[k] == loc[p]:
                        for s in OBJECTS[k]["states"]:
                            if s != state[k]:
                                acts.append(("state", p, k, s))
                        for r in ROOMS:
                            if r != loc[k]:
                                acts.append(("carry", p, k, r))
            kind, p, k, x = rng.choice(acts)
            if kind == "goes":
                loc[p] = x
                changes[p] += 1
                items.append(("S", f"{cap(p)} goes to the {x}."))
            elif kind == "state":
                state[k] = x
                changes[k] += 1
                items.append(("S", f"{cap(p)} {OBJECTS[k]['verb'][x]}."))
            else:
                loc[p] = loc[k] = x
                changes[p] += 1
                changes[k] += 1
                items.append(("S", f"{cap(p)} carries the {k} to the {x}."))
        else:
            changed = [e for e in people + kinds if changes[e] > 0]
            pool = changed if changed else people + kinds
            qs = []
            for e in pool:
                if e in people:
                    qs.append(("where_p", e, None))
                else:
                    qs.append(("where_o", e, None))
                    for s in OBJECTS[e]["states"]:
                        qs.append(("is", e, s))
            for r in ROOMS:
                qs.append(("who", r, None))
            kind, a, b = rng.choice(qs)
            if kind == "where_p":
                text, ans = f"Where is {cap(a)}?", loc[a]
            elif kind == "where_o":
                text, ans = f"Where is the {a}?", loc[a]
            elif kind == "is":
                text = f"Is the {a} {b}?"
                ans = "yes" if state[a] == b else "no"
            else:
                names = sorted(p for p in people if loc[p] == a)
                text, ans = f"Who is in the {a}?", ",".join(names) or "none"
            items.append(("Q", text, ans))

    return items


def render(items):
    out = []
    for n, item in enumerate(items, 1):
        if item[0] == "S":
            out.append(f"{n} {item[1]}")
        else:
            out.append(f"{n} {item[1]}\t{item[2]}")
    return out


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(render(make_story(rng)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
