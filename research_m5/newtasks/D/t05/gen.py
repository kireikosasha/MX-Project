"""Synthetic reading-comprehension generator: changing states.

Objects have states (lamp: on/off, box: open/closed) and locations; people
move, take and put down objects, and switch states. Questions ask the
state of things at that point in the story, so each answer depends on
tracking the sequence of actions.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cy", "Dan"]
ROOMS = ["kitchen", "garden", "hall", "shed"]
STATES = {"lamp": ["on", "off"], "box": ["open", "closed"]}
# (verb phrase in the event sentence, resulting state)
SWITCH = {
    "lamp": [("turned on", "on"), ("turned off", "off")],
    "box": [("opened", "open"), ("closed", "closed")],
}


def make_story(rng):
    people = rng.sample(NAMES, rng.randint(2, 3))
    objects = rng.sample(sorted(STATES), rng.randint(1, 2))
    loc, oloc, holder, state = {}, {}, {}, {}

    init = []
    for p in people:
        loc[p] = rng.choice(ROOMS)
        init.append(f"{p} is in the {loc[p]}.")
    for o in objects:
        oloc[o] = rng.choice(ROOMS)
        holder[o] = None
        state[o] = rng.choice(STATES[o])
        init.append(f"The {o} is in the {oloc[o]}.")
        init.append(f"The {o} is {state[o]}.")
    rng.shuffle(init)
    lines = [(text, None) for text in init]

    def room_of(o):
        return loc[holder[o]] if holder[o] is not None else oloc[o]

    def event():
        cands = []
        for p in people:
            for r in ROOMS:
                if r != loc[p]:
                    cands.append(("walk", p, r))
        for o in objects:
            if holder[o] is None:
                for p in people:
                    if loc[p] == room_of(o):
                        cands.append(("take", p, o))
            else:
                cands.append(("put", holder[o], o))
            for p in people:
                for verb, s in SWITCH[o]:
                    cands.append(("switch", p, o, verb, s))
        c = rng.choice(cands)
        kind = c[0]
        if kind == "walk":
            _, p, r = c
            loc[p] = r
            return (f"{p} walked to the {r}.", None)
        if kind == "take":
            _, p, o = c
            holder[o] = p
            return (f"{p} took the {o}.", None)
        if kind == "put":
            _, p, o = c
            oloc[o] = loc[p]
            holder[o] = None
            return (f"{p} put the {o} down.", None)
        _, p, o, verb, s = c
        state[o] = s
        return (f"{p} {verb} the {o}.", None)

    def question(asked):
        cands = []
        for p in people:
            cands.append((f"Where is {p}?", loc[p]))
        for o in objects:
            cands.append((f"Where is the {o}?", room_of(o)))
            cands.append((f"Who has the {o}?",
                          holder[o].lower() if holder[o] is not None else "nobody"))
            for s in STATES[o]:
                cands.append((f"Is the {o} {s}?", "yes" if state[o] == s else "no"))
        cands = [c for c in cands if c[0] not in asked]
        return rng.choice(cands) if cands else None

    n_events = rng.randint(1, 5)
    slots = sorted(rng.randint(0, n_events) for _ in range(rng.randint(1, 5)))
    asked = set()
    k = 0
    for i in range(n_events + 1):
        while k < len(slots) and slots[k] == i:
            q = question(asked)
            if q is not None:
                lines.append(q)
                asked.add(q[0])
            k += 1
        if i < n_events:
            lines.append(event())
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for num, (text, answer) in enumerate(make_story(rng), start=1):
            if answer is None:
                print(f"{num} {text}")
            else:
                print(f"{num} {text}\t{answer}")


if __name__ == "__main__":
    main()
