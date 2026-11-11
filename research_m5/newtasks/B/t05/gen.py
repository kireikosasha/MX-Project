"""Synthetic state-tracking reading task.

People move between rooms. Devices sit in fixed rooms and switch between a
few states (lamp on/off, door open/closed, box colour). A person can only
change a device while standing in the device's room. Questions ask where a
person is, whether a device is in a state (yes/no), the box colour, or who is
in a room. Every answer is computed by simulating the statements above it.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cy", "Dee"]
ROOMS = ["hall", "kitchen", "garden"]
DEV_ROOM = {"lamp": "kitchen", "door": "hall", "box": "garden"}
DEV_STATES = {
    "lamp": ["on", "off"],
    "door": ["open", "closed"],
    "box": ["white", "red", "blue", "green"],
}


def act_text(person, dev, state):
    if dev == "lamp":
        return f"{person} switches the lamp {state}."
    if dev == "door":
        verb = "opens" if state == "open" else "closes"
        return f"{person} {verb} the door."
    return f"{person} paints the box {state}."


def statement(rng, people, loc, dev):
    """Pick a valid random statement, apply it to the world, return its text."""
    cands = []
    for p in people:
        for r in ROOMS:
            if loc.get(p) != r:
                cands.append(("move", p, r))
    for d, states in DEV_STATES.items():
        for s in states:
            cands.append(("fact", d, s))
    for p in people:
        if p in loc:
            for d, states in DEV_STATES.items():
                if DEV_ROOM[d] == loc[p]:
                    for s in states:
                        cands.append(("act", p, d, s))

    kind, a, b, *rest = rng.choice(cands)
    if kind == "move":
        p, r = a, b
        if p in loc:
            text = f"{p} goes to the {r}."
        else:
            text = f"{p} {rng.choice(['is in the', 'goes to the'])} {r}."
        loc[p] = r
        return text
    if kind == "fact":
        d, s = a, b
        dev[d] = s
        return f"The {d} is {s}."
    p, d, s = a, b, rest[0]
    dev[d] = s
    return act_text(p, d, s)


def ask(rng, people, loc, dev):
    """Return a question line 'Question?<TAB>answer' answerable now, or None."""
    placed = [p for p in people if p in loc]
    kinds = []
    if placed:
        kinds.append("where")
    devs = [d for d in ("lamp", "door") if d in dev]
    if devs:
        kinds.append("state")
    if "box" in dev:
        kinds.append("colour")
    if len(placed) == len(people):
        kinds.append("who")
    if not kinds:
        return None

    k = rng.choice(kinds)
    if k == "where":
        p = rng.choice(placed)
        return f"Where is {p}?\t{loc[p]}"
    if k == "state":
        d = rng.choice(devs)
        s = rng.choice(DEV_STATES[d])
        ans = "yes" if dev[d] == s else "no"
        return f"Is the {d} {s}?\t{ans}"
    if k == "colour":
        return f"What colour is the box?\t{dev['box']}"
    r = rng.choice(ROOMS)
    names = sorted(p.lower() for p in people if loc[p] == r)
    ans = ",".join(names) if names else "nobody"
    return f"Who is in the {r}?\t{ans}"


def build_story(rng):
    """Return a list of lines (no numbers), or None if this attempt failed."""
    people = rng.sample(NAMES, rng.choice([2, 3]))
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    loc, dev = {}, {}
    lines = []
    st = q = 0
    while st < n_st or q < n_q:
        if q < n_q and st >= 1 and (st == n_st or rng.random() < 0.4):
            qa = ask(rng, people, loc, dev)
            if qa is not None:
                lines.append(qa)
                q += 1
                continue
            if st == n_st:
                return None
        lines.append(statement(rng, people, loc, dev))
        st += 1
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
        for i, item in enumerate(story, 1):
            out.append(f"{i} {item}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
