import random
import sys

# Synthetic task: things (gates, valves) get pushed into a state by actions.
# Questions ask the state now, a yes/no check of a state, or who last
# changed a thing. The answer depends on the latest state-changing sentence.

NAMES = ["ana", "ben", "cy", "dan", "eve", "fay"]
THINGS = ["gate", "valve"]
# action verb -> (thing it acts on, state it leaves the thing in)
CHANGES = {
    "lifts": ("gate", "raised"),
    "drops": ("gate", "lowered"),
    "jams": ("valve", "stuck"),
    "frees": ("valve", "free"),
}
IDLE = "watches"  # does not change any state
STATES = {"gate": ["raised", "lowered"], "valve": ["stuck", "free"]}


def candidates(rng, snap, used):
    """Questions that can be answered from the snapshot of the world."""
    out = []
    for thing in sorted(snap):
        state, actor = snap[thing]
        out.append((f"how is the {thing} now?", state))
        out.append((f"who last handled the {thing}?", actor))
        guess = rng.choice(STATES[thing])
        out.append((f"is the {thing} {guess}?", "yes" if guess == state else "no"))
    return [c for c in out if c[0] not in used]


def make_story(rng):
    while True:
        n = rng.randint(3, 12)
        stmts = []
        snaps = []  # world snapshot after each statement
        snap = {}   # thing -> (state, last actor)
        for _ in range(n):
            name = rng.choice(NAMES)
            if rng.random() < 0.25:
                stmts.append(f"{name} {IDLE} the {rng.choice(THINGS)}.")
            else:
                verb = rng.choice(sorted(CHANGES))
                thing, state = CHANGES[verb]
                stmts.append(f"{name} {verb} the {thing}.")
                snap[thing] = (state, name)
            snaps.append(dict(snap))
        if not snap:
            continue

        asks = {i: [] for i in range(1, n + 1)}  # questions after statement i
        used = set()
        target = rng.randint(1, 5)
        for _ in range(60):
            if sum(len(v) for v in asks.values()) >= target:
                break
            p = rng.randint(1, n)
            cands = candidates(rng, snaps[p - 1], used)
            if not cands:
                continue
            text, answer = rng.choice(cands)
            used.add(text)
            asks[p].append(f"{text}\t{answer}")
        if sum(len(v) for v in asks.values()) == 0:
            continue

        lines = []
        for i in range(1, n + 1):
            lines.append(stmts[i - 1])
            lines.extend(asks[i])
        return [f"{k} {line}" for k, line in enumerate(lines, 1)]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    count = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(count):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
