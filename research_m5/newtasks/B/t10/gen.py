"""Synthetic group-membership reading task.

People join and leave groups (clubs, crews). Questions ask which group a
person is in now, whether two people share a group, how many people are in
a group, and who is in a group. Answers come from simulating the events.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cara", "Dan", "Eve", "Finn", "Gus", "Hana"]
GROUPS = ["owls", "foxes", "hawks", "bears"]

JOIN = ["{p} joins the {g}.", "{p} signs up for the {g}."]
LEAVE = ["{p} leaves the {g}.", "{p} quits the {g}."]
Q_WHICH = ["Which group is {p} in?"]
Q_SAME = ["Are {p} and {q} in the same group?", "Is {p} in the same group as {q}?"]
Q_COUNT = ["How many are in the {g}?"]
Q_WHO = ["Who is in the {g}?"]


def pick_question(rng, snap):
    """Choose a question answerable from the prefix state `snap`.

    snap = (state, person_events, group_events): state maps a person to the
    group they are in now; the event counters record how many statements
    so far mention each person or group. Returns (text, answer) or None.
    """
    state, pev, gev = snap
    members = sorted(state)
    options = []
    for p in members:
        if pev.get(p, 0) >= 2:
            options.append(("which", p))
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            if pev.get(a, 0) + pev.get(b, 0) >= 3:
                options.append(("same", a, b))
    for g in sorted(gev):
        if gev[g] < 2:
            continue
        inside = [p for p in members if state[p] == g]
        if inside:
            options.append(("count", g))
        if len(inside) >= 2:
            options.append(("who", g))
    if not options:
        return None

    opt = rng.choice(options)
    if opt[0] == "which":
        p = opt[1]
        return rng.choice(Q_WHICH).format(p=p), state[p]
    if opt[0] == "same":
        a, b = opt[1], opt[2]
        if rng.random() < 0.5:
            a, b = b, a
        ans = "yes" if state[a] == state[b] else "no"
        return rng.choice(Q_SAME).format(p=a, q=b), ans
    g = opt[1]
    if opt[0] == "count":
        n = sum(1 for p in members if state[p] == g)
        return rng.choice(Q_COUNT).format(g=g), str(n)
    names = sorted(p.lower() for p in members if state[p] == g)
    return rng.choice(Q_WHO).format(g=g), ",".join(names)


def build_story(rng):
    """Return a list of output lines for one story, or None to retry."""
    names = rng.sample(NAMES, rng.randint(3, 5))
    groups = rng.sample(GROUPS, rng.randint(2, 3))
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)

    state = {}
    person_ev = {}
    group_ev = {}
    snaps = [({}, {}, {})]
    texts = []
    for _ in range(n_st):
        inside = sorted(state)
        outside = [p for p in names if p not in state]
        if inside and (not outside or rng.random() < 0.4):
            p = rng.choice(inside)
            g = state.pop(p)
            texts.append(rng.choice(LEAVE).format(p=p, g=g))
        else:
            p = rng.choice(outside)
            g = rng.choice(groups)
            state[p] = g
            texts.append(rng.choice(JOIN).format(p=p, g=g))
        person_ev[p] = person_ev.get(p, 0) + 1
        group_ev[g] = group_ev.get(g, 0) + 1
        snaps.append((dict(state), dict(person_ev), dict(group_ev)))

    positions = sorted(rng.randint(2, n_st) for _ in range(n_q))
    questions = {}
    asked = set()
    for pos in positions:
        q = pick_question(rng, snaps[pos])
        if q is None or q[0] in asked:
            return None
        asked.add(q[0])
        questions.setdefault(pos, []).append(q)

    lines = []
    for i in range(1, n_st + 1):
        lines.append(texts[i - 1])
        for text, ans in questions.get(i, []):
            lines.append(text + "\t" + ans)
    return ["%d %s" % (k + 1, line) for k, line in enumerate(lines)]


def make_story(rng):
    while True:
        story = build_story(rng)
        if story is not None:
            return story


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
