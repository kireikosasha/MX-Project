import random
import sys

NAMES = ["ann", "bob", "cat", "dan", "eve", "fay"]
GROUPS = ["red", "blue", "green", "gold"]


def random_event(rng, state, names, groups):
    actions = []
    for n in names:
        cur = state[n]
        if cur is None:
            for g in groups:
                actions.append(("join", n, g, None))
        else:
            actions.append(("leave", n, cur, None))
            for h in groups:
                if h != cur:
                    actions.append(("move", n, cur, h))
    return rng.choice(actions)


def event_text(ev):
    kind, n, a, b = ev
    N = n.capitalize()
    if kind == "join":
        return f"{N} joins {a}."
    if kind == "leave":
        return f"{N} leaves {a}."
    return f"{N} moves from {a} to {b}."


def apply_event(state, ev):
    kind, n, a, b = ev
    if kind == "join":
        state[n] = a
    elif kind == "leave":
        state[n] = None
    else:
        state[n] = b


def mentions_name(ev, n):
    return ev[1] == n


def mentions_group(ev, g):
    return g in (ev[2], ev[3])


def make_question(rng, state, hist, names, groups):
    for _ in range(40):
        kind = rng.choice(["which", "isin", "who", "how"])
        if kind in ("which", "isin"):
            n = rng.choice(names)
            if sum(mentions_name(e, n) for e in hist) < 2:
                continue
            N = n.capitalize()
            if kind == "which":
                return f"Which group is {N} in?", (state[n] or "none")
            g = rng.choice(groups)
            return f"Is {N} in {g}?", ("yes" if state[n] == g else "no")
        g = rng.choice(groups)
        if sum(mentions_group(e, g) for e in hist) < 2:
            continue
        members = [n for n in names if state[n] == g]
        if kind == "who":
            return f"Who is in {g}?", (",".join(sorted(members)) or "none")
        return f"How many are in {g}?", str(len(members))
    return None


def try_story(rng):
    names = rng.sample(NAMES, rng.randint(3, 4))
    groups = rng.sample(GROUPS, rng.randint(2, 4))
    total = rng.randint(3, 12)
    q_count = min(rng.randint(1, 5), total - 1)
    points = set(rng.sample(range(2, total + 1), q_count))
    state = {n: None for n in names}
    hist = []
    lines = []
    for k in range(1, total + 1):
        ev = random_event(rng, state, names, groups)
        apply_event(state, ev)
        hist.append(ev)
        lines.append(event_text(ev))
        if k in points:
            q = make_question(rng, state, hist, names, groups)
            if q is None:
                return None
            lines.append(f"{q[0]}\t{q[1]}")
    return lines


def gen_story(rng):
    while True:
        story = try_story(rng)
        if story:
            return story


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(gen_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
