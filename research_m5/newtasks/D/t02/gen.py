import random
import sys
from collections import namedtuple

NAMES = ["ann", "ben", "cal", "dee", "eve"]
THINGS = ["ball", "hat", "cup", "book", "pen"]

Stmt = namedtuple("Stmt", "text names things event")


def try_story(rng):
    people = rng.sample(NAMES, rng.randint(2, 4))
    things = rng.sample(THINGS, rng.randint(2, 4))
    total = rng.randint(3, 12)
    owner = {t: None for t in things}
    stmts = []
    snaps = [dict(owner)]  # snaps[j] = holdings after j statements

    # Initial possessions (only before any exchange).
    for t in things:
        if len(stmts) >= total:
            break
        if rng.random() < 0.6:
            p = rng.choice(people)
            owner[t] = p
            stmts.append(Stmt(f"{p} has a {t}.", {p}, {t}, False))
            snaps.append(dict(owner))

    # Exchanges: gives and swaps.
    while len(stmts) < total:
        held = [t for t in things if owner[t] is not None]
        if not held:
            return None
        p, q = rng.sample(people, 2)
        if rng.random() < 0.4 and any(owner[t] in (p, q) for t in things):
            for t in things:
                if owner[t] == p:
                    owner[t] = q
                elif owner[t] == q:
                    owner[t] = p
            stmts.append(Stmt(f"{p} swaps with {q}.", {p, q}, set(), True))
        else:
            t = rng.choice(held)
            giver = owner[t]
            receiver = rng.choice([x for x in people if x != giver])
            owner[t] = receiver
            stmts.append(Stmt(f"{giver} gives the {t} to {receiver}.",
                              {giver, receiver}, {t}, True))
        snaps.append(dict(owner))

    # Questions, placed after statement j (j >= 1).
    n = len(stmts)
    k = min(rng.randint(1, 5), n)
    qpos = sorted(rng.sample(range(1, n + 1), k))
    qs = {}
    for j in qpos:
        above = stmts[:j]
        if not any(s.event for s in above):
            return None
        question = make_question(rng, people, things, above, snaps[j])
        if question is None:
            return None
        qs[j] = question
    return stmts, qs


def make_question(rng, people, things, above, snap):
    options = []
    for t in things:
        if sum(t in s.things for s in above) >= 2:
            options.append(("who", None, t))
    for p in people:
        if sum(p in s.names for s in above) >= 2:
            options.append(("what", p, None))
        for t in things:
            if (sum(t in s.things for s in above) >= 1
                    and sum(p in s.names for s in above) >= 1
                    and sum((p in s.names) or (t in s.things) for s in above) >= 2):
                options.append(("does", p, t))
    if not options:
        return None
    kind, p, t = rng.choice(options)
    if kind == "who":
        holder = snap[t]
        return f"who has the {t}?", holder if holder else "nobody"
    if kind == "what":
        mine = sorted(x for x, o in snap.items() if o == p)
        return f"what does {p} have?", ",".join(mine) if mine else "nothing"
    answer = "yes" if snap[t] == p else "no"
    return f"does {p} have the {t}?", answer


def render(stmts, qs):
    lines = []
    num = 0
    for j, s in enumerate(stmts, start=1):
        num += 1
        lines.append(f"{num} {s.text}")
        if j in qs:
            num += 1
            text, answer = qs[j]
            lines.append(f"{num} {text}\t{answer}")
    return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        story = None
        while story is None:
            story = try_story(rng)
        out.extend(render(*story))
    print("\n".join(out))


if __name__ == "__main__":
    main()
