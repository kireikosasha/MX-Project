"""Generate synthetic chain-of-relations reading tasks.

Usage: python gen.py N SEED

Stories describe people linked by two relations: "parent of" (family) and
"boss of" (work). Questions need two relation links followed in a row,
for example "Who is the grandparent of X?" or "Who is the boss of the parent
of X?". Answers are computed from a simulation of the statements above each
question. Lists are written as lowercase names separated by commas.
"""
import random
import sys

NAMES = ["Ann", "Bob", "Carl", "Dana", "Eve", "Fred", "Gina", "Hank", "Ivy", "Jack"]


def up(edges, x):
    """All nodes above x: u such that there is a chain u -> ... -> x."""
    seen = set()
    stack = [x]
    while stack:
        cur = stack.pop()
        for u, v in edges:
            if v == cur and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


def parents(world, x):
    return {p for (p, c) in world["parent"] if c == x}


def bosses(world, x):
    return {b for (b, w) in world["boss"] if w == x}


def grandparents(world, x):
    return {g for p in parents(world, x) for g in parents(world, p)}


def boss_of_parents(world, x):
    return {b for p in parents(world, x) for b in bosses(world, p)}


def parent_of_bosses(world, x):
    return {p for b in bosses(world, x) for p in parents(world, b)}


def boss_of_bosses(world, x):
    return {bb for b in bosses(world, x) for bb in bosses(world, b)}


WHO_TEMPLATES = [
    ("Who is the grandparent of {x}?", grandparents),
    ("Who is the boss of the parent of {x}?", boss_of_parents),
    ("Who is the parent of the boss of {x}?", parent_of_bosses),
    ("Who is the boss of the boss of {x}?", boss_of_bosses),
]

YN_TEMPLATES = [
    # "Is x the grandparent of y?" -> x is in grandparents(y)
    ("Is {x} the grandparent of {y}?", lambda world, x, y: x in grandparents(world, y)),
    # "Is x the boss of the parent of y?" -> x is in boss_of_parents(y)
    ("Is {x} the boss of the parent of {y}?", lambda world, x, y: x in boss_of_parents(world, y)),
]


def add_fact(rng, world, names):
    """Add one new fact to the world and return its sentence, or None."""
    for _ in range(60):
        kind = rng.choice(["parent", "boss"])
        a, b = rng.sample(names, 2)
        key = "parent" if kind == "parent" else "boss"
        edges = world[key]
        # a is above b in this relation. Reject duplicates and cycles.
        if (a, b) in edges or b in up(edges, a):
            continue
        edges.add((a, b))
        if kind == "parent":
            text = rng.choice(["{a} is the parent of {b}.", "{b} is a child of {a}."])
        else:
            text = rng.choice(["{a} is the boss of {b}.", "{b} works for {a}."])
        return text.format(a=a, b=b)
    return None


def make_question(rng, world, names):
    """Return (question_text, answer) computed from the current world, or None."""
    for _ in range(40):
        if rng.random() < 0.7:
            template, fn = rng.choice(WHO_TEMPLATES)
            x = rng.choice(names)
            ans = fn(world, x)
            if not ans:
                continue
            return template.format(x=x), ",".join(sorted(n.lower() for n in ans))
        template, fn = rng.choice(YN_TEMPLATES)
        x, y = rng.sample(names, 2)
        yes = fn(world, x, y)
        return template.format(x=x, y=y), ("yes" if yes else "no")
    return None


def make_story(rng):
    while True:
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        names = rng.sample(NAMES, rng.randint(4, 7))
        world = {"parent": set(), "boss": set()}
        lines = []
        stmts = 0
        qs = 0
        while stmts < n_stmt or qs < n_q:
            want_q = qs < n_q and stmts >= 2 and (stmts >= n_stmt or rng.random() < 0.4)
            if want_q:
                q = make_question(rng, world, names)
                if q is not None:
                    lines.append(q)
                    qs += 1
                    continue
            if stmts < n_stmt:
                s = add_fact(rng, world, names)
                if s is not None:
                    lines.append((s, None))
                    stmts += 1
                    continue
            break
        if stmts == n_stmt and qs >= 1:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = make_story(rng)
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
