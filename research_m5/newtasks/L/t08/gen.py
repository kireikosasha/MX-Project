"""Synthetic reading task: places on a map and the directions between them.

Usage: python gen.py N SEED

Each story places 3-6 places on a map (a random north/south order and an
independent east/west order). Statements are true on that map. Questions are
answered from the statements above them only, under the rule that the
statements are the whole map: a fact holds if it follows from the statements
by inverting a direction or chaining same-axis directions. Every question's
pair of places is not stated directly, so each answer needs two or more
statements.
"""
import random
import sys

PLACES = ["park", "school", "lake", "shop", "farm", "castle", "bridge", "tower"]
DIRS = ["north", "south", "east", "west"]


def closure(edges):
    """Transitive closure of a set of (a, b) pairs meaning a is before b."""
    rel = set(edges)
    changed = True
    while changed:
        changed = False
        for a, b in list(rel):
            for c, d in list(rel):
                if b == c and a != d and (a, d) not in rel:
                    rel.add((a, d))
                    changed = True
    return rel


def candidates(k, prefix, names):
    """All unambiguous, multi-statement questions answerable from prefix.

    Returns a list of (question_text, answer).
    """
    north, east, direct = set(), set(), set()
    for s, d, o, _ in prefix:
        direct.add(frozenset((s, o)))
        if d == "north":
            north.add((s, o))
        elif d == "south":
            north.add((o, s))
        elif d == "east":
            east.add((s, o))
        else:
            east.add((o, s))
    n_rel = closure(north)
    e_rel = closure(east)
    rel = {
        "north": n_rel,
        "south": {(b, a) for a, b in n_rel},
        "east": e_rel,
        "west": {(b, a) for a, b in e_rel},
    }

    out = []
    # "What is <dir> of the X?" : exactly one place lies that way from X.
    for d in DIRS:
        for x in range(k):
            found = [y for y in range(k) if (y, x) in rel[d]]
            if len(found) == 1 and frozenset((found[0], x)) not in direct:
                out.append((f"What is {d} of the {names[x]}?", names[found[0]]))
    for a in range(k):
        for b in range(k):
            if a == b or frozenset((a, b)) in direct:
                continue
            # "Is the A <dir> of the B?" : yes if entailed, otherwise no.
            for d in DIRS:
                ans = "yes" if (a, b) in rel[d] else "no"
                out.append((f"Is the {names[a]} {d} of the {names[b]}?", ans))
            # "Which direction is the A from the B?" : all entailed directions.
            dirs = [d for d in DIRS if (a, b) in rel[d]]
            if dirs:
                out.append((f"Which direction is the {names[a]} from the {names[b]}?",
                            ",".join(sorted(dirs))))
    return out


def statement_text(names, s, d, o, pat):
    if pat == 1:
        return f"The {names[s]} is {d} of the {names[o]}."
    return f"The {names[s]} is to the {d} of the {names[o]}."


def make_story(rng):
    while True:
        k = rng.randint(3, 6)
        names = rng.sample(PLACES, k)
        # Positions: north = larger y, east = larger x. Every pair differs on both axes.
        coords = {"ns": rng.sample(range(k), k), "ew": rng.sample(range(k), k)}
        keys = [(i, j, ax) for i in range(k) for j in range(i + 1, k) for ax in ("ns", "ew")]
        rng.shuffle(keys)
        n_stmt = rng.randint(3, min(12, len(keys)))

        stmts = []
        for i, j, ax in keys[:n_stmt]:
            s, o = (i, j) if rng.random() < 0.5 else (j, i)
            pos = coords[ax]
            if ax == "ns":
                d = "north" if pos[s] > pos[o] else "south"
            else:
                d = "east" if pos[s] > pos[o] else "west"
            stmts.append((s, d, o, rng.choice((1, 2))))

        n_q = rng.randint(1, 5)
        slots = sorted(rng.randint(2, n_stmt) for _ in range(n_q))
        used = set()
        questions = []  # (slot, text, answer); slot = statements above the question
        ok = True
        for slot in slots:
            cands = [c for c in candidates(k, stmts[:slot], names) if c[0] not in used]
            if not cands:
                ok = False
                break
            text, ans = rng.choice(cands)
            used.add(text)
            questions.append((slot, text, ans))
        if not ok:
            continue

        lines = []
        for idx, (s, d, o, pat) in enumerate(stmts):
            lines.append(statement_text(names, s, d, o, pat))
            for slot, text, ans in questions:
                if slot == idx + 1:
                    lines.append(text + "\t" + ans)
        return [f"{i} {line}" for i, line in enumerate(lines, 1)]


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
