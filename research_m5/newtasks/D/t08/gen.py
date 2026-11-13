#!/usr/bin/env python3
"""Synthetic places-on-a-map reading task (own vocabulary, bAbI-style format).

Usage: python gen.py N SEED

Each story places a few named places on a hidden grid and states true
relations "A is north of B" (also south, east, west, in two sentence
patterns). A question is answered only from the statements above it, by
closing the stated relations under transitivity along each axis
(north/south and east/west). A question is kept only when its answer is
logically settled and needs at least two statements to derive.
"""
import random
import sys

PLACES = ["park", "school", "shop", "lake", "farm", "mill", "hill", "bank"]
DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}

STATEMENT_PATTERNS = [
    "The {a} is {d} of the {b}",
    "The {a} lies to the {d} of the {b}",
]
LIST_PATTERNS = [
    "What is {d} of the {x}",
    "What lies to the {d} of the {x}",
]
YESNO_PATTERN = "Is the {a} {d} of the {b}"
DIR_PATTERN = "Which direction is the {a} from the {b}"


def canon(a, d, b):
    """Stored form of 'a is d of b': (axis, x, y) meaning x is further north/east than y."""
    if d == "north":
        return ("north", a, b)
    if d == "south":
        return ("north", b, a)
    if d == "east":
        return ("east", a, b)
    return ("east", b, a)  # west


def closure(facts):
    """Transitive closure of the stated north and east relations."""
    cl = {}
    for axis in ("north", "east"):
        pairs = {(x, y) for (ax, x, y) in facts if ax == axis}
        changed = True
        while changed:
            changed = False
            for (x, y) in list(pairs):
                for (u, v) in list(pairs):
                    if y == u and x != v and (x, v) not in pairs:
                        pairs.add((x, v))
                        changed = True
        cl[axis] = pairs
    return cl


def holds(cl, d, a, b):
    """True if 'a is d of b' follows from the closed relations."""
    axis, x, y = canon(a, d, b)
    return (x, y) in cl[axis]


def settled(cl, d, a, b):
    """True if the follow-up relation 'a is d of b' or its opposite is derivable."""
    return holds(cl, d, a, b) or holds(cl, OPP[d], a, b)


def candidates(facts, places):
    """All questions (kind, a, b, d, answer) answerable from the given canonical facts."""
    cl = closure(facts)
    out = []
    # "What is d of x?" -- every other place must be settled, the list must be
    # non-empty, and at least one item must need a chain of statements.
    for x in places:
        others = [p for p in places if p != x]
        for d in DIRS:
            if not all(settled(cl, d, p, x) for p in others):
                continue
            full = sorted(p for p in others if holds(cl, d, p, x))
            direct = sorted(p for p in others if canon(p, d, x) in facts)
            if full and full != direct:
                out.append(("list", None, x, d, ",".join(full)))
    for a in places:
        for b in places:
            if a == b:
                continue
            # "Is a d of b?" -- yes if derivable, no if the opposite is derivable.
            for d in DIRS:
                if holds(cl, d, a, b):
                    if canon(a, d, b) not in facts:
                        out.append(("yesno", a, b, d, "yes"))
                elif holds(cl, OPP[d], a, b) and canon(a, OPP[d], b) not in facts:
                    out.append(("yesno", a, b, d, "no"))
            # "Which direction is a from b?" -- both axes settled, answer = directions holding.
            if settled(cl, "north", a, b) and settled(cl, "east", a, b):
                full = sorted(d for d in DIRS if holds(cl, d, a, b))
                direct = sorted(d for d in DIRS if canon(a, d, b) in facts)
                if full != direct:
                    out.append(("dir", a, b, None, ",".join(full)))
    return out


def render_question(rng, cand):
    kind, a, b, d, ans = cand
    if kind == "list":
        text = rng.choice(LIST_PATTERNS).format(d=d, x=b)
    elif kind == "yesno":
        text = YESNO_PATTERN.format(a=a, d=d, b=b)
    else:
        text = DIR_PATTERN.format(a=a, b=b)
    return text + "?", ans


def make_statement(rng, places, coords, facts):
    """Pick a true, not yet stated relation between two places; return (fact, sentence) or None."""
    for _ in range(100):
        a, b = rng.sample(places, 2)
        (xa, ya), (xb, yb) = coords[a], coords[b]
        options = []
        if ya > yb:
            options.append("north")
        if ya < yb:
            options.append("south")
        if xa > xb:
            options.append("east")
        if xa < xb:
            options.append("west")
        d = rng.choice(options)  # cells are distinct, so options is never empty
        fact = canon(a, d, b)
        if fact in facts:
            continue
        sentence = rng.choice(STATEMENT_PATTERNS).format(a=a, d=d, b=b) + "."
        return fact, sentence
    return None


def make_story(rng):
    """Return the story as a list of line bodies (without numbers)."""
    while True:
        places = rng.sample(PLACES, rng.randint(3, 6))
        cells = rng.sample([(x, y) for x in range(4) for y in range(4)], len(places))
        coords = dict(zip(places, cells))
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)

        stmts = []  # (canonical fact, sentence)
        facts = set()
        for _ in range(n_stmt):
            made = make_statement(rng, places, coords, facts)
            if made is None:
                break
            fact, sentence = made
            facts.add(fact)
            stmts.append((fact, sentence))
        if len(stmts) < 3:
            continue

        total = len(stmts) + n_q
        if n_q > total - 1:
            continue
        q_pos = set(rng.sample(range(1, total), n_q))  # never the first line

        lines = []
        si = 0
        ok = True
        for pos in range(total):
            if pos in q_pos:
                prefix = {f for f, _ in stmts[:si]}
                cands = candidates(prefix, places)
                if not cands:
                    ok = False
                    break
                text, ans = render_question(rng, rng.choice(cands))
                lines.append(f"{text}\t{ans}")
            else:
                lines.append(stmts[si][1])
                si += 1
        if ok:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
