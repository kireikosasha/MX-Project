#!/usr/bin/env python3
"""Synthetic map-direction reading-comprehension task generator.

Usage: python gen.py N SEED

Places lie in one of four directions (north, south, east, west) from each
other. Each story is generated from a hidden layout of places (distinct x and
y coordinates). Statements are read off the layout. A question is only asked
when its answer is forced by the statements above it, in every layout that is
consistent with them, and the forced answer is then checked against the layout.
"""
import random
import sys

PLACES = ["park", "school", "market", "tower", "bridge", "garden",
          "mill", "harbor", "castle", "farm", "lake", "museum"]
DIRS = ["north", "south", "east", "west"]
AXIS = {"north": "y", "south": "y", "east": "x", "west": "x"}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def dir_rel(d, a, b):
    """Return (axis, hi, lo) such that 'a is d of b' <=> hi lies above lo on axis."""
    if d == "north":
        return ("y", a, b)
    if d == "south":
        return ("y", b, a)
    if d == "east":
        return ("x", a, b)
    return ("x", b, a)  # west


def closure(facts):
    """Transitive closure of 'hi above lo' facts, per axis."""
    clo = {}
    for ax in ("x", "y"):
        rel = {(h, l) for (a, h, l) in facts if a == ax}
        changed = True
        while changed:
            changed = False
            for (p, q) in list(rel):
                for (r, s) in list(rel):
                    if q == r and (p, s) not in rel:
                        rel.add((p, s))
                        changed = True
        clo[ax] = rel
    return clo


def questions(facts, universe, truth):
    """All (question, answer) pairs forced by the facts above the question."""
    direct = {"x": set(), "y": set()}
    for ax, h, l in facts:
        direct[ax].add((h, l))
    clo = closure(facts)
    out = []

    # Yes/no: 'Is the A d of the B?' forced yes or forced no, by a chain.
    for a in universe:
        for b in universe:
            if a == b:
                continue
            for d in DIRS:
                ax, h, l = dir_rel(d, a, b)
                if (h, l) in clo[ax] and (h, l) not in direct[ax]:
                    assert truth(d, a, b)
                    out.append((f"Is the {a} {d} of the {b}?", "yes"))
                elif (l, h) in clo[ax] and (l, h) not in direct[ax]:
                    assert not truth(d, a, b)
                    out.append((f"Is the {a} {d} of the {b}?", "no"))

    # What: 'What is d1 of the X and d2 of the Y?' with one place forced to
    # satisfy both, every other place forced out of at least one of them.
    for x in universe:
        for y in universe:
            if x == y:
                continue
            for d1 in DIRS:
                for d2 in DIRS:
                    if AXIS[d1] == AXIS[d2]:
                        continue
                    forced = []
                    for r in universe:
                        if r in (x, y):
                            continue
                        ax1, h1, l1 = dir_rel(d1, r, x)
                        ax2, h2, l2 = dir_rel(d2, r, y)
                        if (h1, l1) in clo[ax1] and (h2, l2) in clo[ax2]:
                            forced.append(r)
                    if len(forced) != 1:
                        continue
                    r = forced[0]
                    ax1, h1, l1 = dir_rel(d1, r, x)
                    ax2, h2, l2 = dir_rel(d2, r, y)
                    if (h1, l1) in direct[ax1] and (h2, l2) in direct[ax2]:
                        continue  # both lookups, no reasoning needed
                    ok = True
                    for z in universe:
                        if z in (x, y, r):
                            continue
                        ax1, h1, l1 = dir_rel(d1, z, x)
                        ax2, h2, l2 = dir_rel(d2, z, y)
                        if (l1, h1) in clo[ax1] or (l2, h2) in clo[ax2]:
                            continue
                        ok = False
                        break
                    if not ok:
                        continue
                    true_set = [z for z in universe
                                if z not in (x, y)
                                and truth(d1, z, x) and truth(d2, z, y)]
                    assert true_set == [r]
                    out.append((f"What is {d1} of the {x} and {d2} of the {y}?", r))
    return out


def try_story(rng):
    k = rng.randint(4, 7)
    places = rng.sample(PLACES, k)
    xs = dict(zip(places, rng.sample(range(100), k)))
    ys = dict(zip(places, rng.sample(range(100), k)))

    def truth(d, a, b):
        if d == "north":
            return ys[a] > ys[b]
        if d == "south":
            return ys[a] < ys[b]
        if d == "east":
            return xs[a] > xs[b]
        return xs[a] < xs[b]  # west

    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    slots = ["S"] * n_st + ["Q"] * n_q
    while True:
        rng.shuffle(slots)
        if slots[0] == "S" and slots[1] == "S":
            break

    facts = []
    used = set()
    mentioned = set()
    asked = set()
    lines = []
    for num, slot in enumerate(slots, 1):
        if slot == "S":
            while True:
                a, b = rng.sample(places, 2)
                d = rng.choice(DIRS)
                if not truth(d, a, b):
                    d = OPP[d]
                if (a, b, d) not in used:
                    break
            used.add((a, b, d))
            facts.append(dir_rel(d, a, b))
            mentioned.update((a, b))
            pat = rng.choice(["is", "lies"])
            lines.append(f"{num} The {a} {pat} {d} of the {b}.")
        else:
            cands = questions(facts, sorted(mentioned), truth)
            cands = [c for c in cands if c[0] not in asked]
            if not cands:
                return None
            text, ans = rng.choice(cands)
            asked.add(text)
            lines.append(f"{num} {text}\t{ans}")
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = try_story(rng)
        out.extend(story)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
