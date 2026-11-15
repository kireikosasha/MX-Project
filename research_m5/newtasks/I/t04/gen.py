#!/usr/bin/env python3
"""Synthetic task generator: comparison on a scale (height).

Usage: python gen.py N SEED

Each story has a few people with hidden heights. Statements say that one
person is taller or shorter than another (always true). Questions are only
asked when their answer is determined by the statements above them, and
yes/no questions are never about a pair that was compared directly, so the
answer needs a chain of several comparisons.
"""
import random
import sys

NAMES = ["ann", "bob", "cy", "dan", "eve", "fay", "gus", "hal"]


def cap(name):
    return name.capitalize()


def closure(facts):
    """facts: set of (a, b) meaning a is taller than b. Returns transitive closure."""
    taller = set(facts)
    changed = True
    while changed:
        changed = False
        for a, b in list(taller):
            for c, d in list(taller):
                if b == c and a != d and (a, d) not in taller:
                    taller.add((a, d))
                    changed = True
    return taller


def candidates(people, facts, direct):
    """Return (question, answer) pairs whose answers follow from the facts."""
    T = closure(facts)

    def rel(a, b):
        if (a, b) in T:
            return 1
        if (b, a) in T:
            return -1
        return 0

    out = []
    for x in people:
        others = [y for y in people if y != x]
        if all(rel(x, y) == 1 for y in others):
            out.append(("Who is the tallest?", x))
        if all(rel(y, x) == 1 for y in others):
            out.append(("Who is the shortest?", x))

    for a in people:
        for b in people:
            if a == b or frozenset((a, b)) in direct:
                continue
            r = rel(a, b)
            if r == 0:
                continue
            out.append((f"Is {cap(a)} taller than {cap(b)}?", "yes" if r == 1 else "no"))
            out.append((f"Is {cap(a)} shorter than {cap(b)}?", "yes" if r == -1 else "no"))

    for a in people:
        others = [y for y in people if y != a]
        if not all(rel(y, a) != 0 for y in others):
            continue
        taller_ = sorted(y for y in others if rel(y, a) == 1)
        shorter_ = sorted(y for y in others if rel(y, a) == -1)
        if taller_:
            out.append((f"Who is taller than {cap(a)}?", ",".join(taller_)))
        if shorter_:
            out.append((f"Who is shorter than {cap(a)}?", ",".join(shorter_)))
        out.append((f"How many people are taller than {cap(a)}?", str(len(taller_))))
        out.append((f"How many people are shorter than {cap(a)}?", str(len(shorter_))))
    return [(q, ans) for q, ans in out]


def make_story(rng):
    k = rng.randint(3, 5)
    people = rng.sample(NAMES, k)
    heights = rng.sample(range(k), k)
    height = dict(zip(people, heights))
    S = rng.randint(3, 12)   # target number of statements
    Q = rng.randint(1, 5)    # target number of questions

    lines = []               # (text, answer or None)
    facts = set()
    direct = set()
    stmts = 0
    qs = 0
    while True:
        need_q = qs < Q
        if need_q and stmts >= 2 and (stmts >= S or rng.random() < 0.35):
            cands = candidates(people, facts, direct)
            if cands:
                text, ans = rng.choice(cands)
                lines.append((text, ans))
                qs += 1
                continue
        if stmts < 12 and (stmts < S or need_q):
            a, b = rng.sample(people, 2)
            t, s = (a, b) if height[a] > height[b] else (b, a)
            if rng.random() < 0.5:
                text = f"{cap(t)} is taller than {cap(s)}."
            else:
                text = f"{cap(s)} is shorter than {cap(t)}."
            lines.append((text, None))
            facts.add((t, s))
            direct.add(frozenset((a, b)))
            stmts += 1
            continue
        break

    if stmts < 3 or qs < 1:
        return None
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = make_story(rng)
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
