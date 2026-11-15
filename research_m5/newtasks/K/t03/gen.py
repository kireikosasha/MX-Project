#!/usr/bin/env python3
"""Synthetic reading-comprehension task: comparisons on one scale.

Each story has a small cast of people compared pairwise on one scale
(height or weight). Questions need several comparisons chained together:
superlatives ("Who is the tallest?"), comparisons never stated directly
("Is Ann taller than Bob?") and lists ("Who is shorter than Ann?").

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cy", "Dan", "Eve", "Fay", "Gus", "Hal"]

# (positive comparative, negative comparative, greatest, least)
SCALES = [
    ("taller", "shorter", "tallest", "shortest"),
    ("heavier", "lighter", "heaviest", "lightest"),
]


def closure(cast, facts):
    """gt[i][j] is True when cast[i] is known to be greater than cast[j]."""
    idx = {name: i for i, name in enumerate(cast)}
    m = len(cast)
    gt = [[False] * m for _ in range(m)]
    for big, small in facts:
        gt[idx[big]][idx[small]] = True
    for k in range(m):
        for i in range(m):
            if gt[i][k]:
                for j in range(m):
                    if gt[k][j]:
                        gt[i][j] = True
    return idx, gt


def make_question(rng, cast, world, facts, scale):
    """Return (question_text, answer) if the answer follows from facts, else None."""
    pos, neg, top_word, bottom_word = scale
    idx, gt = closure(cast, facts)
    direct = set(facts)

    def beats(a, b):
        return gt[idx[a]][idx[b]]

    def known(a, b):
        return beats(a, b) or beats(b, a)

    kinds = ["top", "bottom", "pair", "list"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "top":
            target = world[0]
            if all(beats(target, o) for o in cast if o != target):
                return f"Who is the {top_word}?", target.lower()
        elif kind == "bottom":
            target = world[-1]
            if all(beats(o, target) for o in cast if o != target):
                return f"Who is the {bottom_word}?", target.lower()
        elif kind == "pair":
            a, b = rng.sample(cast, 2)
            if (a, b) in direct or (b, a) in direct:
                continue  # answerable from a single sentence: not interesting
            if not known(a, b):
                continue
            answer = "yes" if world.index(a) < world.index(b) else "no"
            return f"Is {a} {pos} than {b}?", answer
        else:
            a = rng.choice(cast)
            others = [o for o in cast if o != a]
            if not all(known(a, o) for o in others):
                continue
            smaller = sorted(o for o in others if world.index(a) < world.index(o))
            if not smaller:
                continue
            return (f"Who is {neg} than {a}?",
                    ",".join(x.lower() for x in smaller))
    return None


def try_story(rng):
    """Return a list of (text, answer) lines, or None if this attempt failed."""
    scale = rng.choice(SCALES)
    pos, neg, _, _ = scale
    cast = rng.sample(NAMES, rng.randint(3, 5))
    world = cast[:]
    rng.shuffle(world)  # world[0] is the greatest, world[-1] the least
    n_stmt = rng.randint(3, 12)
    left_q = rng.randint(1, 5)
    facts = []  # (greater, lesser) pairs stated so far
    lines = []  # (text, answer); answer is None for statements
    asked = set()
    while len(facts) < n_stmt or left_q > 0:
        if left_q > 0 and facts and (len(facts) == n_stmt or rng.random() < 0.4):
            q = make_question(rng, cast, world, facts, scale)
            if q is not None and q[0] not in asked:
                asked.add(q[0])
                lines.append(q)
                left_q -= 1
                continue
            if len(facts) == n_stmt:
                return None
        a, b = rng.sample(cast, 2)
        big, small = (a, b) if world.index(a) < world.index(b) else (b, a)
        facts.append((big, small))
        if rng.random() < 0.5:
            lines.append((f"{big} is {pos} than {small}.", None))
        else:
            lines.append((f"{small} is {neg} than {big}.", None))
    return lines


def main():
    count, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        lines = None
        while lines is None:
            lines = try_story(rng)
        for number, (text, answer) in enumerate(lines, 1):
            if answer is None:
                out.append(f"{number} {text}")
            else:
                out.append(f"{number} {text}\t{answer}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
