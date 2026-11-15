#!/usr/bin/env python3
"""Synthetic task: comparisons on one scale (taller, older, faster).

Usage: python gen.py N SEED   (prints N stories to standard output)
"""
import itertools
import random
import sys

NAMES = ["ann", "bob", "cara", "dan", "eve", "finn", "gus", "hana", "ivo", "joy"]

# scale -> (greater comparative, lesser comparative, greatest superlative, least superlative)
SCALES = {
    "height": ("taller", "shorter", "tallest", "shortest"),
    "age": ("older", "younger", "oldest", "youngest"),
    "speed": ("faster", "slower", "fastest", "slowest"),
}


def closure(pairs):
    """Transitive closure of (greater, lesser) pairs."""
    reach = set(pairs)
    while True:
        new = {(a, d) for (a, b) in reach for (c, d) in reach if b == c and a != d}
        if new <= reach:
            return reach
        reach |= new


def statement(rng, more, less, x, y):
    """x is greater than y, said with one of two fixed patterns."""
    if rng.random() < 0.5:
        return f"{x} is {more} than {y}."
    return f"{y} is {less} than {x}."


def question(rng, people, rank, reach, direct, more, less, most, least):
    """Return a question line whose answer follows from the statements so far, or None.

    Answers come from the hidden rank (the simulated world); the statements
    so far (reach = their closure) must determine them.
    """
    options = []
    # yes/no: is x more than y?  Needs a chain, so the pair is not stated directly.
    for x, y in itertools.permutations(people, 2):
        if ((x, y) in reach or (y, x) in reach) and (x, y) not in direct and (y, x) not in direct:
            ans = "yes" if rank[x] < rank[y] else "no"
            options.append(f"is {x} {more} than {y}?\t{ans}")
    # superlative over three people: who is the tallest / shortest of a, b and c?
    for trio in itertools.combinations(people, 3):
        for a in trio:
            rest = [o for o in trio if o != a]
            if all((a, o) in reach for o in rest) and any((a, o) not in direct for o in rest):
                names = list(trio)
                rng.shuffle(names)
                ans = min(trio, key=rank.get)
                options.append(f"who is the {most} of {' and '.join(names)}?\t{ans}")
            if all((o, a) in reach for o in rest) and any((o, a) not in direct for o in rest):
                names = list(trio)
                rng.shuffle(names)
                ans = max(trio, key=rank.get)
                options.append(f"who is the {least} of {' and '.join(names)}?\t{ans}")
    # lists: who is taller / shorter than x?  Every other person must be placed relative to x.
    for x in people:
        others = [y for y in people if y != x]
        if all((y, x) in reach or (x, y) in reach for y in others):
            above = sorted(y for y in others if rank[y] < rank[x])
            below = sorted(y for y in others if rank[y] > rank[x])
            if above:
                options.append(f"who is {more} than {x}?\t{','.join(above)}")
            if below:
                options.append(f"who is {less} than {x}?\t{','.join(below)}")
    return rng.choice(options) if options else None


def try_story(rng):
    """Build one story as a list of lines, or None if this attempt got stuck."""
    more, less, most, least = SCALES[rng.choice(sorted(SCALES))]
    people = rng.sample(NAMES, rng.randint(3, 6))
    order = rng.sample(people, len(people))  # hidden world: order[0] is the greatest
    rank = {p: i for i, p in enumerate(order)}
    n_st, n_q = rng.randint(3, 12), rng.randint(1, 5)
    lines, direct, st, q = [], set(), 0, 0
    while st < n_st or q < n_q:
        reach = closure(direct)
        if q < n_q and st >= 2 and (st == n_st or rng.random() < 0.4):
            text = question(rng, people, rank, reach, direct, more, less, most, least)
            if text is not None:
                lines.append(text)
                q += 1
                continue
            if st == n_st:
                return None
        cands = [(x, y) for x, y in itertools.permutations(people, 2)
                 if rank[x] < rank[y] and (x, y) not in reach]
        if not cands:
            cands = [(x, y) for x, y in itertools.permutations(people, 2) if rank[x] < rank[y]]
        x, y = rng.choice(cands)
        direct.add((x, y))
        lines.append(statement(rng, more, less, x, y))
        st += 1
    return lines


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lines = None
        while lines is None:
            lines = try_story(rng)
        out.extend(f"{i} {line}" for i, line in enumerate(lines, 1))
    print("\n".join(out))


if __name__ == "__main__":
    main()
