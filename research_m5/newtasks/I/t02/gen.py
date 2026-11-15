import itertools
import random
import sys

# Task: comparison on a scale. Each story has a few people with a hidden
# order on one scale (height, age or speed). Statements reveal pairwise
# comparisons; questions need chaining of several statements.

NAMES = ["ann", "ben", "cara", "dan", "eli", "fay", "gus", "hal", "ivy", "jon"]

# (greater word, lesser word, superlative of greater, superlative of lesser)
SCALES = [
    ("taller", "shorter", "tallest", "shortest"),
    ("older", "younger", "oldest", "youngest"),
    ("faster", "slower", "fastest", "slowest"),
]


def closure(direct):
    """Transitive closure of known facts (a, b) meaning a is greater than b."""
    reach = set(direct)
    while True:
        new = {(a, d) for (a, b) in reach for (c, d) in reach
               if b == c and a != d} - reach
        if not new:
            return reach
        reach |= new


def questions(people, direct, reach, scale):
    """Questions whose answer is fixed by the known facts and needs chaining.

    Each candidate is (question text, answer). Only answers that follow from
    the closure but are not stated directly are kept.
    """
    greater, _, top, bottom = scale
    out = []
    for m in people:
        others = [z for z in people if z != m]
        if all((m, z) in reach for z in others) and \
                any((m, z) not in direct for z in others):
            out.append(("who is the %s?" % top, m))
        if all((z, m) in reach for z in others) and \
                any((z, m) not in direct for z in others):
            out.append(("who is the %s?" % bottom, m))
    for a in people:
        for b in people:
            if a == b or (a, b) in direct or (b, a) in direct:
                continue
            if (a, b) in reach:
                out.append(("is %s %s than %s?" % (a, greater, b), "yes"))
            elif (b, a) in reach:
                out.append(("is %s %s than %s?" % (a, greater, b), "no"))
    for x in people:
        others = [z for z in people if z != x]
        if all((z, x) in reach or (x, z) in reach for z in others) and \
                any((z, x) in reach and (z, x) not in direct for z in others):
            n = sum((z, x) in reach for z in others)
            out.append(("how many people are %s than %s?" % (greater, x), str(n)))
    return out


def try_story(rng):
    scale = rng.choice(SCALES)
    greater, lesser, _, _ = scale
    k = rng.randint(4, 6)
    people = rng.sample(NAMES, k)
    hidden = people[:]          # hidden[0] is the greatest on the scale
    rng.shuffle(hidden)
    rank = {p: i for i, p in enumerate(hidden)}
    pairs = list(itertools.combinations(people, 2))
    s_left = rng.randint(3, min(12, len(pairs)))
    q_left = rng.randint(1, 5)
    lines, direct, used, asked = [], set(), set(), set()
    while s_left or q_left:
        if q_left and lines and (not s_left or rng.random() < 0.5):
            cands = [c for c in questions(people, direct, closure(direct), scale)
                     if c[0] not in asked]
            if cands:
                text, ans = rng.choice(cands)
                asked.add(text)
                lines.append("%s\t%s" % (text, ans))
                q_left -= 1
                continue
            if not s_left:
                return None
        free = [p for p in pairs if p not in used]
        a, b = rng.choice(free)
        used.add((a, b))
        g, l = (a, b) if rank[a] < rank[b] else (b, a)
        direct.add((g, l))
        if rng.random() < 0.5:
            lines.append("%s is %s than %s." % (g, greater, l))
        else:
            lines.append("%s is %s than %s." % (l, lesser, g))
        s_left -= 1
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
