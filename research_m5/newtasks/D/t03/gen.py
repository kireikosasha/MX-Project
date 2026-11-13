"""Generate synthetic reading-comprehension stories: comparison on a scale.

Usage: python gen.py N SEED

Each story compares people on one scale (height, age or weight). A hidden
ranking of the people is simulated; statements reveal some pairwise
comparisons. A question is asked only when its answer follows from the
statements above it, by transitivity, and never from a single statement.
Question kinds: yes/no comparisons between two people, and "who is the
tallest/oldest/heaviest" (or shortest/youngest/lightest) over the people
named in the story so far.
"""
import random
import sys

NAMES = ["alice", "bob", "carl", "dana", "eli", "fay",
         "gus", "hana", "ivan", "june", "kai", "lena"]

# positive comparative, negative comparative, top superlative, bottom superlative
SCALES = [
    ("taller", "shorter", "tallest", "shortest"),
    ("older", "younger", "oldest", "youngest"),
    ("heavier", "lighter", "heaviest", "lightest"),
]


def closure(n, direct):
    """gt[i][j] is True when i is above j, by transitivity of the stated facts."""
    gt = [[False] * n for _ in range(n)]
    for i, j in direct:
        gt[i][j] = True
    for k in range(n):
        for i in range(n):
            if gt[i][k]:
                for j in range(n):
                    if gt[k][j]:
                        gt[i][j] = True
    return gt


def capitalise(text):
    return text[0].upper() + text[1:]


def pick_question(rng, gt, direct, mentioned, people, pos, neg, top, bottom):
    """Return (sentence, answer) for a question settled by the statements, or None."""
    m = sorted(mentioned)
    yn = [(a, b) for a in m for b in m
          if a != b and (gt[a][b] or gt[b][a])
          and (a, b) not in direct and (b, a) not in direct]
    options = []
    if yn:
        options.append("yn")
    if len(m) >= 3:
        for x in m:
            others = [y for y in m if y != x]
            if all(gt[x][y] for y in others) and any((x, y) not in direct for y in others):
                options.append(("top", x))
            if all(gt[y][x] for y in others) and any((y, x) not in direct for y in others):
                options.append(("bottom", x))
    if not options:
        return None
    choice = rng.choice(options)
    if choice == "yn":
        a, b = rng.choice(yn)
        if rng.random() < 0.5:
            return (f"Is {people[a]} {pos} than {people[b]}?",
                    "yes" if gt[a][b] else "no")
        return (f"Is {people[a]} {neg} than {people[b]}?",
                "yes" if gt[b][a] else "no")
    kind, x = choice
    if kind == "top":
        return (f"Who is the {top}?", people[x])
    return (f"Who is the {bottom}?", people[x])


def make_story(rng):
    """Return a list of (sentence, answer-or-None) items, or None if this try failed."""
    pos, neg, top, bottom = rng.choice(SCALES)
    k = rng.randint(3, 6)
    people = rng.sample(NAMES, k)
    rank = list(range(k))
    rng.shuffle(rank)  # hidden scale value per person; larger = more
    n_stmts = rng.randint(3, 12)
    n_qs = rng.randint(1, 5)

    direct = set()       # (i, j): person i is stated to be above person j
    mentioned = set()    # people named in the statements so far
    items = []
    stmts = qs = 0

    while stmts < n_stmts or qs < n_qs:
        gt = closure(k, direct)
        if qs < n_qs and stmts >= 2 and (stmts >= n_stmts or rng.random() < 0.35):
            q = pick_question(rng, gt, direct, mentioned, people, pos, neg, top, bottom)
            if q is not None:
                items.append(q)
                qs += 1
                continue
            if stmts >= n_stmts:
                return None
        if stmts >= n_stmts:
            return None
        # prefer a pair that the statements so far do not settle
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        fresh = [(a, b) for a, b in pairs if not gt[a][b] and not gt[b][a]]
        a, b = rng.choice(fresh or pairs)
        hi, lo = (a, b) if rank[a] > rank[b] else (b, a)
        direct.add((hi, lo))
        mentioned.update((hi, lo))
        if rng.random() < 0.5:
            text = f"{people[hi]} is {pos} than {people[lo]}."
        else:
            text = f"{people[lo]} is {neg} than {people[hi]}."
        items.append((text, None))
        stmts += 1
    return items


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        items = None
        while items is None:
            items = make_story(rng)
        for i, (text, ans) in enumerate(items, 1):
            if ans is None:
                out.append(f"{i} {capitalise(text)}")
            else:
                out.append(f"{i} {capitalise(text)}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
