"""Generate synthetic comparison-on-a-scale reading tasks.

Usage: python gen.py N SEED

Each story hides a total order of a few people on one scale (for example
height). Statements reveal true pairwise comparisons. Questions are asked only
when the statements above them determine the answer through transitive
chaining, and answers are read from the hidden order.
"""
import random
import sys

NAMES = ["ann", "bob", "cy", "dan", "eve", "fay", "gus", "hal", "ivy", "jon"]

# (greater comparative, lesser comparative, greatest, least)
SCALES = [
    ("taller", "shorter", "tallest", "shortest"),
    ("older", "younger", "oldest", "youngest"),
    ("heavier", "lighter", "heaviest", "lightest"),
]


def closure(direct, names):
    """reach[x] = set of names known to be below x (transitive closure)."""
    reach = {}
    for x in names:
        seen = set()
        stack = [x]
        while stack:
            u = stack.pop()
            for v in direct.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        reach[x] = seen
    return reach


def make_question(rng, direct, rank, comp, opp, sup, inf):
    names = sorted({x for x in direct} | {y for ys in direct.values() for y in ys})
    reach = closure(direct, names)
    kind = rng.choice(["yn", "ext", "set"])

    if kind == "yn":
        x, y = rng.sample(names, 2)
        w = rng.choice([comp, opp])
        if y in direct.get(x, ()) or x in direct.get(y, ()):
            return None  # answer is a single stated fact, not a chain
        if not (y in reach[x] or x in reach[y]):
            return None  # not determined yet
        if w == comp:
            truth = rank[x] < rank[y]
        else:
            truth = rank[y] < rank[x]
        return f"is {x} {w} than {y}?\t{'yes' if truth else 'no'}"

    if len(names) < 3:
        return None

    if kind == "ext":
        if rng.random() < 0.5:
            cand = min(names, key=lambda n: rank[n])
            if not all(y in reach[cand] for y in names if y != cand):
                return None
            return f"who is the {sup}?\t{cand}"
        cand = max(names, key=lambda n: rank[n])
        if not all(cand in reach[y] for y in names if y != cand):
            return None
        return f"who is the {inf}?\t{cand}"

    # kind == "set"
    x = rng.choice(names)
    w = rng.choice([comp, opp])
    others = [y for y in names if y != x]
    if not all((y in reach[x]) or (x in reach[y]) for y in others):
        return None
    if w == comp:
        answer = sorted(y for y in others if rank[y] < rank[x])
    else:
        answer = sorted(y for y in others if rank[y] > rank[x])
    ans = ",".join(answer) if answer else "nobody"
    return f"who is {w} than {x}?\t{ans}"


def attempt(rng):
    k = rng.randint(3, 5)
    people = rng.sample(NAMES, k)
    rank = {p: i for i, p in enumerate(people)}  # rank 0 = greatest on scale
    comp, opp, sup, inf = rng.choice(SCALES)
    target_s = rng.randint(3, 12)
    target_q = rng.randint(1, 5)

    direct = {}   # direct[a] = set of names b with a stated greater than b
    stated = set()
    lines = []
    s = q = 0
    while s < target_s or q < target_q:
        if q < target_q and s >= 2 and (s >= target_s or rng.random() < 0.4):
            qline = make_question(rng, direct, rank, comp, opp, sup, inf)
            if qline is not None:
                lines.append(f"{len(lines) + 1} {qline}")
                q += 1
                continue
        if s >= target_s:
            return None
        pairs = [(a, b) for a in people for b in people
                 if rank[a] < rank[b] and frozenset((a, b)) not in stated]
        if not pairs:
            return None
        a, b = rng.choice(pairs)  # a is greater on the scale
        if rng.random() < 0.5:
            text = f"{a} is {comp} than {b}."
        else:
            text = f"{b} is {opp} than {a}."
        lines.append(f"{len(lines) + 1} {text}")
        direct.setdefault(a, set()).add(b)
        stated.add(frozenset((a, b)))
        s += 1
    return lines


def make_story(rng):
    while True:
        lines = attempt(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
