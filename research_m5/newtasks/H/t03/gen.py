"""Synthetic comparison-on-a-scale reading task generator.

Usage: python gen.py N SEED

Each story hides a total order of 3-6 people on one scale (height, age or
speed). Statements reveal true pairwise comparisons. Questions are asked only
when the answer follows from the statements above them, and at least one
needed comparison requires chaining two or more statements.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli", "Fay", "Gus", "Hana"]

# (greater word, lesser word, greatest word, least word)
SCALES = [
    ("taller", "shorter", "tallest", "shortest"),
    ("older", "younger", "oldest", "youngest"),
    ("faster", "slower", "fastest", "slowest"),
]


def candidates(known, edges, comp, opp, top, bottom):
    """Return (question, answer) pairs that the statements so far determine."""
    # Shortest path lengths in the "is greater than" graph, among known people.
    dist = {}
    for s in known:
        d, seen, frontier, level = {}, {s}, [s], 0
        while frontier:
            level += 1
            nxt = []
            for u in frontier:
                for v in edges.get(u, ()):
                    if v not in seen:
                        seen.add(v)
                        d[v] = level
                        nxt.append(v)
            frontier = nxt
        dist[s] = d

    def better(a, b):
        return dist[a].get(b)  # path length if a is provably greater than b

    out = []

    # Superlatives: one person provably above (or below) all other known people.
    for word, sign in ((top, 1), (bottom, -1)):
        for x in known:
            others = [q for q in known if q != x]
            if not others:
                continue
            if sign == 1:
                ds = [better(x, q) for q in others]
            else:
                ds = [better(q, x) for q in others]
            if all(d is not None for d in ds) and max(ds) >= 2:
                out.append((f"Who is the {word}?", x.lower()))

    # Yes/no comparisons between two known people.
    for x in known:
        for y in known:
            if x == y:
                continue
            for w in (comp, opp):
                a, b = (x, y) if w == comp else (y, x)
                d1, d2 = better(a, b), better(b, a)
                if d1 is not None:
                    if d1 >= 2:
                        out.append((f"Is {x} {w} than {y}?", "yes"))
                elif d2 is not None and d2 >= 2:
                    out.append((f"Is {x} {w} than {y}?", "no"))

    # Counting and listing: every other known person must be comparable to x.
    for x in known:
        others = [q for q in known if q != x]
        if not others:
            continue
        for w in (comp, opp):
            ok, members, maxd = True, [], 0
            for q in others:
                d1, d2 = better(q, x), better(x, q)
                if d1 is None and d2 is None:
                    ok = False
                    break
                if (w == comp and d1 is not None) or (w == opp and d2 is not None):
                    members.append(q)
                maxd = max(maxd, d1 if d1 is not None else d2)
            if not ok or maxd < 2:
                continue
            out.append((f"How many people are {w} than {x}?", str(len(members))))
            answer = ",".join(sorted(m.lower() for m in members)) or "none"
            out.append((f"Who is {w} than {x}?", answer))

    return out


def make_story(rng):
    comp, opp, top, bottom = rng.choice(SCALES)
    k = rng.randint(3, 6)
    people = rng.sample(NAMES, k)
    order = people[:]
    rng.shuffle(order)
    rank = {p: i for i, p in enumerate(order)}  # smaller rank = greater on the scale
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)

    lines, edges, known, asked = [], {}, [], set()
    st = qu = 0
    while st + qu < n_st + n_q:
        rem_st = n_st - st
        rem_q = n_q - qu
        if rem_q > 0 and st >= 2 and (rem_st == 0 or rng.random() < 0.4):
            cands = [c for c in candidates(known, edges, comp, opp, top, bottom)
                     if c[0] not in asked]
            if cands:
                text, ans = rng.choice(cands)
                asked.add(text)
                lines.append(f"{text}\t{ans}")
                qu += 1
                continue
            if rem_st == 0:
                return None  # cannot place the required questions; retry
        a, b = rng.sample(people, 2)
        g, l = (a, b) if rank[a] < rank[b] else (b, a)
        if rng.random() < 0.5:
            lines.append(f"{g} is {comp} than {l}.")
        else:
            lines.append(f"{l} is {opp} than {g}.")
        edges.setdefault(g, set()).add(l)
        for p in (g, l):
            if p not in known:
                known.append(p)
        st += 1
    return lines


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = make_story(rng)
        for i, line in enumerate(story, 1):
            out.append(f"{i} {line}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
