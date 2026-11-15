import random
import sys
from itertools import combinations

# Task: comparison on a scale. People are compared pairwise on one scale
# (taller/shorter or older/younger). Questions need several comparisons
# chained together (transitivity), so they cannot be answered from one sentence.

NAMES = ["ann", "bob", "cy", "dan", "eve", "fay", "gus", "hal", "ivy", "jo", "kim", "leo"]

# (greater word, lesser word, top word, bottom word)
ATTRS = [
    ("taller", "shorter", "tallest", "shortest"),
    ("older", "younger", "oldest", "youngest"),
]


def closure(edges):
    """edges: set of (winner, loser). Returns the transitive closure."""
    rel = set(edges)
    while True:
        new = set()
        for (a, b) in rel:
            for (c, d) in rel:
                if b == c and (a, d) not in rel:
                    new.add((a, d))
        if not new:
            return rel
        rel |= new


def make_question(rng, mentioned, edges, stated, attr):
    hi, lo, top, bot = attr
    rel = closure(edges)
    for _ in range(30):
        kind = rng.choice(["yesno", "extreme", "list"])
        if kind == "yesno":
            if len(mentioned) < 2:
                continue
            a, b = rng.sample(mentioned, 2)
            # must need a chain: the pair is not stated directly
            if frozenset((a, b)) in stated:
                continue
            if (a, b) in rel:
                a_bigger = True
            elif (b, a) in rel:
                a_bigger = False
            else:
                continue
            if rng.random() < 0.5:
                ans = "yes" if a_bigger else "no"
                q = f"Is {a.capitalize()} {hi} than {b.capitalize()}?"
            else:
                ans = "no" if a_bigger else "yes"
                q = f"Is {a.capitalize()} {lo} than {b.capitalize()}?"
            return f"{q}\t{ans}"
        if kind == "extreme":
            if len(mentioned) < 3:
                continue
            want_max = rng.random() < 0.5
            found = None
            for x in mentioned:
                if all(((x, y) in rel) if want_max else ((y, x) in rel)
                       for y in mentioned if y != x):
                    found = x
                    break
            if found is None:
                continue
            word = top if want_max else bot
            return f"Who is the {word}?\t{found}"
        # list kind
        if len(mentioned) < 3:
            continue
        b = rng.choice(mentioned)
        others = [m for m in mentioned if m != b]
        if not all((x, b) in rel or (b, x) in rel for x in others):
            continue
        if rng.random() < 0.5:
            winners = [x for x in others if (x, b) in rel]
            word = hi
        else:
            winners = [x for x in others if (b, x) in rel]
            word = lo
        if not winners:
            continue
        ans = ",".join(sorted(winners))
        return f"Who is {word} than {b.capitalize()}?\t{ans}"
    return None


def make_story(rng):
    while True:
        attr = rng.choice(ATTRS)
        hi, lo, top, bot = attr
        k = rng.randint(3, 6)
        people = rng.sample(NAMES, k)
        order = people[:]
        rng.shuffle(order)
        rank = {p: i for i, p in enumerate(order)}  # smaller rank = greater on the scale
        max_s = min(12, k * (k - 1) // 2)
        s_target = rng.randint(3, max_s)
        q_target = rng.randint(1, 5)

        items = []
        edges = set()
        stated = set()
        mentioned = []
        n_s = 0
        n_q = 0
        while n_s < s_target:
            pairs = [p for p in combinations(people, 2) if frozenset(p) not in stated]
            a, b = rng.choice(pairs)
            stated.add(frozenset((a, b)))
            w, l = (a, b) if rank[a] < rank[b] else (b, a)
            edges.add((w, l))
            for p in (w, l):
                if p not in mentioned:
                    mentioned.append(p)
            if rng.random() < 0.5:
                items.append(f"{w.capitalize()} is {hi} than {l.capitalize()}.")
            else:
                items.append(f"{l.capitalize()} is {lo} than {w.capitalize()}.")
            n_s += 1
            if n_q < q_target and rng.random() < 0.4:
                q = make_question(rng, mentioned, edges, stated, attr)
                if q is not None:
                    items.append(q)
                    n_q += 1
        while n_q < q_target:
            q = make_question(rng, mentioned, edges, stated, attr)
            if q is None:
                break
            items.append(q)
            n_q += 1
        if n_q >= 1:
            return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for i, item in enumerate(make_story(rng), 1):
            lines.append(f"{i} {item}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
