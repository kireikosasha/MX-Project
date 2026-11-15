#!/usr/bin/env python3
"""Synthetic time-order reading task.

Usage: python gen.py N SEED

Each story has a hidden timeline of visits: a person goes to a place at a
hidden time (a single digit). The statements reveal visits either with their
exact time, or as before/after another visit, in shuffled order. Questions ask
whether one visit came before another, or which of two places a person went
to first. A question is only asked when the statements above it fix the
answer (checked by transitive closure over those statements); the answer
itself is read from the hidden timeline.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cal", "Dee", "Eve", "Fay", "Gus", "Hal"]
PLACES = ["park", "shop", "hall", "lab", "farm", "pool"]
LABELS = list(range(1, 10))
MAX_Q = 5


def closure(nodes, edges):
    """Return a function before(a, b) that is True when a is entailed to come before b."""
    idx = {n: i for i, n in enumerate(nodes)}
    k = len(nodes)
    reach = [[False] * k for _ in range(k)]
    for a, b in edges:
        reach[idx[a]][idx[b]] = True
    for m in range(k):
        for i in range(k):
            if reach[i][m]:
                for j in range(k):
                    if reach[m][j]:
                        reach[i][j] = True
    return lambda a, b: reach[idx[a]][idx[b]]


def analyse(prefix):
    """Collect visits, ordering edges and directly related pairs from statements."""
    nodes, timed, edges, direct = [], {}, set(), set()
    for st in prefix:
        mvs = [st["move"]] if st["kind"] == "time" else [st["x"], st["y"]]
        for mv in mvs:
            if mv not in nodes:
                nodes.append(mv)
        if st["kind"] == "time":
            timed[st["move"]] = st["label"]
        else:
            x, y = st["x"], st["y"]
            direct.add(frozenset((x, y)))
            if st["rel"] == "before":
                edges.add((x, y))
            else:
                edges.add((y, x))
    for a in timed:
        for b in timed:
            if timed[a] < timed[b]:
                edges.add((a, b))
    return nodes, edges, direct


def stmt_text(st):
    if st["kind"] == "time":
        p, l = st["move"]
        return f"{p} went to the {l} at time {st['label']}."
    (p1, l1), (p2, l2) = st["x"], st["y"]
    return f"{p1} went to the {l1} {st['rel']} {p2} went to the {l2}."


def rel_stmt(a, b, label):
    """Relation statement a ? b, true in the hidden timeline."""
    return {"kind": "rel", "x": a, "y": b,
            "rel": "before" if label[a] < label[b] else "after"}


def pick_question(prefix, label, rng):
    """Pick a question whose answer the prefix fixes; return (text, answer) or None."""
    nodes, edges, direct = analyse(prefix)
    before = closure(nodes, edges)
    cands = [(x, y) for x in nodes for y in nodes
             if x != y and before(x, y) and frozenset((x, y)) not in direct]
    if not cands:
        return None
    x, y = rng.choice(cands)  # entailed: x comes before y
    if x[0] == y[0] and rng.random() < 0.5:
        first, second = (x, y) if label[x] < label[y] else (y, x)
        places = [first[1], second[1]]
        rng.shuffle(places)
        text = f"Which did {x[0]} go to first, the {places[0]} or the {places[1]}?"
        return text, first[1]
    a, b = (x, y) if rng.random() < 0.5 else (y, x)
    ans = "yes" if label[a] < label[b] else "no"
    return f"Did {a[0]} go to the {a[1]} before {b[0]} went to the {b[1]}?", ans


def try_story(rng):
    persons = rng.sample(NAMES, rng.randint(2, 4))
    places = rng.sample(PLACES, rng.randint(2, 4))
    pairs = [(p, l) for p in persons for l in places]
    m = rng.randint(3, min(5, len(pairs)))
    moves = rng.sample(pairs, m)
    label = dict(zip(moves, rng.sample(LABELS, m)))

    order = moves[:]
    rng.shuffle(order)
    stmts, intro = [], []
    for mv in order:
        if not intro or rng.random() < 0.5:
            stmts.append({"kind": "time", "move": mv, "label": label[mv]})
        else:
            stmts.append(rel_stmt(mv, rng.choice(intro), label))
        intro.append(mv)
    for _ in range(rng.randint(0, 3)):
        x, y = rng.sample(intro, 2)
        stmts.append(rel_stmt(x, y, label))

    items, n_q = [], 0
    for i, st in enumerate(stmts):
        items.append(("s", stmt_text(st)))
        if i >= 1 and n_q < MAX_Q and rng.random() < 0.5:
            q = pick_question(stmts[: i + 1], label, rng)
            if q is not None:
                items.append(("q", q))
                n_q += 1
    if n_q == 0:
        q = pick_question(stmts, label, rng)
        if q is None:
            return None
        items.append(("q", q))

    out = []
    for n, (kind, val) in enumerate(items, 1):
        if kind == "s":
            out.append(f"{n} {val}")
        else:
            text, ans = val
            out.append(f"{n} {text}\t{ans}")
    return out


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        story = None
        while story is None:
            story = try_story(rng)
        lines.extend(story)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
