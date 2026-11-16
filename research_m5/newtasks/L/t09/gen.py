#!/usr/bin/env python3
"""Synthetic chain-of-relations reading task.

Each story describes a small forest of people linked by one relation
(parent-of or boss-of). Questions ask who sits two links up a chain
(a grandparent, or a boss's boss). A question is only asked once both links
have been stated above it, and its answer is computed from the stated links.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "bob", "cal", "dee", "eli", "fay",
         "gus", "hal", "ivy", "jon", "kit", "lou"]


def make_forest(rng):
    """Return (parent dict child->parent, list of two-link chains (x, p, g))."""
    while True:
        n = rng.randint(4, 12)
        order = rng.sample(NAMES, n)
        parent = {}
        for i in range(1, n):
            if rng.random() < 0.8:
                parent[order[i]] = rng.choice(order[:i])
        chains = [(x, parent[x], parent[parent[x]])
                  for x in parent if parent[x] in parent]
        if len(parent) >= 3 and chains:
            return parent, chains


def statement(rng, rel, child, par):
    if rel == "parent":
        if rng.random() < 0.5:
            return f"{par} is the parent of {child}."
        return f"{child} is a child of {par}."
    if rng.random() < 0.5:
        return f"{par} is the boss of {child}."
    return f"{child} works for {par}."


def question(rng, rel, known, asked):
    """Pick an unasked two-link question answerable from the known links."""
    options = []
    for x in known:
        if known[x] in known:
            text = f"Who is the {rel} of the {rel} of {x}?"
            if text not in asked:
                options.append((text, known[known[x]]))
    if not options:
        return None
    return rng.choice(options)


def make_story(rng):
    rel = rng.choice(["parent", "boss"])
    parent, chains = make_forest(rng)
    edges = list(parent.items())
    x, p, g = rng.choice(chains)
    chain_edges = [(x, p), (p, g)]
    others = [e for e in edges if e not in chain_edges]
    s = rng.randint(3, min(12, len(edges)))
    chosen = chain_edges + rng.sample(others, s - 2)
    rng.shuffle(chosen)

    q_total = rng.randint(1, 5)
    items = []
    known = {}
    asked = set()
    for idx, (child, par) in enumerate(chosen):
        known[child] = par
        items.append(statement(rng, rel, child, par))
        if len(asked) < q_total and idx < len(chosen) - 1 and rng.random() < 0.4:
            q = question(rng, rel, known, asked)
            if q:
                asked.add(q[0])
                items.append(q)
    while len(asked) < q_total:
        q = question(rng, rel, known, asked)
        if q is None:
            break
        asked.add(q[0])
        items.append(q)
    return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        number = 0
        for item in make_story(rng):
            number += 1
            if isinstance(item, str):
                out.append(f"{number} {item}")
            else:
                text, answer = item
                out.append(f"{number} {text}\t{answer}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
