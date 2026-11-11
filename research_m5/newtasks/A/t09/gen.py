#!/usr/bin/env python3
"""Synthetic chain-of-relation reading task.

World: a forest of people. Each person has at most one superior, and every
story uses one kind of relation (parent, boss or teacher). The true world is
simulated as a dict `sup` (subordinate -> superior).

Story: the superior links are revealed one at a time as statements. Questions
are asked between statements about indirect relations (grandparent-like
"parent of the parent", the top of a chain, and yes/no checks of a two-step
relation). Answers are computed from the links revealed so far, under the usual
closed-world convention: the statements above a question are all the
information there is.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ada", "Bob", "Cy", "Dee", "Eve", "Fay", "Gus", "Hal", "Ivy", "Jo", "Kit", "Lou"]
RELS = ["parent", "boss", "teacher"]
MAX_QUESTIONS = 5


def build_world(rng, n_edges):
    """Return sup: subordinate -> superior, a forest with n_edges links."""
    roots = rng.randint(1, min(3, 12 - n_edges))
    people = rng.sample(NAMES, n_edges + roots)
    sup = {}
    for i in range(roots, len(people)):
        if rng.random() < 0.6:
            sup[people[i]] = people[i - 1]  # extend the newest chain
        else:
            sup[people[i]] = rng.choice(people[:i])
    return sup


def statement(rng, top, sub, rel):
    if rng.random() < 0.5:
        return f"{top} is the {rel} of {sub}."
    return f"{sub} has {top} as a {rel}."


def chain_up(known, x):
    """Superiors of x, nearest first, from the links revealed so far."""
    chain = []
    cur = x
    while cur in known:
        cur = known[cur]
        chain.append(cur)
    return chain


def people_of(known):
    seen = []
    for sub, top in known.items():
        for p in (sub, top):
            if p not in seen:
                seen.append(p)
    return seen


def ask(rng, known, rel):
    """Return (question_text, answer) for a question the statements so far settle, or None."""
    cands = [x for x in people_of(known) if len(chain_up(known, x)) >= 2]
    if not cands:
        return None
    x = rng.choice(cands)
    anc = chain_up(known, x)
    kind = rng.randrange(3)
    if kind == 0:
        return f"Who is the {rel} of the {rel} of {x}?", anc[1].lower()
    if kind == 1:
        return f"Who is the top {rel} of {x}?", anc[-1].lower()
    grand = anc[1]
    others = [p for p in people_of(known) if p != x and p != grand]
    if rng.random() < 0.5 or not others:
        return f"Is {grand} the {rel} of the {rel} of {x}?", "yes"
    return f"Is {rng.choice(others)} the {rel} of the {rel} of {x}?", "no"


def make_story(rng):
    while True:
        n_edges = rng.randint(3, 11)
        rel = rng.choice(RELS)
        sup = build_world(rng, n_edges)
        pairs = list(sup.items())
        rng.shuffle(pairs)
        known = {}
        items = []
        asked = set()
        nq = 0

        def try_question():
            nonlocal nq
            q = ask(rng, known, rel)
            if q and q[0] not in asked:
                asked.add(q[0])
                items.append(("q", q))
                nq += 1

        for sub, top in pairs:
            known[sub] = top
            items.append(("s", statement(rng, top, sub, rel)))
            if nq < MAX_QUESTIONS and rng.random() < 0.4:
                try_question()
        if nq == 0:
            try_question()
        if nq == 0:
            continue  # no chain of two links anywhere: draw another world
        return items


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(2)
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n):
        for i, (kind, payload) in enumerate(make_story(rng), 1):
            if kind == "s":
                out.append(f"{i} {payload}")
            else:
                q, a = payload
                out.append(f"{i} {q}\t{a}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
