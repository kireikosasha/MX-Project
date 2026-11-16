#!/usr/bin/env python3
"""Synthetic reading task: family trees with chains of 'parent of' relations.

Usage: python gen.py N SEED   (prints N stories; all randomness from random.Random(SEED))

Each story states the parent of every person (one fact per person, in shuffled
order). Questions may appear between facts and only use facts stated above them.
Answers are computed by simulating what the known facts establish; a question is
asked only when its answer is fully determined by those facts.
"""
import random
import sys

NAMES = ["ann", "bob", "cara", "dan", "eli", "fay", "gus", "hal", "ivy", "jo", "kim", "lee"]


def ancestor_chain(known, person):
    """Ancestors of person established by the known facts, nearest first.

    Returns (chain, complete). complete is True when the top of the chain is
    known to have no parent, so nothing else can be an ancestor.
    """
    chain = []
    cur = person
    while True:
        if cur not in known:
            return chain, False
        par = known[cur]
        if par is None:
            return chain, True
        chain.append(par)
        cur = par


def fact_text(fact):
    if fact[0] == "root":
        return "%s has no parent." % fact[1]
    _, parent, child, style = fact
    if style == "parent":
        return "%s is the parent of %s." % (parent, child)
    return "%s is the child of %s." % (child, parent)


def apply_fact(known, fact):
    if fact[0] == "root":
        known[fact[1]] = None
    else:
        _, parent, child, _ = fact
        known[child] = parent


def candidates(known):
    """All (question, answer) pairs that the known facts determine and that need 2+ links."""
    out = []
    people = sorted(known)
    for x in people:
        chain, _ = ancestor_chain(known, x)
        if len(chain) >= 2:
            out.append(("Who is the grandparent of %s?" % x, chain[1]))
        if len(chain) >= 3:
            out.append(("Who is the great grandparent of %s?" % x, chain[2]))
    for a in people:
        for b in people:
            if a == b:
                continue
            chain, complete = ancestor_chain(known, b)
            if a in chain:
                if chain.index(a) >= 1:
                    out.append(("Is %s an ancestor of %s?" % (a, b), "yes"))
            elif complete and len(chain) >= 2:
                out.append(("Is %s an ancestor of %s?" % (a, b), "no"))
    return out


def try_story(rng):
    n = rng.randint(4, 8)
    people = rng.sample(NAMES, n)
    rng.shuffle(people)

    # Random forest: each person has no parent, or an earlier person as parent.
    parent = {}
    for i, p in enumerate(people):
        if i == 0 or rng.random() < 0.3:
            parent[p] = None
        else:
            parent[p] = people[rng.randrange(i)]

    facts = []
    for p in people:
        par = parent[p]
        if par is None:
            facts.append(("root", p))
        else:
            facts.append(("edge", par, p, rng.choice(["parent", "child"])))
    rng.shuffle(facts)

    target = rng.randint(1, 5)
    known = {}
    items = []  # (text, answer or None)
    asked = set()

    def try_question():
        if sum(1 for _, a in items if a is not None) >= target:
            return
        cands = [c for c in candidates(known) if c[0] not in asked]
        if not cands:
            return
        q, a = rng.choice(cands)
        asked.add(q)
        items.append((q, a))

    for f in facts:
        apply_fact(known, f)
        items.append((fact_text(f), None))
        if rng.random() < 0.5:
            try_question()
    while sum(1 for _, a in items if a is not None) < target:
        before = len(items)
        try_question()
        if len(items) == before:
            break

    if sum(1 for _, a in items if a is not None) == 0:
        return None
    return items


def make_story(rng):
    while True:
        items = try_story(rng)
        if items is not None:
            return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        items = make_story(rng)
        for i, (text, ans) in enumerate(items, 1):
            if ans is None:
                out.append("%d %s" % (i, text))
            else:
                out.append("%d %s\t%s" % (i, text, ans))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
