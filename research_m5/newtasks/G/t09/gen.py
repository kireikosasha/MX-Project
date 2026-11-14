#!/usr/bin/env python3
"""Generate synthetic family-chain reading-comprehension stories.

Usage: python gen.py N SEED

Each story states parent links between people in simple sentences. Questions
ask about grandparents and ancestors, so the answer has to be found by
following a chain of two or more stated links. Answers are computed from the
simulated family tree, never written by hand.
"""
import random
import sys

NAMES = ["ann", "bob", "cara", "dan", "eve", "finn", "gina", "hugo",
         "ivy", "jack", "kate", "leo", "mia", "noah"]


def fact_text(rng, parent, child):
    """Return one of two fixed sentence patterns for the parent link."""
    if rng.random() < 0.5:
        return f"{parent} is the parent of {child}."
    return f"{child} is the child of {parent}."


def ancestors(parent_of, person):
    """Ancestors of person, nearest first (follows the chain upward)."""
    result = []
    p = parent_of.get(person)
    while p is not None:
        result.append(p)
        p = parent_of.get(p)
    return result


def subtree(children_of, person):
    """The person and all people below them."""
    seen = {person}
    stack = [person]
    while stack:
        cur = stack.pop()
        for c in children_of[cur]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def make_question(rng, parent_of, mentioned):
    """Return (question text without '?', answer) about the stated facts."""
    with_parent = [x for x in mentioned if x in parent_of]
    if with_parent and rng.random() < 0.5:
        x = rng.choice(with_parent)
        g = parent_of.get(parent_of[x])
        answer = g if g is not None else "none"
        return f"who is the grandparent of {x}", answer

    deep = [(a, b) for b in mentioned for a in ancestors(parent_of, b)[1:]]
    if deep and rng.random() < 0.6:
        a, b = rng.choice(deep)
    else:
        a, b = rng.sample(mentioned, 2)
    answer = "yes" if a in ancestors(parent_of, b) else "no"
    return f"is {a} the ancestor of {b}", answer


def make_story(rng):
    """Return the numbered lines of one story."""
    n_stmts = rng.randint(3, 12)
    n_qs = rng.randint(1, 5)
    n_people = min(len(NAMES), n_stmts + 2 + rng.randint(0, 3))
    people = rng.sample(NAMES, n_people)

    parent_of = {}                          # child -> parent (stated facts)
    children_of = {p: [] for p in people}
    mentioned = []                          # people named so far
    texts = []                              # (text, answer or None)

    left_stmts, left_qs = n_stmts, n_qs
    first = True
    while left_stmts or left_qs:
        # The first sentence is always a statement, so every question has
        # at least one fact above it.
        make_q = (not first) and left_qs > 0 and (
            left_stmts == 0
            or rng.random() < left_qs / (left_qs + left_stmts))
        if make_q:
            text, answer = make_question(rng, parent_of, mentioned)
            texts.append((text + "?", answer))
            left_qs -= 1
        else:
            # Pick an unparented person as child and any person outside
            # that child's subtree as parent, so no cycle can form.
            roots = [p for p in people if p not in parent_of]
            child = rng.choice(roots)
            sub = subtree(children_of, child)
            parent = rng.choice([p for p in people if p not in sub])
            parent_of[child] = parent
            children_of[parent].append(child)
            for name in (parent, child):
                if name not in mentioned:
                    mentioned.append(name)
            texts.append((fact_text(rng, parent, child), None))
            left_stmts -= 1
        first = False

    lines = []
    for i, (text, answer) in enumerate(texts, start=1):
        if answer is None:
            lines.append(f"{i} {text}")
        else:
            lines.append(f"{i} {text}\t{answer}")
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
