#!/usr/bin/env python3
"""Synthetic 'height ranking' reading-comprehension task generator.

Usage: python gen.py N SEED

Each story is a set of statements about the relative height of a few people
(taller / shorter, always in one of two fixed patterns), followed by questions
that can only be answered by chaining several statements together:
  - "Is X taller than Y?"  (pairs never compared directly in the story)
  - "Who is the tallest of X and Y and Z?"
Answers are computed by a closure over the statements placed above each
question, so they are always consistent with a hidden true height order.
"""
import itertools
import random
import sys

NAMES = ["Mira", "Tomas", "Lena", "Omar", "Iris", "Felix",
         "Nadia", "Ruben", "Clara", "Hugo", "Yuki", "Ada"]
MIN_STMT, MAX_STMT = 3, 12
MIN_Q, MAX_Q = 1, 5
MAX_FACTS = 9
MAX_ATTEMPTS = 10000


class Retry(Exception):
    pass


def fact_text(taller, shorter, rng):
    # Two fixed patterns that express the same comparison.
    if rng.random() < 0.5:
        return f"{taller} is taller than {shorter}."
    return f"{shorter} is shorter than {taller}."


def closure(edges):
    """Transitive closure of (taller, shorter) pairs."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    reach = set()
    for start in adj:
        seen = set()
        stack = list(adj[start])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj.get(x, ()))
        for x in seen:
            reach.add((start, x))
    return reach


def make_question(prefix, direct, names, rng):
    """Return (question_text, answer) answerable from prefix, or None."""
    reach = closure(prefix)
    kinds = []

    # Yes/no about a pair related only through a chain.
    yn = sorted((a, b) for (a, b) in reach
                if frozenset((a, b)) not in direct)
    if yn:
        kinds.append("yn")

    # Tallest of three, where the winner beats both others and at least
    # one of those comparisons needs a chain.
    cand = []
    for x in sorted(names):
        others = [m for m in sorted(names) if m != x]
        for y, z in itertools.combinations(others, 2):
            if (x, y) in reach and (x, z) in reach:
                if not (frozenset((x, y)) in direct and
                        frozenset((x, z)) in direct):
                    cand.append((x, y, z))
    if cand:
        kinds.append("tallest")

    if not kinds:
        return None
    kind = rng.choice(kinds)

    if kind == "yn":
        a, b = rng.choice(yn)  # a is taller than b in the closure
        if rng.random() < 0.5:
            return f"Is {a} taller than {b}?", "yes"
        return f"Is {b} taller than {a}?", "no"

    x, y, z = rng.choice(cand)
    p, q, r = rng.sample([x, y, z], 3)
    return f"Who is the tallest of {p} and {q} and {r}?", x.lower()


def build_story(rng):
    n = rng.randint(3, 6)
    names = rng.sample(NAMES, n)
    order = names[:]  # hidden true order, order[0] is tallest
    all_pairs = [(order[i], order[j])
                 for i in range(n) for j in range(i + 1, n)]

    Q = rng.randint(MIN_Q, MAX_Q)
    max_f = min(MAX_FACTS, len(all_pairs), MAX_STMT - Q)
    F = rng.randint(2, max_f)
    facts = rng.sample(all_pairs, F)
    direct = {frozenset(p) for p in facts}

    T = F + Q
    q_slots = set(rng.sample(range(1, T), Q))  # slot 0 is always a fact

    prefix = []
    entries = []
    fi = 0
    for slot in range(T):
        if slot in q_slots:
            q = make_question(prefix, direct, names, rng)
            if q is None:
                raise Retry()
            text, ans = q
            entries.append(f"{text}\t{ans}")
        else:
            a, b = facts[fi]
            fi += 1
            prefix.append((a, b))
            entries.append(fact_text(a, b, rng))
    return entries


def make_story(rng):
    for _ in range(MAX_ATTEMPTS):
        try:
            return build_story(rng)
        except Retry:
            continue
    raise RuntimeError("could not build a story")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(2)
    n_stories = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n_stories):
        for i, entry in enumerate(make_story(rng), 1):
            out.append(f"{i} {entry}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
