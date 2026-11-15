#!/usr/bin/env python3
"""Synthetic reading-comprehension stories about the order of events in time.

Usage: python gen.py N SEED

Each story tells the order of some moves ("ann went to park before bob went
to shop.") and where people were at the start. Questions at the end ask where
someone was before or after a move, whether a move happened before another
move, or who went somewhere before a move. Answers are computed by simulating
every ordering of the story's moves that agrees with the statements; a
question is kept only if all those orderings give the same answer.
"""
import itertools
import random
import sys

NAMES = ["ann", "bob", "cat", "dan", "eve"]
PLACES = ["house", "park", "shop", "school", "farm", "lake"]


def statement_text(fact, rng):
    if fact[0] == "order":
        a, b = fact[1], fact[2]  # a happened earlier than b
        if rng.random() < 0.5:
            return "%s went to %s before %s went to %s." % (a[0], a[1], b[0], b[1])
        return "%s went to %s after %s went to %s." % (b[0], b[1], a[0], a[1])
    return "at start %s was at %s." % (fact[1], fact[2])


def question_text(q):
    if q[0] == "where":
        _, x, r, after = q
        return "where was %s %s %s went to %s?" % (
            x, "after" if after else "before", r[0], r[1])
    if q[0] == "did":
        _, x, pl, r = q
        return "did %s go to %s before %s went to %s?" % (x, pl, r[0], r[1])
    _, pl, r = q
    return "who went to %s before %s went to %s?" % (pl, r[0], r[1])


def answer(q, perm, starts):
    """Answer question q in one ordering perm of the moves.

    Returns None when the answer is unknown (no known start and no earlier
    move), or "" when a list answer is empty.
    """
    if q[0] == "where":
        _, x, r, after = q
        cut = perm.index(r) + (1 if after else 0)
        loc = starts.get(x)
        for m in perm[:cut]:
            if m[0] == x:
                loc = m[1]
        return loc
    if q[0] == "did":
        _, x, pl, r = q
        return "yes" if perm.index((x, pl)) < perm.index(r) else "no"
    _, pl, r = q
    idx = perm.index(r)
    names = sorted({m[0] for m in perm[:idx] if m[1] == pl})
    return ",".join(names)


def determined(q, perms, starts):
    """Return the answer if it is the same in every consistent ordering."""
    vals = set()
    for perm in perms:
        v = answer(q, perm, starts)
        if not v:
            return None
        vals.add(v)
        if len(vals) > 1:
            return None
    return vals.pop() if vals else None


def try_story(rng):
    people = rng.sample(NAMES, rng.randint(2, 4))
    pairs = [(p, pl) for p in people for pl in PLACES]
    moves = rng.sample(pairs, rng.randint(3, 5))  # true time order: list order
    start = {p: rng.choice(PLACES) for p in people}

    cands = []
    for i in range(len(moves)):
        for j in range(i + 1, len(moves)):
            cands.append(("order", moves[i], moves[j]))
    for p in people:
        cands.append(("start", p, start[p]))
    k = rng.randint(3, 10)
    if k > len(cands):
        return None
    facts = rng.sample(cands, k)

    # The world is the set of moves the statements mention.
    mentioned = set()
    for f in facts:
        if f[0] == "order":
            mentioned.add(f[1])
            mentioned.add(f[2])
    world = [m for m in moves if m in mentioned]
    if len(world) < 2:
        return None
    known_start = {f[1]: f[2] for f in facts if f[0] == "start"}
    constraints = [(f[1], f[2]) for f in facts if f[0] == "order"]
    perms = [p for p in itertools.permutations(world)
             if all(p.index(a) < p.index(b) for a, b in constraints)]
    true_perm = tuple(world)

    cands_q = []
    for x in people:
        for r in world:
            cands_q.append(("where", x, r, False))
            cands_q.append(("where", x, r, True))
    for m in world:
        for r in world:
            if m != r:
                cands_q.append(("did", m[0], m[1], r))
    for pl in PLACES:
        for r in world:
            cands_q.append(("who", pl, r))
    rng.shuffle(cands_q)

    questions = []
    for q in cands_q:
        det = determined(q, perms, known_start)
        if det is None:
            continue
        # The simulation of the true world must agree with the determined answer.
        if answer(q, true_perm, start) != det:
            continue
        questions.append((q, det))
        if len(questions) == 5:
            break
    if not questions:
        return None
    questions = questions[:rng.randint(1, len(questions))]

    lines = []
    n = 0
    for f in facts:
        n += 1
        lines.append("%d %s" % (n, statement_text(f, rng)))
    for q, a in questions:
        n += 1
        lines.append("%d %s\t%s" % (n, question_text(q), a))
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
