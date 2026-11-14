#!/usr/bin/env python3
"""Synthetic ordering task.

Each story has a hidden order of a few chores (a bakery day). Statements give
pairwise "done before / done after" facts. Questions are yes/no questions about
the order of two chores that is only implied by chaining several statements
(never stated directly), or "What is done last?" over the chores named so far.
Answers are computed from the statements above each question, so they are
unambiguous.
"""
import random
import sys

TASKS = ["mix", "stir", "bake", "cool", "frost", "pack", "wash", "dry"]


def closure(direct):
    """Transitive closure of the 'is done before' relation (set of (a, b))."""
    rel = set(direct)
    changed = True
    while changed:
        changed = False
        for a, b in list(rel):
            for c, d in list(rel):
                if b == c and a != d and (a, d) not in rel:
                    rel.add((a, d))
                    changed = True
    return rel


def statement(rng, a, b):
    """Say that task a is done before task b, with one of two patterns."""
    if rng.random() < 0.5:
        return "The %s is done before the %s." % (a, b)
    return "The %s is done after the %s." % (b, a)


def candidates(direct, mentioned):
    """All (question, answer) pairs that the statements so far determine."""
    rel = closure(direct)
    cands = []
    for x in mentioned:
        for y in mentioned:
            if x == y or (x, y) in direct or (y, x) in direct:
                continue  # skip self pairs and pairs stated directly
            if (x, y) not in rel and (y, x) not in rel:
                continue  # order not determined yet
            before_ans = "yes" if (x, y) in rel else "no"
            after_ans = "yes" if (x, y) in rel else "no"  # "y after x" iff x before y
            cands.append(("Is the %s done before the %s?" % (x, y), before_ans))
            cands.append(("Is the %s done after the %s?" % (y, x), after_ans))
    if len(mentioned) >= 3:
        for x in mentioned:
            if all((y, x) in rel for y in mentioned if y != x):
                cands.append(("What is done last?", x))
    return cands


def make_story(rng):
    while True:
        m = rng.randint(3, len(TASKS))
        tasks = rng.sample(TASKS, m)
        rank = {t: i for i, t in enumerate(tasks)}
        pairs = [(a, b) for a in tasks for b in tasks if rank[a] < rank[b]]
        n_st = rng.randint(3, min(12, len(pairs)))
        chosen = rng.sample(pairs, n_st)

        lines = []
        direct = set()
        mentioned = []
        q = 0
        for a, b in chosen:
            direct.add((a, b))
            for t in (a, b):
                if t not in mentioned:
                    mentioned.append(t)
            lines.append(statement(rng, a, b))
            if q < 5 and rng.random() < 0.4:
                cands = candidates(direct, mentioned)
                if cands:
                    text, ans = rng.choice(cands)
                    lines.append(text + "\t" + ans)
                    q += 1

        if q == 0:
            cands = candidates(direct, mentioned)
            if not cands:
                continue  # nothing answerable: draw another story
            text, ans = rng.choice(cands)
            lines.append(text + "\t" + ans)
            q += 1

        return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
