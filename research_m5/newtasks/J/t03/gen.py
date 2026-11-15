#!/usr/bin/env python3
"""Synthetic order-in-time reading comprehension stories.

Usage: python gen.py N SEED

Each story: people start in places. Each person's moves happen at hidden
times 1..m. The moves are told out of order, either with an absolute time
("... at time 3.") or relative to another move ("... before/after ...").
Questions come after all statements. A question is kept only if its answer
is the same in every ordering of the moves consistent with the statements,
and if no single move statement already determines it.
"""
import random
import sys
from itertools import permutations

NAMES = ["sam", "mary", "john", "anna", "tom", "lily"]
PLACES = ["kitchen", "garden", "hall", "office", "cellar"]
MOVE_COUNTS = [3, 4, 5]


def satisfies(perm, constraints):
    """perm[i] is the time of move i. Constraints: ("abs", i, t) or ("lt", a, b)."""
    for c in constraints:
        if c[0] == "abs":
            if perm[c[1]] != c[2]:
                return False
        elif perm[c[1]] >= perm[c[2]]:
            return False
    return True


def try_story(rng):
    people = rng.sample(NAMES, rng.choice([2, 3]))
    starts = {p: rng.choice(PLACES) for p in people}
    m = rng.choice(MOVE_COUNTS)
    events = rng.sample([(p, l) for p in people for l in PLACES], m)
    times = list(range(1, m + 1))
    rng.shuffle(times)

    stmts = []  # (text, constraint or None)
    for p in people:
        stmts.append((f"{p.capitalize()} started in the {starts[p]}.", None))
    for i, (p, l) in enumerate(events):
        P = p.capitalize()
        if rng.random() < 0.5:
            text = f"{P} went to the {l} at time {times[i]}."
            con = ("abs", i, times[i])
        else:
            j = rng.choice([x for x in range(m) if x != i])
            q, l2 = events[j]
            Q = q.capitalize()
            if times[i] < times[j]:
                text = f"{P} went to the {l} before {Q} went to the {l2}."
                con = ("lt", i, j)
            else:
                text = f"{P} went to the {l} after {Q} went to the {l2}."
                con = ("lt", j, i)
        stmts.append((text, con))
    rng.shuffle(stmts)
    cons = [c for _, c in stmts if c is not None]

    all_perms = list(permutations(range(1, m + 1)))
    consistent = [pm for pm in all_perms if satisfies(pm, cons)]
    assert tuple(times) in consistent

    def build(pm):
        # table[k] = location of every person after time k
        table = []
        for k in range(m + 1):
            loc = dict(starts)
            for i in sorted(range(m), key=lambda x: pm[x]):
                if pm[i] <= k:
                    loc[events[i][0]] = events[i][1]
            table.append(loc)
        return table

    tbl = {pm: build(pm) for pm in all_perms}

    def answer(q, pm):
        kind = q[0]
        if kind == "where":
            return tbl[pm][q[2]][q[1]]
        if kind == "was":
            return "yes" if tbl[pm][q[3]][q[1]] == q[2] else "no"
        if kind == "who":
            loc = tbl[pm][q[2]]
            names = sorted(p for p in people if loc[p] == q[1])
            return ",".join(names) if names else "nobody"
        return "yes" if pm[q[1]] < pm[q[2]] else "no"

    def qtext(q):
        kind = q[0]
        if kind == "where":
            return f"Where was {q[1].capitalize()} at time {q[2]}"
        if kind == "was":
            return f"Was {q[1].capitalize()} in the {q[2]} at time {q[3]}"
        if kind == "who":
            return f"Who was in the {q[1]} at time {q[2]}"
        i, j = q[1], q[2]
        return (f"Did {events[i][0].capitalize()} go to the {events[i][1]} "
                f"before {events[j][0].capitalize()} went to the {events[j][1]}")

    cands = []
    for p in people:
        for k in range(1, m + 1):
            cands.append(("where", p, k))
            for l in PLACES:
                cands.append(("was", p, l, k))
    for l in PLACES:
        for k in range(1, m + 1):
            cands.append(("who", l, k))
    for i in range(m):
        for j in range(m):
            if i != j:
                cands.append(("did", i, j))

    single_sets = [[pm for pm in all_perms if satisfies(pm, [c])] for c in cons]

    valid = []
    for q in cands:
        got = {answer(q, pm) for pm in consistent}
        if len(got) != 1:
            continue
        if any(len({answer(q, pm) for pm in S}) == 1 for S in single_sets):
            continue
        valid.append((q, got.pop()))
    if not valid:
        return None

    chosen = rng.sample(valid, rng.randint(1, min(5, len(valid))))
    lines = [text for text, _ in stmts]
    for q, a in chosen:
        lines.append(qtext(q) + "?\t" + a)
    return [f"{n} {line}" for n, line in enumerate(lines, 1)]


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 1
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    stories = ["\n".join(make_story(rng)) for _ in range(n)]
    sys.stdout.write("\n\n".join(stories) + ("\n" if stories else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
