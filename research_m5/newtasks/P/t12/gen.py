#!/usr/bin/env python3
"""Synthetic set-algebra task written as records.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10, default 1)

An episode is a log of operations on named sets (m1..m8) that hold items
(k1..k8), followed by queries about the sets as they stand at that point.
Answers come from simulating the log.
"""

import random
import sys

ADD, DROP, UNION, INTER, COPY = "qa", "vo", "ze", "xi", "cp"
HAS, COUNT, LIST = "hv", "nu", "ls"
REC_OPS = [ADD, DROP, UNION, INTER, COPY]
QRY_OPS = [HAS, COUNT, LIST]
ITEMS = ["k%d" % i for i in range(1, 9)]
SETS = ["m%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)
MAX_EPISODE_TRIES = 2000
MAX_QUERY_TRIES = 200


class Retry(Exception):
    pass


def writer(recs, name, item, upto):
    """Index of the last record before `upto` that decides whether `item`
    is in set `name`, or None if no record has done so."""
    for i in range(upto - 1, -1, -1):
        op, a = recs[i]
        if op in (ADD, DROP):
            if a[0] == name and a[1] == item:
                return i
        elif a[0] == name:  # union / intersect / copy overwrite the whole set
            return i
    return None


def support(recs, name, item, upto):
    """Set of record indices that the membership of `item` in `name` depends on."""
    i = writer(recs, name, item, upto)
    if i is None:
        return set()
    op, a = recs[i]
    out = {i}
    if op in (UNION, INTER):
        out |= support(recs, a[1], item, i) | support(recs, a[2], item, i)
    elif op == COPY:
        out |= support(recs, a[1], item, i)
    return out


def apply(state, op, a):
    if op == ADD:
        state[a[0]].add(a[1])
    elif op == DROP:
        state[a[0]].discard(a[1])
    elif op == UNION:
        state[a[0]] = state[a[1]] | state[a[2]]
    elif op == INTER:
        state[a[0]] = state[a[1]] & state[a[2]]
    elif op == COPY:
        state[a[0]] = set(state[a[1]])


def random_record(rng):
    op = rng.choices(REC_OPS, weights=[5, 2, 2, 1, 2])[0]
    if op in (ADD, DROP):
        return op, (rng.choice(SETS), rng.choice(ITEMS))
    if op in (UNION, INTER):
        return op, tuple(rng.sample(SETS, 3))
    return op, tuple(rng.sample(SETS, 2))


def render_record(op, a):
    if op in (ADD, DROP):
        return "%s %s %s." % (op, a[0], a[1])
    return " ".join([op] + list(a)) + "."


def random_query(rng, recs, state):
    """Return (query text, answer) for a query that needs at least two records."""
    for _ in range(MAX_QUERY_TRIES):
        op = rng.choice(QRY_OPS)
        name = rng.choice(SETS)
        if op == HAS:
            item = rng.choice(ITEMS)
            if len(support(recs, name, item, len(recs))) < 2:
                continue
            ans = "yes" if item in state[name] else "no"
            return "%s %s %s?" % (HAS, name, item), ans
        sup = set()
        for item in ITEMS:
            sup |= support(recs, name, item, len(recs))
        if len(sup) < 2:
            continue
        if op == COUNT:
            return "%s %s?" % (COUNT, name), str(len(state[name]))
        if not state[name]:  # an empty list has no code to answer with
            continue
        # single-digit item codes, so lexicographic order is numeric order
        return "%s %s?" % (LIST, name), ",".join(sorted(state[name]))
    raise Retry()


def build_episode(rng, size):
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    kinds = ["r"] * n_rec + ["q"] * n_q
    rng.shuffle(kinds)
    kinds.insert(0, kinds.pop(kinds.index("r")))  # the log starts with a record

    recs = []
    state = {m: set() for m in SETS}
    lines = []
    for kind in kinds:
        row = len(lines) + 1
        if kind == "r":
            op, a = random_record(rng)
            recs.append((op, a))
            apply(state, op, a)
            lines.append("%d %s" % (row, render_record(op, a)))
        else:
            text, ans = random_query(rng, recs, state)
            lines.append("%d %s\t%s" % (row, text, ans))
    return lines


def episode(rng, size):
    for _ in range(MAX_EPISODE_TRIES):
        try:
            return build_episode(rng, size)
        except Retry:
            continue
    raise RuntimeError("could not build an episode")


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2

    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            lines = episode(rng, size)
            text = "\n".join(lines)
            if text not in seen:
                break
        seen.add(text)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
