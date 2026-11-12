#!/usr/bin/env python3
"""Synthetic record task: signal propagation through logic gates.

Rows drive wires to 0 or 1: a set row gives a wire a value, AND/OR rows combine
two wires into a third, NOT rows invert one wire into another. A later row that
drives a wire overwrites its earlier value. Queries ask whether a wire is high at
that point in the log, or whether two wires agree there.

Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

WIRES = ["w%d" % i for i in range(8)]
SET, AND, OR, NOT = "sv", "ja", "ku", "ne"
Q_HIGH, Q_SAME = "hx", "tx"
SIZES = (1, 3, 10)


def make_records(rng, size):
    """Return the record rows and snapshots; snaps[p] is the state after p rows."""
    target = rng.randint(3 * size, 12 * size)
    state = {}  # wire -> (value, frozenset of record indices that determined it)
    rows = []
    snaps = [dict(state)]
    for idx in range(1, target + 1):
        driven = sorted(state)
        if len(driven) < 2 or rng.random() < 0.2:
            op = SET
        else:
            op = rng.choices([AND, OR, NOT], weights=[4, 4, 2])[0]
        out = rng.choice(WIRES)
        if op == SET:
            v = rng.randint(0, 1)
            state[out] = (v, frozenset([idx]))
            tokens = [SET, out, str(v)]
        elif op == NOT:
            a = rng.choice(driven)
            va, pa = state[a]
            state[out] = (1 - va, pa | frozenset([idx]))
            tokens = [NOT, a, out]
        else:
            a, b = rng.sample(driven, 2)
            va, pa = state[a]
            vb, pb = state[b]
            v = (va & vb) if op == AND else (va | vb)
            state[out] = (v, pa | pb | frozenset([idx]))
            tokens = [op, a, b, out]
        rows.append(" ".join(tokens) + ".")
        snaps.append(dict(state))
    return rows, snaps


def make_queries(rng, snaps, nq):
    """Pick up to nq queries whose answers depend on at least two records."""
    cands = []
    for p in range(2, len(snaps)):
        snap = snaps[p]
        wires = sorted(snap)
        for w in wires:
            if len(snap[w][1]) >= 2:
                cands.append((p, Q_HIGH, (w,)))
        for i, a in enumerate(wires):
            for b in wires[i + 1:]:
                if len(snap[a][1] | snap[b][1]) >= 2:
                    cands.append((p, Q_SAME, (a, b)))
    if not cands:
        return []
    return rng.sample(cands, min(nq, len(cands)))


def answer(snap, q, args):
    if q == Q_HIGH:
        return "yes" if snap[args[0]][0] == 1 else "no"
    return "yes" if snap[args[0]][0] == snap[args[1]][0] else "no"


def make_episode(rng, size):
    rows, snaps = make_records(rng, size)
    queries = make_queries(rng, snaps, rng.randint(1, 5))
    if not queries:
        return None
    by_pos = {}
    for p, q, args in queries:
        by_pos.setdefault(p, []).append((q, args))
    out = []
    for p in range(1, len(rows) + 1):
        out.append(rows[p - 1])
        for q, args in by_pos.get(p, []):
            text = " ".join([q] + list(args)) + "?"
            out.append(text + "\t" + answer(snaps[p], q, args))
    return ["%d %s" % (i, line) for i, line in enumerate(out, 1)]


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in SIZES:
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            ep = make_episode(rng, size)
            if ep is None or tuple(ep) in seen:
                continue
            seen.add(tuple(ep))
            break
        for line in ep:
            print(line)


if __name__ == "__main__":
    main(sys.argv)
