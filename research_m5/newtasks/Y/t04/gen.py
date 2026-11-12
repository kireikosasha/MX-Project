#!/usr/bin/env python3
"""Synthetic record-log task: pairwise match tallies between teams.

Usage: python gen.py N SEED [SIZE]      SIZE is 1, 3 or 10 (default 1).

Record row:  pex A B K.     K matches were played between teams A and B
                             (order of A and B is irrelevant).
Queries:     ruv A B?  -> yes/no  were any matches played between A and B?
             zim A B?  -> number  total matches played between A and B
             duk A?    -> list    teams that A has played, sorted, comma-separated
"""
import random
import sys

RECORD_OP = "pex"
QUERY_OPS = ("ruv", "zim", "duk")
TEAMS = ["t%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)


def pair_records(prefix, x, y):
    return [r for r in prefix if {r[0], r[1]} == {x, y}]


def touching(prefix, x):
    return [r for r in prefix if x in (r[0], r[1])]


def candidates(prefix, teams):
    """Queries answerable from prefix that need at least two records.

    Returns a list of (op, args, answer) with canonical args.
    """
    out = []
    for i, x in enumerate(teams):
        tx = touching(prefix, x)
        if len(tx) >= 2:
            opp = sorted({b if a == x else a for a, b, _ in tx})
            out.append(("duk", (x,), ",".join(opp)))
        for y in teams[i + 1:]:
            pr = pair_records(prefix, x, y)
            if len(pr) >= 2:
                out.append(("ruv", (x, y), "yes"))
                out.append(("zim", (x, y), str(sum(r[2] for r in pr))))
            elif not pr and len(tx) + len(touching(prefix, y)) >= 2:
                out.append(("ruv", (x, y), "no"))
    return out


def render_record(rng, a, b, k):
    if rng.random() < 0.5:
        a, b = b, a
    return "%s %s %s %d." % (RECORD_OP, a, b, k)


def render_query(rng, op, args, ans):
    if op == "duk":
        text = "duk %s?" % args[0]
    else:
        x, y = args
        if rng.random() < 0.5:
            x, y = y, x
        text = "%s %s %s?" % (op, x, y)
    return text + "\t" + ans


def episode(rng, size):
    while True:
        teams = sorted(rng.sample(TEAMS, rng.randint(3, 8)))
        n = rng.randint(3 * size, 12 * size)
        recs = []
        for _ in range(n):
            a, b = rng.sample(teams, 2)
            recs.append((a, b, rng.randint(1, 9 * size)))
        nq = rng.randint(1, 5)
        cuts = sorted(rng.randint(2, n) for _ in range(nq))

        rows = []
        used = set()
        pos = 0
        ok = True
        for cut in cuts:
            for a, b, k in recs[pos:cut]:
                rows.append(render_record(rng, a, b, k))
            pos = cut
            cands = [c for c in candidates(recs[:cut], teams)
                     if (c[0], c[1]) not in used]
            if not cands:
                ok = False
                break
            op, args, ans = rng.choice(cands)
            used.add((op, args))
            rows.append(render_query(rng, op, args, ans))
        if not ok:
            continue
        for a, b, k in recs[pos:]:
            rows.append(render_record(rng, a, b, k))
        return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n_ep = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    lines = []
    for _ in range(n_ep):
        for i, row in enumerate(episode(rng, size), 1):
            lines.append("%d %s" % (i, row))
    sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
