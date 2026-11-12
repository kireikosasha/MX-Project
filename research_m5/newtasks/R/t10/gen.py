#!/usr/bin/env python3
"""Synthetic GROUPS record task.

Persons (p1..p8) join, leave and move between groups (g1..g4). Rows are
log-like records; some rows are queries asking which group a person is in
now, who is in a group, or how many people are in a group. Answers come from
a simulation of the records above each query.

Usage: python gen.py N SEED [SIZE]     SIZE in {1, 3, 10}, default 1
"""
import random
import sys

PERSONS = ["p%d" % i for i in range(1, 9)]
GROUPS = ["g%d" % i for i in range(1, 5)]

# Record ops: jn = join a group (not in one), xt = leave current group,
# mv = move from current group to another group.
REC_OPS = ["jn", "xt", "mv"]
REC_WEIGHTS = {"jn": 4, "xt": 2, "mv": 4}

# Query ops: wh = which group is person in, mb = members of group,
# ct = number of members of group.
QRY_OPS = ["wh", "mb", "ct"]

SIZES = (1, 3, 10)


def make_records(rng, n):
    """Simulate n record rows.

    Returns (rows, snaps): rows is a list of token lists; snaps[i] is the
    state (person -> group or None) after the first i records.
    """
    state = {p: None for p in PERSONS}
    rows = []
    snaps = [dict(state)]
    for _ in range(n):
        free = [p for p in PERSONS if state[p] is None]
        inside = [p for p in PERSONS if state[p] is not None]
        options = []
        if free:
            options.append("jn")
        if inside:
            options.append("xt")
            options.append("mv")
        weights = [REC_WEIGHTS[o] for o in options]
        op = rng.choices(options, weights)[0]
        if op == "jn":
            p = rng.choice(free)
            g = rng.choice(GROUPS)
            state[p] = g
            rows.append(["jn", p, g])
        elif op == "xt":
            p = rng.choice(inside)
            rows.append(["xt", p])
            state[p] = None
        else:
            p = rng.choice(inside)
            g = rng.choice([x for x in GROUPS if x != state[p]])
            state[p] = g
            rows.append(["mv", p, g])
        snaps.append(dict(state))
    return rows, snaps


def try_query(rng, rows, snaps):
    """Pick one random query whose answer is defined and needs >= 2 records.

    Returns (cut, op, subject, answer) where cut is the number of records
    above the query, or None if this draw does not work.
    """
    op = rng.choice(QRY_OPS)
    k = rng.randint(1, len(rows))
    st = snaps[k]
    if op == "wh":
        subj = rng.choice(PERSONS)
        ans = st[subj]
        if ans is None:
            return None
        relevant = {subj}
    else:
        subj = rng.choice(GROUPS)
        members = [p for p in PERSONS if st[p] == subj]
        if op == "mb":
            if not members:
                return None
            ans = ",".join(members)
        else:
            ans = str(len(members))
        relevant = {subj} | set(members)
    support = sum(1 for toks in rows[:k] if relevant.intersection(toks))
    if support < 2:
        return None
    return k, op, subj, ans


def make_episode(rng, size):
    while True:
        n_rec = rng.randint(3 * size, 12 * size)
        n_qry = rng.randint(1, 5)
        rows, snaps = make_records(rng, n_rec)
        queries = []
        tries = 0
        while len(queries) < n_qry and tries < 400:
            tries += 1
            q = try_query(rng, rows, snaps)
            if q is not None:
                queries.append(q)
        if len(queries) == n_qry:
            break

    by_cut = {}
    for q in queries:
        by_cut.setdefault(q[0], []).append(q)

    body = []
    for i, toks in enumerate(rows, start=1):
        body.append(" ".join(toks) + ".")
        for _, op, subj, ans in by_cut.get(i, []):
            body.append(op + " " + subj + "?\t" + ans)

    return "\n".join("%d %s" % (num, line)
                     for num, line in enumerate(body, start=1))


def main(argv):
    if len(argv) < 3 or len(argv) > 4:
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
            ep = make_episode(rng, size)
            if ep not in seen:
                break
        seen.add(ep)
        print(ep)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
