#!/usr/bin/env python3
"""Synthetic ledger task given as records.

A ledger holds amounts for holders. Records set an amount, add to it, remove
from it, or move amounts between two holders. Queries ask how much one holder
has now, which holder of a group has the most, or which holders have at least
a threshold. Every answer comes from a simulation of the records above it.

Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

HOLDERS = ["h%d" % i for i in range(1, 9)]          # one entity kind: letter h, digits 1-8
REC_OPS = [("zar", 15), ("pol", 30), ("kes", 20), ("vut", 35)]  # set, add, remove, move
QRY_OPS = ["ket", "mox", "lis"]                      # how much, most, at least


def amount_cap(size):
    return 20 * size


def step_cap(size):
    return 9 * size


def make_record(rng, state, ep_holders, size):
    """Pick and apply one record to state. Returns (text, holders mentioned)."""
    ops, weights = zip(*REC_OPS)
    op = rng.choices(ops, weights)[0]
    nonzero = [h for h in ep_holders if state[h] > 0]
    if op in ("kes", "vut") and not nonzero:
        op = "pol"
    if op == "zar":
        h = rng.choice(ep_holders)
        v = rng.randint(0, amount_cap(size))
        state[h] = v
        return "zar %s %d." % (h, v), (h,)
    if op == "pol":
        h = rng.choice(ep_holders)
        n = rng.randint(1, step_cap(size))
        state[h] += n
        return "pol %s %d." % (h, n), (h,)
    if op == "kes":
        h = rng.choice(nonzero)
        n = rng.randint(1, min(state[h], step_cap(size)))
        state[h] -= n
        return "kes %s %d." % (h, n), (h,)
    src = rng.choice(nonzero)
    dst = rng.choice([h for h in ep_holders if h != src])
    n = rng.randint(1, min(state[src], step_cap(size)))
    state[src] -= n
    state[dst] += n
    return "vut %s %s %d." % (src, dst, n), (src, dst)


def make_query(rng, state, touches, k, ep_holders):
    """Build one query from a snapshot after k records. Returns (text, answer) or None."""
    op = rng.choice(QRY_OPS)
    if op == "ket":
        h = rng.choice(ep_holders)
        if touches[h] < 2:
            return None
        return "ket %s?" % h, str(state[h])
    if k < 3:
        return None
    if op == "mox":
        cand = rng.sample(ep_holders, rng.randint(2, min(4, len(ep_holders))))
        vals = [state[h] for h in cand]
        m = max(vals)
        if m == 0 or vals.count(m) != 1:
            return None
        return "mox %s?" % " ".join(cand), cand[vals.index(m)]
    t = rng.randint(1, max(1, max(state.values())))
    ans = [h for h in ep_holders if state[h] >= t]
    if not ans or len(ans) == len(ep_holders):
        return None
    return "lis %d?" % t, ",".join(ans)


def try_episode(rng, size):
    """Return the rows of one episode (without numbers), or None if queries could not be made."""
    n_rec = rng.randint(3, 12) * size
    n_qry = rng.randint(1, 5)
    ep_holders = sorted(rng.sample(HOLDERS, rng.randint(3, 6)))
    state = {h: 0 for h in ep_holders}
    touches = {h: 0 for h in ep_holders}
    texts = []
    snaps = [(dict(state), dict(touches))]
    for _ in range(n_rec):
        text, who = make_record(rng, state, ep_holders, size)
        for h in who:
            touches[h] += 1
        texts.append(text)
        snaps.append((dict(state), dict(touches)))

    slots = {}
    seen_q = set()
    for _ in range(n_qry):
        for _attempt in range(40):
            p = rng.randint(2, n_rec)
            st, tc = snaps[p]
            q = make_query(rng, st, tc, p, ep_holders)
            if q is not None and q[0] not in seen_q:
                break
        else:
            return None
        seen_q.add(q[0])
        slots.setdefault(p, []).append(q)

    rows = []
    for p in range(n_rec + 1):
        for text, answer in slots.get(p, []):
            rows.append(text + "\t" + answer)
        if p < n_rec:
            rows.append(texts[p])
    return rows


def make_episode(rng, size, seen):
    while True:
        rows = try_episode(rng, size)
        if rows is None:
            continue
        key = "\n".join(rows)
        if key not in seen:
            seen.add(key)
            return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED SIZE  (SIZE 1, 3 or 10; default 1)\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        rows = make_episode(rng, size, seen)
        for i, row in enumerate(rows, 1):
            out.append("%d %s" % (i, row))
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
