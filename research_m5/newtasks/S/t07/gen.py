#!/usr/bin/env python3
"""Synthetic ledger-record reasoning episodes (records of codes and numbers only).

Usage: python gen.py N SEED [SIZE]
  N     number of episodes to print
  SEED  seed for random.Random(SEED); all randomness comes from it
  SIZE  1, 3 or 10 (default 1)
"""
import random
import sys

HOLDERS = ["p%d" % i for i in range(1, 9)]  # the only entity kind: holders p1..p8

SET, ADD, REMOVE, MOVE = "zet", "fip", "gom", "vrq"  # record operations
HAS, AT_LEAST, MOST = "hav", "geq", "mxo"            # query operations
RECORD_WEIGHTS = {SET: 1, ADD: 3, REMOVE: 2, MOVE: 3}


def make_record(rng, bal, cnt, size):
    top = 9 * size
    holding = [h for h in HOLDERS if bal[h] > 0]
    kind = rng.choices(list(RECORD_WEIGHTS), weights=list(RECORD_WEIGHTS.values()))[0]
    if kind in (REMOVE, MOVE) and not holding:
        kind = ADD
    if kind == SET:
        h = rng.choice(HOLDERS)
        v = rng.randint(1, top)
        bal[h] = v
        cnt[h] = 1
        return "%s %s %d." % (SET, h, v)
    if kind == ADD:
        h = rng.choice(HOLDERS)
        a = rng.randint(1, top)
        bal[h] += a
        cnt[h] += 1
        return "%s %s %d." % (ADD, h, a)
    if kind == REMOVE:
        h = rng.choice(holding)
        a = rng.randint(1, bal[h])
        bal[h] -= a
        cnt[h] += 1
        return "%s %s %d." % (REMOVE, h, a)
    src = rng.choice(holding)
    dst = rng.choice([h for h in HOLDERS if h != src])
    a = rng.randint(1, bal[src])
    bal[src] -= a
    bal[dst] += a
    cnt[src] += 1
    cnt[dst] += 1
    return "%s %s %s %d." % (MOVE, src, dst, a)


def make_query(rng, bal, cnt, size):
    """Return (query_text, answer) answered from the current state, or None."""
    for _ in range(300):
        op = rng.choice([HAS, AT_LEAST, MOST])
        if op == MOST:
            cands = sorted(rng.sample(HOLDERS, rng.randint(2, 4)))
            best = max(bal[c] for c in cands)
            winners = [c for c in cands if bal[c] == best]
            if len(winners) != 1 or sum(cnt[c] for c in cands) < 2:
                continue
            return "%s %s?" % (MOST, " ".join(cands)), winners[0]
        h = rng.choice(HOLDERS)
        if cnt[h] < 2:  # the value must rest on several records
            continue
        if op == HAS:
            return "%s %s?" % (HAS, h), str(bal[h])
        t = max(1, bal[h] + rng.randint(-2 * size - 1, 2 * size + 1))
        return "%s %s %d?" % (AT_LEAST, h, t), ("yes" if bal[h] >= t else "no")
    return None


def build_episode(rng, size):
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    total = n_rec + n_q
    q_rows = set(rng.sample(range(3, total), n_q))  # queries come after some records
    bal = dict.fromkeys(HOLDERS, 0)
    cnt = dict.fromkeys(HOLDERS, 0)
    rows = []
    for i in range(total):
        if i in q_rows:
            q = make_query(rng, bal, cnt, size)
            if q is None:
                return None  # caller draws a fresh episode
            rows.append("%s\t%s" % q)
        else:
            rows.append(make_record(rng, bal, cnt, size))
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write(__doc__)
        return 2
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        rows = None
        while rows is None:
            rows = build_episode(rng, size)
        lines.extend("%d %s" % (i, r) for i, r in enumerate(rows, 1))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
