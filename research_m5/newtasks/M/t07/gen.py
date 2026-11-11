#!/usr/bin/env python3
"""Synthetic ledger task given as records: holders hold amounts that are set,
added to, removed and moved; queries ask a holder's amount now, which holder
has the most among some candidates, or whether a holder has at least N.

Usage: python gen.py N SEED [SIZE]     SIZE is 1 (default), 3 or 10.
Prints N episodes. Standard library only; all randomness from random.Random(SEED).
"""
import random
import sys

HOLDERS = ["h%d" % i for i in range(1, 9)]  # fixed pool of 8 holder codes at every SIZE
SET_OP, ADD_OP, REMOVE_OP, MOVE_OP = "dzu", "pix", "rsk", "mvj"
AMOUNT_Q, MOST_Q, ATLEAST_Q = "kqh", "mxq", "gte"


class Ledger(object):
    def __init__(self):
        self.amount = {}
        self.touches = {}

    def get(self, holder):
        return self.amount.get(holder, 0)

    def touch(self, holder):
        self.touches[holder] = self.touches.get(holder, 0) + 1

    def touched(self, holder):
        return self.touches.get(holder, 0)


def make_record(rng, ledger, active, maxnum):
    holders_with = [h for h in active if ledger.get(h) > 0]
    kinds = ["set", "add"]
    weights = [2, 3]
    if holders_with:
        kinds += ["remove", "move"]
        weights += [2, 2]
    kind = rng.choices(kinds, weights=weights)[0]
    if kind == "set":
        h = rng.choice(active)
        n = rng.randint(1, maxnum)
        ledger.amount[h] = n
        ledger.touch(h)
        return [SET_OP, h, str(n)]
    if kind == "add":
        h = rng.choice(active)
        n = rng.randint(1, maxnum)
        ledger.amount[h] = ledger.get(h) + n
        ledger.touch(h)
        return [ADD_OP, h, str(n)]
    if kind == "remove":
        h = rng.choice(holders_with)
        n = rng.randint(1, ledger.get(h))
        ledger.amount[h] -= n
        ledger.touch(h)
        return [REMOVE_OP, h, str(n)]
    a = rng.choice(holders_with)
    b = rng.choice([x for x in active if x != a])
    n = rng.randint(1, ledger.get(a))
    ledger.amount[a] -= n
    ledger.amount[b] = ledger.get(b) + n
    ledger.touch(a)
    ledger.touch(b)
    return [MOVE_OP, a, b, str(n)]


def make_query(rng, ledger, active):
    """Return (question, answer) or None if this query would not need several records."""
    kind = rng.choice([AMOUNT_Q, MOST_Q, ATLEAST_Q])
    if kind == AMOUNT_Q:
        h = rng.choice(active)
        if ledger.touched(h) < 2:
            return None
        return "%s %s?" % (kind, h), str(ledger.get(h))
    if kind == MOST_Q:
        cands = sorted(rng.sample(active, rng.randint(2, min(4, len(active)))))
        if sum(ledger.touched(h) for h in cands) < 2:
            return None
        vals = [ledger.get(h) for h in cands]
        top = max(vals)
        if vals.count(top) != 1:  # answer must be unambiguous
            return None
        return "%s %s?" % (kind, " ".join(cands)), cands[vals.index(top)]
    h = rng.choice(active)
    if ledger.touched(h) < 2:
        return None
    n = max(1, ledger.get(h) + rng.randint(-2, 2))
    answer = "yes" if ledger.get(h) >= n else "no"
    return "%s %s %d?" % (kind, h, n), answer


def make_episode(rng, size):
    """Return the list of row strings (without numbers), or None if it could not be placed."""
    ledger = Ledger()
    n_active = rng.randint(3, 5) if size == 1 else rng.randint(3, 8)
    active = sorted(rng.sample(HOLDERS, n_active))
    maxnum = 9 * size
    rem_r = rng.randint(3 * size, 12 * size)
    rem_q = rng.randint(1, 5)
    rows = []
    n_records = 0
    while rem_r > 0 or rem_q > 0:
        if rem_q > 0 and n_records >= 2:
            if rem_r == 0 or rng.random() < rem_q / float(rem_r + rem_q):
                q = None
                for _ in range(50):
                    q = make_query(rng, ledger, active)
                    if q is not None:
                        break
                if q is not None:
                    rows.append(q[0] + "\t" + q[1])
                    rem_q -= 1
                    continue
                if rem_r == 0:
                    return None
        rows.append(" ".join(make_record(rng, ledger, active, maxnum)) + ".")
        n_records += 1
        rem_r -= 1
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    count = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(count):
        while True:
            rows = make_episode(rng, size)
            if rows is None:
                continue
            key = "\n".join(rows)
            if key not in seen:
                seen.add(key)
                break
        for i, row in enumerate(rows, 1):
            out.append("%d %s" % (i, row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv)
