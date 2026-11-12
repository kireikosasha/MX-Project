#!/usr/bin/env python3
"""Synthetic ledger task: running net debts between people, given as records.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)

Each record changes the net balance between two people:
  dbt a b n.   a owes b n more
  pyt a b n.   a pays b n (reduces what a owes b, may reverse it)
  clr a b.     the balance between a and b is cleared to zero
Net balance net(a, b) is the amount a owes b (negative: b owes a).

Each query asks about the state after the records above it:
  hds a b?     does a owe b (net > 0)?            yes / no
  ttl a?       total a owes, summed over all b     number
  wdb a?       all b who owe a (net(b, a) > 0)     sorted entity codes, comma-separated
"""
import random
import sys

PEOPLE = ["p%d" % i for i in range(1, 9)]
MAX_AMOUNT = {1: 9, 3: 40, 10: 400}
REC_OPS = ["dbt", "pyt", "clr"]
QRY_OPS = ["hds", "ttl", "wdb"]


def find_query(rng, net, touched):
    """Pick a query whose answer rests on at least two earlier records."""

    def owed(a, b):
        return net.get((a, b), 0)

    for _ in range(60):
        op = rng.choice(QRY_OPS)
        if op == "hds":
            a, b = rng.sample(PEOPLE, 2)
            support = sum(1 for t in touched if set(t) == {a, b})
            if support < 2:
                continue
            ans = "yes" if owed(a, b) > 0 else "no"
            return "hds %s %s?" % (a, b), ans
        if op == "ttl":
            a = rng.choice(PEOPLE)
            total = sum(max(0, owed(a, b)) for b in PEOPLE if b != a)
            support = sum(1 for t in touched if a in t)
            if total <= 0 or support < 2:
                continue
            return "ttl %s?" % a, str(total)
        a = rng.choice(PEOPLE)
        owers = [b for b in PEOPLE if b != a and owed(b, a) > 0]
        support = sum(1 for t in touched if a in t)
        if not owers or support < 2:
            continue
        return "wdb %s?" % a, ",".join(owers)
    return None


def build_episode(rng, size):
    """Return a list of row bodies (without numbers), or None if it failed."""
    n_rec = rng.randint(3 * size, 12 * size)
    n_qry = rng.randint(1, 5)
    max_amt = MAX_AMOUNT[size]

    net = {}
    touched = []
    rows = []
    recs_left, qry_left = n_rec, n_qry

    while recs_left > 0 or qry_left > 0:
        want_query = qry_left > 0 and len(touched) >= 2 and (
            recs_left == 0 or rng.random() < qry_left / float(qry_left + recs_left))
        if want_query:
            found = find_query(rng, net, touched)
            if found is not None:
                text, ans = found
                rows.append("%s\t%s" % (text, ans))
                qry_left -= 1
                continue
        if recs_left == 0:
            return None
        op = rng.choice(REC_OPS)
        a, b = rng.sample(PEOPLE, 2)
        if op == "clr":
            text = "clr %s %s." % (a, b)
            net[(a, b)] = 0
            net[(b, a)] = 0
        else:
            n = rng.randint(1, max_amt)
            text = "%s %s %s %d." % (op, a, b, n)
            delta = n if op == "dbt" else -n
            net[(a, b)] = net.get((a, b), 0) + delta
            net[(b, a)] = net.get((b, a), 0) - delta
        touched.append((a, b))
        rows.append(text)
        recs_left -= 1

    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n_episodes = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in MAX_AMOUNT:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2

    rng = random.Random(seed)
    seen = set()
    count = 0
    while count < n_episodes:
        rows = build_episode(rng, size)
        if rows is None:
            continue
        lines = ["%d %s" % (i, r) for i, r in enumerate(rows, 1)]
        key = "\n".join(lines)
        if key in seen:
            continue
        seen.add(key)
        for line in lines:
            sys.stdout.write(line + "\n")
        count += 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
