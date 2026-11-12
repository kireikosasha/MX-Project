#!/usr/bin/env python3
"""Synthetic records task: a two-table join over ownership and app use.

Record rows:
  gk P D.   person P owns device D (each device has at most one owner)
  rq D A.   device D runs app A (a device may run several apps)

Query rows (answer after a TAB):
  wq A?      sorted list of persons who own a device running app A
  nq A?      number of such persons
  uq P A?    yes/no: does person P own a device that runs app A

Usage: python gen.py N SEED SIZE   (SIZE in 1, 3, 10; default 1)
"""
import random
import sys

PERSONS = ["p%d" % i for i in range(1, 9)]
DEVICES = ["d%d" % i for i in range(1, 9)]
APPS = ["a%d" % i for i in range(1, 9)]


def make_episode(rng, size):
    records_left = rng.randint(3, 12) * size
    queries_left = rng.randint(1, 5)
    owner = {}      # device -> person (fixed once set)
    runs = set()    # (device, app) pairs
    rows = []       # (is_query, text, answer)

    def users_of(app):
        return sorted({owner[d] for (d, a) in runs if a == app and d in owner})

    def make_query():
        apps_with_users = [a for a in APPS if users_of(a)]
        choice = rng.choice(["wq", "nq", "uq"])
        if choice == "wq" and apps_with_users:
            a = rng.choice(apps_with_users)
            return (True, "wq %s?" % a, ",".join(users_of(a)))
        if choice == "nq":
            a = rng.choice(APPS)
            return (True, "nq %s?" % a, str(len(users_of(a))))
        p = rng.choice(PERSONS)
        a = rng.choice(APPS)
        yes = any(owner.get(d) == p and (d, a) in runs for d in DEVICES)
        return (True, "uq %s %s?" % (p, a), "yes" if yes else "no")

    while records_left > 0 or queries_left > 0:
        gkc = [d for d in DEVICES if d not in owner]
        rqc = [(d, a) for d in DEVICES for a in APPS if (d, a) not in runs]
        if records_left > 0 and not gkc and not rqc:
            records_left = 0
            continue
        ready = bool(owner) and bool(runs)
        remaining = records_left + queries_left
        if queries_left > 0 and ready and (
                records_left == 0 or rng.random() < queries_left / remaining):
            rows.append(make_query())
            queries_left -= 1
            continue
        # Emit a record; the first two records are always one of each kind,
        # so queries have data above them.
        if not owner:
            kind = "gk"
        elif not runs:
            kind = "rq"
        else:
            kind = "gk" if rng.random() < 0.4 else "rq"
        if kind == "gk" and not gkc:
            kind = "rq"
        if kind == "rq" and not rqc:
            kind = "gk"
        if kind == "gk":
            d = rng.choice(gkc)
            p = rng.choice(PERSONS)
            owner[d] = p
            rows.append((False, "gk %s %s." % (p, d), None))
        else:
            d, a = rng.choice(rqc)
            runs.add((d, a))
            rows.append((False, "rq %s %s." % (d, a), None))
        records_left -= 1
    return rows


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (is_query, text, answer) in enumerate(make_episode(rng, size), 1):
            if is_query:
                out.append("%d %s\t%s" % (i, text, answer))
            else:
                out.append("%d %s" % (i, text))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main(sys.argv)
