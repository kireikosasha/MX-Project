#!/usr/bin/env python3
"""Synthetic records task: class membership by rules given as records.

Records (rows with a full stop):
  ruv X Y.      every individual with class X also has class Y
  gok X Y Z.    every individual with classes X and Y also has class Z
  tiq K X.      individual K has class X

Queries (rows with a question mark, a TAB and the answer), answered only from
the records above them in the same episode, by forward chaining:
  qiv K X?      does K have class X?          -> yes / no
  nuz X?        which individuals have X?     -> sorted codes, comma separated
  wex X?        how many individuals have X?  -> number

Usage: python gen.py N SEED SIZE   (SIZE in 1, 3, 10; default 1)
"""
import random
import sys

INDS = ["k%d" % i for i in range(1, 9)]
CLS = ["c%d" % i for i in range(1, 9)]


def closure(records):
    """Return (have, direct): have[k] is the set of classes k ends up with."""
    direct = set()
    single = set()
    conj = []
    for r in records:
        if r[0] == "tiq":
            direct.add((r[1], r[2]))
        elif r[0] == "ruv":
            single.add((r[1], r[2]))
        elif r[0] == "gok":
            conj.append((r[1], r[2], r[3]))
    have = {k: set() for k in INDS}
    for k, c in direct:
        have[k].add(c)
    changed = True
    while changed:
        changed = False
        for k in INDS:
            cur = have[k]
            new = set()
            for x, y in single:
                if x in cur and y not in cur:
                    new.add(y)
            for x, y, z in conj:
                if x in cur and y in cur and z not in cur:
                    new.add(z)
            if new:
                cur |= new
                changed = True
    return have, direct


def random_record(rng):
    u = rng.random()
    if u < 0.4:
        return ("tiq", rng.choice(INDS), rng.choice(CLS))
    if u < 0.75:
        a, b = rng.sample(CLS, 2)
        return ("ruv", a, b)
    a, b, c = rng.sample(CLS, 3)
    return ("gok", a, b, c)


def make_episode(rng, size):
    nrec = rng.randint(3, 12) * size
    nq_target = rng.randint(1, 5)
    # Chains of rules get longer with SIZE (1..2 at SIZE 1, up to 7 at SIZE 10).
    maxchain = min(7, size + 1)
    chain_len = min(rng.randint(1, maxchain), nrec - 1)

    recs = []
    seen = set()

    def add(r):
        if r not in seen:
            seen.add(r)
            recs.append(r)

    path = rng.sample(CLS, chain_len + 1)
    for i in range(chain_len):
        add(("ruv", path[i], path[i + 1]))
    add(("tiq", rng.choice(INDS), path[0]))

    tries = 0
    while len(recs) < nrec and tries < 10000:
        tries += 1
        add(random_record(rng))
    if len(recs) < nrec:
        return None
    rng.shuffle(recs)

    queries = []
    qseen = set()
    attempts = 0
    while len(queries) < nq_target and attempts < 300:
        attempts += 1
        p = rng.randint(2, len(recs))
        prefix = recs[:p]
        have, direct = closure(prefix)
        op = rng.choice(("qiv", "nuz", "wex"))
        if op == "qiv":
            derived = [(k, c) for k in INDS for c in sorted(have[k])
                       if (k, c) not in direct]
            if derived and rng.random() < 0.5:
                k, c = rng.choice(derived)
            else:
                k, c = rng.choice(INDS), rng.choice(CLS)
            yes = c in have[k]
            if yes and (k, c) in direct:
                continue
            if not yes and not any(r[0] == "tiq" and r[1] == k for r in prefix):
                continue
            text = "qiv %s %s?" % (k, c)
            ans = "yes" if yes else "no"
        else:
            c = rng.choice(CLS)
            mem = [k for k in INDS if c in have[k]]
            if not mem or all((k, c) in direct for k in mem):
                continue
            if op == "nuz":
                text = "nuz %s?" % c
                ans = ",".join(mem)
            else:
                text = "wex %s?" % c
                ans = str(len(mem))
        if text in qseen:
            continue
        qseen.add(text)
        queries.append((p, text, ans))

    if not queries:
        return None

    rows = []
    qs = sorted(queries, key=lambda q: q[0])
    qi = 0
    n = 0
    for i in range(len(recs) + 1):
        while qi < len(qs) and qs[qi][0] == i:
            _, text, ans = qs[qi]
            qi += 1
            n += 1
            rows.append("%d %s\t%s" % (n, text, ans))
        if i < len(recs):
            n += 1
            rows.append("%d %s." % (n, " ".join(recs[i])))
    return rows


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen_eps = set()
    out = []
    while len(out) < n:
        ep = make_episode(rng, size)
        if ep is None:
            continue
        key = "\n".join(ep)
        if key in seen_eps:
            continue
        seen_eps.add(key)
        out.append(ep)
    for ep in out:
        for row in ep:
            print(row)


if __name__ == "__main__":
    main(sys.argv)
