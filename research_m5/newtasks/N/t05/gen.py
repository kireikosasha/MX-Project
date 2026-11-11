#!/usr/bin/env python3
"""Synthetic changing-states task written as records (no natural language).

Usage: python gen.py N SEED SIZE      (SIZE in 1, 3, 10; default 1)

Entities and the state each one carries:
  items   i1..i6  -> location (a place code, p1..p4)
  devices d1..d4  -> status   (a status code, s1..s3)
  vessels v1..v4  -> level    (a number)
Records change states over time. Queries, interleaved with the records, ask
the present state of one entity, which entities hold a given place or status,
or whether an entity is in a given state. Answers come from simulating the
records above each query.
"""
import random
import sys

ITEMS = ["i%d" % k for k in range(1, 7)]
DEVICES = ["d%d" % k for k in range(1, 5)]
VESSELS = ["v%d" % k for k in range(1, 5)]
PLACES = ["p%d" % k for k in range(1, 5)]
STATUSES = ["s%d" % k for k in range(1, 4)]

# Record operations (6 codes).
MOVE, SWAP, STATUS, SET, ADD, TAKE = "dzo", "kiv", "pex", "fom", "vum", "gaz"
# Query operations (3 codes).
WHERE, WHICH, IS = "qo", "ziv", "tek"

# Largest number used in one record, by SIZE. Vessel levels stay within 0..2*max.
MAXNUM = {1: 9, 3: 30, 10: 99}


class World:
    def __init__(self, size):
        self.m = MAXNUM[size]
        self.cap = 2 * self.m
        self.loc = {}      # item -> place
        self.status = {}   # device -> status
        self.level = {}    # vessel -> int
        self.touch = {}    # entity -> number of records that touched it

    def touched(self, *ents):
        for e in ents:
            self.touch[e] = self.touch.get(e, 0) + 1

    def stated(self):
        return sorted(set(self.loc) | set(self.status) | set(self.level))

    def value(self, e):
        if e in self.loc:
            return self.loc[e]
        if e in self.status:
            return self.status[e]
        return str(self.level[e])

    def holders(self):
        """Map each place/status value to the sorted entities holding it now."""
        out = {}
        for e, p in self.loc.items():
            out.setdefault(p, []).append(e)
        for e, s in self.status.items():
            out.setdefault(s, []).append(e)
        return {v: sorted(es) for v, es in out.items()}


def random_record(rng, w):
    options = [(3, "move"), (2, "status"), (2, "set")]
    located = [i for i in ITEMS if i in w.loc]
    grow = [v for v in VESSELS if v in w.level and w.level[v] < w.cap]
    shrink = [v for v in VESSELS if v in w.level and w.level[v] >= 1]
    if len(located) >= 2:
        options.append((1, "swap"))
    if grow:
        options.append((2, "add"))
    if shrink:
        options.append((2, "take"))
    kind = rng.choices([k for _, k in options], [wt for wt, _ in options])[0]

    if kind == "move":
        item = rng.choice(ITEMS)
        place = rng.choice([p for p in PLACES if p != w.loc.get(item)])
        w.loc[item] = place
        w.touched(item)
        return "%s %s %s." % (MOVE, item, place)
    if kind == "swap":
        a, b = rng.sample(located, 2)
        w.loc[a], w.loc[b] = w.loc[b], w.loc[a]
        w.touched(a, b)
        return "%s %s %s." % (SWAP, a, b)
    if kind == "status":
        dev = rng.choice(DEVICES)
        new = rng.choice([s for s in STATUSES if s != w.status.get(dev)])
        w.status[dev] = new
        w.touched(dev)
        return "%s %s %s." % (STATUS, dev, new)
    if kind == "set":
        v = rng.choice(VESSELS)
        n = rng.randint(0, w.m)
        w.level[v] = n
        w.touched(v)
        return "%s %s %d." % (SET, v, n)
    if kind == "add":
        v = rng.choice(grow)
        n = rng.randint(1, min(w.m, w.cap - w.level[v]))
        w.level[v] += n
        w.touched(v)
        return "%s %s %d." % (ADD, v, n)
    v = rng.choice(shrink)
    n = rng.randint(1, min(w.m, w.level[v]))
    w.level[v] -= n
    w.touched(v)
    return "%s %s %d." % (TAKE, v, n)


def random_query(rng, w, used):
    """Return (question, answer) about the current state, or None if no
    well-founded question exists (each subject needs >= 2 records behind it)."""
    stated = [e for e in w.stated() if w.touch.get(e, 0) >= 2]
    holders = w.holders()
    which = [x for x, hs in holders.items() if sum(w.touch.get(h, 0) for h in hs) >= 2]
    kinds = []
    if stated:
        kinds += [WHERE, IS]
    if which:
        kinds.append(WHICH)
    if not kinds:
        return None
    kind = rng.choice(kinds)
    if kind == WHERE:
        e = rng.choice(stated)
        question, ans = "%s %s?" % (WHERE, e), w.value(e)
    elif kind == WHICH:
        x = rng.choice(sorted(which))
        question, ans = "%s %s?" % (WHICH, x), ",".join(holders[x])
    else:
        e = rng.choice(stated)
        cur = w.value(e)
        if rng.random() < 0.5:
            x = cur
        elif e in w.level:
            x = str(rng.randint(0, w.cap))
        else:
            x = rng.choice(PLACES if e in w.loc else STATUSES)
        question = "%s %s %s?" % (IS, e, x)
        ans = "yes" if x == cur else "no"
    if question in used:
        return None
    return question, ans


def make_episode(rng, size):
    while True:
        w = World(size)
        nrec = rng.randint(3, 12) * size
        nq = rng.randint(1, 5)
        qpos = sorted(rng.randint(2, nrec) for _ in range(nq))
        rows, used, qi, failed = [], set(), 0, False
        for j in range(1, nrec + 1):
            rows.append(random_record(rng, w))
            while qi < nq and qpos[qi] == j:
                q = random_query(rng, w, used)
                if q is None:
                    failed = True
                    break
                used.add(q[0])
                rows.append("%s\t%s" % q)
                qi += 1
            if failed:
                break
        if not failed:
            return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in MAXNUM:
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen, out = set(), []
    for _ in range(n):
        while True:
            rows = make_episode(rng, size)
            key = tuple(rows)
            if key not in seen:
                seen.add(key)
                break
        out.extend("%d %s" % (k, row) for k, row in enumerate(rows, 1))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv)
