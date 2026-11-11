#!/usr/bin/env python3
"""Synthetic GROUPS task given as records: people join, leave and move between groups.

Usage: python gen.py N SEED [SIZE]      (SIZE is 1, 3 or 10; default 1)

Record rows (3 tokens each, applied in order):
  jn p g   person p joins group g (p is in no group now)
  lv p g   person p leaves group g (p is in g now)
  mv p g   person p moves from its current group to group g (p is elsewhere now)
  mg a b   every current member of group a moves into group b; a becomes empty

Query rows (answered from the records above them in the same episode):
  qw p?    which group is p in now           -> one group code
  qx g?    who is in group g now             -> person codes, sorted, comma-separated
  qn g?    how many people are in group g now -> a number in digits
"""
import random
import sys

PERSONS = ["p%d" % i for i in range(1, 9)]
GROUPS = ["g%d" % i for i in range(1, 5)]
REC_OPS = ("jn", "lv", "mv", "mg")
QRY_OPS = ("qw", "qx", "qn")


def new_state():
    return {
        "where": {p: None for p in PERSONS},
        "members": {g: set() for g in GROUPS},
        "touches": {e: 0 for e in PERSONS + GROUPS},  # records that changed each entity
    }


def apply_record(st, rec):
    op, a, b = rec
    touched = {a, b}
    if op == "jn":
        st["where"][a] = b
        st["members"][b].add(a)
    elif op == "lv":
        st["where"][a] = None
        st["members"][b].discard(a)
    elif op == "mv":
        old = st["where"][a]
        st["members"][old].discard(a)
        st["members"][b].add(a)
        st["where"][a] = b
        touched.add(old)
    elif op == "mg":
        moved = sorted(st["members"][a])
        for p in moved:
            st["where"][p] = b
            touched.add(p)
        st["members"][b].update(moved)
        st["members"][a].clear()
    for e in touched:
        st["touches"][e] += 1


def pick_record(rng, st):
    where, mem = st["where"], st["members"]
    free = [p for p in PERSONS if where[p] is None]
    busy = [p for p in PERSONS if where[p] is not None]
    full = [g for g in GROUPS if mem[g]]
    kinds = []
    if free:
        kinds.append(("jn", 3))
    if busy:
        kinds.append(("lv", 2))
        kinds.append(("mv", 2))
    if full:
        kinds.append(("mg", 1))
    op = rng.choices([k for k, _ in kinds], weights=[w for _, w in kinds], k=1)[0]
    if op == "jn":
        return ("jn", rng.choice(free), rng.choice(GROUPS))
    if op == "lv":
        p = rng.choice(busy)
        return ("lv", p, where[p])
    if op == "mv":
        p = rng.choice(busy)
        return ("mv", p, rng.choice([g for g in GROUPS if g != where[p]]))
    a = rng.choice(full)
    return ("mg", a, rng.choice([g for g in GROUPS if g != a]))


def pick_query(rng, st):
    """Return (op, target, answer) for a query that needs at least two records, or None."""
    where, mem, t = st["where"], st["members"], st["touches"]
    cands = {
        "qw": [p for p in PERSONS if where[p] is not None and t[p] >= 2],
        "qx": [g for g in GROUPS if mem[g] and t[g] >= 2],
        "qn": [g for g in GROUPS if t[g] >= 2],
    }
    ops = [o for o in QRY_OPS if cands[o]]
    if not ops:
        return None
    op = rng.choice(ops)
    target = rng.choice(cands[op])
    if op == "qw":
        ans = where[target]
    elif op == "qx":
        ans = ",".join(sorted(mem[target]))
    else:
        ans = str(len(mem[target]))
    return op, target, ans


def attempt_episode(rng, size):
    """One try at an episode; returns its rows, or None if the queries could not be placed."""
    st = new_state()
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    rows = []
    rec_done = q_done = 0
    while rec_done < n_rec or q_done < n_q:
        rem_r, rem_q = n_rec - rec_done, n_q - q_done
        want_q = rem_r == 0 or (rem_q > 0 and rng.random() < rem_q / (rem_q + rem_r))
        if want_q:
            res = pick_query(rng, st)
            if res is not None:
                op, target, ans = res
                rows.append("%d %s %s?\t%s" % (len(rows) + 1, op, target, ans))
                q_done += 1
                continue
            if rem_r == 0:
                return None
        rec = pick_record(rng, st)
        apply_record(st, rec)
        rows.append("%d %s %s %s." % (len(rows) + 1, rec[0], rec[1], rec[2]))
        rec_done += 1
    return rows


def gen_episode(rng, size):
    while True:
        rows = attempt_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        rows = gen_episode(rng, size)
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(rows)
    for rows in episodes:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main(sys.argv)
