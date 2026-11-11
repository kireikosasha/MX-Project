#!/usr/bin/env python3
"""Generate synthetic group-membership episodes written as records.

Usage:  python gen.py N SEED [SIZE]      SIZE is 1, 3 or 10 (default 1)

Entities: people p1..p8 and groups g1..g8. A person is in at most one group.
Record operations (state changes):
    jn p g.    person p joins group g (p must be in no group)
    mv p g.    person p moves to group g (g must differ from p's group)
    lv p.      person p leaves its group
Query operations (answered from the records above them):
    wg p?      which group p is in now            -> one group code
    mb g?      who is in group g now, sorted      -> person codes, comma-separated
    ct g?      how many are in group g now        -> a number
"""
import random
import sys

PEOPLE = ["p%d" % i for i in range(1, 9)]
GROUPS = ["g%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)


def make_records(rng, size):
    """Random record sequence plus the membership state after each prefix.

    Returns (records, snaps, people, groups). snaps[k] maps each person to
    its group (or None) after the first k records.
    """
    n_people = rng.randint(3, 6)
    n_groups = rng.randint(2, 4)
    people = sorted(rng.sample(PEOPLE, n_people))
    groups = sorted(rng.sample(GROUPS, n_groups))
    if size == 1:
        n_rec = rng.randint(3, 12)
    else:
        n_rec = rng.randint(3 * size, 12 * size)

    member = {p: None for p in people}
    records = []
    snaps = [dict(member)]
    for _ in range(n_rec):
        ungrouped = [p for p in people if member[p] is None]
        grouped = [p for p in people if member[p] is not None]
        ops, weights = [], []
        if ungrouped:
            ops.append("jn")
            weights.append(4)
        if grouped:
            ops.append("lv")
            weights.append(2)
            ops.append("mv")  # n_groups >= 2, so a different group exists
            weights.append(3)
        op = rng.choices(ops, weights)[0]
        if op == "jn":
            p = rng.choice(ungrouped)
            g = rng.choice(groups)
            rec = {"p": p, "from": None, "to": g, "text": "jn %s %s." % (p, g)}
        elif op == "mv":
            p = rng.choice(grouped)
            g = rng.choice([x for x in groups if x != member[p]])
            rec = {"p": p, "from": member[p], "to": g, "text": "mv %s %s." % (p, g)}
        else:
            p = rng.choice(grouped)
            rec = {"p": p, "from": member[p], "to": None, "text": "lv %s." % p}
        member[p] = rec["to"]
        records.append(rec)
        snaps.append(dict(member))
    return records, snaps, people, groups


def person_hits(records, k, p):
    """Number of the first k records that change person p's membership."""
    return sum(1 for r in records[:k] if r["p"] == p)


def group_hits(records, k, g):
    """Number of the first k records that change membership of group g."""
    return sum(1 for r in records[:k] if r["to"] == g or r["from"] == g)


def make_episode(rng, size):
    """One episode as a list of numbered rows. Every query needs at least
    two records about its entity or group, so each answer needs several
    records to be worked out."""
    for _ in range(1000):
        records, snaps, people, groups = make_records(rng, size)
        n_rec = len(records)
        n_q = rng.randint(1, 5)
        slots = list(range(3, n_rec + 1))  # a query sits after record k
        rng.shuffle(slots)
        chosen = {}  # k -> (query text, answer)
        for k in slots:
            if len(chosen) == n_q:
                break
            snap = snaps[k]
            cands = []
            for p in people:
                if snap[p] is not None and person_hits(records, k, p) >= 2:
                    cands.append(("wg", p, snap[p]))
            for g in groups:
                if group_hits(records, k, g) >= 2:
                    mem = sorted(p for p in people if snap[p] == g)
                    cands.append(("ct", g, str(len(mem))))
                    if mem:
                        cands.append(("mb", g, ",".join(mem)))
            if cands:
                op, ent, ans = rng.choice(cands)
                chosen[k] = ("%s %s?" % (op, ent), ans)
        if len(chosen) < n_q:
            continue
        lines = []
        for i, rec in enumerate(records, start=1):
            lines.append(rec["text"])
            if i in chosen:
                q, ans = chosen[i]
                lines.append(q + "\t" + ans)
        return ["%d %s" % (n + 1, t) for n, t in enumerate(lines)]
    raise RuntimeError("could not build an episode")


def main(argv):
    if len(argv) not in (3, 4):
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
            lines = make_episode(rng, size)
            key = "\n".join(lines)
            if key not in seen:
                break
        seen.add(key)
        sys.stdout.write(key + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
