"""Synthetic GROUPS task written as records.

Usage: python gen.py N SEED [SIZE]   (SIZE in 1, 3, 10; default 1)

Persons p1..p8 join, leave and move between groups g1..g8.
Record ops: jn (a person outside any group joins a group),
            lv (a person leaves the group it is in),
            mv (a person moves from its group to another group).
Query ops:  wg (which group is this person in now),
            mb (who is in this group now, sorted),
            ct (how many are in this group now).
"""
import random
import sys

PEOPLE = [f"p{i}" for i in range(1, 9)]
GROUPS = [f"g{i}" for i in range(1, 9)]


def make_attempt(rng, size):
    people = sorted(rng.sample(PEOPLE, rng.randint(3, 5)))
    groups = sorted(rng.sample(GROUPS, rng.randint(2, 4)))
    n_records = rng.randint(3, 12) * size
    n_queries = rng.randint(1, 5)

    where = {p: None for p in people}     # person -> its group, or None
    members = {g: set() for g in groups}  # group -> persons in it now
    mentions = {}                         # code -> number of records naming it
    lines = []
    left = n_queries

    def pick_person(cands):
        seen = [p for p in cands if mentions.get(p, 0) > 0]
        if seen and rng.random() < 0.6:
            return rng.choice(seen)
        return rng.choice(cands)

    def emit_record():
        out = [p for p in people if where[p] is None]
        inn = [p for p in people if where[p] is not None]
        ops, weights = [], []
        if out:
            ops.append("jn")
            weights.append(0.45)
        if inn:
            ops += ["lv", "mv"]
            weights += [0.2, 0.35]
        op = rng.choices(ops, weights)[0]
        if op == "jn":
            p = pick_person(out)
            g = rng.choice(groups)
            where[p] = g
            members[g].add(p)
            text = f"jn {p} {g}."
        elif op == "lv":
            p = pick_person(inn)
            g = where[p]
            members[g].discard(p)
            where[p] = None
            text = f"lv {p} {g}."
        else:
            p = pick_person(inn)
            old = where[p]
            g = rng.choice([x for x in groups if x != old])
            members[old].discard(p)
            members[g].add(p)
            where[p] = g
            text = f"mv {p} {g}."
        for tok in text[:-1].split()[1:]:
            mentions[tok] = mentions.get(tok, 0) + 1
        lines.append(text)

    asked = set()

    def try_query():
        # Only ask about codes named by at least two earlier records,
        # so the answer needs several records to work out. No question
        # is asked twice in one episode.
        cands = []
        for p in people:
            if where[p] is not None and mentions.get(p, 0) >= 2:
                cands.append(f"wg {p}?\t{where[p]}")
        for g in groups:
            if mentions.get(g, 0) >= 2:
                if members[g]:
                    cands.append(f"mb {g}?\t{','.join(sorted(members[g]))}")
                cands.append(f"ct {g}?\t{len(members[g])}")
        cands = [c for c in cands if c.split("\t")[0] not in asked]
        if not cands:
            return False
        line = rng.choice(cands)
        asked.add(line.split("\t")[0])
        lines.append(line)
        return True

    for _ in range(n_records):
        emit_record()
        if left and rng.random() < 0.35 and try_query():
            left -= 1
    while left:
        if not try_query():
            return None  # no valid query left; reject and retry
        left -= 1
    return lines


def episode(rng, size, seen):
    while True:
        lines = make_attempt(rng, size)
        if lines is None:
            continue
        key = "\n".join(lines)
        if key in seen:
            continue
        seen.add(key)
        return lines


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        for i, line in enumerate(episode(rng, size, seen), 1):
            print(f"{i} {line}")


if __name__ == "__main__":
    main(sys.argv)
