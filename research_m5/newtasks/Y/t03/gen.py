#!/usr/bin/env python3
"""Synthetic record-log task: which agent knows which message.

Records (operation codes kn, pz, fg):
  kn a m.   agent a learns message m
  pz a b.   agent a passes on everything it knows at this moment to agent b
  fg a m.   agent a forgets message m

Queries (operation codes hs, ct, wh):
  hs a m?   does agent a know message m          -> yes / no
  ct a?     how many messages agent a knows      -> number
  wh m?     which agents know message m          -> sorted codes, comma separated

Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

AGENTS = ["a%d" % i for i in range(1, 9)]
MSGS = ["m%d" % i for i in range(1, 9)]


def mentions_of(code, mentions):
    return {i for i, used in enumerate(mentions) if code in used}


def record(rng, known, idx):
    """Create one record, update the simulation, return (body, codes used)."""
    has = [a for a in AGENTS if known[a]]
    r = rng.random()
    if r < 0.4 or not has:
        a = rng.choice(AGENTS)
        m = rng.choice(MSGS)
        known[a][m] = {idx}
        return "kn %s %s." % (a, m), {a, m}
    if r < 0.8:
        a = rng.choice(has)
        b = rng.choice([x for x in AGENTS if x != a])
        snap = {m: set(s) for m, s in known[a].items()}
        for m, supp in snap.items():
            known[b][m] = known[b].get(m, set()) | supp | {idx}
        return "pz %s %s." % (a, b), {a, b}
    a = rng.choice(has)
    m = rng.choice(sorted(known[a]))
    del known[a][m]
    return "fg %s %s." % (a, m), {a, m}


def query(rng, known, mentions):
    """Create one query whose answer depends on at least two records."""
    for _ in range(60):
        kind = rng.random()
        if kind < 0.45:
            a = rng.choice(AGENTS)
            m = rng.choice(MSGS)
            if m in known[a]:
                ans = "yes"
                deps = set(known[a][m])
            else:
                ans = "no"
                deps = mentions_of(a, mentions) | mentions_of(m, mentions)
            body = "hs %s %s?" % (a, m)
        elif kind < 0.75:
            a = rng.choice(AGENTS)
            ans = str(len(known[a]))
            deps = mentions_of(a, mentions)
            for supp in known[a].values():
                deps |= supp
            body = "ct %s?" % a
        else:
            m = rng.choice(MSGS)
            knowers = [a for a in AGENTS if m in known[a]]
            if not knowers:
                continue
            ans = ",".join(knowers)
            deps = mentions_of(m, mentions)
            for a in knowers:
                deps |= known[a][m]
            body = "wh %s?" % m
        if len(deps) >= 2:
            return body + "\t" + ans
    return None


def attempt(rng, size):
    """Try to build one episode; return a list of numbered rows or None."""
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    total = n_rec + n_q
    q_slots = set(rng.sample(range(2, total), n_q))
    known = {a: {} for a in AGENTS}
    mentions = []
    rows = []
    for step in range(total):
        number = step + 1
        if step in q_slots:
            q = query(rng, known, mentions)
            if q is None:
                return None
            rows.append("%d %s" % (number, q))
        else:
            body, used = record(rng, known, len(mentions))
            mentions.append(used)
            rows.append("%d %s" % (number, body))
    return rows


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        rows = attempt(rng, size)
        if rows is None:
            continue
        text = "\n".join(rows)
        if text in seen:
            continue
        seen.add(text)
        episodes.append(rows)
    for rows in episodes:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main(sys.argv)
