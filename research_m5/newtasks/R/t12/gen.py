#!/usr/bin/env python3
"""Synthetic record-log reasoning task: who knows which fact after gossip.

Records (one row each):
  kn p f    person p learns fact f directly
  tl a b    person a tells person b everything a knows at that moment
  fg p f    person p forgets fact f

Queries (answered from the state at their own position in the episode):
  cn p      how many facts person p knows
  hk p f    does person p know fact f (yes / no)
  wh f      which people know fact f (sorted, comma-separated)

Usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1).
"""

import random
import sys

RECORD_OPS = ("kn", "tl", "fg")
QUERY_OPS = ("cn", "hk", "wh")
SIZES = (1, 3, 10)
PEOPLE = ["p%d" % i for i in range(1, 9)]
FACTS = ["f%d" % i for i in range(1, 9)]


class State:
    def __init__(self, people):
        self.knows = {p: set() for p in people}
        # why[p][f] = record numbers that together gave p the fact f
        self.why = {p: {} for p in people}

    def learn(self, p, f, num):
        self.knows[p].add(f)
        self.why[p][f] = {num}

    def forget(self, p, f):
        self.knows[p].discard(f)
        del self.why[p][f]


def record(rng, st, people, facts, num):
    holders = [p for p in people if st.knows[p]]
    unknown = [(p, f) for p in people for f in facts if f not in st.knows[p]]
    tells = [(a, b) for a in holders for b in people
             if b != a and st.knows[a] - st.knows[b]]
    options = []
    if unknown:
        options.append(("kn", 0.3))
    if tells:
        options.append(("tl", 0.45))
    if holders:
        options.append(("fg", 0.25))
    op = rng.choices([o for o, _ in options], weights=[w for _, w in options])[0]

    if op == "kn":
        p, f = rng.choice(unknown)
        st.learn(p, f, num)
        return "kn %s %s." % (p, f)
    if op == "tl":
        a, b = rng.choice(tells)
        for f in sorted(st.knows[a] - st.knows[b]):
            st.knows[b].add(f)
            st.why[b][f] = {num} | st.why[a][f]
        return "tl %s %s." % (a, b)
    p = rng.choice(holders)
    f = rng.choice(sorted(st.knows[p]))
    st.forget(p, f)
    return "fg %s %s." % (p, f)


def query(rng, st, people, facts):
    # Prefer questions whose answer needs several records (a tell chain),
    # so that the answer is not a single lookup.
    for _ in range(30):
        op = rng.choice(QUERY_OPS)
        if op == "cn":
            p = rng.choice(people)
            if len(st.knows[p]) >= 2:
                return "cn %s?\t%d" % (p, len(st.knows[p]))
        elif op == "hk":
            p = rng.choice(people)
            f = rng.choice(facts)
            if f not in st.knows[p]:
                return "hk %s %s?\tno" % (p, f)
            if len(st.why[p][f]) >= 2:
                return "hk %s %s?\tyes" % (p, f)
        else:
            f = rng.choice(facts)
            holders = [p for p in people if f in st.knows[p]]
            if holders and any(len(st.why[p][f]) >= 2 for p in holders):
                return "wh %s?\t%s" % (f, ",".join(sorted(holders)))
    # Fallback (rarely reached): a plain yes/no question.
    p = rng.choice(people)
    f = rng.choice(facts)
    return "hk %s %s?\t%s" % (p, f, "yes" if f in st.knows[p] else "no")


def episode(rng, size):
    people = [PEOPLE[i] for i in sorted(rng.sample(range(8), rng.randint(3, 6)))]
    facts = [FACTS[i] for i in sorted(rng.sample(range(8), rng.randint(2, 5)))]
    st = State(people)
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    q_at = sorted(rng.randint(1, n_rec) for _ in range(n_q))

    rows = []
    num = 0
    qi = 0
    for r in range(1, n_rec + 1):
        num += 1
        rows.append("%d %s" % (num, record(rng, st, people, facts, num)))
        while qi < n_q and q_at[qi] == r:
            num += 1
            rows.append("%d %s" % (num, query(rng, st, people, facts)))
            qi += 1
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        rows = episode(rng, size)
        text = "\n".join(rows)
        while text in seen:  # episodes must differ from each other
            rows = episode(rng, size)
            text = "\n".join(rows)
        seen.add(text)
        for row in rows:
            print(row)


if __name__ == "__main__":
    main(sys.argv)
