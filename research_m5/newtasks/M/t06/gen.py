#!/usr/bin/env python3
"""Synthetic order-in-time task given as records.

Each record says that a mover was at a place at a time, in the form
"vx MOVER PLACE TIME." The records of an episode appear in random order,
so the visit sequence of a mover must be rebuilt by sorting on TIME.
Queries ask about the place just before or just after a given place, or
whether one place was reached before another. A query is answered only
from the records above it.

Usage: python gen.py N SEED [SIZE]
"""
import random
import sys

MOVERS = ["m1", "m2", "m3", "m4"]
PLACES = ["p1", "p2", "p3", "p4", "p5", "p6"]
REC_OP = "vx"                    # record: mover MOVER was at PLACE at TIME
QUERY_OPS = ("pv", "nx", "fs")   # pv: place before; nx: place after; fs: yes/no came first


def make_records(rng, size):
    """Return (time, mover, place) triples in random line order."""
    n = rng.randint(3, 12) * size
    times = rng.sample(range(1, 100 * size + 1), n)  # distinct times, so order is total
    return [(t, rng.choice(MOVERS), rng.choice(PLACES)) for t in times]


def visits_of(prefix, mover):
    """Visits of one mover in time order, as (time, place)."""
    return sorted((t, p) for (t, m, p) in prefix if m == mover)


def candidate_queries(prefix):
    """All queries that the records in prefix answer unambiguously.

    A (mover, place) pair is used only when the mover was at that place
    exactly once in the prefix, so every answer is unique.
    """
    out = []
    for m in MOVERS:
        vis = visits_of(prefix, m)
        places = [p for _, p in vis]
        once = [p for p in PLACES if places.count(p) == 1]
        time_of = {p: t for t, p in vis}
        for i, (_, p) in enumerate(vis):
            if p not in once:
                continue
            if i > 0:
                out.append((f"pv {m} {p}?", vis[i - 1][1]))
            if i < len(vis) - 1:
                out.append((f"nx {m} {p}?", vis[i + 1][1]))
        for p in once:
            for q in once:
                if p != q:
                    ans = "yes" if time_of[p] < time_of[q] else "no"
                    out.append((f"fs {m} {p} {q}?", ans))
    return out


def make_episode(rng, size):
    """Return the list of output lines (without numbers) for one episode."""
    while True:
        recs = make_records(rng, size)
        n = len(recs)
        nq = rng.randint(1, 5)
        cuts = sorted(rng.randint(2, n) for _ in range(nq))

        rows = []          # (is_query, text, answer)
        used = set()
        ci = 0
        for i, (t, m, p) in enumerate(recs):
            rows.append((False, f"{REC_OP} {m} {p} {t}.", None))
            count = i + 1
            while ci < len(cuts) and cuts[ci] == count:
                ci += 1
                cands = [c for c in candidate_queries(recs[:count]) if c[0] not in used]
                if cands:
                    q, a = rng.choice(cands)
                    used.add(q)
                    rows.append((True, q, a))

        n_queries = sum(1 for r in rows if r[0])
        if n_queries == 0:
            continue  # retry: every episode needs at least one query

        out = []
        for num, (is_q, text, ans) in enumerate(rows, 1):
            if is_q:
                out.append(f"{num} {text}\t{ans}")
            else:
                out.append(f"{num} {text}")
        return out


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n_episodes = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2

    rng = random.Random(seed)
    seen = set()
    for _ in range(n_episodes):
        while True:
            ep = make_episode(rng, size)
            key = "\n".join(ep)
            if key not in seen:
                break
        seen.add(key)
        for line in ep:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
