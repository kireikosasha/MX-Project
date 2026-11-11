#!/usr/bin/env python3
"""Synthetic ORDER-IN-TIME reasoning episodes, written as records only.

Usage: python gen.py N SEED [SIZE]      SIZE in {1, 3, 10}, default 1.

Model (internal, never printed as words): movers k1..k6 move through
places p1..p6. Line order is time order.

Records:
  ruk k p.   mover k arrives at place p (it leaves whatever place it was at).
  fip k.     mover k leaves its current place and is nowhere until its next ruk.

Queries (answer computed from the simulation of the lines above):
  zoq k p?   the place k occupied just before its arrival at p. Asked only when
             k arrives at p exactly once and that arrival was not its first.
  bik x y p? which of movers x and y arrived at p first. Asked only when both
             arrive at p exactly once.
  nxt k p?   the place k occupied just after its arrival at p. Asked only when
             k arrives at p exactly once and that arrival was not its last.

Each episode is numbered from 1; a new episode starts when numbering restarts.
"""
import random
import sys

MOVERS = ["k%d" % i for i in range(1, 7)]
PLACES = ["p%d" % i for i in range(1, 7)]
SIZES = (1, 3, 10)


def query_candidates(arr):
    """arr: mover -> list of (time, place) arrivals. Returns (body, answer) pairs."""
    out = []
    for m in MOVERS:
        a = arr[m]
        if not a:
            continue
        places = [pl for _, pl in a]
        for p in PLACES:
            if places.count(p) == 1:
                i = places.index(p)
                if i >= 1:
                    out.append(("zoq %s %s" % (m, p), places[i - 1]))
                if i + 1 < len(places):
                    out.append(("nxt %s %s" % (m, p), places[i + 1]))
    for x in MOVERS:
        for y in MOVERS:
            if x == y:
                continue
            for p in PLACES:
                tx = [t for t, pl in arr[x] if pl == p]
                ty = [t for t, pl in arr[y] if pl == p]
                if len(tx) == 1 and len(ty) == 1:
                    ans = x if tx[0] < ty[0] else y
                    out.append(("bik %s %s %s" % (x, y, p), ans))
    return out


def _attempt(rng, n_records, n_queries):
    """Build one episode as a list of numbered lines, or None if it has no query."""
    at = {m: None for m in MOVERS}          # current place, None when nowhere
    arr = {m: [] for m in MOVERS}           # arrivals: (line number, place)
    lines = []
    seen = set()
    recs_done = 0
    recs_left = n_records
    qs_left = n_queries
    while True:
        num = len(lines) + 1
        if qs_left > 0 and recs_done >= 2 and (recs_left == 0 or rng.random() < 0.35):
            cands = [c for c in query_candidates(arr) if c[0] not in seen]
            if cands:
                body, ans = rng.choice(cands)
                seen.add(body)
                lines.append("%d %s?\t%s" % (num, body, ans))
                qs_left -= 1
                continue
        if recs_left == 0:
            break
        movers_in = [m for m in MOVERS if at[m] is not None]
        if movers_in and rng.random() < 0.25:
            m = rng.choice(movers_in)
            at[m] = None
            lines.append("%d fip %s." % (num, m))
        else:
            m = rng.choice(MOVERS)
            p = rng.choice([q for q in PLACES if q != at[m]])
            at[m] = p
            arr[m].append((num, p))
            lines.append("%d ruk %s %s." % (num, m, p))
        recs_left -= 1
        recs_done += 1
    if qs_left == n_queries:
        return None
    return lines


def make_episode(rng, size, seen_eps):
    base = rng.randint(3, 12)
    n_records = base * size
    n_queries = rng.randint(1, 5)
    for _ in range(10000):
        lines = _attempt(rng, n_records, n_queries)
        if lines is None:
            continue
        key = "\n".join(lines)
        if key in seen_eps:
            continue
        seen_eps.add(key)
        return lines
    raise RuntimeError("could not build a distinct episode")


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
    seen_eps = set()
    out = []
    for _ in range(n):
        out.extend(make_episode(rng, size, seen_eps))
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
