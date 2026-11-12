#!/usr/bin/env python3
"""Synthetic place-direction records with queries (codes and numbers only).

Records: "vn A B." = A lies north of B (same column, A higher);
         "vs", "ve", "vw" = south, east, west likewise.
Queries:
  "ln A B?"  yes/no: A and B lie on one straight line, linked by a chain of
             records that all run north-south or all run east-west.
  "rt A B?"  list: the steps of the way from A to B (unique shortest chain of
             records, places in order, comma-separated).
  "hp A B?"  number: number of records (steps) on that unique shortest chain.
All answers are computed from the records above the query only.
"""
import random
import sys
from collections import deque

NS_OPS = {"vn", "vs"}
EW_OPS = {"ve", "vw"}
QUERY_OPS = ("rt", "ln", "hp")
PLACE_POOL = ["p%d" % i for i in range(1, 9)]
GRID = [(x, y) for x in range(3) for y in range(3)]


def make_pool(rng, k):
    """All same-line place pairs of a random placement, one record each."""
    places = rng.sample(PLACE_POOL, k)
    pos = dict(zip(places, rng.sample(GRID, k)))
    pool = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = places[i], places[j]
            if rng.random() < 0.5:
                a, b = b, a
            (ax, ay), (bx, by) = pos[a], pos[b]
            if ax == bx:
                op = "vn" if ay > by else "vs"
            elif ay == by:
                op = "ve" if ax > bx else "vw"
            else:
                continue
            pool.append((op, a, b))
    return pool


def shortest(edges, src, dst):
    """Return (distance, number of shortest paths, one shortest path) or None."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    if src not in adj or dst not in adj:
        return None
    dist = {src: 0}
    ways = {src: 1}
    pred = {src: []}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                ways[v] = 0
                pred[v] = []
                queue.append(v)
            if dist[v] == dist[u] + 1:
                ways[v] += ways[u]
                pred[v].append(u)
    if dst not in dist:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]][0])
    path.reverse()
    return dist[dst], ways[dst], path


def make_query(rng, recs):
    """Return (op, A, B, answer) for a query that needs several records, or None."""
    places = sorted({p for _, a, b in recs for p in (a, b)})
    if len(places) < 2:
        return None
    a, b = rng.sample(places, 2)
    op = rng.choice(QUERY_OPS)
    if op == "ln":
        ns = [(x, y) for o, x, y in recs if o in NS_OPS]
        ew = [(x, y) for o, x, y in recs if o in EW_OPS]
        r_ns = shortest(ns, a, b)
        r_ew = shortest(ew, a, b)
        for r in (r_ns, r_ew):
            if r is not None and r[0] == 1:
                return None  # a single record answers it; not a reasoning task
        yes = r_ns is not None or r_ew is not None
        if not yes and rng.random() < 0.6:
            return None  # keep answers mostly "yes" so the chains matter
        return op, a, b, "yes" if yes else "no"
    r = shortest([(x, y) for _, x, y in recs], a, b)
    if r is None or r[1] != 1 or r[0] < 2:
        return None  # no route, several routes, or a single record
    dist, _, path = r
    if op == "hp":
        return op, a, b, str(dist)
    return op, a, b, ",".join(path)


def make_episode(rng, size):
    """Return the rows of one episode (without numbers), or None if it failed."""
    n_rec = rng.randint(3, 12) * size
    pool = []
    for _ in range(50):  # prefer a placement with enough distinct records
        pool = make_pool(rng, rng.randint(4, 8))
        if len(pool) >= n_rec:
            break
    if not pool:
        return None
    recs = []
    while len(recs) < n_rec:  # distinct records first, then evenly spread repeats
        batch = list(pool)
        rng.shuffle(batch)
        recs.extend(batch)
    recs = recs[:n_rec]
    n_q = rng.randint(1, 5)
    queries = []
    tries = 0
    while len(queries) < n_q and tries < 300:
        tries += 1
        cut = rng.randint(2, n_rec)  # query sits after this many records
        q = make_query(rng, recs[:cut])
        if q is not None:
            queries.append((cut,) + q)
    if len(queries) < n_q:
        return None
    queries.sort(key=lambda t: t[0])
    rows = []
    qi = 0
    for i, (o, x, y) in enumerate(recs, 1):
        rows.append("%s %s %s." % (o, x, y))
        while qi < len(queries) and queries[qi][0] == i:
            _, qo, qa, qb, ans = queries[qi]
            rows.append("%s %s %s?\t%s" % (qo, qa, qb, ans))
            qi += 1
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        rows = make_episode(rng, size)
        if rows is None:
            continue
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(rows)
    for rows in episodes:
        for i, row in enumerate(rows, 1):
            print("%d %s" % (i, row))


if __name__ == "__main__":
    main()
