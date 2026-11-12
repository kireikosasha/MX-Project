#!/usr/bin/env python3
"""Synthetic reachability task over a small directed network, given as records.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)

Records are one-way links between node codes n1..n8:
  "ln a b."  a wire from a to b (a can send to b)
  "rl a b."  a relay from a to b (a can send to b)
A message may travel along any chain of links of either kind.

Queries are answered from the whole network above them:
  "xr a b?"  can a reach b (yes / no)
  "zw b?"    which nodes can reach b (sorted, comma-separated, no spaces)
"""
import random
import sys
from collections import deque

RECORD_OPS = ("ln", "rl")
QUERY_OPS = ("xr", "zw")
POOL = ["n%d" % i for i in range(1, 9)]
SIZES = (1, 3, 10)


def distances(start, adj):
    """Breadth-first hop counts from start over the directed links."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def simulate_query(query, adj, nodes):
    """Compute the answer by simulation; None means the query is rejected."""
    if query[0] == "xr":
        _, a, b = query
        d = distances(a, adj)
        if b not in d:
            return "no"
        if d[b] < 2:
            return None  # one record answers it: not a multi-record task
        return "yes"
    _, b = query
    found = sorted(x for x in nodes if x != b and b in distances(x, adj))
    if not found:
        return None
    return ",".join(found)


def make_episode(rng, size):
    target = rng.randint(3, 12) * size
    k_min = 4
    while k_min < 8 and 2 * k_min * (k_min - 1) < target:
        k_min += 1
    k = rng.randint(k_min, 8)
    nodes = sorted(rng.sample(POOL, k), key=lambda c: int(c[1:]))
    target = min(target, 2 * k * (k - 1))

    # A planted chain of 2 or more links, so that the chain query needs
    # several records. Longer chains are allowed at larger sizes.
    max_len = min(k - 1, 2 + size)
    length = rng.randint(2, max_len)
    path = rng.sample(nodes, length + 1)
    start, end = path[0], path[-1]
    links = [(rng.choice(RECORD_OPS), a, b) for a, b in zip(path, path[1:])]
    target = max(target, len(links))

    # Fill with other links; the start->end pair is kept free of direct
    # links so the chain query is never a single-record lookup.
    used = set(links)
    free = [(op, a, b) for op in RECORD_OPS for a in nodes for b in nodes
            if a != b and (a, b) != (start, end) and (op, a, b) not in used]
    need = min(target - len(links), len(free))
    records = links + rng.sample(free, need)
    rng.shuffle(records)

    adj = {}
    for _, a, b in records:
        adj.setdefault(a, []).append(b)

    queries = [("xr", start, end)]
    q_target = rng.randint(1, 5)
    tries = 0
    while len(queries) < q_target and tries < 300:
        tries += 1
        if rng.random() < 0.6:
            a, b = rng.sample(nodes, 2)
            q = ("xr", a, b)
        else:
            q = ("zw", rng.choice(nodes))
        if q in queries or simulate_query(q, adj, nodes) is None:
            continue
        queries.append(q)
    rng.shuffle(queries)

    rows = ["%s %s %s." % r for r in records]
    for q in queries:
        ans = simulate_query(q, adj, nodes)
        if q[0] == "xr":
            rows.append("xr %s %s?\t%s" % (q[1], q[2], ans))
        else:
            rows.append("zw %s?\t%s" % (q[1], ans))
    return rows


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
    out = []
    for _ in range(n):
        for i, row in enumerate(make_episode(rng, size), 1):
            out.append("%d %s" % (i, row))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
