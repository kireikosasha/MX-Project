#!/usr/bin/env python3
"""Synthetic records task: places on a map, four fixed direction codes.

Usage: python gen.py N SEED [SIZE]      SIZE in {1, 3, 10}, default 1

Records (one place lies a number of steps from another, in a fixed direction):
    vek A B dK n.     A lies exactly n steps in direction dK from B
Queries (answered only from the records above them in the same episode):
    qab A C?<TAB>dK   in which direction A lies from C
    nuf A C?<TAB>n    how many steps A lies from C (on that line)
    zep A C?<TAB>dK,dL,...   the directions of the steps of the one route A -> C

Fixed direction codes: d1 = north (0,+1), d2 = east (+1,0),
                       d3 = south (0,-1), d4 = west (-1,0).
"""

import random
import sys
from collections import deque

RECORD_OP = "vek"
QUERY_OPS = ("qab", "nuf", "zep")
VEC = {"d1": (0, 1), "d2": (1, 0), "d3": (0, -1), "d4": (-1, 0)}
OPPOSITE = {"d1": "d3", "d2": "d4", "d3": "d1", "d4": "d2"}
PLACE_CODES = ["p%d" % i for i in range(1, 9)]

SIZE_PARAMS = {
    1: {"places": (4, 6), "records": (3, 12), "gap": 3},
    3: {"places": (5, 8), "records": (9, 36), "gap": 9},
    10: {"places": (6, 8), "records": (30, 120), "gap": 30},
}
QUERY_COUNT = (1, 5)


def make_layout(rng, m, gap):
    """Place m distinct points on a small grid with random, increasing gaps."""
    while True:
        nr = rng.choice((2, 3))
        nc = rng.choice((3, 4))
        if nr * nc >= m:
            break
    ys = [0]
    for _ in range(nr - 1):
        ys.append(ys[-1] + rng.randint(1, gap))
    xs = [0]
    for _ in range(nc - 1):
        xs.append(xs[-1] + rng.randint(1, gap))
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    return [(xs[c], ys[r]) for r, c in rng.sample(cells, m)]


def relation(src, ref):
    """Return (direction, steps) if src lies on a straight line from ref, else None."""
    dx = src[0] - ref[0]
    dy = src[1] - ref[1]
    if dx == 0 and dy > 0:
        return "d1", dy
    if dy == 0 and dx > 0:
        return "d2", dx
    if dx == 0 and dy < 0:
        return "d3", -dy
    if dy == 0 and dx < 0:
        return "d4", -dx
    return None


def aligned_pairs(pos):
    out = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if pos[i][0] == pos[j][0] or pos[i][1] == pos[j][1]:
                out.append((i, j))
    return out


def spanning_split(rng, m, pairs):
    """Random spanning tree over aligned pairs; returns (tree, rest) or (None, None)."""
    order = list(pairs)
    rng.shuffle(order)
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree, rest = [], []
    for i, j in order:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))
        else:
            rest.append((i, j))
    if len(tree) != m - 1:
        return None, None
    return tree, rest


def key(a, b):
    return (a, b) if a < b else (b, a)


def step_vec(rec, u, v):
    """Vector added when moving from place u to adjacent place v along record rec."""
    s, r, d, n = rec
    vx, vy = VEC[d]
    if s == v and r == u:          # s = r + n*d, so moving r -> s goes d
        return (n * vx, n * vy)
    return (-n * vx, -n * vy)      # moving s -> r goes opposite


def step_dir(rec, u, v):
    """Direction code of the step from place u to adjacent place v."""
    s, r, d, n = rec
    if s == v and r == u:
        return d
    return OPPOSITE[d]


def simple_paths(adj, src, dst, limit=2):
    found = []

    def dfs(u, path, seen):
        if len(found) >= limit:
            return
        if u == dst:
            found.append(list(path))
            return
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                path.append(v)
                dfs(v, path, seen)
                path.pop()
                seen.discard(v)

    dfs(src, [src], {src})
    return found


def bfs_path(adj, src, dst):
    prev = {src: None}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        if u == dst:
            break
        for v in sorted(adj[u]):
            if v not in prev:
                prev[v] = u
                queue.append(v)
    path = [dst]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return list(reversed(path))


def make_episode(rng, size):
    par = SIZE_PARAMS[size]
    while True:
        m = rng.randint(*par["places"])
        pos = make_layout(rng, m, par["gap"])
        pairs = aligned_pairs(pos)
        if len(pairs) < m:
            continue
        tree, rest = spanning_split(rng, m, pairs)
        if tree is None:
            continue
        codes = rng.sample(PLACE_CODES, m)

        # Distinct recorded pairs: a spanning tree plus extras; always leave
        # at least one aligned pair unrecorded so that queries need reasoning.
        target = max(rng.randint(*par["records"]), m - 1)
        distinct_n = min(target, len(pairs) - 1)
        chosen = tree + rest[: distinct_n - (m - 1)]

        recs = []
        for i, j in chosen:
            if rng.random() < 0.5:
                i, j = j, i
            d, n = relation(pos[i], pos[j])
            recs.append((i, j, d, n))
        # Restate some facts in the opposite orientation to reach the target count.
        total = min(target, 2 * len(chosen))
        for i, j, d, n in rng.sample(recs, total - len(recs)):
            recs.append((j, i, OPPOSITE[d], n))
        rng.shuffle(recs)
        R = len(recs)

        first = {}
        adj = {v: set() for v in range(m)}
        for k, (s, r, _, _) in enumerate(recs):
            first.setdefault(key(s, r), k)
            adj[s].add(r)
            adj[r].add(s)

        cands = []
        for a in range(m):
            for c in range(m):
                if a == c:
                    continue
                # Shortest record path c -> a; the answer follows from these records.
                path = bfs_path(adj, c, a)
                idxs = [first[key(path[t], path[t + 1])] for t in range(len(path) - 1)]
                D = (0, 0)
                for t in range(len(path) - 1):
                    sx, sy = step_vec(recs[first[key(path[t], path[t + 1])]], path[t], path[t + 1])
                    D = (D[0] + sx, D[1] + sy)
                assert D == (pos[a][0] - pos[c][0], pos[a][1] - pos[c][1])
                if key(a, c) not in first:
                    rel = relation(D, (0, 0))
                    if rel is not None:
                        cands.append(("qab", a, c, rel[0], idxs))
                        cands.append(("nuf", a, c, str(rel[1]), idxs))
                paths = simple_paths(adj, a, c)
                if len(paths) == 1 and len(paths[0]) >= 3:
                    p = paths[0]
                    steps = []
                    ridx = []
                    for t in range(len(p) - 1):
                        k = first[key(p[t], p[t + 1])]
                        ridx.append(k)
                        steps.append(step_dir(recs[k], p[t], p[t + 1]))
                    cands.append(("zep", a, c, ",".join(steps), ridx))

        if not cands:
            continue
        rng.shuffle(cands)
        picked = cands[: min(rng.randint(*QUERY_COUNT), len(cands))]

        buckets = [[] for _ in range(R + 1)]
        for op, a, c, ans, idxs in picked:
            slot = rng.randint(max(idxs) + 1, R)
            buckets[slot].append("%s %s %s?\t%s" % (op, codes[a], codes[c], ans))

        body = []
        for i in range(R + 1):
            body.extend(buckets[i])
            if i < R:
                s, r, d, n = recs[i]
                body.append("%s %s %s %s %d." % (RECORD_OP, codes[s], codes[r], d, n))
        return tuple(body)


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZE_PARAMS:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        while True:
            ep = make_episode(rng, size)
            if ep not in seen:
                break
        seen.add(ep)
        for number, line in enumerate(ep, 1):
            out.append("%d %s" % (number, line))
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
