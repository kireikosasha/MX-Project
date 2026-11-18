#!/usr/bin/env python3
"""Which tie-break makes the marked way: candidates are breadth-first parents (first found) for every order of the
four moves, searched from S or from G; and A* with Manhattan distance, ties by g or by insertion. Kept: a candidate
that reproduces all 1000 training ways exactly; then the test."""
import csv, heapq, itertools, sys
from collections import deque
W = 30
def rows(p):
    r = csv.reader(open(p, newline="")); next(r); return [(q, a) for _, q, a, _ in r]
def bfs_way(q, s, g, order):
    par = {s: None}; todo = deque([s])
    while todo:
        i = todo.popleft()
        if i == g: break
        y, x = divmod(i, W)
        for dy, dx in order:
            if 0 <= y+dy < W and 0 <= x+dx < W:
                j = (y+dy)*W + x+dx
                if j not in par and q[j] != "#":
                    par[j] = i; todo.append(j)
    way, i = set(), g
    while i is not None: way.add(i); i = par.get(i)
    return way
def astar_way(q, s, g, order, tie):
    gy, gx = divmod(g, W); h = lambda i: abs(i//W-gy)+abs(i%W-gx)
    cnt = itertools.count(); best = {s: 0}; par = {s: None}
    heap = [(h(s), (0 if tie == "g" else -0), next(cnt), s)]
    while heap:
        f, _, _, i = heapq.heappop(heap)
        if i == g: break
        y, x = divmod(i, W)
        for dy, dx in order:
            if 0 <= y+dy < W and 0 <= x+dx < W:
                j = (y+dy)*W + x+dx
                if q[j] != "#" and best[i]+1 < best.get(j, 10**9):
                    best[j] = best[i]+1; par[j] = i
                    k = {"g": best[j], "-g": -best[j], "fifo": 0}[tie]
                    heapq.heappush(heap, (best[j]+h(j), k, next(cnt), j))
    way, i = set(), g
    while i is not None: way.add(i); i = par.get(i)
    return way
def render(q, way):
    return "".join("o" if i in way and q[i] == " " else q[i] for i in range(len(q)))
teach, test = rows(sys.argv[1]), rows(sys.argv[2])
N4 = [(0,1),(0,-1),(1,0),(-1,0)]
cands = []
for order in itertools.permutations(N4):
    cands.append(("bfs from S", order, lambda q, s, g, o=order: bfs_way(q, s, g, o)))
    cands.append(("bfs from G", order, lambda q, s, g, o=order: bfs_way(q, g, s, o)))
    for tie in ("g", "-g", "fifo"):
        cands.append((f"A* tie {tie} from S", order, lambda q, s, g, o=order, t=tie: astar_way(q, s, g, o, t)))
        cands.append((f"A* tie {tie} from G", order, lambda q, s, g, o=order, t=tie: astar_way(q, g, s, o, t)))
scored = []
for name, order, f in cands:
    n = 0
    for q, a in teach[:100]:
        n += render(q, f(q, q.index("S"), q.index("G"))) == a
    scored.append((n, name, order, f))
scored.sort(key=lambda t: -t[0])
for n, name, order, _ in scored[:5]:
    print(f"{name} {order}: {n}/100 training ways reproduced")
n, name, order, f = scored[0]
full = sum(render(q, f(q, q.index("S"), q.index("G"))) == a for q, a in teach)
right = sum(render(q, f(q, q.index("S"), q.index("G"))) == a for q, a in test)
print(f"kept {name} {order}: training {full}/1000; test {right}/1000")
