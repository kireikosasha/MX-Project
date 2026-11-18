#!/usr/bin/env python3
"""
Maze-Hard by mined invariants and one monotone operator, nothing said of mazes.

From the 1000 solved training mazes (question grid -> answer grid with the way marked):
  1. which question symbols the way may stand on (the symbol under a marked cell is ever that one)
  2. which neighbourhood the way is a chain in: of a few candidate offset sets, the one under which every marked cell
     has exactly two neighbours on the way and the two ends exactly one, in every training maze
  3. whether the way is a shortest one (its length equals the distance by the monotone min-plus turn, which only
     lowers distances and so must come to a fixed point), and whether it is the only shortest one
Then a test maze is answered by the same min-plus turn from both ends: the cells with dS + dG = d(S, G).

    python maze.py TRAIN.csv TEST.csv
"""
import csv
import sys
from collections import deque

W = 30
N4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
N8 = N4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def rows(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        return [(q, a) for _, q, a, _ in r]


def nb(i, D):
    y, x = divmod(i, W)
    for dy, dx in D:
        if 0 <= y + dy < W and 0 <= x + dx < W:
            yield (y + dy) * W + x + dx


def dist(q, start, ok, D):
    """the monotone turn: a distance only ever goes down (min over neighbours + 1); breadth first is its fastest order"""
    d = [None] * len(q)
    d[start] = 0
    todo = deque([start])
    while todo:
        i = todo.popleft()
        for j in nb(i, D):
            if d[j] is None and q[j] in ok:
                d[j] = d[i] + 1
                todo.append(j)
    return d


def main():
    teach, test = rows(sys.argv[1]), rows(sys.argv[2])
    mark = sorted(set(teach[0][1]) - set(teach[0][0]))[0]
    under = {q[i] for q, a in teach for i in range(len(q)) if a[i] == mark}
    ends = sorted(set(teach[0][0]) - {" ", "#"})                 # (symbols met once in every question)
    ends = [s for s in sorted(set(c for q, _ in teach for c in q)) if all(q.count(s) == 1 for q, _ in teach)]
    print(f"marked by {mark!r}; the way stands on {sorted(under)}; symbols met once in every maze: {ends}")
    hood = None
    for name, D in (("4 neighbours", N4), ("8 neighbours", N8)):
        good = True
        for q, a in teach:
            way = {i for i in range(len(a)) if a[i] == mark or q[i] in ends}
            for i in way:
                k = sum(j in way for j in nb(i, D))
                if k != (1 if q[i] in ends else 2):
                    good = False
                    break
            if not good:
                break
        print(f"  the way is a chain under {name}: {'in every training maze' if good else 'no'}")
        if good and hood is None:
            hood = D
    ok = under | set(ends)
    shortest = only = 0
    for q, a in teach:
        s, g = q.index(ends[0]), q.index(ends[1])
        ds, dg = dist(q, s, ok, hood), dist(q, g, ok, hood)
        length = sum(c == mark for c in a) + 1
        shortest += ds[g] == length
        on = [i for i in range(len(q)) if ds[i] is not None and dg[i] is not None and ds[i] + dg[i] == ds[g]]
        only += len(on) == ds[g] + 1
    print(f"  the way is a shortest one in {shortest} of {len(teach)} training mazes; the only shortest one in {only}")
    right = 0
    for q, a in test:
        s, g = q.index(ends[0]), q.index(ends[1])
        ds, dg = dist(q, s, ok, hood), dist(q, g, ok, hood)
        got = "".join(mark if q[i] not in ends and ds[i] is not None and dg[i] is not None and ds[i] + dg[i] == ds[g] else q[i]
                      for i in range(len(q)))
        right += got == a
    print(f"test mazes {len(test)}: exactly right {right} ({100 * right / len(test):.1f}%)")


if __name__ == "__main__":
    main()
