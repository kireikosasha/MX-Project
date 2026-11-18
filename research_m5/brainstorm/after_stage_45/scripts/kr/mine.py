#!/usr/bin/env python3
"""
Rules from solutions, then an exact search that knows nothing of the puzzle.

From n solved grids of the training split (the answers only), keep every pair of cells that is never equal in any of
them: that is the whole of what is learnt. The alphabet is the set of symbols seen. Then every test puzzle is solved
by a generic search over the learnt pairs: a cell's options are the symbols, a decided cell removes its symbol from
the options of the cells it is paired with (a monotone step: options only shrink, so the propagation always comes to
a fixed point), and where the fixed point leaves choices, the cell with fewest options is branched on (DPLL).

    python mine.py TRAIN.csv TEST.csv [how many test puzzles]
"""
import csv
import sys
import time


def rows(path, n):
    out = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 3 or len(row[1]) != 81 or len(row[2]) != 81:
                continue                                         # (the last line of a partial download)
            out.append((row[1], row[2]))
            if len(out) == n:
                break
    return out


def mine(solutions):
    cells = len(solutions[0])
    pairs = set((i, j) for i in range(cells) for j in range(i + 1, cells))
    for s in solutions:
        pairs = {(i, j) for i, j in pairs if s[i] != s[j]}
    return pairs


def peers_true():
    same = lambda i, j: i // 9 == j // 9 or i % 9 == j % 9 or (i // 27 == j // 27 and i % 9 // 3 == j % 9 // 3)
    return {(i, j) for i in range(81) for j in range(i + 1, 81) if same(i, j)}


def solver(pairs, symbols, cells=81):
    nb = [[] for _ in range(cells)]
    for i, j in pairs:
        nb[i].append(j)
        nb[j].append(i)
    full = (1 << len(symbols)) - 1
    index = {c: k for k, c in enumerate(symbols)}

    def propagate(dom, todo):
        while todo:
            i = todo.pop()
            bit = dom[i]
            for j in nb[i]:
                if dom[j] & bit:
                    dom[j] &= ~bit
                    if dom[j] == 0:
                        return False
                    if dom[j] & (dom[j] - 1) == 0:
                        todo.append(j)
        return True

    def search(dom):
        best, bc = -1, 99
        for i in range(cells):
            d = dom[i]
            if d & (d - 1):
                c = bin(d).count("1")
                if c < bc:
                    best, bc = i, c
        if best < 0:
            return dom
        d = dom[best]
        while d:
            bit = d & -d
            d ^= bit
            new = dom[:]
            new[best] = bit
            if propagate(new, [best]):
                got = search(new)
                if got:
                    return got
        return None

    def solve(puzzle):
        dom = [full] * cells
        todo = []
        for i, ch in enumerate(puzzle):
            if ch in index:
                dom[i] = 1 << index[ch]
                todo.append(i)
        if not propagate(dom, todo):
            return None
        got = search(dom)
        if got is None:
            return None
        return "".join(symbols[d.bit_length() - 1] for d in got)

    return solve


def main():
    train, test = sys.argv[1], sys.argv[2]
    many = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    teach = rows(train, 1000)
    truth = peers_true()
    print(f"true pairs of cells that must differ: {len(truth)}")
    for n in (5, 10, 20, 40, 60, 80, 100, 200, 1000):
        got = mine([s for _, s in teach[:n]])
        print(f"from {n:4d} solutions: kept {len(got)} pairs; true ones among them {len(got & truth)}, spurious {len(got - truth)}")
    pairs = mine([s for _, s in teach])
    symbols = sorted({c for _, s in teach for c in s})
    solve = solver(pairs, symbols)
    tried = rows(test, many)
    t0, right = time.time(), 0
    for puzzle, answer in tried:
        right += solve(puzzle) == answer
    print(f"rules from 1000 solutions; test puzzles {len(tried)}: exactly right {right} ({100 * right / len(tried):.2f}%), "
          f"{1000 * (time.time() - t0) / len(tried):.1f} ms a puzzle on one core")


if __name__ == "__main__":
    main()
