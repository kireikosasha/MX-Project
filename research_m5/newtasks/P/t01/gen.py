#!/usr/bin/env python3
"""Synthetic containment-tracking task given as records.

Rooms (r), containers (c) and items (i) form a containment forest.
Records move things: zor puts a thing into a container or room,
xtk takes a thing out of its container into the room, swp exchanges
the places of two things. Queries ask where a thing is now: qrm gives
the room, qcn counts the items inside a container (at any depth), and
qin asks whether a thing is inside a container (at any depth).

Usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1).
"""
import random
import sys

ROOMS = ['r1', 'r2', 'r3']
CONTS = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
ITEMS = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8']
MOVABLE = CONTS + ITEMS


def chain(parent, x):
    """Ancestors of x, nearest first, ending at a room."""
    out = []
    p = parent.get(x)
    while p is not None:
        out.append(p)
        p = parent.get(p)
    return out


def try_record(rng, parent):
    op = rng.choice(['zor', 'zor', 'zor', 'xtk', 'swp'])
    if op == 'zor':
        x = rng.choice(MOVABLE)
        y = rng.choice(CONTS + ROOMS)
        # a target container must already sit in a room (or inside one that does)
        if y in CONTS and y not in parent:
            return None
        if y == x or x in chain(parent, y) or parent.get(x) == y:
            return None
        parent[x] = y
        return f"zor {x} {y}."
    if op == 'xtk':
        inside = [e for e in MOVABLE if e in parent and parent[e][0] == 'c']
        if not inside:
            return None
        x = rng.choice(inside)
        parent[x] = chain(parent, x)[-1]
        return f"xtk {x}."
    placed = [e for e in MOVABLE if e in parent]
    if len(placed) < 2:
        return None
    a, b = rng.sample(placed, 2)
    if parent[a] == parent[b] or a in chain(parent, b) or b in chain(parent, a):
        return None
    parent[a], parent[b] = parent[b], parent[a]
    return f"swp {a} {b}."


def try_query(rng, parent):
    op = rng.choice(['qrm', 'qcn', 'qin'])
    placed = [e for e in MOVABLE if e in parent]
    if op == 'qrm':
        cands = [e for e in placed if len(chain(parent, e)) >= 2]
        if not cands:
            return None
        x = rng.choice(cands)
        return f"qrm {x}?\t{chain(parent, x)[-1]}"
    if op == 'qcn':
        counts = []
        for c in CONTS:
            if c in parent:
                n = sum(1 for i in ITEMS if c in chain(parent, i))
                if n >= 2:
                    counts.append((c, n))
        if not counts:
            return None
        c, n = rng.choice(counts)
        return f"qcn {c}?\t{n}"
    if not placed:
        return None
    x = rng.choice(placed)
    cs = [c for c in CONTS if c in parent and c != x]
    if not cs:
        return None
    c = rng.choice(cs)
    return f"qin {x} {c}?\t{'yes' if c in chain(parent, x) else 'no'}"


def episode(rng, size, seen):
    while True:
        parent = {}
        target = rng.randint(3, 12) * size
        nq = rng.randint(1, 5)
        rows, recs, qs, tries, ok = [], 0, 0, 0, True
        while recs < target or qs < nq:
            tries += 1
            if tries > 20000:
                ok = False
                break
            if qs < nq and (recs >= target or rng.random() < 0.35):
                q = try_query(rng, parent)
                if q is not None:
                    rows.append(q)
                    qs += 1
                    continue
                if recs >= target:
                    ok = False
                    break
            r = try_record(rng, parent)
            if r is not None:
                rows.append(r)
                recs += 1
        if not ok:
            continue
        text = "\n".join(rows)
        if text in seen:
            continue
        seen.add(text)
        return rows


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: gen.py N SEED [SIZE]")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    lines = []
    for _ in range(n):
        for i, row in enumerate(episode(rng, size, seen), 1):
            lines.append(f"{i} {row}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
