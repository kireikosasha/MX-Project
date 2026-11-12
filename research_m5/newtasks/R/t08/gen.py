#!/usr/bin/env python3
"""Synthetic place-map episodes written as records and queries.

Usage: python gen.py N SEED [SIZE]      SIZE is 1, 3 or 10 (default 1).

Places p1..p8 sit on an integer grid. Record "vek a D b n." says place a lies
n steps in direction D from place b. Direction codes: 1 north, 2 east,
3 south, 4 west. Queries:
  qeb a b?  direction code of a from b (only when a and b share a row or column)
  zul a b?  number of grid steps between a and b (|dx| + |dy|)
  mup a b?  places inside the box spanned by a and b, sorted (skipped if none)
Records form a connected tree plus restated true facts, so every query is
answerable from the records above it, and the answers come from the layout.
"""
import random
import sys

SIZES = (1, 3, 10)
MAX_PLACES = 8
REC_OP = "vek"
Q_DIR = "qeb"
Q_STEPS = "zul"
Q_BETWEEN = "mup"
VEC = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}


def offset_to_direction(dx, dy):
    """(direction code, steps) for an axis-aligned offset, otherwise None."""
    if dx == 0 and dy != 0:
        return (1, dy) if dy > 0 else (3, -dy)
    if dy == 0 and dx != 0:
        return (2, dx) if dx > 0 else (4, -dx)
    return None


def try_layout(rng, n_pl, reach):
    """Grow a random tree of places; each new place is a straight run from a
    earlier place. Returns (positions, tree edges) or None if stuck."""
    pos = [(0, 0)]
    tree = []
    for i in range(1, n_pl):
        cell = None
        for _ in range(100):
            parent = rng.randrange(i)
            code = rng.choice(sorted(VEC))
            steps = rng.randint(1, reach)
            vx, vy = VEC[code]
            cand = (pos[parent][0] + vx * steps, pos[parent][1] + vy * steps)
            if cand not in pos:
                cell = cand
                break
        if cell is None:
            return None
        pos.append(cell)
        tree.append((i, parent, code, steps))
    return pos, tree


def make_episode(rng, size):
    """Return the rows of one episode (records, then queries)."""
    n_rec = rng.randint(3 * size, 12 * size)
    n_pl = rng.randint(3, min(MAX_PLACES, n_rec + 1))
    labels = rng.sample(range(1, MAX_PLACES + 1), n_pl)

    def name(i):
        return "p%d" % labels[i]

    while True:
        layout = None
        while layout is None:
            layout = try_layout(rng, n_pl, 3 * size)
        pos, tree = layout
        records = list(tree)

        # Every record is a true aligned fact, so restated facts are allowed.
        aligned = []
        for a in range(n_pl):
            for b in range(n_pl):
                if a != b:
                    r = offset_to_direction(pos[a][0] - pos[b][0],
                                            pos[a][1] - pos[b][1])
                    if r is not None:
                        aligned.append((a, b, r[0], r[1]))
        while len(records) < n_rec:
            records.append(rng.choice(aligned))
        rng.shuffle(records)

        # Queries must not be answered by one direct record.
        linked = {frozenset((a, b)) for a, b, _, _ in records}
        cands = []
        for a in range(n_pl):
            for b in range(n_pl):
                if a == b or frozenset((a, b)) in linked:
                    continue
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                r = offset_to_direction(dx, dy)
                if r is not None:
                    cands.append((Q_DIR, a, b, str(r[0])))
                cands.append((Q_STEPS, a, b, str(abs(dx) + abs(dy))))
                x0, x1 = sorted((pos[a][0], pos[b][0]))
                y0, y1 = sorted((pos[a][1], pos[b][1]))
                inside = [c for c in range(n_pl)
                          if c not in (a, b)
                          and x0 <= pos[c][0] <= x1
                          and y0 <= pos[c][1] <= y1]
                if inside:
                    cands.append((Q_BETWEEN, a, b,
                                  ",".join(sorted(name(c) for c in inside))))
        if cands:
            break

    rng.shuffle(cands)
    chosen = cands[: rng.randint(1, 5)]
    rows = ["%s %s %d %s %d." % (REC_OP, name(a), code, name(b), steps)
            for a, b, code, steps in records]
    rows += ["%s %s %s?\t%s" % (op, name(a), name(b), ans)
             for op, a, b, ans in chosen]
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n_ep = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    out = []
    for _ in range(n_ep):
        for idx, row in enumerate(make_episode(rng, size), 1):
            out.append("%d %s" % (idx, row))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
