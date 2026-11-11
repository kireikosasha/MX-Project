#!/usr/bin/env python3
"""Synthetic containment task given as records.

Usage: python gen.py N SEED [SIZE]     SIZE in {1, 3, 10}, default 1

Entities: boxes b1-b4 (containers that can be moved), objects o1-o6 (weighted
things), rooms r1-r3 (outermost containers, never moved).

Record operations:
  mek X [W]   create entity X; an object o* gets weight W. It starts loose.
  zib X Y     put X (object or box) into container Y (box or room). Whatever is
              inside X moves with it.
  kol X       take object or box X out of the box it is in; it is placed where
              that box is now.

Query operations (answered from the simulated state at that row):
  ker X?      the room X is in now (X is inside a box inside a room, or deeper)
  mot R?      total weight of all objects inside room R now, at any depth
  lis R?      objects inside room R now, sorted, separated by commas
"""
import random
import sys

BOXES = ["b1", "b2", "b3", "b4"]
OBJECTS = ["o1", "o2", "o3", "o4", "o5", "o6"]
ROOMS = ["r1", "r2", "r3"]
ALL = BOXES + OBJECTS + ROOMS


class World:
    def __init__(self, size):
        self.max_weight = 9 if size == 1 else 9 * size
        self.parent = {}  # existing entity -> container code, or None if loose
        self.weight = {}  # object -> weight

    def ancestors(self, e):
        out = []
        p = self.parent[e]
        while p is not None:
            out.append(p)
            p = self.parent[p]
        return out

    def inside(self, e):
        """Every entity inside e, directly or indirectly."""
        found = []
        todo = [x for x, p in self.parent.items() if p == e]
        while todo:
            x = todo.pop()
            found.append(x)
            todo.extend(y for y, p in self.parent.items() if p == x)
        return found


def can_put(w, x, y):
    if x not in w.parent or y not in w.parent:
        return False
    if x[0] not in "ob" or y[0] not in "br":
        return False
    if x == y or w.parent[x] == y:
        return False
    if x in w.ancestors(y):  # would put x inside itself
        return False
    return True


def choose_record(w, rng):
    """Pick and apply one random record. Returns its text, or None if stuck."""
    uncreated = [e for e in ALL if e not in w.parent]
    moves = [(x, y) for x in w.parent for y in w.parent if can_put(w, x, y)]
    takeout = [x for x in w.parent
               if x[0] in "ob" and w.parent[x] is not None and w.parent[x][0] == "b"]

    options = []
    if uncreated and (len(w.parent) < 2 or rng.random() < 0.25):
        options.append((4, "mek"))
    if moves:
        options.append((6, "zib"))
    if takeout:
        options.append((2, "kol"))
    if not options:
        return None

    weights, names = zip(*options)
    choice = rng.choices(names, weights)[0]
    if choice == "mek":
        e = rng.choice(uncreated)
        w.parent[e] = None
        if e[0] == "o":
            w.weight[e] = rng.randint(1, w.max_weight)
            return f"mek {e} {w.weight[e]}."
        return f"mek {e}."
    if choice == "zib":
        x, y = rng.choice(moves)
        w.parent[x] = y
        return f"zib {x} {y}."
    x = rng.choice(takeout)
    w.parent[x] = w.parent[w.parent[x]]
    return f"kol {x}."


def query_options(w):
    opts = []
    for e in w.parent:
        if e[0] in "ob":
            anc = w.ancestors(e)
            if len(anc) >= 2 and anc[-1][0] == "r":
                opts.append((f"ker {e}?", anc[-1]))
    for r in ROOMS:
        if r in w.parent:
            objs = sorted(x for x in w.inside(r) if x[0] == "o")
            if len(objs) >= 2:
                opts.append((f"mot {r}?", str(sum(w.weight[o] for o in objs))))
                opts.append((f"lis {r}?", ",".join(objs)))
    return opts


def pick_query(w, rng, asked):
    opts = [o for o in query_options(w) if o[0] not in asked]
    if not opts:
        return None
    text, ans = rng.choice(opts)
    asked.add(text)
    return text, ans


def try_episode(rng, size):
    w = World(size)
    target_records = rng.randint(3, 12) * size
    target_queries = rng.randint(1, 5)
    rows = []  # (text, answer); answer is None for records
    asked = set()
    records = queries = 0
    while records < target_records:
        if records > 0 and queries < target_queries and rng.random() < 0.3:
            q = pick_query(w, rng, asked)
            if q is not None:
                rows.append(q)
                queries += 1
        rec = choose_record(w, rng)
        if rec is None:
            return None
        rows.append((rec, None))
        records += 1
    while queries < target_queries:
        q = pick_query(w, rng, asked)
        if q is None:
            return None
        rows.append(q)
        queries += 1
    return rows


def make_episode(rng, size):
    while True:
        rows = try_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for num, (text, ans) in enumerate(make_episode(rng, size), 1):
            if ans is None:
                lines.append(f"{num} {text}")
            else:
                lines.append(f"{num} {text}\t{ans}")
    if lines:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
