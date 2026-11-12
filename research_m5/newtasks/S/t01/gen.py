#!/usr/bin/env python3
"""Synthetic container-tracking task given as records.

Items sit inside boxes, boxes sit inside boxes or on places. Moving a box
moves everything inside it. Queries ask where a thing is now.

Usage: python gen.py N SEED [SIZE]
"""
import random
import sys

ITEMS = ["k1", "k2", "k3", "k4"]
BOXES = ["b1", "b2", "b3"]
PLACES = ["p1", "p2", "p3"]

# Record operations
#   dek X Y : put X (item or box) directly into Y (box or place)
#   fuz X   : take X out of its box and leave it directly on its top place
#   gor B P : move box B, with everything inside it, to place P
REC_OPS = ["dek", "fuz", "gor"]

# Query operations
#   zuv X   : which place is X at now
#   qeb X Y : are X and Y at the same place now (yes / no)
#   pimo P  : how many items are at place P now, including inside boxes
QRY_OPS = ["zuv", "qeb", "pimo"]


class World:
    def __init__(self, items, boxes, places):
        self.items = list(items)
        self.boxes = list(boxes)
        self.place_list = list(places)
        self.places = set(places)
        self.loc = {e: None for e in self.items + self.boxes}

    def ancestors(self, e):
        """Holders above e, nearest first, ending at a place or None."""
        out = []
        cur = self.loc.get(e)
        while cur is not None:
            out.append(cur)
            cur = self.loc.get(cur)
        return out

    def top(self, e):
        """The place where e is now, or None if e is not placed."""
        if e in self.places:
            return e
        anc = self.ancestors(e)
        if anc and anc[-1] in self.places:
            return anc[-1]
        return None

    def depth(self, e):
        return len(self.ancestors(e))


def record_candidates(w):
    cands = {"dek": [], "fuz": [], "gor": []}
    for x in w.items + w.boxes:
        for y in w.boxes + w.place_list:
            if x != y and w.loc[x] != y and x not in w.ancestors(y):
                cands["dek"].append((x, y))
        if w.loc[x] in w.boxes and w.top(x) is not None:
            cands["fuz"].append((x,))
    for b in w.boxes:
        for p in w.place_list:
            if w.loc[b] != p:
                cands["gor"].append((b, p))
    return cands


def query_candidates(w):
    cands = {"zuv": [], "qeb": [], "pimo": []}
    for x in w.items + w.boxes:
        if w.top(x) is not None and w.depth(x) >= 2:
            cands["zuv"].append((x,))
    for x in w.items + w.boxes:
        for y in w.items + w.boxes:
            if x != y and w.top(x) is not None and w.top(y) is not None:
                cands["qeb"].append((x, y))
    for p in w.place_list:
        cands["pimo"].append((p,))
    return cands


def apply_record(w, op, args):
    if op == "dek":
        w.loc[args[0]] = args[1]
    elif op == "fuz":
        w.loc[args[0]] = w.top(args[0])
    elif op == "gor":
        w.loc[args[0]] = args[1]


def answer(w, op, args):
    if op == "zuv":
        return w.top(args[0])
    if op == "qeb":
        return "yes" if w.top(args[0]) == w.top(args[1]) else "no"
    if op == "pimo":
        return str(sum(1 for i in w.items if w.top(i) == args[0]))
    raise ValueError(op)


def make_episode(rng, size):
    items = sorted(rng.sample(ITEMS, rng.randint(2, 4)))
    boxes = sorted(rng.sample(BOXES, rng.randint(1, 3)))
    places = sorted(rng.sample(PLACES, rng.randint(2, 3)))
    w = World(items, boxes, places)

    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    rows = []
    recs = 0
    qs = 0
    while recs < n_rec or qs < n_q:
        want_q = qs < n_q and (recs >= n_rec or (recs >= 2 and rng.random() < 0.3))
        if want_q:
            qc = query_candidates(w)
            op = rng.choice([o for o in QRY_OPS if qc[o]])
            args = rng.choice(qc[op])
            ans = answer(w, op, args)
            rows.append(" ".join([op, *args]) + "?\t" + ans)
            qs += 1
        else:
            rc = record_candidates(w)
            op = rng.choice([o for o in REC_OPS if rc[o]])
            args = rng.choice(rc[op])
            apply_record(w, op, args)
            rows.append(" ".join([op, *args]) + ".")
            recs += 1
    return [f"{i} {r}" for i, r in enumerate(rows, 1)]


def main(argv):
    if len(argv) < 3:
        print("usage: python gen.py N SEED [SIZE]", file=sys.stderr)
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        while True:
            ep = make_episode(rng, size)
            key = "\n".join(ep)
            if key not in seen:
                break
        seen.add(key)
        out.extend(ep)
    if out:
        print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
