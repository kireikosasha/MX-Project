import random
import sys

# Changing states: items sit in boxes (location state), boxes get a status
# (status state that is overwritten over time) and boxes accumulate stock
# (number state). Queries ask the present state of an item, the items that
# are in boxes with a given status, or the total stock of boxes with a status.

RECORD_OPS = ["mv", "sw", "dp"]   # mv k b: item k moves to box b
                                  # sw b s: box b takes status s
                                  # dp b n: n units are added to the stock of box b
QUERY_OPS = ["ws", "ls", "tt"]    # ws k?: status of the box holding item k
                                  # ls s?: items whose box currently has status s
                                  # tt s?: total stock of boxes currently with status s

ITEMS = ["k1", "k2", "k3", "k4", "k5", "k6"]
BOXES = ["b1", "b2", "b3", "b4"]
STATUSES = ["s1", "s2", "s3"]


class World:
    def __init__(self):
        self.loc = {}     # item -> box
        self.status = {}  # box -> status
        self.stock = {}   # box -> number


def make_record(rng, w, items, boxes, stats, size):
    kind = rng.choices(RECORD_OPS, weights=[4, 3, 3])[0]
    if kind == "mv":
        k = rng.choice(items)
        choices = [b for b in boxes if b != w.loc.get(k)]
        b = rng.choice(choices)
        w.loc[k] = b
        return f"mv {k} {b}."
    if kind == "sw":
        b = rng.choice(boxes)
        choices = [s for s in stats if s != w.status.get(b)]
        s = rng.choice(choices)
        w.status[b] = s
        return f"sw {b} {s}."
    b = rng.choice(boxes)
    n = rng.randint(1, 5 * size)
    w.stock[b] = w.stock.get(b, 0) + n
    return f"dp {b} {n}."


def make_query(rng, w):
    kind = rng.choice(QUERY_OPS)
    if kind == "ws":
        cands = sorted(k for k, b in w.loc.items() if b in w.status)
        if not cands:
            return None
        k = rng.choice(cands)
        return f"ws {k}?\t{w.status[w.loc[k]]}"
    if kind == "ls":
        opts = sorted({w.status[b] for b in w.loc.values() if b in w.status})
        if not opts:
            return None
        s = rng.choice(opts)
        ans = [k for k in sorted(w.loc) if w.status.get(w.loc[k]) == s]
        return f"ls {s}?\t{','.join(ans)}"
    opts = sorted(set(w.status.values()))
    if not opts:
        return None
    s = rng.choice(opts)
    total = sum(w.stock.get(b, 0) for b, st in w.status.items() if st == s)
    if total == 0:
        return None
    return f"tt {s}?\t{total}"


def make_episode(rng, size):
    while True:
        items = sorted(rng.sample(ITEMS, rng.randint(2, 5)))
        boxes = sorted(rng.sample(BOXES, rng.randint(2, 4)))
        stats = sorted(rng.sample(STATUSES, rng.randint(2, 3)))
        n_rec = rng.randint(3 * size, 12 * size)
        n_q = rng.randint(1, 5)
        w = World()
        lines = []
        asked = set()
        recs = qs = 0
        ok = True
        while recs < n_rec or qs < n_q:
            want_query = qs < n_q and (
                recs >= n_rec or rng.random() < n_q / (n_q + n_rec))
            if want_query:
                q = make_query(rng, w)
                if q is not None and q.split("\t")[0] not in asked:
                    asked.add(q.split("\t")[0])
                    lines.append(q)
                    qs += 1
                    continue
                if recs >= n_rec:
                    ok = False
                    break
            lines.append(make_record(rng, w, items, boxes, stats, size))
            recs += 1
        if ok:
            return lines


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < n:
        lines = make_episode(rng, size)
        key = tuple(lines)
        if key in seen:
            continue
        seen.add(key)
        out.append(lines)
    text = []
    for lines in out:
        for i, line in enumerate(lines, 1):
            text.append(f"{i} {line}")
    sys.stdout.write("\n".join(text) + "\n")


if __name__ == "__main__":
    main(sys.argv)
