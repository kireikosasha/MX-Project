#!/usr/bin/env python3
"""Synthetic EXCHANGE episodes written as rows of codes.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)

Record rows (items change hands between holders):
  ob item holder units.   introduce an item (a lot of units) held by holder
  gv item holder.         the item's current holder gives it to holder
  sw item item.           the two items swap their holders
  bk item.                the item goes back to the holder it had before its last change
Query rows (answer computed from the records above):
  wh item?   holder of the item now
  hl holder? items held by the holder now, sorted, comma-separated
  cn holder? units held by the holder now
"""
import random
import sys

HOLDER_KIND = "a"      # holders a1..a8
ITEM_KIND = "o"        # items o1..o8
MAX_CODES = 8
SIZES = (1, 3, 10)


def fmt(tokens, end):
    """Join tokens with single spaces; the end mark attaches to the last token."""
    return " ".join(tokens[:-1] + [tokens[-1] + end])


def make_record(rng, size, n, holders, items, st):
    held, prev, units, events = st["held"], st["prev"], st["units"], st["events"]
    intro = sorted(i for i in items if i in held)
    fresh = [i for i in items if i not in held]
    swappable = [(x, y) for x in intro for y in intro
                 if x < y and held[x] != held[y]]
    undoable = [i for i in intro if prev[i] is not None]
    ops = []
    if fresh:
        ops += ["ob", "ob"]
    if intro:
        ops += ["gv"] * 4
    if swappable:
        ops += ["sw"] * 2
    if undoable:
        ops.append("bk")
    op = rng.choice(ops)

    if op == "ob":
        item = rng.choice(fresh)
        holder = rng.choice(holders)
        amount = rng.randint(1, 9 * size)
        held[item] = holder
        prev[item] = None
        units[item] = amount
        events[item] = [n]
        return fmt(["ob", item, holder, str(amount)], ".")

    if op == "gv":
        item = rng.choice(intro)
        new = rng.choice([h for h in holders if h != held[item]])
        prev[item] = held[item]
        held[item] = new
        events[item].append(n)
        return fmt(["gv", item, new], ".")

    if op == "sw":
        x, y = rng.choice(swappable)
        hx, hy = held[x], held[y]
        held[x], held[y] = hy, hx
        prev[x], prev[y] = hx, hy
        events[x].append(n)
        events[y].append(n)
        return fmt(["sw", x, y], ".")

    item = rng.choice(undoable)
    old = held[item]
    held[item] = prev[item]
    prev[item] = old
    events[item].append(n)
    return fmt(["bk", item], ".")


def make_query(rng, st):
    """A not-yet-asked question whose answer rests on at least two records."""
    held, units, events, asked = st["held"], st["units"], st["events"], st["asked"]
    options = {"wh": [], "hl": [], "cn": []}
    for item in sorted(held):
        if len(events[item]) >= 2 and ("wh", item) not in asked:
            options["wh"].append((["wh", item], held[item]))
    for h in sorted(set(held.values())):
        its = sorted(i for i in held if held[i] == h)
        rel = set()
        for i in its:
            rel.update(events[i])
        if len(rel) < 2:
            continue
        if ("hl", h) not in asked:
            options["hl"].append((["hl", h], ",".join(its)))
        if ("cn", h) not in asked:
            options["cn"].append((["cn", h], str(sum(units[i] for i in its))))
    kinds = [k for k in ("wh", "hl", "cn") if options[k]]
    if not kinds:
        return None
    kind = rng.choice(kinds)
    tokens, answer = rng.choice(options[kind])
    asked.add((tokens[0], tokens[1]))
    return fmt(tokens, "?") + "\t" + answer


def make_episode(rng, size):
    """Return the rows of one episode (without numbers), or None if it must be retried."""
    n_rec = rng.randint(3, 12) * size
    n_qry = rng.randint(1, 5)
    holders = ["%s%d" % (HOLDER_KIND, d) for d in
               sorted(rng.sample(range(1, MAX_CODES + 1), rng.randint(2, MAX_CODES)))]
    items = ["%s%d" % (ITEM_KIND, d) for d in
             sorted(rng.sample(range(1, MAX_CODES + 1), rng.randint(2, MAX_CODES)))]
    st = {"held": {}, "prev": {}, "units": {}, "events": {}, "asked": set()}
    rows = []
    n = 0
    q_left = n_qry
    while n < n_rec or q_left:
        if q_left and n >= 2 and (n >= n_rec or rng.random() < 0.3):
            q = make_query(rng, st)
            if q is not None:
                rows.append(q)
                q_left -= 1
                continue
            if n >= n_rec:
                return None
        if n < n_rec:
            n += 1
            rows.append(make_record(rng, size, n, holders, items, st))
            continue
        return None
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n_eps, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2

    rng = random.Random(seed)
    seen = set()
    lines = []
    for _ in range(n_eps):
        while True:
            rows = None
            while rows is None:
                rows = make_episode(rng, size)
            key = "\n".join(rows)
            if key not in seen:
                break
        seen.add(key)
        for i, row in enumerate(rows, 1):
            lines.append("%d %s" % (i, row))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
