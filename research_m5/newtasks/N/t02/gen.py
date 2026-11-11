#!/usr/bin/env python3
"""Synthetic EXCHANGE task written as records (rows of codes and numbers).

Usage: python gen.py N SEED SIZE      (SIZE in {1, 3, 10}, default 1)

Holders (a1..a5) pass items (o1..o8) to each other. Records change who holds
what; queries ask who holds an item, what a holder holds, or how many items a
holder holds. Answers come from replaying the records above each query.
"""
import itertools
import random
import sys

HOLDERS = ["a1", "a2", "a3", "a4", "a5"]
ITEMS = ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"]
SIZES = (1, 3, 10)


def holdings(owner, h):
    return sorted(i for i, o in owner.items() if o == h)


def step(rng, owner):
    """Choose a legal exchange record, apply it to owner.

    Returns (record text, set of items whose holder changed, set of holders
    whose holdings changed).
    Record operations: nw (item enters a holding), gv (give an item),
    tk (take an item from its holder), sw (two holders swap all items).
    """
    uncreated = [i for i in ITEMS if i not in owner]
    options = []
    if uncreated:
        options.append(("nw", 2))
    if owner:
        options += [("gv", 4), ("tk", 3), ("sw", 2)]
    kind = rng.choices([k for k, _ in options], [w for _, w in options])[0]

    if kind == "nw":
        item = rng.choice(uncreated)
        h = rng.choice(HOLDERS)
        owner[item] = h
        return f"nw {item} {h}.", {item}, {h}

    if kind == "gv":
        item = rng.choice(sorted(owner))
        giver = owner[item]
        receiver = rng.choice([h for h in HOLDERS if h != giver])
        owner[item] = receiver
        return f"gv {giver} {item} {receiver}.", {item}, {giver, receiver}

    if kind == "tk":
        item = rng.choice(sorted(owner))
        prev = owner[item]
        taker = rng.choice([h for h in HOLDERS if h != prev])
        owner[item] = taker
        return f"tk {taker} {item}.", {item}, {prev, taker}

    # sw: exchange all items between two holders, at least one of them non-empty
    pairs = [(a, b) for a, b in itertools.combinations(HOLDERS, 2)
             if holdings(owner, a) or holdings(owner, b)]
    a, b = rng.choice(pairs)
    moved = set(holdings(owner, a)) | set(holdings(owner, b))
    for i in moved:
        owner[i] = b if owner[i] == a else a
    return f"sw {a} {b}.", moved, {a, b}


def queries_for(owner, item_touch, holder_touch):
    """Queries whose answer depends on at least two records each.

    Returns (op, argument, answer) triples.
    """
    out = []
    for item in sorted(owner):
        if item_touch[item] >= 2:
            out.append(("wh", item, owner[item]))
    for h in HOLDERS:
        if holder_touch[h] >= 2:
            mine = holdings(owner, h)
            out.append(("ct", h, str(len(mine))))
            if mine:
                out.append(("hs", h, ",".join(mine)))
    return out


def make_episode(rng, size, seen):
    """Return the numbered rows of one new episode."""
    while True:
        n = rng.randint(3, 12) * size
        owner = {}
        item_touch = {i: 0 for i in ITEMS}
        holder_touch = {h: 0 for h in HOLDERS}
        records = []
        # snaps[j] = state after the first j records
        snaps = [None]
        for _ in range(n):
            text, items, holders = step(rng, owner)
            records.append(text)
            for i in items:
                item_touch[i] += 1
            for h in holders:
                holder_touch[h] += 1
            snaps.append((dict(owner), dict(item_touch), dict(holder_touch)))

        wanted = rng.randint(1, 5)
        chosen = {}   # j -> query row placed after record j
        used = set()  # (op, argument) pairs already asked
        for _ in range(4 * wanted + 8):
            if len(chosen) >= wanted:
                break
            j = rng.randint(1, n)
            if j in chosen:
                continue
            o, it, ht = snaps[j]
            cands = [c for c in queries_for(o, it, ht) if c[:2] not in used]
            if not cands:
                continue
            op, arg, ans = rng.choice(cands)
            used.add((op, arg))
            chosen[j] = f"{op} {arg}?\t{ans}"
        if not chosen:
            continue

        rows = []
        for k in range(1, n + 1):
            rows.append(records[k - 1])
            if k in chosen:
                rows.append(chosen[k])
        lines = [f"{i} {r}" for i, r in enumerate(rows, 1)]
        key = "\n".join(lines)
        if key in seen:
            continue
        seen.add(key)
        return lines


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n_episodes = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    for _ in range(n_episodes):
        for line in make_episode(rng, size, seen):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
