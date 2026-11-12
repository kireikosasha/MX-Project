#!/usr/bin/env python3
"""Synthetic EXCHANGE task given as records.

Items (b-codes) change hands between holders (p-codes) through placement,
gives, takes and swaps. Queries ask who holds an item now, what a holder
holds, or how many items a holder holds. Answers come from a simulation.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

ITEMS = ["b1", "b2", "b3", "b4"]
HOLDERS = ["p1", "p2", "p3", "p4"]

# Record operations: ptn = place an item with a holder,
# zof = giver hands item to receiver, nup = taker takes item from giver,
# kel = two holders swap the items they hold.
REC_OPS = ["ptn", "zof", "nup", "kel"]

# Query operations: qh = holder of item, qi = items of holder,
# qn = number of items of holder.
QRY_OPS = ["qh", "qi", "qn"]


def pick_record(rng, items, holders, owner):
    """Return (op, tokens) for one valid record, updating nothing."""
    unplaced = [x for x in items if x not in owner]
    placed = [x for x in items if x in owner]
    cands = []
    if unplaced:
        cands.append("ptn")
    if placed:
        cands += ["zof", "nup"]
        if any(owner[a] != owner[b] for a in placed for b in placed if a < b):
            cands.append("kel")
    op = rng.choice(cands)

    if op == "ptn":
        item = rng.choice(unplaced)
        holder = rng.choice(holders)
        return op, [op, item, holder]
    if op == "zof":
        item = rng.choice(placed)
        giver = owner[item]
        receiver = rng.choice([h for h in holders if h != giver])
        return op, [op, giver, item, receiver]
    if op == "nup":
        item = rng.choice(placed)
        taker = rng.choice([h for h in holders if h != owner[item]])
        return op, [op, taker, item, owner[item]]
    # kel: pick two placed items held by different holders
    pairs = [(a, b) for a in placed for b in placed
             if a < b and owner[a] != owner[b]]
    a, b = rng.choice(pairs)
    return op, [op, owner[a], a, owner[b], b]


def apply_record(op, tokens, owner):
    if op == "ptn":
        owner[tokens[1]] = tokens[2]
    elif op == "zof":
        owner[tokens[2]] = tokens[3]
    elif op == "nup":
        owner[tokens[2]] = tokens[1]
    elif op == "kel":
        a, b = tokens[2], tokens[4]
        owner[a], owner[b] = owner[b], owner[a]


def pick_query(rng, items, holders, owner, mentions):
    """Return a query row whose answer needs at least two records, or None."""
    cands = []
    for x in items:
        if x in owner and mentions.get(x, 0) >= 2:
            cands.append(f"qh {x}?\t{owner[x]}")
    for h in holders:
        if mentions.get(h, 0) >= 2:
            held = sorted(x for x in items if owner.get(x) == h)
            if held:
                cands.append(f"qi {h}?\t{','.join(held)}")
            cands.append(f"qn {h}?\t{len(held)}")
    if not cands:
        return None
    return rng.choice(cands)


def try_episode(rng, size):
    items = rng.sample(ITEMS, rng.randint(2, len(ITEMS)))
    holders = rng.sample(HOLDERS, rng.randint(2, len(HOLDERS)))
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    # Queries are inserted after a given number of records (2 or more).
    q_at = sorted(rng.sample(range(2, n_rec + 1), min(n_q, n_rec - 1)))

    owner = {}
    mentions = {}
    rows = []
    emitted = 0
    q_ptr = 0
    while True:
        while q_ptr < len(q_at) and q_at[q_ptr] == emitted:
            q = pick_query(rng, items, holders, owner, mentions)
            if q is not None:
                rows.append(q)
            q_ptr += 1
        if emitted == n_rec:
            break
        op, tokens = pick_record(rng, items, holders, owner)
        apply_record(op, tokens, owner)
        for code in tokens[1:]:
            mentions[code] = mentions.get(code, 0) + 1
        rows.append(" ".join(tokens) + ".")
        emitted += 1

    if not any("\t" in r for r in rows):
        return None
    return rows


def make_episode(rng, size):
    while True:
        rows = try_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    if len(argv) not in (3, 4):
        raise SystemExit("usage: python gen.py N SEED [SIZE]")
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")

    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        rows = make_episode(rng, size)
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(rows)

    for rows in episodes:
        for i, row in enumerate(rows, 1):
            print(f"{i} {row}")


if __name__ == "__main__":
    main(sys.argv)
