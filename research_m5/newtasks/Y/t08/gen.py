#!/usr/bin/env python3
"""Synthetic record task: tracking which item sits in which slot after swaps.

Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)

Each episode places a few items (m1..m4) into slots (p1..p4), then swaps the
contents of pairs of occupied slots. Queries ask which item is in a slot, where
an item is, or whether an item is in a slot. Answers come from simulating the
records above the query.

Operation codes: ps (place item in slot), sw (swap two slots),
wa (which item is at slot), wh (which slot holds item), yn (is item in slot).
"""

import random
import sys

ITEMS = ["m1", "m2", "m3", "m4"]   # fixed pool, identical at every SIZE
SLOTS = ["p1", "p2", "p3", "p4"]   # fixed pool, identical at every SIZE


def episode(rng, size):
    k = rng.randint(2, 4)                       # number of occupied slots
    items = rng.sample(ITEMS, k)
    slots = rng.sample(SLOTS, k)

    state = {}                                  # slot -> item
    lines = []                                  # (text, answer or None)
    for item, slot in zip(items, slots):
        state[slot] = item
        lines.append((f"ps {item} {slot}.", None))

    # Records after the placements: 2*size .. (12-k)*size swaps.
    # At SIZE 1 the episode has 3 to 12 records in total.
    n_swaps = rng.randint(2 * size, (12 - k) * size)
    n_queries = rng.randint(1, 5)
    kinds = ["sw"] * n_swaps + ["q"] * n_queries
    rng.shuffle(kinds)

    for kind in kinds:
        if kind == "sw":
            a, b = rng.sample(sorted(state), 2)
            state[a], state[b] = state[b], state[a]
            lines.append((f"sw {a} {b}.", None))
            continue

        qtype = rng.choice(["wa", "wh", "yn"])
        if qtype == "wa":
            slot = rng.choice(sorted(state))
            text, ans = f"wa {slot}?", state[slot]
        elif qtype == "wh":
            item = rng.choice(sorted(state.values()))
            where = [s for s in state if state[s] == item]
            text, ans = f"wh {item}?", where[0]
        else:
            item = rng.choice(sorted(state.values()))
            here = [s for s in state if state[s] == item][0]
            if rng.random() < 0.5:
                slot = here
            else:
                slot = rng.choice([s for s in sorted(state) if s != here])
            text = f"yn {item} {slot}?"
            ans = "yes" if state[slot] == item else "no"
        lines.append((text, ans))

    out = []
    for i, (text, ans) in enumerate(lines, start=1):
        if ans is None:
            out.append(f"{i} {text}")
        else:
            out.append(f"{i} {text}\t{ans}")
    return out


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2

    rng = random.Random(seed)
    for _ in range(n):
        for row in episode(rng, size):
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
