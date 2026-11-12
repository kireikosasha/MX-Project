#!/usr/bin/env python3
"""Synthetic record task: room bookings, blocks and overlap queries.

Usage: python gen.py N SEED [SIZE]   (SIZE in 1, 3, 10; default 1)

Records (rows of codes and hour numbers):
  bok p r a b.   person p books room r for hours [a, b)
  blk r a b.     room r is closed for hours [a, b)
Queries (answer after the rows above them):
  fre r a b?     yes/no: is room r free for all of [a, b)?
  hld r h?       which person holds room r at hour h (always exactly one)
  ovl p q?       yes/no: do p and q have bookings overlapping in time (any rooms)?
Within one room, no two bookings or blocks overlap, so holders are unique.
"""
import random
import sys

REC_OPS = ["bok", "blk"]
QRY_OPS = ["fre", "hld", "ovl"]
PEOPLE = ["p%d" % i for i in range(1, 7)]
ROOMS = ["r%d" % i for i in range(1, 5)]


def overlaps(a1, b1, a2, b2):
    return a1 < b2 and a2 < b1


def try_record(rng, H, rooms, bookings):
    """Return (row_text, state_update) for a valid new record, or None."""
    for _ in range(200):
        r = rng.choice(ROOMS)
        a = rng.randrange(0, H)
        b = min(H, a + rng.randint(1, 4))
        if b <= a:
            continue
        if any(overlaps(a, b, x, y) for x, y, _, _ in rooms[r]):
            continue
        if rng.random() < 0.7:
            p = rng.choice(PEOPLE)
            rooms[r].append((a, b, "bok", p))
            bookings.append((p, r, a, b))
            return "bok %s %s %d %d." % (p, r, a, b)
        rooms[r].append((a, b, "blk", None))
        return "blk %s %d %d." % (r, a, b)
    return None


def make_query(rng, H, rooms, bookings):
    """Return (query_text, answer), answer computed from the current state."""
    qop = rng.choice(QRY_OPS)
    if qop == "hld" and bookings:
        p, r, a0, b0 = rng.choice(bookings)
        h = rng.randrange(a0, b0)
        # Bookings in a room are disjoint and blocks never overlap them,
        # so exactly one booking covers hour h.
        return "hld %s %d?" % (r, h), p
    if qop == "ovl":
        p, q = rng.sample(PEOPLE, 2)
        hit = any(
            p1 == p and q1 == q and overlaps(a1, b1, a2, b2)
            for p1, _, a1, b1 in bookings
            for q1, _, a2, b2 in bookings
        )
        return "ovl %s %s?" % (p, q), ("yes" if hit else "no")
    # fre (also the fallback when no booking exists yet)
    r = rng.choice(ROOMS)
    a = rng.randrange(0, H)
    b = min(H, a + rng.randint(1, 4))
    busy = any(overlaps(a, b, x, y) for x, y, _, _ in rooms[r])
    return "fre %s %d %d?" % (r, a, b), ("no" if busy else "yes")


def gen_episode(rng, size, seen):
    H = 12 * size
    while True:
        n_rec = rng.randint(3, 12) * size
        k = rng.randint(1, 5)
        # First three rows are records, so every query has records above it.
        rest = ["R"] * (n_rec - 3) + ["Q"] * k
        rng.shuffle(rest)
        kinds = ["R", "R", "R"] + rest

        rooms = {r: [] for r in ROOMS}
        bookings = []
        body = []
        ok = True
        for kind in kinds:
            if kind == "R":
                text = try_record(rng, H, rooms, bookings)
                if text is None:
                    ok = False
                    break
            else:
                text, ans = make_query(rng, H, rooms, bookings)
                text = text + "\t" + ans
            body.append(text)
        if not ok:
            continue
        key = "\n".join(body)
        if key in seen:
            continue
        seen.add(key)
        return body


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        body = gen_episode(rng, size, seen)
        for i, line in enumerate(body, 1):
            out.append("%d %s" % (i, line))
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv)
