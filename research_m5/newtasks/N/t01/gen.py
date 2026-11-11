#!/usr/bin/env python3
"""Synthetic containment task given as records: boxes, things and rooms.

Records move things and boxes around (put into a box or room, take out,
swap places). Queries ask where something is now (its room, yes/no on
whether it is in a room, how many things are inside a box). Containers
carry their contents when they move. Answers come from a simulation.

Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

REC_OPS = ("mv", "ot", "sw")
REC_WEIGHTS = (6, 2, 3)
QRY_OPS = ("qt", "qi", "qc")


def acyclic(loc):
    """True when no entity is (transitively) inside itself."""
    for e in loc:
        cur = loc.get(e)
        steps = 0
        while cur is not None:
            if cur == e or steps > 20:
                return False
            cur = loc.get(cur)
            steps += 1
    return True


def try_apply(op, args, loc):
    """Return the new location map after a record, or None if it is invalid."""
    new = dict(loc)
    if op == "mv":
        x, y = args
        if x == y or loc.get(x) == y:
            return None
        new[x] = y
    elif op == "ot":
        (x,) = args
        if loc.get(x) is None:
            return None
        del new[x]
    elif op == "sw":
        x, y = args
        if x == y or loc.get(x) is None or loc.get(y) is None:
            return None
        if loc[x] == loc[y]:
            return None
        new[x], new[y] = loc[y], loc[x]
    else:
        return None
    return new if acyclic(new) else None


def top(loc, e):
    """The room an entity is ultimately in, or None if not placed in a room."""
    if loc.get(e) is None:
        return None
    cur = e
    while loc.get(cur) is not None:
        cur = loc[cur]
    return cur if cur[0] == "r" else None


def inside(loc, e, box):
    cur = loc.get(e)
    while cur is not None:
        if cur == box:
            return True
        cur = loc.get(cur)
    return False


def depth_in_box(loc, e):
    """True when e sits inside a box (so the answer needs at least two hops)."""
    p = loc.get(e)
    return p is not None and p[0] == "b"


def gen_record(rng, loc, boxes, things, rooms):
    movable = boxes + things
    for _ in range(200):
        op = rng.choices(REC_OPS, weights=REC_WEIGHTS)[0]
        if op == "mv":
            args = [rng.choice(movable), rng.choice(boxes + rooms)]
        elif op == "ot":
            args = [rng.choice(movable)]
        else:
            args = rng.sample(movable, 2)
        new = try_apply(op, args, loc)
        if new is not None:
            return op, args, new
    return None


def count_things_in(loc, things, box):
    return sum(1 for t in things if inside(loc, t, box))


def gen_query(rng, loc, boxes, things, rooms):
    """Return (query_text_without_number, answer) or None."""
    ops = list(QRY_OPS)
    rng.shuffle(ops)
    for op in ops:
        if op == "qt":
            xs = [x for x in boxes + things if depth_in_box(loc, x) and top(loc, x)]
            if xs:
                x = rng.choice(xs)
                return f"qt {x}?", top(loc, x)
        elif op == "qi":
            xs = [x for x in boxes + things if depth_in_box(loc, x) and top(loc, x)]
            if xs:
                x = rng.choice(xs)
                t = top(loc, x)
                if rng.random() < 0.5:
                    r = t
                else:
                    r = rng.choice([q for q in rooms if q != t])
                return f"qi {x} {r}?", ("yes" if r == t else "no")
        else:
            cands = [b for b in boxes if count_things_in(loc, things, b) >= 1]
            if cands:
                b = rng.choice(cands)
                return f"qc {b}?", str(count_things_in(loc, things, b))
    return None


def build_rows(rng, boxes, things, rooms, nrec, nq):
    """Simulate one episode. Return a list of numbered rows, or None to retry."""
    loc = {}
    rows = []
    asked = set()
    recs_done = 0
    recs_left = nrec
    q_left = nq
    while recs_left > 0 or q_left > 0:
        n = len(rows) + 1
        want_query = q_left > 0 and recs_done >= 1 and (
            recs_left == 0 or rng.random() < 0.35
        )
        if want_query:
            q = gen_query(rng, loc, boxes, things, rooms)
            if q is not None and q[0] not in asked:
                text, ans = q
                asked.add(text)
                rows.append(f"{n} {text}\t{ans}")
                q_left -= 1
                continue
            if recs_left == 0:
                return None
        if recs_left == 0:
            return None
        r = gen_record(rng, loc, boxes, things, rooms)
        if r is None:
            return None
        op, args, loc = r
        rows.append(f"{n} {op} {' '.join(args)}.")
        recs_left -= 1
        recs_done += 1
    return rows


def gen_episode(rng, size, seen):
    for _ in range(10000):
        nb = rng.randint(2, min(8, 2 + 2 * size))
        nt = rng.randint(2, min(8, 3 + 2 * size))
        nr = rng.randint(2, 3)
        boxes = [f"b{i}" for i in sorted(rng.sample(range(1, 9), nb))]
        things = [f"t{i}" for i in sorted(rng.sample(range(1, 9), nt))]
        rooms = [f"r{i}" for i in sorted(rng.sample(range(1, 9), nr))]
        nrec = rng.randint(3 * size, 12 * size)
        nq = rng.randint(1, 5)
        rows = build_rows(rng, boxes, things, rooms, nrec, nq)
        if rows is None:
            continue
        text = "\n".join(rows)
        if text in seen:
            continue
        seen.add(text)
        return rows
    raise RuntimeError("could not generate a new episode")


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
    seen = set()
    out = []
    for _ in range(n):
        out.extend(gen_episode(rng, size, seen))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
