#!/usr/bin/env python3
"""Synthetic time-ordering reading task (bAbI-style), standard library only.

Usage: python gen.py N SEED

Each story: a start statement ("Everyone started in the hall."), then K people
(K = 3 or 4), each making exactly one move to one of four places. The moves
happen one after another: move t happens at time t, and time 0 is the start.
Statements give the times, the order or the destinations of the moves, shuffled.
Questions ask where someone was at a time, where someone was just before another
person moved, who moved at a time, or who was in a place at a time.

Answers come from a simulation of the world. A question is kept only when the
statements above it fix its answer (checked by enumerating every move order the
statements allow) and no single statement fixes it on its own.
"""
import random
import sys
from itertools import permutations, product

NAMES = ["alice", "bob", "carol", "dan", "eve"]
PLACES = ["kitchen", "garden", "office", "park"]
START = "hall"


def cap(text):
    return text[0].upper() + text[1:]


def render_fact(f):
    k = f[0]
    if k == "start":
        return "Everyone started in the %s." % START
    if k == "abs":
        return "%s moved at time %d." % (f[1], f[2])
    if k == "before":
        return "%s moved before %s." % (f[1], f[2])
    if k == "after":
        return "%s moved after %s." % (f[1], f[2])
    if k == "first":
        return "%s moved first." % f[1]
    if k == "last":
        return "%s moved last." % f[1]
    if k == "dest":
        return "%s went to the %s." % (f[1], f[2])
    if k == "abs_dest":
        return "At time %d %s went to the %s." % (f[2], f[1], f[3])
    raise ValueError(k)


def render_q(q):
    k = q[0]
    if k == "where":
        return "where was %s at time %d" % (q[1], q[2])
    if k == "where_before":
        return "where was %s just before %s moved" % (q[1], q[2])
    if k == "who_moved":
        return "who moved at time %d" % q[1]
    return "who was in the %s at time %d" % (q[1], q[2])


def make_world(rng):
    k = rng.choice([3, 4])
    names = rng.sample(NAMES, k)
    order = names[:]
    rng.shuffle(order)
    pos = {n: i + 1 for i, n in enumerate(order)}
    dest = {n: rng.choice(PLACES) for n in names}
    loc = {n: START for n in names}
    hist = [dict(loc)]  # hist[t] = locations after t moves
    for n in order:
        loc[n] = dest[n]
        hist.append(dict(loc))
    return {"K": k, "names": names, "order": order, "pos": pos,
            "dest": dest, "hist": hist}


def make_pool(rng, w):
    names, pos, dest, k = w["names"], w["pos"], w["dest"], w["K"]
    pool = []
    for n in names:
        pool.append(("abs", n, pos[n]))
        pool.append(("dest", n, dest[n]))
        pool.append(("abs_dest", n, pos[n], dest[n]))
        if pos[n] == 1:
            pool.append(("first", n))
        if pos[n] == k:
            pool.append(("last", n))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            early, late = (a, b) if pos[a] < pos[b] else (b, a)
            if rng.random() < 0.5:
                pool.append(("before", early, late))
            else:
                pool.append(("after", late, early))
    return pool


def timing_ok(f, pos):
    k = f[0]
    if k in ("abs", "abs_dest"):
        return pos[f[1]] == f[2]
    if k == "before":
        return pos[f[1]] < pos[f[2]]
    if k == "after":
        return pos[f[1]] > pos[f[2]]
    if k == "first":
        return pos[f[1]] == 1
    if k == "last":
        return pos[f[1]] == len(pos)
    return True


def dest_sets(facts, names):
    allowed = {n: list(PLACES) for n in names}
    for f in facts:
        if f[0] == "dest":
            n, p = f[1], f[2]
        elif f[0] == "abs_dest":
            n, p = f[1], f[3]
        else:
            continue
        allowed[n] = [d for d in allowed[n] if d == p]
    return allowed


def loc_opts(x, t, pos, allowed):
    if pos[x] > t:
        return {START}
    return set(allowed[x])


def possible(q, pos, allowed, names):
    k = q[0]
    if k == "where":
        return loc_opts(q[1], q[2], pos, allowed)
    if k == "where_before":
        return loc_opts(q[1], pos[q[2]] - 1, pos, allowed)
    if k == "who_moved":
        return {n for n in names if pos[n] == q[1]}
    p, t = q[1], q[2]
    opts = []
    for n in names:
        if pos[n] > t:
            opts.append([False])
        else:
            opts.append(sorted({d == p for d in allowed[n]}))
    res = set()
    for combo in product(*opts):
        inside = sorted(n for n, flag in zip(names, combo) if flag)
        res.add(",".join(inside) if inside else "nobody")
    return res


def solve(q, facts, names):
    """All answers to q over every world the facts allow."""
    allowed = dest_sets(facts, names)
    out = set()
    for order in permutations(names):
        pos = {n: i + 1 for i, n in enumerate(order)}
        if all(timing_ok(f, pos) for f in facts):
            out |= possible(q, pos, allowed, names)
    return out


def truth(q, w):
    """Answer from the simulation of the true world."""
    k = q[0]
    if k == "where":
        return w["hist"][q[2]][q[1]]
    if k == "where_before":
        return w["hist"][w["pos"][q[2]] - 1][q[1]]
    if k == "who_moved":
        return w["order"][q[1] - 1]
    p, t = q[1], q[2]
    inside = sorted(n for n in w["names"] if w["hist"][t][n] == p)
    return ",".join(inside) if inside else "nobody"


def random_question(rng, names, k):
    kind = rng.randrange(4)
    if kind == 0:
        return ("where", rng.choice(names), rng.randint(1, k))
    if kind == 1:
        x, y = rng.sample(names, 2)
        return ("where_before", x, y)
    if kind == 2:
        return ("who_moved", rng.randint(1, k))
    return ("who_in", rng.choice(PLACES), rng.randint(1, k))


def accept(q, prefix, names, ans):
    if solve(q, prefix, names) != {ans}:
        return False
    if len(solve(q, [], names)) == 1:
        return False
    for f in prefix[1:]:
        if len(solve(q, [f], names)) == 1:
            return False
    return True


def build_lines(statements, placed, w):
    lines = []
    num = 0
    for k in range(len(statements) + 1):
        for cut, q in placed:
            if cut == k:
                num += 1
                lines.append("%d %s?\t%s" % (num, cap(render_q(q)), truth(q, w)))
        if k < len(statements):
            num += 1
            lines.append("%d %s" % (num, cap(render_fact(statements[k]))))
    return lines


def make_story(rng):
    while True:
        w = make_world(rng)
        names, k = w["names"], w["K"]
        pool = make_pool(rng, w)
        m = rng.randint(2, min(11, len(pool)))
        statements = [("start",)] + rng.sample(pool, m)
        nq = rng.randint(1, 5)
        placed = []
        seen = set()
        for _ in range(100):
            if len(placed) == nq:
                break
            q = random_question(rng, names, k)
            if q in seen:
                continue
            cut = rng.randint(1, len(statements))
            if accept(q, statements[:cut], names, truth(q, w)):
                seen.add(q)
                placed.append((cut, q))
        if len(placed) == nq:
            return build_lines(statements, placed, w)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(2)
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
