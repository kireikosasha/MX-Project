#!/usr/bin/env python3
"""Synthetic lock-ownership task given as records.

Agents (a1..a8) take (tk) or drop (dp) locks (l1..l8). A lock that is held
makes a taker wait in a FIFO queue; dropping a lock hands it to the first
waiter. Queries ask: is a lock free (fr), who is waiting for it in queue
order (wt), how many locks an agent holds (ct).

Usage: python gen.py N SEED SIZE   (SIZE in 1, 3, 10; default 1)
"""
import random
import sys

AGENTS = ["a%d" % i for i in range(1, 9)]
LOCKS = ["l%d" % i for i in range(1, 9)]


def new_state():
    return {
        "holder": {l: None for l in LOCKS},
        "queue": {l: [] for l in LOCKS},
        "touches": {e: 0 for e in AGENTS + LOCKS},
    }


def make_record(rng, st):
    holder, queue, touches = st["holder"], st["queue"], st["touches"]
    held = [l for l in LOCKS if holder[l] is not None]
    cands = [(a, l) for a in AGENTS for l in LOCKS
             if holder[l] != a and a not in queue[l]]
    if held and (not cands or rng.random() < 0.35):
        l = rng.choice(held)
        a = holder[l]
        holder[l] = queue[l].pop(0) if queue[l] else None
        text = "dp %s %s." % (a, l)
    else:
        a, l = rng.choice(cands)
        if holder[l] is None:
            holder[l] = a
        else:
            queue[l].append(a)
        text = "tk %s %s." % (a, l)
    touches[a] += 1
    touches[l] += 1
    return text


def make_query(rng, st, asked):
    holder, queue, touches = st["holder"], st["queue"], st["touches"]
    opts = []
    for l in LOCKS:
        if touches[l] >= 2:
            opts.append(("fr", l))
            if queue[l]:
                opts.append(("wt", l))
    for a in AGENTS:
        if touches[a] >= 2:
            opts.append(("ct", a))
    opts = [o for o in opts if "%s %s" % o not in asked]
    if not opts:
        return None
    op, e = rng.choice(opts)
    asked.add("%s %s" % (op, e))
    if op == "fr":
        ans = "yes" if holder[e] is None else "no"
    elif op == "wt":
        ans = ",".join(queue[e])
    else:
        ans = str(sum(1 for l in LOCKS if holder[l] == e))
    return "%s %s?\t%s" % (op, e, ans)


def make_episode(rng, size):
    st = new_state()
    rec_target = rng.randint(3, 12) * size
    q_target = rng.randint(1, 5)
    asked = set()
    lines = []
    rec_done = q_done = 0
    while rec_done < rec_target or q_done < q_target:
        if q_done < q_target and rec_done > 0 and (
                rec_done >= rec_target or rng.random() < 0.4):
            q = make_query(rng, st, asked)
            if q is not None:
                lines.append(q)
                q_done += 1
                continue
            if rec_done >= rec_target:
                return None  # cannot place the queries in the record budget
        lines.append(make_record(rng, st))
        rec_done += 1
    return lines


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: gen.py N SEED [SIZE]")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        while True:
            lines = make_episode(rng, size)
            if lines is None:
                continue
            key = "\n".join(lines)
            if key in seen:
                continue
            seen.add(key)
            break
        for i, line in enumerate(lines, 1):
            out.append("%d %s" % (i, line))
    print("\n".join(out))


if __name__ == "__main__":
    main()
