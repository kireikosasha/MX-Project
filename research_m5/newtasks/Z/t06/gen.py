#!/usr/bin/env python3
"""Synthetic relay-baton records task.

Usage: python gen.py N SEED SIZE   (SIZE in 1, 3, 10; default 1)

Each episode is a log of baton handoffs in a relay of runners r1..r8.
Records:
  stg X.      the baton starts with runner X
  hov X Y.    the baton is handed from X (the current holder) to Y
  dsq X.      runner X is disqualified (never holds or receives again)
Queries (answered only from the rows above them):
  hld X?      is X holding the baton now?            -> yes / no
  cnt X?      how many handoffs X has received so far -> number
  lst X?      runners X has handed the baton to, sorted -> codes joined by commas
"""
import random
import sys

RUNNERS = ["r%d" % i for i in range(1, 9)]
MAX_DSQ = 5          # keeps at least 3 runners in the race at every size
MIN_ACTIVE = 3


def gen_records(rng, size):
    target = rng.randint(3, 12) * size
    start = rng.choice(RUNNERS)
    recs = [("stg", start)]
    holder = start
    active = set(RUNNERS)
    dsq_n = 0
    while len(recs) < target:
        others = sorted(active - {holder})
        can_dsq = bool(others) and len(active) > MIN_ACTIVE and dsq_n < MAX_DSQ
        if others and (not can_dsq or rng.random() < 0.75):
            nxt = rng.choice(others)
            recs.append(("hov", holder, nxt))
            holder = nxt
        elif can_dsq:
            x = rng.choice(others)
            active.discard(x)
            dsq_n += 1
            recs.append(("dsq", x))
        else:
            break
    return recs


def simulate(recs):
    holder = None
    received = {}
    edges = []
    hovs = 0
    for r in recs:
        if r[0] == "stg":
            holder = r[1]
        elif r[0] == "hov":
            holder = r[2]
            received[r[2]] = received.get(r[2], 0) + 1
            edges.append((r[1], r[2]))
            hovs += 1
    return holder, received, edges, hovs


def render_record(r):
    if r[0] == "stg":
        return "stg %s." % r[1]
    if r[0] == "hov":
        return "hov %s %s." % (r[1], r[2])
    return "dsq %s." % r[1]


def make_query(rng, recs, pos):
    holder, received, edges, hovs = simulate(recs[:pos])
    if hovs < 1:
        return None
    op = rng.choice(["hld", "cnt", "lst"])
    if op == "hld":
        if rng.random() < 0.5:
            x = holder
        else:
            x = rng.choice([r for r in RUNNERS if r != holder])
        ans = "yes" if x == holder else "no"
    elif op == "cnt":
        x = rng.choice(RUNNERS)
        if received.get(x, 0) == 0:
            return None
        ans = str(received[x])
    else:
        x = rng.choice(RUNNERS)
        outs = sorted({b for a, b in edges if a == x})
        if not outs:
            return None
        ans = ",".join(outs)
    return op, x, ans


def episode(rng, size):
    while True:
        recs = gen_records(rng, size)
        n_q = rng.randint(1, 5)
        queries = []
        tries = 0
        while len(queries) < n_q and tries < 500:
            tries += 1
            pos = rng.randint(2, len(recs))
            q = make_query(rng, recs, pos)
            if q:
                queries.append((pos,) + q)
        if len(queries) < n_q:
            continue
        queries.sort(key=lambda t: t[0])
        lines = []
        for i in range(len(recs) + 1):
            for pos, op, x, ans in queries:
                if pos == i:
                    lines.append("%s %s?\t%s" % (op, x, ans))
            if i < len(recs):
                lines.append(render_record(recs[i]))
        return "\n".join("%d %s" % (k, ln) for k, ln in enumerate(lines, 1))


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        while True:
            text = episode(rng, size)
            if text not in seen:
                seen.add(text)
                break
        out.append(text)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv)
