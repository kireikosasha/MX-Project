import random
import sys

# Order-in-time task: movers (k) arrive at places (p) in a sequence of records.
# "gox K P." = mover K arrives at place P; "tik N." = the clock advances by N.
# Queries ask about the FIRST arrival of a mover at a place, so answers are unambiguous.

MOVERS = ["k1", "k2", "k3", "k4"]
PLACES = ["p1", "p2", "p3", "p4", "p5", "p6"]
MOVE = "gox"
TICK = "tik"


def simulate(prefix):
    """Return {(mover, place): (record_index, clock_time, previous_place)} for first arrivals."""
    clock = 0
    cur = {}
    first = {}
    for idx, rec in enumerate(prefix):
        if rec[0] == TICK:
            clock += rec[1]
        else:
            _, k, p = rec
            if (k, p) not in first:
                first[(k, p)] = (idx, clock, cur.get(k))
            cur[k] = p
    return first


def evaluate(query, prefix):
    """Answer a query text (without number and tab) using only the records in prefix."""
    body = query[:-1].split()  # drop the trailing '?'
    op = body[0]
    first = simulate(prefix)
    if op == "vor":  # place the mover was in just before its first arrival at place
        e = first.get((body[1], body[2]))
        if e is None or e[2] is None:
            return None
        return e[2]
    if op == "clk":  # clock time of the mover's first arrival at place
        e = first.get((body[1], body[2]))
        if e is None:
            return None
        return str(e[1])
    if op == "fst":  # which of two movers arrived first at place
        p, k1, k2 = body[1], body[2], body[3]
        e1 = first.get((k1, p))
        e2 = first.get((k2, p))
        if e1 is None or e2 is None:
            return None
        return k1 if e1[0] < e2[0] else k2
    return None


def candidates(recs):
    """All answerable queries over the full record list, with the record index they need."""
    first = simulate(recs)
    out = []
    for (k, p), (idx, _clock, prev) in first.items():
        if prev is not None:
            out.append((f"vor {k} {p}?", idx))
        out.append((f"clk {k} {p}?", idx))
    for p in PLACES:
        for i, k1 in enumerate(MOVERS):
            for k2 in MOVERS[i + 1:]:
                if (k1, p) in first and (k2, p) in first:
                    cutoff = max(first[(k1, p)][0], first[(k2, p)][0])
                    out.append((f"fst {p} {k1} {k2}?", cutoff))
    return out


def render(rec):
    if rec[0] == TICK:
        return f"{TICK} {rec[1]}."
    return f"{MOVE} {rec[1]} {rec[2]}."


def make_episode(rng, size, seen):
    while True:
        n = rng.randint(3, 12) * size
        recs = []
        cur = {}
        while len(recs) < n:
            if rng.random() < 0.2:
                recs.append((TICK, rng.randint(1, 9 * size)))
            else:
                k = rng.choice(MOVERS)
                p = rng.choice([x for x in PLACES if x != cur.get(k)])
                recs.append((MOVE, k, p))
                cur[k] = p
        cands = candidates(recs)
        q = rng.randint(1, 5)
        if len(cands) < q:
            continue
        chosen = rng.sample(cands, q)
        slots = []
        for text, cutoff in chosen:
            slots.append((rng.randint(cutoff, n - 1), text))
        lines = []
        num = 0
        for i, rec in enumerate(recs):
            num += 1
            lines.append(f"{num} {render(rec)}")
            for pos, text in slots:
                if pos == i:
                    ans = evaluate(text, recs[: i + 1])
                    if ans is None:
                        raise RuntimeError("query not answerable from its prefix")
                    num += 1
                    lines.append(f"{num} {text}\t{ans}")
        episode = "\n".join(lines)
        if episode in seen:
            continue
        seen.add(episode)
        return lines


def main():
    args = sys.argv[1:]
    n_episodes = int(args[0]) if len(args) > 0 else 1
    seed = int(args[1]) if len(args) > 1 else 0
    size = int(args[2]) if len(args) > 2 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n_episodes):
        out.extend(make_episode(rng, size, seen))
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    main()
