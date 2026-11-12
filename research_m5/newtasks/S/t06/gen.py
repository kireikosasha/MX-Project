#!/usr/bin/env python3
"""Synthetic ORDER-IN-TIME task written as log records.

Entities e1..e4 move through places p1..p5. Each record line
    zom <entity> <place>.
says the entity arrives at the place; line order is time.

Queries (answered only from the lines above them):
    rav <entity> <place>?   place the entity was in just before its first arrival at <place>
    tuk <entity> <place>?   place the entity went to just after its first arrival at <place>
    kor <place> <a> <b>?    which of the two entities arrived at <place> first

Usage: python gen.py N SEED [SIZE]     SIZE in {1, 3, 10}, default 1
"""
import random
import sys

ENTITIES = ["e1", "e2", "e3", "e4"]
PLACES = ["p1", "p2", "p3", "p4", "p5"]
REC_OP = "zom"


def candidates(state, rng):
    """All valid (query_tokens, answer) pairs for the current state."""
    out = []
    for e in ENTITIES:
        tr = state[e]  # list of (place, line_index), in time order
        seen = set()
        for idx, (p, _) in enumerate(tr):
            if p in seen:
                continue  # only the first arrival at p counts
            seen.add(p)
            if idx >= 1:
                out.append(((["rav", e, p]), tr[idx - 1][0]))
            if idx < len(tr) - 1:
                out.append(((["tuk", e, p]), tr[idx + 1][0]))
    for p in PLACES:
        firsts = []
        for e in ENTITIES:
            for q, line in state[e]:
                if q == p:
                    firsts.append((line, e))
                    break
        for i in range(len(firsts)):
            for j in range(i + 1, len(firsts)):
                (la, a), (lb, b) = firsts[i], firsts[j]
                pair = [a, b]
                rng.shuffle(pair)
                ans = a if la < lb else b
                out.append((["kor", p, pair[0], pair[1]], ans))
    return out


def attempt(rng, n_rec, n_q):
    """One try at an episode; None if a query slot has no valid query."""
    total = n_rec + n_q
    qslots = set(rng.sample(range(2, total), n_q))  # never the first row
    state = {e: [] for e in ENTITIES}
    rows = []  # (text, answer or None)
    for i in range(total):
        if i in qslots:
            cands = candidates(state, rng)
            if not cands:
                return None
            args, ans = rng.choice(cands)
            rows.append((" ".join(args) + "?", ans))
        else:
            e = rng.choice(ENTITIES)
            cur = state[e][-1][0] if state[e] else None
            p = rng.choice([x for x in PLACES if x != cur])
            state[e].append((p, i))
            rows.append((REC_OP + " " + e + " " + p + ".", None))
    return rows


def make_episode(rng, size):
    while True:
        n_rec = rng.randint(3 * size, 12 * size)
        n_q = rng.randint(1, 5)
        rows = attempt(rng, n_rec, n_q)
        if rows is not None:
            return rows


def render(rows):
    lines = []
    for n, (text, ans) in enumerate(rows, 1):
        if ans is None:
            lines.append(f"{n} {text}")
        else:
            lines.append(f"{n} {text}\t{ans}")
    return "\n".join(lines)


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        text = render(make_episode(rng, size))
        if text in seen:
            continue
        seen.add(text)
        episodes.append(text)
    print("\n".join(episodes))


if __name__ == "__main__":
    main(sys.argv)
