#!/usr/bin/env python3
"""Synthetic order-in-time reading task.

Usage: python gen.py N SEED

People walk between places. Each move is an event with a hidden place in
time. The story tells the order of some pairs of moves with "before" and
"after" sentences, in shuffled order. Questions ask where someone was just
before or after a move, which place someone went to first or last, whether
one move came before another, and who was in a place before a move. The
answers come from the hidden timeline. A question is only placed where the
statements above it settle the answer through a chain of facts.
"""
import random
import sys
from itertools import combinations

PERSONS = ["ann", "bob", "cat", "dan", "eve"]
PLACES = ["hall", "kitchen", "garden", "office", "shop"]


def closure(n, direct):
    """Transitive closure of the known 'before' relation."""
    k = [row[:] for row in direct]
    for m in range(n):
        for i in range(n):
            if k[i][m]:
                for j in range(n):
                    if k[m][j]:
                        k[i][j] = True
    return k


def truth_nearest(xs, e, rank, before):
    """Simulation: the latest (before) or earliest (after) event in xs
    relative to event e, using the hidden timeline."""
    xs = [x for x in xs if x != e]
    if before:
        c = [x for x in xs if rank[x] < rank[e]]
        return max(c, key=lambda x: rank[x]) if c else None
    c = [x for x in xs if rank[x] > rank[e]]
    return min(c, key=lambda x: rank[x]) if c else None


def nearest(xs, e, known, before):
    """Reasoning from known facts. Returns (status, event, pairs) when the
    facts settle the answer, else None. status is 'loc' (event found) or
    'none' (no event on that side of e). pairs lists the relations used."""
    xs = [x for x in xs if x != e]
    if before:
        cand = [x for x in xs if known[x][e]]
        if not cand:
            if all(known[e][x] for x in xs):
                return ("none", None, [(e, x) for x in xs])
            return None
        for b in cand:
            pairs, good = [], True
            for x in xs:
                if x == b:
                    continue
                if known[x][b]:
                    pairs.append((x, b))
                elif known[e][x]:
                    pairs.append((e, x))
                else:
                    good = False
                    break
            if good:
                return ("loc", b, pairs + [(b, e)])
        return None
    cand = [x for x in xs if known[e][x]]
    if not cand:
        if all(known[x][e] for x in xs):
            return ("none", None, [(x, e) for x in xs])
        return None
    for b in cand:
        pairs, good = [], True
        for x in xs:
            if x == b:
                continue
            if known[b][x]:
                pairs.append((b, x))
            elif known[x][e]:
                pairs.append((x, e))
            else:
                good = False
                break
        if good:
            return ("loc", b, pairs + [(e, b)])
    return None


def candidates(events, n, rank, direct, mentioned):
    """All questions the statements so far settle, as (text, answer)."""
    known = closure(n, direct)
    persons = sorted(set(p for p, _ in events))
    ev_of = {p: [e for e in range(n) if events[e][0] == p] for p in persons}
    places = sorted(set(l for _, l in events))

    def derived(pairs):
        # at least one relation was reached through a chain, not stated directly
        return any(known[u][v] and not direct[u][v] for u, v in pairs)

    def full(p):
        return all(e in mentioned for e in ev_of[p])

    out = []

    # Where was X before/after <move E>?
    for e in range(n):
        if e not in mentioned:
            continue
        pe, le = events[e]
        for x in persons:
            if x == pe or not full(x):
                continue
            for before in (True, False):
                r = nearest(ev_of[x], e, known, before)
                if r is None or r[0] != "loc" or not derived(r[2]):
                    continue
                t = truth_nearest(ev_of[x], e, rank, before)
                if t is None or t != r[1]:
                    continue
                word = "before" if before else "after"
                out.append((f"Where was {x} {word} {pe} went to the {le}?",
                            events[t][1]))

        # Who was in <place> before <move E>?
        for l in places:
            res, pairs, ok = [], [], True
            for x in persons:
                if not full(x):
                    ok = False
                    break
                r = nearest(ev_of[x], e, known, True)
                if r is None:
                    ok = False
                    break
                pairs += r[2]
                if r[0] == "loc" and events[r[1]][1] == l:
                    res.append(x)
            if not ok or not res or not derived(pairs):
                continue
            truth = []
            for x in persons:
                t = truth_nearest(ev_of[x], e, rank, True)
                if t is not None and events[t][1] == l:
                    truth.append(x)
            if sorted(res) != sorted(truth):
                continue
            out.append((f"Who was in the {l} before {pe} went to the {le}?",
                        ",".join(sorted(res))))

    # Which place did X go to first / last?
    for x in persons:
        if not full(x):
            continue
        xs = ev_of[x]
        for b in xs:
            others = [y for y in xs if y != b]
            if others and all(known[b][y] for y in others) \
                    and derived([(b, y) for y in others]):
                out.append((f"Which place did {x} go to first?", events[b][1]))
            if others and all(known[y][b] for y in others) \
                    and derived([(y, b) for y in others]):
                out.append((f"Which place did {x} go to last?", events[b][1]))

    # Did A go to P before B go to Q?
    for a in range(n):
        for b in range(n):
            if a == b or a not in mentioned or b not in mentioned:
                continue
            if known[a][b]:
                if direct[a][b]:
                    continue
                ans = "yes"
            elif known[b][a]:
                if direct[b][a]:
                    continue
                ans = "no"
            else:
                continue
            pa, la = events[a]
            pb, lb = events[b]
            out.append((f"Did {pa} go to the {la} before {pb} go to the {lb}?", ans))
    return out


def try_story(rng):
    n = rng.randint(4, 6)
    events, used = [], set()
    while len(events) < n:
        pair = (rng.choice(PERSONS), rng.choice(PLACES))
        if pair not in used:
            used.add(pair)
            events.append(pair)

    # hidden timeline: rank[e] is the time of event e
    order = list(range(n))
    rng.shuffle(order)
    rank = [0] * n
    for t, e in enumerate(order):
        rank[e] = t

    # each unordered pair gives two sentence forms that state the same order
    cands = []
    for a, b in combinations(range(n), 2):
        if rank[a] > rank[b]:
            a, b = b, a
        pa, la = events[a]
        pb, lb = events[b]
        cands.append((a, b, f"{pa} went to the {la} before {pb} went to the {lb}."))
        cands.append((a, b, f"{pb} went to the {lb} after {pa} went to the {la}."))
    rng.shuffle(cands)
    count = rng.randint(3, min(12, len(cands)))
    stmts = cands[:count]

    direct = [[False] * n for _ in range(n)]
    mentioned = set()
    items, asked = [], set()
    qtarget = rng.randint(1, 5)
    state = {"nq": 0}

    def add_question():
        pool = [c for c in candidates(events, n, rank, direct, mentioned)
                if c[0] not in asked]
        if not pool:
            return
        q, ans = rng.choice(pool)
        asked.add(q)
        items.append(f"{q}\t{ans}")
        state["nq"] += 1

    for idx, (a, b, text) in enumerate(stmts):
        items.append(text)
        direct[a][b] = True
        mentioned.update((a, b))
        if state["nq"] < qtarget and idx >= 1 and rng.random() < 0.5:
            add_question()
    tries = 0
    while state["nq"] < qtarget and tries < 10:
        tries += 1
        add_question()

    if state["nq"] == 0 or len(mentioned) != n:
        return None
    return items


def make_story(rng):
    while True:
        items = try_story(rng)
        if items is not None:
            return items


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n_stories):
        items = make_story(rng)
        for k, item in enumerate(items, 1):
            lines.append(f"{k} {item}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
