#!/usr/bin/env python3
"""Synthetic order-in-time reading comprehension task.

Usage: python gen.py N SEED

A story is a set of events. Each event is one person going to one place;
each (person, place) pair occurs at most once per story, so an event can be
named by its pair. Every event is told in one statement, either with its time
step ("at time 3") or as just after / just before another event. Statements
are told in shuffled order. Questions are answered by simulating the true
timeline, and a question is kept only when the statements above it fix its
answer uniquely (checked by an exhaustive search over orderings).
"""
import random
import sys

NAMES = ["ann", "ben", "cal", "dee"]
PLACES = ["park", "cafe", "shop", "school"]
HOME = "home"


def allowed_answers(events, cons, query):
    """Set of answers over all timelines that satisfy the constraints."""
    n = len(events)
    full = (1 << n) - 1
    pre = [0] * n  # pre[b]: bit set for each event that must come before b
    pos = [0] * n  # pos[e]: fixed time step of e, or 0 if unknown
    for c in cons:
        if c[0] == "time":
            pos[c[1]] = c[2]
        else:
            pre[c[2]] |= 1 << c[1]
    count = [bin(m).count("1") for m in range(full + 1)]
    kind = query[0]
    qperson = query[1] if kind != "first" else None

    def can_place(mask, e):
        if (mask >> e) & 1 or (pre[e] & ~mask):
            return False
        return (not pos[e]) or count[mask] == pos[e] - 1

    done = {}

    def completable(mask):
        if mask == full:
            return True
        if mask not in done:
            done[mask] = any(
                can_place(mask, e) and completable(mask | (1 << e))
                for e in range(n)
            )
        return done[mask]

    def placed_at(mask, place):
        return any((mask >> k) & 1 and events[k][1] == place for k in range(n))

    memo = {}

    def walk(mask, ploc):
        key = (mask, ploc)
        if key in memo:
            return memo[key]
        res = set()
        if kind == "time" and count[mask] == query[2]:
            if completable(mask):
                res.add(ploc)
        else:
            for e in range(n):
                if not can_place(mask, e):
                    continue
                nmask = mask | (1 << e)
                if not completable(nmask):
                    continue
                person, place = events[e]
                if kind == "before" and e == query[2]:
                    res.add(ploc)
                elif kind == "first" and place == query[1] and not placed_at(mask, place):
                    res.add(person)
                else:
                    nloc = place if person == qperson else ploc
                    res |= walk(nmask, nloc)
        memo[key] = res
        return res

    return walk(0, None if kind == "first" else HOME)


def simulate(events, order, query):
    """Answer a query by running the true timeline (order = event ids in time order)."""
    kind = query[0]
    qperson = query[1] if kind != "first" else None
    ploc = HOME
    for step, e in enumerate(order, 1):
        person, place = events[e]
        if kind == "time" and step > query[2]:
            break
        if kind == "before" and e == query[2]:
            return ploc
        if kind == "first" and place == query[1]:
            return person
        if person == qperson:
            ploc = place
    return ploc


def constraint(stmt):
    if stmt[0] == "time":
        return ("time", stmt[1], stmt[2])
    _, i, j, word = stmt
    # "i after j" means j comes before i; "i before j" means i comes before j.
    return ("pre", j, i) if word == "after" else ("pre", i, j)


def statement_text(events, stmt):
    if stmt[0] == "time":
        _, i, t, v = stmt
        person, place = events[i]
        if v == 0:
            s = f"{person} went to the {place} at time {t}."
        else:
            s = f"at time {t} {person} went to the {place}."
    else:
        _, i, j, word = stmt
        pi, li = events[i]
        pj, lj = events[j]
        s = f"{pi} went to the {li} {word} {pj} went to the {lj}."
    return s[0].upper() + s[1:]


def question_text(events, query):
    kind = query[0]
    if kind == "time":
        s = f"where was {query[1]} at time {query[2]}?"
    elif kind == "before":
        person, place = events[query[2]]
        s = f"where was {query[1]} before {person} went to the {place}?"
    else:
        s = f"who went to the {query[1]} first?"
    return s[0].upper() + s[1:]


def minimal_support(events, cons, query, truth, rng):
    """Greedily drop statements while the answer stays fixed; return what is left."""
    trials = list(range(len(cons)))
    rng.shuffle(trials)
    current = list(range(len(cons)))
    for idx in trials:
        trial = [k for k in current if k != idx]
        if allowed_answers(events, [cons[k] for k in trial], query) == {truth}:
            current = trial
    return current


def attempt(rng):
    n = rng.randint(3, 12)
    events = rng.sample([(p, l) for p in NAMES for l in PLACES], n)
    order = list(range(n))
    rng.shuffle(order)
    step = [0] * n
    for k, e in enumerate(order, 1):
        step[e] = k

    stmts = []
    for i in range(n):
        if rng.random() < 0.5:
            stmts.append(("time", i, step[i], rng.randint(0, 1)))
        else:
            j = rng.choice([x for x in range(n) if x != i])
            word = "after" if step[i] > step[j] else "before"
            stmts.append(("rel", i, j, word))
    rng.shuffle(stmts)
    cons = [constraint(s) for s in stmts]

    queries = [("time", p, t) for p in NAMES for t in range(1, n + 1)]
    queries += [("before", p, x) for x in range(n) for p in NAMES if p != events[x][0]]
    queries += [("first", l) for l in PLACES if any(pl == l for _, pl in events)]
    rng.shuffle(queries)

    want = rng.randint(1, 5)
    chosen = []
    for q in queries:
        if len(chosen) == want:
            break
        truth = simulate(events, order, q)
        if allowed_answers(events, cons, q) != {truth}:
            continue
        if len(minimal_support(events, cons, q, truth, rng)) < 2:
            continue  # a single statement would answer it; need several
        chosen.append((q, truth))
    if not chosen:
        return None

    lines = [f"{num} {statement_text(events, s)}" for num, s in enumerate(stmts, 1)]
    for num, (q, truth) in enumerate(chosen, n + 1):
        lines.append(f"{num} {question_text(events, q)}\t{truth}")
    return lines


def make_story(rng):
    while True:
        lines = attempt(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
