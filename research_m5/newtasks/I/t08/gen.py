"""Synthetic order-in-time reading task.

People move between places. Some moves are stated alone, others are stated
as coming before or after another move. Questions ask where someone was when
an event happened, whether one move came before another, or who went to a
place first. Answers are found by enumerating every timeline that agrees with
the statements shown so far; a question is only asked when all those timelines
give the same answer, and when that answer needs the ordering statements.

Usage: python gen.py N SEED
"""

import random
import sys
from itertools import permutations

NAMES = ["ann", "bob", "carl", "dana", "eve"]
PLACES = ["park", "school", "shop", "lake", "home"]


def event_text(ev):
    person, place = ev
    return f"{person} went to {place}"


def settle(fn, constrained, everything):
    """Return the answer of fn if it is the same in all constrained timelines
    and differs somewhere among all timelines; otherwise None."""
    first = None
    for i, order in enumerate(constrained):
        value = fn(order)
        if i == 0:
            first = value
        elif value != first:
            return None
    if first is None:
        return None
    for order in everything:
        if fn(order) != first:
            return first
    return None


def candidates(events, revealed, rel):
    everything = list(permutations(revealed))
    constrained = [o for o in everything
                   if all(o.index(a) < o.index(b) for a, b in rel)]
    out = []

    for x in revealed:
        qx, mx = events[x]
        for p in NAMES:
            if p == qx:
                continue

            def where_fn(o, p=p, x=x):
                prior = [e for e in o[:o.index(x)] if events[e][0] == p]
                return events[prior[-1]][1] if prior else None

            ans = settle(where_fn, constrained, everything)
            if ans is not None:
                out.append((f"where was {p} when {event_text(events[x])}?", ans))

    for x in revealed:
        px, lx = events[x]
        for y in revealed:
            if x == y:
                continue

            def before_fn(o, x=x, y=y):
                return "yes" if o.index(x) < o.index(y) else "no"

            ans = settle(before_fn, constrained, everything)
            if ans is not None:
                out.append((f"did {px} go to {lx} before {event_text(events[y])}?", ans))

    for l in PLACES:
        at = [e for e in revealed if events[e][1] == l]
        if len(at) < 2:
            continue

        def first_fn(o, at=at):
            return events[min(at, key=o.index)][0]

        ans = settle(first_fn, constrained, everything)
        if ans is not None:
            out.append((f"who went to {l} first?", ans))

    return out


def add_statement(rng, events, revealed, rel, lines):
    k = len(events)
    unrevealed = [i for i in range(k) if i not in revealed]
    if unrevealed and rng.random() < 0.4:
        i = rng.choice(unrevealed)
        lines.append(event_text(events[i]) + ".")
        revealed.append(i)
        return
    a, b = rng.sample(range(k), 2)
    i, j = min(a, b), max(a, b)  # event index equals true time order
    if rng.random() < 0.5:
        lines.append(f"{event_text(events[i])} before {event_text(events[j])}.")
    else:
        lines.append(f"{event_text(events[j])} after {event_text(events[i])}.")
    rel.append((i, j))
    for e in (i, j):
        if e not in revealed:
            revealed.append(e)


def try_story(rng):
    k = rng.randint(3, 6)
    events = []
    while len(events) < k:
        e = (rng.choice(NAMES), rng.choice(PLACES))
        if e not in events:
            events.append(e)  # list position = true time

    n_stmt = rng.randint(3, 10)
    n_q = rng.randint(1, 4)
    lines = []
    revealed = []
    rel = []
    asked = set()
    n_s = n_q_done = 0
    attempts = 0

    while n_s < n_stmt or n_q_done < n_q:
        attempts += 1
        if attempts > 300:
            return None
        if n_q_done < n_q and n_s >= 2 and (n_s >= n_stmt or rng.random() < 0.4):
            cands = [c for c in candidates(events, revealed, rel) if c[0] not in asked]
            if cands:
                text, ans = rng.choice(cands)
                asked.add(text)
                lines.append(f"{text}\t{ans}")
                n_q_done += 1
                continue
            if n_s >= n_stmt:
                return None
        if n_s < n_stmt:
            add_statement(rng, events, revealed, rel, lines)
            n_s += 1
    return lines


def make_story(rng):
    while True:
        story = try_story(rng)
        if story is not None:
            return story


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for num, line in enumerate(make_story(rng), 1):
            print(f"{num} {line}")


if __name__ == "__main__":
    main()
