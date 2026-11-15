#!/usr/bin/env python3
"""Synthetic order-in-time reading task.

Each story has a hidden timeline of events: a person goes to a place at a
distinct time 1..n. The statements tell these events out of order, each either
with its absolute time ("at time 3") or relative to another event ("before" /
"after"). Questions (placed at the end) ask where someone was at a time, where
someone was before a given event, who was in a place at a time, or whether
someone was in a place at a time.

Answers come from the hidden timeline. A question is kept only if every
timeline that agrees with the statements gives the same answer, so every
answer is unambiguous.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Mary", "John", "Sandra", "Fred", "Kim", "Lou"]
PLACES = ["kitchen", "garden", "hall", "office", "park"]


def consistent_orders(n, fixed, prec):
    """All timelines (lists of event ids, index 0 = time 1) that agree with
    the statements. fixed: {event: time}; prec: [(a, b)] meaning a is before b."""
    need = [[] for _ in range(n)]
    for a, b in prec:
        need[b].append(a)
    fixed_at = {t: e for e, t in fixed.items()}
    used = [False] * n
    seq = []
    out = []

    def place(r):
        if r > n:
            out.append(list(seq))
            return
        if r in fixed_at:
            cands = [fixed_at[r]]
        else:
            cands = [e for e in range(n) if e not in fixed]
        for e in cands:
            if used[e] or any(not used[a] for a in need[e]):
                continue
            used[e] = True
            seq.append(e)
            place(r + 1)
            seq.pop()
            used[e] = False

    place(1)
    return out


def where(order, events, person, t):
    """Place of person after times 1..t, or None if no event yet."""
    loc = None
    for e in order[:t]:
        if events[e][0] == person:
            loc = events[e][1]
    return loc


def make_question(rng, events, people, n):
    kind = rng.randrange(4)
    if kind == 0:
        p = rng.choice(people)
        t = rng.randint(1, n)
        text = f"Where was {p} at time {t}?"

        def fn(order, p=p, t=t):
            return where(order, events, p, t)
    elif kind == 1:
        p = rng.choice(people)
        k = rng.randrange(n)
        q, m = events[k]
        text = f"Where was {p} before {q} went to the {m}?"

        def fn(order, p=p, k=k):
            pos = order.index(k) + 1
            return where(order, events, p, pos - 1)
    elif kind == 2:
        l = rng.choice(PLACES)
        t = rng.randint(1, n)
        text = f"Who was in the {l} at time {t}?"

        def fn(order, l=l, t=t):
            names = sorted(p for p in people if where(order, events, p, t) == l)
            return ",".join(x.lower() for x in names) or None
    else:
        p = rng.choice(people)
        l = rng.choice(PLACES)
        t = rng.randint(1, n)
        text = f"Was {p} in the {l} at time {t}?"

        def fn(order, p=p, l=l, t=t):
            return "yes" if where(order, events, p, t) == l else "no"
    return text, fn


def try_story(rng):
    n = rng.randint(3, 7)
    people = rng.sample(NAMES, rng.randint(2, 4))
    pairs = [(p, l) for p in people for l in PLACES]
    events = rng.sample(pairs, n)  # distinct (person, place) per event

    true_time = list(range(1, n + 1))
    rng.shuffle(true_time)  # true_time[e] = hidden time of event e
    abs_ids = set(rng.sample(range(n), rng.randint(1, n // 2 + 1)))

    tell = list(range(n))
    rng.shuffle(tell)
    fixed = {}
    prec = []
    stmts = []
    for e in tell:
        p, l = events[e]
        if e in abs_ids:
            fixed[e] = true_time[e]
            stmts.append(f"{p} went to the {l} at time {true_time[e]}.")
        else:
            j = rng.choice([x for x in range(n) if x != e])
            q, m = events[j]
            if true_time[e] < true_time[j]:
                prec.append((e, j))
                word = "before"
            else:
                prec.append((j, e))
                word = "after"
            stmts.append(f"{p} went to the {l} {word} {q} went to the {m}.")

    order_truth = sorted(range(n), key=lambda e: true_time[e])
    orders = consistent_orders(n, fixed, prec)

    qs = []
    target = rng.randint(1, 5)
    for _ in range(40):
        if len(qs) >= target:
            break
        text, fn = make_question(rng, events, people, n)
        if any(text == t for t, _ in qs):
            continue
        truth = fn(order_truth)
        if truth is None:
            continue
        if all(fn(o) == truth for o in orders):
            qs.append((text, truth))
    if not qs:
        return None

    lines = [f"{i + 1} {s}" for i, s in enumerate(stmts)]
    base = len(stmts)
    lines += [f"{base + i + 1} {t}\t{a}" for i, (t, a) in enumerate(qs)]
    return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(count):
        story = None
        while story is None:
            story = try_story(rng)
        for line in story:
            print(line)


if __name__ == "__main__":
    main()
