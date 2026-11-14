"""Synthetic time-ordering reading-comprehension stories.

Usage: python gen.py N SEED

Each story has K events (K = 3..6). An event is a person going to a place.
Every event is stated exactly once, either with its time ("at time T") or
relative to another event ("before" / "after"). The statements are shuffled,
so events are told out of order. Questions come at the end of the story and
are answered by enumerating every timeline (ordering of the events) that
agrees with the statements; a question is kept only if all those timelines
give the same answer, and no single statement alone already fixes that answer.
"""
import itertools
import random
import sys

NAMES = ["ann", "bob", "cal", "dan", "eve"]
PLACES = ["kitchen", "garden", "hall", "barn"]


def solve(events, cons):
    """Every assignment of times 1..K to the events that satisfies cons."""
    k = len(events)
    result = []
    for perm in itertools.permutations(range(1, k + 1)):
        p = dict(zip(events, perm))
        good = True
        for c in cons:
            if c[0] == "at":
                if p[c[1]] != c[2]:
                    good = False
                    break
            else:
                first_is_earlier = p[c[1]] < p[c[2]]
                if first_is_earlier != (c[3] == "before"):
                    good = False
                    break
        if good:
            result.append(p)
    return result


def loc(p, events, name, t):
    """Place of the latest event of name at or before time t (None if none)."""
    best, best_pos = None, 0
    for e in events:
        if e[0] == name and best_pos < p[e] <= t:
            best, best_pos = e[1], p[e]
    return best


def spec_text(spec):
    k = spec[0]
    if k == "where":
        return "where was %s at time %d" % (spec[1], spec[2])
    if k == "yesno":
        return "was %s in the %s at time %d" % (spec[1], spec[2], spec[3])
    if k == "who":
        return "who was in the %s at time %d" % (spec[1], spec[2])
    if k == "count":
        return "how many people were in the %s at time %d" % (spec[1], spec[2])
    if k == "first":
        return "who went to the %s first" % spec[1]
    if k == "when":
        return "where was %s when %s went to the %s" % (spec[1], spec[2], spec[3])
    raise ValueError(k)


def spec_answer(spec, p, events):
    """Answer under one timeline p; None means the answer is undefined there."""
    k = spec[0]
    if k == "where":
        return loc(p, events, spec[1], spec[2])
    if k == "yesno":
        here = loc(p, events, spec[1], spec[3])
        return "yes" if here == spec[2] else "no"
    if k == "who":
        names = [n for n in NAMES if loc(p, events, n, spec[2]) == spec[1]]
        return ",".join(names) if names else None
    if k == "count":
        return str(sum(1 for n in NAMES if loc(p, events, n, spec[2]) == spec[1]))
    if k == "first":
        visits = [e for e in events if e[1] == spec[1]]
        if not visits:
            return None
        return min(visits, key=lambda e: p[e])[0]
    if k == "when":
        other_event = (spec[2], spec[3])
        return loc(p, events, spec[1], p[other_event] - 1)
    raise ValueError(k)


def determined(timelines, spec, events):
    """The single answer that all timelines agree on, or None."""
    value = None
    for p in timelines:
        a = spec_answer(spec, p, events)
        if a is None:
            return None
        if value is None:
            value = a
        elif a != value:
            return None
    return value


def all_specs(events, k):
    specs = []
    for n in NAMES:
        for t in range(1, k + 1):
            specs.append(("where", n, t))
            for pl in PLACES:
                specs.append(("yesno", n, pl, t))
    for pl in PLACES:
        specs.append(("first", pl))
        for t in range(1, k + 1):
            specs.append(("who", pl, t))
            specs.append(("count", pl, t))
    for other, pl2 in events:
        for n in NAMES:
            if n != other:
                specs.append(("when", n, other, pl2))
    return specs


def make_story(rng):
    while True:
        k = rng.randint(3, 6)
        events = []
        while len(events) < k:
            e = (rng.choice(NAMES), rng.choice(PLACES))
            if e not in events:
                events.append(e)

        order = events[:]
        rng.shuffle(order)
        truepos = {e: i + 1 for i, e in enumerate(order)}

        stated = events[:]
        rng.shuffle(stated)
        cons = []
        for e in stated:
            if rng.random() < 0.5:
                cons.append(("at", e, truepos[e]))
            else:
                others = [x for x in events if x != e]
                r = rng.choice(others)
                rel = "before" if truepos[e] < truepos[r] else "after"
                cons.append(("rel", e, r, rel))

        full = solve(events, cons)
        singles = [solve(events, [c]) for c in cons]

        specs = all_specs(events, k)
        rng.shuffle(specs)
        target = rng.randint(1, 5)
        chosen = []
        seen = set()
        for spec in specs:
            if len(chosen) >= target:
                break
            text = spec_text(spec)
            if text in seen:
                continue
            answer = determined(full, spec, events)
            if answer is None:
                continue
            if any(determined(s, spec, events) is not None for s in singles):
                continue
            seen.add(text)
            chosen.append((text, answer))

        if chosen:
            return events, cons, chosen


def render_statement(c):
    if c[0] == "at":
        e = c[1]
        return "%s went to the %s at time %d." % (e[0], e[1], c[2])
    e, r = c[1], c[2]
    return "%s went to the %s %s %s went to the %s." % (
        e[0], e[1], c[3], r[0], r[1])


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    stories = []
    for _ in range(n):
        events, cons, chosen = make_story(rng)
        lines = []
        i = 0
        for c in cons:
            i += 1
            lines.append("%d %s" % (i, render_statement(c)))
        for text, answer in chosen:
            i += 1
            lines.append("%d %s?\t%s" % (i, text, answer))
        stories.append("\n".join(lines))
    sys.stdout.write("\n".join(stories) + "\n")


if __name__ == "__main__":
    main()
