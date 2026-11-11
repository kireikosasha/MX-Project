#!/usr/bin/env python3
"""Synthetic time-order reading-comprehension stories.

Usage: python gen.py N SEED

Each story is generated from a hidden world: a few visits (person, place),
each on one weekday (Monday..Friday). Statements reveal either the weekday
of a visit or an order between two visits, in shuffled order. Questions ask
where a person was before/after another visit, or on a given weekday. Answers
are computed by brute-force enumeration of all weekday assignments that agree
with the statements above the question.
"""
import itertools
import random
import sys

NAMES = ["Ann", "Ben", "Cy", "Dana", "Eli", "Fay", "Gus"]
PLACES = ["park", "shop", "farm", "school", "lake"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
NDAYS = len(DAYS)


# ---------------------------------------------------------------- facts
# A fact is (rel, a, b):
#   ("on", event, day_index)   event a happened on day b
#   ("before", a, b)           day(a) < day(b)
#   ("after", a, b)            day(a) > day(b)
# An event is a (person, place) pair; each pair occurs at most once per story.

def fact_events(fact):
    rel, a, b = fact
    return [a] if rel == "on" else [a, b]


def ev_text(e):
    return "%s went to the %s" % (e[0], e[1])


def fact_text(rng, fact):
    rel, a, b = fact
    if rel == "on":
        day = DAYS[b]
        if rng.random() < 0.5:
            return "%s went to the %s on %s." % (a[0], a[1], day)
        return "On %s %s went to the %s." % (day, a[0], a[1])
    return "%s %s %s." % (ev_text(a), rel, ev_text(b))


def holds(w, fact):
    rel, a, b = fact
    if rel == "on":
        return w[a] == b
    if rel == "before":
        return w[a] < w[b]
    return w[a] > w[b]


def consistent_worlds(facts):
    """All day assignments to the events mentioned in facts that satisfy them."""
    universe = []
    for f in facts:
        for e in fact_events(f):
            if e not in universe:
                universe.append(e)
    worlds = []
    for days in itertools.product(range(NDAYS), repeat=len(universe)):
        w = dict(zip(universe, days))
        if all(holds(w, f) for f in facts):
            worlds.append(w)
    return worlds


# ---------------------------------------------------------------- questions
# A question is (kind, person, x): x is a day index for "on", else an event.

def answer_in(w, q):
    kind, p, x = q
    if kind == "on":
        hits = [e for e in w if e[0] == p and w[e] == x]
    else:
        if x not in w:
            return None
        t = w[x]
        if kind == "before":
            hits = [e for e in w if e[0] == p and w[e] < t]
            if not hits:
                return None
            best = max(w[e] for e in hits)
        else:
            hits = [e for e in w if e[0] == p and w[e] > t]
            if not hits:
                return None
            best = min(w[e] for e in hits)
        hits = [e for e in hits if w[e] == best]
    if len(hits) != 1:
        return None
    return hits[0][1]


def answer_over(worlds, q):
    res = set()
    for w in worlds:
        a = answer_in(w, q)
        if a is None:
            return None
        res.add(a)
    if len(res) != 1:
        return None
    return res.pop()


def q_text(q):
    kind, p, x = q
    if kind == "on":
        return "Where was %s on %s?" % (p, DAYS[x])
    return "Where was %s %s %s went to the %s?" % (p, kind, x[0], x[1])


def try_question(rng, facts, day, used):
    """Find a question about facts that is answered the same in every consistent
    world, matches the hidden world, and is not answered by any single statement."""
    universe = []
    for f in facts:
        for e in fact_events(f):
            if e not in universe:
                universe.append(e)
    persons = []
    for e in universe:
        if e[0] not in persons:
            persons.append(e[0])
    cands = []
    for p in persons:
        for e in universe:
            if e[0] != p:
                cands.append(("before", p, e))
                cands.append(("after", p, e))
        for d in range(NDAYS):
            cands.append(("on", p, d))
    rng.shuffle(cands)

    worlds = consistent_worlds(facts)
    singles = [consistent_worlds([f]) for f in facts]
    full = dict(day)
    for q in cands:
        if q_text(q) in used:
            continue
        a = answer_over(worlds, q)
        if a is None:
            continue
        if answer_in(full, q) != a:
            continue
        if any(answer_over(sw, q) is not None for sw in singles):
            continue
        return q, a
    return None


# ---------------------------------------------------------------- stories
def random_fact(rng, events, day):
    if rng.random() < 0.4:
        e = rng.choice(events)
        return ("on", e, day[e])
    a, b = rng.sample(events, 2)
    if day[a] == day[b]:
        return None
    if day[a] < day[b]:
        return ("before", a, b)
    return ("after", a, b)


def try_story(rng):
    people = rng.sample(NAMES, rng.randint(3, 5))
    places = rng.sample(PLACES, rng.randint(3, 4))
    n_ev = rng.randint(3, 5)
    pairs = [(p, l) for p in people for l in places]
    events = rng.sample(pairs, n_ev)
    day = {e: rng.randrange(NDAYS) for e in events}

    n_st = rng.randint(3, 10)
    facts, texts = [], []
    tries = 0
    while len(facts) < n_st and tries < 200:
        tries += 1
        f = random_fact(rng, events, day)
        if f is None:
            continue
        t = fact_text(rng, f)
        if t in texts:
            continue
        facts.append(f)
        texts.append(t)
    if len(facts) < 3:
        return None

    target = rng.randint(1, 5)
    used = set()
    out = []
    qcount = 0
    for i, t in enumerate(texts):
        out.append((t, None))
        if i >= 1 and qcount < target and rng.random() < 0.5:
            r = try_question(rng, facts[: i + 1], day, used)
            if r:
                q, a = r
                used.add(q_text(q))
                out.append((q_text(q), a))
                qcount += 1
    attempts = 0
    while qcount < target and attempts < 4:
        attempts += 1
        r = try_question(rng, facts, day, used)
        if r:
            q, a = r
            used.add(q_text(q))
            out.append((q_text(q), a))
            qcount += 1
    if qcount == 0:
        return None
    return out


def make_story(rng):
    while True:
        s = try_story(rng)
        if s is not None:
            return s


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        story = make_story(rng)
        for num, (text, ans) in enumerate(story, 1):
            if ans is None:
                lines.append("%d %s\n" % (num, text))
            else:
                lines.append("%d %s\t%s\n" % (num, text, ans))
    sys.stdout.write("".join(lines))


if __name__ == "__main__":
    main()
