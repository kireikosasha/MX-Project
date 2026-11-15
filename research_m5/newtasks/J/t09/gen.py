#!/usr/bin/env python3
"""Synthetic order-in-time reading task generator.

Usage: python gen.py N SEED

Each story is a set of events: a person came to or went to a place at a
clock slot (a digit 1-9). The events are told in shuffled order, so the
reader must sort them by time. Questions ask where someone was at a time,
who was in a place at a time, where someone went first or last, or whether
someone was in a place at a time. Each question is placed only after every
statement it depends on, and its answer is computed from the statements above.
"""
import random
import sys

NAMES = ["Ana", "Ben", "Cy", "Dee", "Eli", "Fay"]
PLACES = ["park", "shop", "school", "lake", "farm", "hall"]
TIMES = list(range(1, 10))


def make_events(rng):
    """Return a list of (person, time, place, verb); times are distinct per person."""
    n_events = rng.randint(3, 12)
    people = rng.sample(NAMES, rng.randint(2, 4))
    used = {p: set() for p in people}
    raw = []
    for _ in range(n_events):
        open_people = [p for p in people if len(used[p]) < len(TIMES)]
        person = rng.choice(open_people)
        t = rng.choice([x for x in TIMES if x not in used[person]])
        used[person].add(t)
        raw.append((person, t, rng.choice(PLACES)))
    first = {}
    for person, t, _ in raw:
        first[person] = min(first.get(person, 10), t)
    events = []
    for person, t, place in raw:
        verb = "came" if t == first[person] else rng.choice(["came", "went"])
        events.append((person, t, place, verb))
    return events


def telling_order(rng, events):
    """Shuffle the events so they are not told in time order."""
    order = list(events)
    by_time = sorted(events, key=lambda e: e[1])
    for _ in range(20):
        rng.shuffle(order)
        if order != by_time:
            break
    return order


def relevant(kind, person, t, events):
    """Events the answer depends on; all of them must be stated above the question."""
    if kind == "who":
        return [e for e in events if e[1] <= t]
    if kind in ("first", "last"):
        return [e for e in events if e[0] == person]
    return [e for e in events if e[0] == person and e[1] <= t]


def answer(kind, person, place, t, seen):
    """Compute the answer from the statements shown so far."""
    if kind == "where":
        own = [e for e in seen if e[0] == person and e[1] <= t]
        if not own:
            return "nowhere"
        return max(own, key=lambda e: e[1])[2]
    if kind == "who":
        latest = {}
        for e in seen:
            if e[1] <= t and (e[0] not in latest or e[1] > latest[e[0]][1]):
                latest[e[0]] = e
        names = sorted(p.lower() for p, e in latest.items() if e[2] == place)
        return ",".join(names) if names else "nobody"
    if kind == "first":
        own = [e for e in seen if e[0] == person]
        return min(own, key=lambda e: e[1])[2]
    if kind == "last":
        own = [e for e in seen if e[0] == person]
        return max(own, key=lambda e: e[1])[2]
    own = [e for e in seen if e[0] == person and e[1] <= t]
    if not own:
        return "no"
    return "yes" if max(own, key=lambda e: e[1])[2] == place else "no"


def question_text(kind, person, place, t):
    if kind == "where":
        return f"Where was {person} at {t}?"
    if kind == "who":
        return f"Who was in the {place} at {t}?"
    if kind == "first":
        return f"Where did {person} go first?"
    if kind == "last":
        return f"Where did {person} go last?"
    return f"Was {person} in the {place} at {t}?"


def make_question(rng, events, order):
    kind = rng.choice(["where", "who", "first", "last", "was"])
    person = rng.choice(sorted({e[0] for e in events}))
    place = rng.choice(PLACES)
    t = rng.choice(TIMES)
    rel = relevant(kind, person, t, events)
    kmin = max((order.index(e) + 1 for e in rel), default=0)
    return kind, person, place, t, kmin


def make_story(rng):
    while True:
        events = make_events(rng)
        order = telling_order(rng, events)
        target = rng.randint(1, 5)
        questions = []
        texts = set()
        for _ in range(60):
            if len(questions) >= target:
                break
            kind, person, place, t, kmin = make_question(rng, events, order)
            text = question_text(kind, person, place, t)
            if text in texts:
                continue
            pos = rng.randint(max(1, kmin), len(order))
            ans = answer(kind, person, place, t, order[:pos])
            texts.add(text)
            questions.append((pos, text, ans))
        if questions:
            break
    questions.sort(key=lambda q: q[0])
    lines = []
    qi = 0
    for i in range(len(order) + 1):
        while qi < len(questions) and questions[qi][0] == i:
            _, text, ans = questions[qi]
            lines.append(f"{text}\t{ans}")
            qi += 1
        if i < len(order):
            person, t, place, verb = order[i]
            lines.append(f"At {t} {person} {verb} to the {place}.")
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for k, line in enumerate(make_story(rng), 1):
            out.append(f"{k} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
