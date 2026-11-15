#!/usr/bin/env python3
"""Synthetic time-order reading-comprehension stories.

Usage: python gen.py N SEED

Each story is a set of events in which a person goes to a place. The
events are told out of order. Some statements give the time of the event
("... at time 4."), others only say what it came before or after
("... after bob went to the park."). Questions ask where a person was
when an event happened, or whether one event came before another.

The true timeline is simulated first. Each question is then checked
against what the reader can deduce from the statements above it; the
question is kept only if that deduction is unambiguous and gives the
simulated answer.
"""
import random
import sys

NAMES = ["ann", "bob", "cat", "dan", "eve", "fay"]
PLACES = ["shop", "park", "hall", "lab", "barn", "garden"]
TIMED_PROB = 0.4


class Event:
    def __init__(self, idx, person, place):
        self.idx = idx          # position in the true timeline, 0-based
        self.person = person
        self.place = place
        self.kind = "timed"     # "timed", "after" or "before"
        self.ref = None         # index of the other event for after/before


def statement(ev, events):
    base = f"{ev.person} went to the {ev.place}"
    if ev.kind == "timed":
        return f"{base} at time {ev.idx + 1}."
    other = events[ev.ref]
    return f"{base} {ev.kind} {other.person} went to the {other.place}."


def known_order(events, shown):
    """Events the reader knows about, and the 'known before' relation.

    rows[a] is a bitmask: bit b is set when event a is known to come
    before event b, using only the statements in `shown`. The relation is
    transitively closed.
    """
    K = len(events)
    rows = [0] * K
    mentioned = set()
    for s in shown:
        mentioned.add(s)
        ev = events[s]
        if ev.kind == "timed":
            for t in shown:
                tv = events[t]
                if tv.kind == "timed" and tv.idx < ev.idx:
                    rows[t] |= 1 << s
        else:
            mentioned.add(ev.ref)
            if ev.kind == "after":
                rows[ev.ref] |= 1 << s
            else:
                rows[s] |= 1 << ev.ref
    for k in range(K):
        for i in range(K):
            if rows[i] >> k & 1:
                rows[i] |= rows[k]
    return mentioned, rows


def answer_where(events, mentioned, rows, x, ev_id):
    """Where was x just before the event ev_id? None if not deducible."""
    xs = [e for e in mentioned if events[e].person == x and e != ev_id]
    for c in xs:
        if not (rows[c] >> ev_id & 1):
            continue  # c must be known to come before the event
        # c is the latest x-event before the event if every other x-event
        # is known to come before c, or known to come after the event.
        if all(e == c or (rows[e] >> c & 1) or (rows[ev_id] >> e & 1)
               for e in xs):
            earlier = [e for e in range(len(events))
                       if events[e].person == x and e < ev_id]
            if earlier and max(earlier) == c:
                return events[c].place
            return None
    return None


def answer_before(rows, a, b):
    """Did event a come before event b? Returns 'yes', 'no' or None."""
    if rows[a] >> b & 1:
        return "yes"
    if rows[b] >> a & 1:
        return "no"
    return None


def make_question(rng, events, people, shown, mentioned, rows):
    if rng.random() < 0.5:
        ev_id = rng.choice(shown)
        ev = events[ev_id]
        x = rng.choice(people)
        ans = answer_where(events, mentioned, rows, x, ev_id)
        if ans is None:
            return None
        word = rng.choice(["when", "before"])
        return f"where was {x} {word} {ev.person} went to the {ev.place}?", ans
    if len(shown) < 2:
        return None
    a, b = rng.sample(shown, 2)
    ans = answer_before(rows, a, b)
    if ans is None:
        return None
    # the true timeline is the index order, so this must agree with it
    if (a < b) != (ans == "yes"):
        return None
    ea, eb = events[a], events[b]
    text = (f"did {ea.person} go to the {ea.place} before "
            f"{eb.person} went to the {eb.place}?")
    return text, ans


def make_story(rng):
    while True:
        people = rng.sample(NAMES, rng.randint(2, 4))
        places = rng.sample(PLACES, rng.randint(3, 5))
        pairs = [(p, q) for p in people for q in places]
        K = rng.randint(3, 12)
        if len(pairs) < K:
            continue
        # each (person, place) pair happens at most once, so a statement
        # "x went to the y" always names a single event
        chosen = rng.sample(pairs, K)
        events = [Event(i, p, q) for i, (p, q) in enumerate(chosen)]
        for ev in events:
            i = ev.idx
            options = []
            if i > 0:
                options.append("after")
            if i < K - 1:
                options.append("before")
            if rng.random() < TIMED_PROB or not options:
                continue
            ev.kind = rng.choice(options)
            if ev.kind == "after":
                ev.ref = rng.randrange(i)
            else:
                ev.ref = rng.randrange(i + 1, K)

        order = list(range(K))
        rng.shuffle(order)            # statements are told out of order
        ctx = {p: known_order(events, order[:p]) for p in range(2, K + 1)}

        questions = {}                # p -> [(text, answer)], asked after p statements
        asked = set()
        wanted = rng.randint(1, 5)
        found = 0
        for _ in range(60):
            if found >= wanted:
                break
            p = rng.randint(2, K)
            mentioned, rows = ctx[p]
            q = make_question(rng, events, people, order[:p], mentioned, rows)
            if q is None or q[0] in asked:
                continue
            asked.add(q[0])
            questions.setdefault(p, []).append(q)
            found += 1
        if found == 0:
            continue
        return events, order, questions


def render(story):
    events, order, questions = story
    out = []
    for p in range(1, len(order) + 1):
        out.append(statement(events[order[p - 1]], events))
        for text, ans in questions.get(p, []):
            out.append(f"{text}\t{ans}")
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: python gen.py N SEED", file=sys.stderr)
        sys.exit(2)
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    for _ in range(n):
        for i, line in enumerate(render(make_story(rng)), 1):
            print(f"{i} {line}")


if __name__ == "__main__":
    main()
