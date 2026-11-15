#!/usr/bin/env python3
"""Synthetic time-order reading comprehension stories (standard library only).

Usage: python gen.py N SEED

Each story is a set of events "name went to the place at hour", with distinct
hours, told in shuffled order. Questions ask where someone was at an hour,
whether someone went to a place before an hour, who went to a place first, or
who was at a place at an hour. Answers come from simulating the events shown
above the question. A question is kept only if that answer equals the answer
computed from the whole story, so the answer never depends on later statements.
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan"]
PLACES = ["park", "shop", "lake", "school"]
HOURS = list(range(1, 13))


def statement(ev, rng):
    t, n, p = ev
    if rng.random() < 0.5:
        return "At %d %s went to the %s." % (t, n, p)
    return "%s went to the %s at %d." % (n, p, t)


def random_question(rng):
    kind = rng.choice(["where", "did", "first", "who"])
    if kind == "where":
        args = (rng.choice(NAMES), rng.choice(HOURS))
    elif kind == "did":
        args = (rng.choice(NAMES), rng.choice(PLACES), rng.choice(HOURS))
    elif kind == "first":
        args = (rng.choice(PLACES),)
    else:
        args = (rng.choice(PLACES), rng.choice(HOURS))
    return kind, args


def question_text(q):
    kind, a = q
    if kind == "where":
        return "Where was %s at %d?" % a
    if kind == "did":
        return "Did %s go to the %s before %d?" % a
    if kind == "first":
        return "Who went to the %s first?" % a
    return "Who was at the %s at %d?" % a


def answer(q, evs):
    """Answer computed by simulating the events in evs (each is (hour, name, place))."""
    kind, a = q
    if kind == "where":
        n, h = a
        c = [e for e in evs if e[1] == n and e[0] <= h]
        return max(c)[2] if c else "nowhere"
    if kind == "did":
        n, p, h = a
        hit = any(e[1] == n and e[2] == p and e[0] < h for e in evs)
        return "yes" if hit else "no"
    if kind == "first":
        p = a[0]
        c = [e for e in evs if e[2] == p]
        return min(c)[1] if c else None
    p, h = a
    there = []
    for n in NAMES:
        c = [e for e in evs if e[1] == n and e[0] <= h]
        if c and max(c)[2] == p:
            there.append(n)
    return ",".join(there) if there else "nobody"


def make_story(rng):
    while True:
        n = rng.randint(3, 12)
        events = [(t, rng.choice(NAMES), rng.choice(PLACES))
                  for t in rng.sample(HOURS, n)]
        rng.shuffle(events)
        want = rng.randint(1, 5)
        lines = []
        shown = []
        asked = 0
        for i, ev in enumerate(events):
            shown.append(ev)
            lines.append(statement(ev, rng))
            if asked < want and (i == n - 1 or rng.random() < 0.4):
                for _ in range(60):
                    q = random_question(rng)
                    a = answer(q, shown)
                    if a is not None and a == answer(q, events):
                        lines.append("%s\t%s" % (question_text(q), a))
                        asked += 1
                        break
        if asked:
            return lines


def main():
    count, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        for k, line in enumerate(make_story(rng), 1):
            out.append("%d %s" % (k, line))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
