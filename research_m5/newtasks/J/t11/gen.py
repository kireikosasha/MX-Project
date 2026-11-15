#!/usr/bin/env python3
"""Synthetic order-in-time reading stories.

Usage: python gen.py N SEED

A story is a set of moves ("Sam went to the kitchen"). Every move happens at
a different time 1..m. Statements give a move's time exactly ("at time 3") or
say that it came before or after another move. Questions are asked about the
state of the world at a time, or about the order of two moves. A question is
only asked when every assignment of times that fits the statements above it
gives the same answer, and no single statement alone already fixes it.
"""
import itertools
import random
import sys

NAMES = ["Sam", "Ann", "Ben", "Tom", "Mia", "Joe", "Kim"]
PLACES = ["kitchen", "garden", "office", "park", "hall"]
MIN_EVENTS = 3
MAX_EVENTS = 5
MAX_QUESTIONS = 5

# PERMS[m][idx][e] is the time (1..m) of move e under assignment idx.
PERMS = {m: list(itertools.permutations(range(1, m + 1)))
         for m in range(MIN_EVENTS, MAX_EVENTS + 1)}


def location(perm, events, person, when):
    """Place where person was at time `when`, or None if no move has happened."""
    best_place, best_time = None, 0
    for e, (name, place) in enumerate(events):
        t = perm[e]
        if name == person and best_time < t <= when:
            best_place, best_time = place, t
    return best_place


def answer(q, perm, events, persons):
    kind = q[0]
    if kind == "where":
        return location(perm, events, q[1], q[2])
    if kind == "was":
        return "yes" if location(perm, events, q[1], q[3]) == q[2] else "no"
    if kind == "who":
        inside = sorted(p.lower() for p in persons
                        if location(perm, events, p, q[2]) == q[1])
        return ",".join(inside)
    if kind == "when":
        return str(perm[q[1]])
    # "order": was move e1 before (after=False) or after (after=True) move e2?
    e1, e2, after = q[1], q[2], q[3]
    return "yes" if (perm[e1] > perm[e2]) == after else "no"


def question_text(q, events):
    kind = q[0]
    if kind == "where":
        return "Where was %s at time %d?" % (q[1], q[2])
    if kind == "was":
        return "Was %s in the %s at time %d?" % (q[1], q[2], q[3])
    if kind == "who":
        return "Who was in the %s at time %d?" % (q[1], q[2])
    if kind == "when":
        name, place = events[q[1]]
        return "When did %s go to the %s?" % (name, place)
    n1, p1 = events[q[1]]
    n2, p2 = events[q[2]]
    word = "after" if q[3] else "before"
    return "Did %s go to the %s %s %s went to the %s?" % (n1, p1, word, n2, p2)


def statement_text(st, events):
    kind, i, j, val = st
    name, place = events[i]
    if kind == "time":
        return "%s went to the %s at time %d." % (name, place, val)
    other, other_place = events[j]
    word = "after" if val else "before"
    return "%s went to the %s %s %s went to the %s." % (
        name, place, word, other, other_place)


def distinct_answers(q, indices, perms, events, persons, limit=2):
    """Answers q takes over the given assignments; stops early at `limit`."""
    seen = set()
    for idx in indices:
        seen.add(answer(q, perms[idx], events, persons))
        if len(seen) >= limit:
            break
    return seen


def determined(q, alive, sat_prefix, perms, events, persons):
    """The answer to q if the statements above fix it, else None."""
    seen = distinct_answers(q, alive, perms, events, persons)
    if len(seen) != 1:
        return None
    (value,) = seen
    if not value:  # None (no move yet) or empty list
        return None
    # The question must not be guessable from nothing at all ...
    if len(distinct_answers(q, range(len(perms)), perms, events, persons)) < 2:
        return None
    # ... and no single statement may already settle it.
    for s in sat_prefix:
        if len(distinct_answers(q, s, perms, events, persons)) < 2:
            return None
    return value


def pick_question(rng, alive, sat_prefix, perms, events, persons, mentioned, m):
    people = sorted({events[e][0] for e in mentioned})
    cands = []
    for x in people:
        for t in range(1, m + 1):
            cands.append(("where", x, t))
            for pl in PLACES:
                cands.append(("was", x, pl, t))
    for pl in PLACES:
        for t in range(1, m + 1):
            cands.append(("who", pl, t))
    ev = sorted(mentioned)
    for e in ev:
        cands.append(("when", e))
    for e1 in ev:
        for e2 in ev:
            if e1 != e2:
                cands.append(("order", e1, e2, False))
                cands.append(("order", e1, e2, True))
    rng.shuffle(cands)
    for q in cands:
        value = determined(q, alive, sat_prefix, perms, events, persons)
        if value is not None:
            return question_text(q, events), value
    return None


def make_story(rng):
    """Return a list of (text, answer-or-None) lines for one story."""
    while True:
        m = rng.randint(MIN_EVENTS, MAX_EVENTS)
        perms = PERMS[m]
        persons = rng.sample(NAMES, rng.randint(2, 4))
        pairs = [(name, place) for name in persons for place in PLACES]
        events = rng.sample(pairs, m)
        truth = perms[rng.randrange(len(perms))]

        # One statement per move: its exact time, or its order relative to another move.
        stmts = []
        for i in range(m):
            if rng.random() < 0.5:
                stmts.append(("time", i, None, truth[i]))
            else:
                j = rng.choice([x for x in range(m) if x != i])
                stmts.append(("rel", i, j, truth[i] > truth[j]))
        rng.shuffle(stmts)  # told out of order

        sat = []
        for kind, i, j, val in stmts:
            if kind == "time":
                sat.append([idx for idx, pm in enumerate(perms) if pm[i] == val])
            else:
                sat.append([idx for idx, pm in enumerate(perms)
                            if (pm[i] > pm[j]) == val])

        alive = list(range(len(perms)))
        mentioned = set()
        out = []
        questions = 0
        for k, st in enumerate(stmts):
            _, i, j, _ = st
            keep = set(sat[k])
            alive = [idx for idx in alive if idx in keep]
            mentioned.add(i)
            if j is not None:
                mentioned.add(j)
            out.append((statement_text(st, events), None))
            last = k == len(stmts) - 1
            if questions < MAX_QUESTIONS and (last or rng.random() < 0.5):
                found = pick_question(rng, alive, sat[:k + 1], perms, events,
                                      persons, mentioned, m)
                if found is not None:
                    out.append(found)
                    questions += 1
        if questions:
            return out


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 2
    n, seed = int(argv[1]), int(argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for num, (text, ans) in enumerate(make_story(rng), start=1):
            if ans is None:
                lines.append("%d %s" % (num, text))
            else:
                lines.append("%d %s\t%s" % (num, text, ans))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
