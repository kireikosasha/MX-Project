"""Synthetic order-in-time reading task generator.

Usage: python gen.py N SEED

Each story is a timeline of k events (a person goes to a place). Events are
told out of order: some carry an explicit time ("at time T ..."), the others
are placed only relative to another event ("... after/before ..."). Questions
ask where someone was at a time, whether someone was in a place at a time,
who went to a place first, or who was in a place at a time. Every answer is
computed from the true timeline and checked to be the same under every
timeline that is consistent with the statements, so it is unambiguous.
"""
import itertools
import random
import sys

NAMES = ["alice", "bob", "carol", "dan", "eve"]
PLACES = ["kitchen", "garden", "office", "park", "hall"]


def consistent_orders(k, labelled, relations):
    """All timelines (event index -> position 1..k) that fit the statements.

    labelled: dict event -> fixed position (from "at time T").
    relations: list of (later, earlier) event pairs (from "after"/"before").
    """
    used = set(labelled.values())
    free_positions = [p for p in range(1, k + 1) if p not in used]
    free_events = [e for e in range(k) if e not in labelled]
    result = []
    for perm in itertools.permutations(free_events):
        pos = dict(labelled)
        for e, p in zip(perm, free_positions):
            pos[e] = p
        if all(pos[a] > pos[b] for a, b in relations):
            result.append(pos)
    return result


def state_at(tl, person, t):
    """Place of person after the event at position t (None if never moved)."""
    place = None
    for p in range(1, t + 1):
        if tl[p][0] == person:
            place = tl[p][1]
    return place


def answer(kind, tl, k, people, x, p_place, t):
    if kind == 0:  # where was X at time T
        return state_at(tl, x, t) or "nowhere"
    if kind == 1:  # was X in the P at time T
        return "yes" if state_at(tl, x, t) == p_place else "no"
    if kind == 2:  # who went to the P first
        for p in range(1, k + 1):
            if tl[p][1] == p_place:
                return tl[p][0]
        return "nobody"
    # who was in the P at time T
    names = [n for n in people if state_at(tl, n, t) == p_place]
    return ",".join(sorted(names)) if names else "nobody"


def make_story(rng):
    k = rng.randint(3, 9)

    # True timeline: event i happens at position i + 1. (person, place) pairs are unique.
    events = []
    seen = set()
    while len(events) < k:
        pair = (rng.choice(NAMES), rng.choice(PLACES))
        if pair not in seen:
            seen.add(pair)
            events.append(pair)

    # At least half of the events carry a time; at least one does not.
    n_lab = rng.randint((k + 1) // 2, k - 1)
    labelled_ids = sorted(rng.sample(range(k), n_lab))
    labelled = {i: i + 1 for i in labelled_ids}

    statements = {}
    relations = []
    for i in range(k):
        name, place = events[i]
        if i in labelled:
            statements[i] = "at time %d %s went to the %s." % (i + 1, name, place)
        else:
            j = rng.choice([x for x in range(k) if x != i])
            other_name, other_place = events[j]
            if i > j:
                relations.append((i, j))
                word = "after"
            else:
                relations.append((j, i))
                word = "before"
            statements[i] = "%s went to the %s %s %s went to the %s." % (
                name, place, word, other_name, other_place)

    orders = consistent_orders(k, labelled, relations)
    people = sorted({n for n, _ in events})

    tell_order = list(range(k))
    rng.shuffle(tell_order)

    questions = []
    seen_q = set()
    n_q = rng.randint(1, 5)
    attempts = 0
    while len(questions) < n_q and attempts < 100:
        attempts += 1
        kind = rng.randrange(4)
        x = rng.choice(people)
        place = rng.choice(PLACES)
        t = rng.randint(1, k)
        if kind == 0:
            text = "where was %s at time %d?" % (x, t)
        elif kind == 1:
            text = "was %s in the %s at time %d?" % (x, place, t)
        elif kind == 2:
            text = "who went to the %s first?" % place
        else:
            text = "who was in the %s at time %d?" % (place, t)
        if text in seen_q:
            continue
        answers = set()
        for pos in orders:
            tl = [None] * (k + 1)
            for e, p in pos.items():
                tl[p] = events[e]
            answers.add(answer(kind, tl, k, people, x, place, t))
            if len(answers) > 1:
                break
        if len(answers) != 1:
            continue
        seen_q.add(text)
        questions.append((text, answers.pop()))

    if not questions:
        return None

    lines = []
    for num, i in enumerate(tell_order, 1):
        lines.append("%d %s" % (num, statements[i]))
    for num, (text, ans) in enumerate(questions, k + 1):
        lines.append("%d %s\t%s" % (num, text, ans))
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = make_story(rng)
        out.extend(story)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
