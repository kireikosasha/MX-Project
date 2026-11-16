"""Synthetic task: order in time.

Each story has a hidden timeline of 3-6 events (a person going to a place).
Events are told out of order: some with their exact time, some relative to
another event ("before" / "after"), some with no order information.
Questions come after all statements and are kept only when the answer is
forced by the statements (checked by enumerating every timeline that fits)
and not forced by any single statement alone.

Usage: python gen.py N SEED
"""

import itertools
import random
import sys

NAMES = ["ann", "ben", "cal", "dee", "eli", "fay"]
PLACES = ["park", "cafe", "shop", "pool"]
MAX_EVENTS = 6

_PERMS = {}


def perms(n):
    if n not in _PERMS:
        _PERMS[n] = list(itertools.permutations(range(1, n + 1)))
    return _PERMS[n]


def make_world(rng):
    n = rng.randint(3, MAX_EVENTS)
    people = rng.sample(NAMES, rng.randint(2, 4))
    pairs = set()
    events = []
    while len(events) < n:
        pair = (rng.choice(people), rng.choice(PLACES))
        if pair not in pairs:
            pairs.add(pair)
            events.append(pair)
    times = list(range(1, n + 1))
    rng.shuffle(times)  # times[i] is the true time of event i
    return events, times


def make_statements(rng, events, times):
    n = len(events)
    stmts = []
    for i in range(n):
        name, place = events[i]
        r = rng.random()
        if r < 0.3:
            text = "at time %d %s went to the %s." % (times[i], name, place)
            cons = ("time", i, times[i])
        elif r < 0.85:
            j = rng.choice([k for k in range(n) if k != i])
            other_name, other_place = events[j]
            word = "after" if times[i] > times[j] else "before"
            text = "%s went to the %s %s %s went to the %s." % (
                name, place, word, other_name, other_place)
            cons = ("rel", i, j, word)
        else:
            text = "%s went to the %s." % (name, place)
            cons = ("none",)
        stmts.append((text, cons))
    rng.shuffle(stmts)
    return stmts


def satisfies(pos, cons):
    kind = cons[0]
    if kind == "time":
        return pos[cons[1]] == cons[2]
    if kind == "rel":
        i, j, word = cons[1], cons[2], cons[3]
        return pos[i] > pos[j] if word == "after" else pos[i] < pos[j]
    return True


def where(pos, events, name, t):
    best = None
    for i, (n, _) in enumerate(events):
        if n == name and pos[i] <= t and (best is None or pos[i] > pos[best]):
            best = i
    return None if best is None else events[best][1]


def who_in(pos, events, place, t):
    res = []
    for name in sorted(set(n for n, _ in events)):
        if where(pos, events, name, t) == place:
            res.append(name)
    return ",".join(res) if res else None


def first_at(pos, events, place, last):
    idxs = [i for i, (_, p) in enumerate(events) if p == place]
    if not idxs:
        return None
    if last:
        pick = max(idxs, key=lambda i: pos[i])
    else:
        pick = min(idxs, key=lambda i: pos[i])
    return events[pick][0]


def candidates(events):
    cands = []
    n = len(events)
    names = sorted(set(nm for nm, _ in events))
    places = sorted(set(p for _, p in events))
    for name in names:
        for t in range(1, n + 1):
            cands.append(("where was %s at time %d?" % (name, t),
                          lambda pos, name=name, t=t: where(pos, events, name, t)))
    for place in places:
        for t in range(1, n + 1):
            cands.append(("who was in the %s at time %d?" % (place, t),
                          lambda pos, place=place, t=t: who_in(pos, events, place, t)))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = events[i], events[j]
            for word in ("before", "after"):
                text = "did %s go to the %s %s %s went to the %s?" % (
                    a[0], a[1], word, b[0], b[1])

                def ans_fn(pos, i=i, j=j, word=word):
                    lt = pos[i] < pos[j]
                    ans = lt if word == "before" else not lt
                    return "yes" if ans else "no"
                cands.append((text, ans_fn))
    for place in places:
        for last in (False, True):
            word = "last" if last else "first"
            cands.append(("who went to the %s %s?" % (place, word),
                          lambda pos, place=place, last=last: first_at(pos, events, place, last)))
    return cands


def make_story(rng):
    while True:
        events, times = make_world(rng)
        stmts = make_statements(rng, events, times)
        n = len(events)
        cons = [c for _, c in stmts]
        all_p = perms(n)
        sols = [p for p in all_p if all(satisfies(p, c) for c in cons)]
        singles = [[p for p in all_p if satisfies(p, c)] for c in cons]
        truth = tuple(times)

        cands = candidates(events)
        rng.shuffle(cands)
        target = rng.randint(1, 5)
        chosen = []
        seen = set()
        for text, f in cands:
            if len(chosen) >= target:
                break
            if text in seen:
                continue
            ans = f(truth)
            if ans is None:
                continue
            if any(f(p) != ans for p in sols):
                continue  # not forced by all statements
            if any(all(f(p) == ans for p in s) for s in singles):
                continue  # forced by one statement alone
            seen.add(text)
            chosen.append((text, ans))
        if chosen:
            return stmts, chosen


def format_story(stmts, questions):
    lines = []
    k = 0
    for text, _ in stmts:
        k += 1
        lines.append("%d %s" % (k, text))
    for text, ans in questions:
        k += 1
        lines.append("%d %s\t%s" % (k, text, ans))
    return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(2)
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        stmts, questions = make_story(rng)
        out.extend(format_story(stmts, questions))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
