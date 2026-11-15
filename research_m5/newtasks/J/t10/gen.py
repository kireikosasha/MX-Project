#!/usr/bin/env python3
"""Synthetic time-order reading tasks.

Usage: python gen.py N SEED

Each story is a set of visits (person went to a place). Statements are told
out of order, either with a clock time ("at time t", where t is the visit's
rank in the hidden timeline) or relative to another visit ("before" / "after").
Questions are answered by brute force over all timelines consistent with the
statements above them; a question is kept only if its answer is the same for
every consistent timeline, and not determined by any single statement.
"""
import itertools
import random
import sys

NAMES = ["ann", "bob", "cy", "dee", "eli", "fay", "gus"]
PLACES = ["park", "shop", "school", "farm", "lake"]
PAIRS = [(n, p) for n in NAMES for p in PLACES]
MAX_STATEMENTS = 12
MAX_QUESTIONS = 5


def mask_of(perms, pred):
    """Bitmask over timelines (perms) that satisfy pred."""
    m = 0
    for i, r in enumerate(perms):
        if pred(r):
            m |= 1 << i
    return m


def determined(mask, ans, perms):
    """The one answer shared by all timelines in mask, or None if ambiguous/invalid."""
    found = None
    while mask:
        low = mask & -mask
        mask ^= low
        a = ans(perms[low.bit_length() - 1])
        if a is None or (found is not None and a != found):
            return None
        found = a
    return found


def make_question(rng, events):
    """Return (question text, answer function over a timeline r), or None."""
    k = len(events)
    ev = range(k)
    kind = rng.randrange(6)
    if kind == 0:
        p = events[rng.randrange(k)][0]
        t = rng.randint(1, k)
        text = f"where was {p} at time {t}?"

        def ans(r):
            own = [e for e in ev if events[e][0] == p and r[e] <= t]
            return events[max(own, key=lambda e: r[e])][1] if own else None
    elif kind == 1 or kind == 5:
        f = rng.randrange(k)
        q, m = events[f]
        others = [x for x, _ in events if x != q]
        if not others:
            return None
        p = rng.choice(others)
        if kind == 1:
            text = f"where was {p} before {q} went to the {m}?"

            def ans(r):
                own = [e for e in ev if events[e][0] == p and r[e] < r[f]]
                return events[max(own, key=lambda e: r[e])][1] if own else None
        else:
            text = f"which places did {p} visit before {q} went to the {m}?"

            def ans(r):
                own = sorted({events[e][1] for e in ev
                              if events[e][0] == p and r[e] < r[f]})
                return ",".join(own) if own else None
    elif kind == 2:
        a, b = rng.sample(list(ev), 2)
        (pa, la), (pb, lb) = events[a], events[b]
        text = f"did {pa} go to the {la} before {pb} went to the {lb}?"

        def ans(r):
            return "yes" if r[a] < r[b] else "no"
    elif kind == 3:
        place = events[rng.randrange(k)][1]
        text = f"who went to the {place} first?"

        def ans(r):
            at = [e for e in ev if events[e][1] == place]
            return events[min(at, key=lambda e: r[e])][0]
    else:
        e = rng.randrange(k)
        p, place = events[e]
        text = f"what time did {p} go to the {place}?"

        def ans(r):
            return str(r[e])
    return text, ans


def make_valid_question(rng, events, perms, truth, masks, used):
    full = (1 << len(perms)) - 1
    for m in masks:
        full &= m
    for _ in range(300):
        built = make_question(rng, events)
        if built is None:
            continue
        text, ans = built
        if text in used:
            continue
        value = determined(full, ans, perms)
        if value is None or value != ans(truth):
            continue
        if any(determined(m, ans, perms) is not None for m in masks):
            continue
        used.add(text)
        return text, value
    return None


def try_story(rng):
    k = rng.randint(3, 5)
    events = rng.sample(PAIRS, k)
    perms = list(itertools.permutations(range(1, k + 1)))
    truth = rng.choice(perms)  # truth[e] = rank (clock time) of visit e

    # Each candidate: (text, predicate over timelines, visits it mentions).
    cands = []
    for e in range(k):
        p, place = events[e]
        cands.append((f"{p} went to the {place} at time {truth[e]}.",
                      lambda r, e=e: r[e] == truth[e], {e}))
    for a in range(k):
        for b in range(k):
            if a == b or truth[a] > truth[b]:
                continue
            pa, la = events[a]
            pb, lb = events[b]
            cands.append((f"{pa} went to the {la} before {pb} went to the {lb}.",
                          lambda r, a=a, b=b: r[a] < r[b], {a, b}))
            cands.append((f"{pb} went to the {lb} after {pa} went to the {la}.",
                          lambda r, a=a, b=b: r[a] < r[b], {a, b}))

    n = rng.randint(3, min(MAX_STATEMENTS, len(cands)))
    chosen = rng.sample(cands, n)  # random order: events told out of order
    stmts = [c[0] for c in chosen]
    masks = [mask_of(perms, c[1]) for c in chosen]

    # A question may only use visits that the statements above it have mentioned,
    # so the reader knows the full set of visits. Find the first prefix that
    # mentions every visit; questions are placed at or after it.
    seen, cover = set(), None
    for i, c in enumerate(chosen, 1):
        seen |= c[2]
        if len(seen) == k:
            cover = i
            break
    if cover is None:
        return None
    lo = max(2, cover)
    if lo > n:
        return None
    prefixes = sorted(rng.randint(lo, n) for _ in range(rng.randint(1, MAX_QUESTIONS)))
    entries = []
    used = set()
    qi = 0
    for i in range(n + 1):
        while qi < len(prefixes) and prefixes[qi] == i:
            q = make_valid_question(rng, events, perms, truth, masks[:i], used)
            if q is None:
                return None
            entries.append(q)
            qi += 1
        if i < n:
            entries.append((stmts[i], None))
    return entries


def make_story(rng):
    while True:
        entries = try_story(rng)
        if entries is not None:
            return entries


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(2)
    n_stories = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n_stories):
        for idx, (text, answer) in enumerate(make_story(rng), 1):
            if answer is None:
                out.append(f"{idx} {text}")
            else:
                out.append(f"{idx} {text}\t{answer}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
