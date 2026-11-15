"""Synthetic order-in-time reading task generator.

Usage: python gen.py N SEED

Each story has a few events of the form "X went to the P". Events are told out
of order: either with an absolute time slot ("at time T X went to the P.") or
relative to another event ("X went to the P after Y went to the Q.").
Questions ask where someone was at a time, who was in a place at a time, where
someone was just before an event, or whether one event came after another.
Answers are computed from the true timeline and checked by brute force: a
question is only kept if all orders of the story's events that agree with the
statements above it give the same answer, and no single statement alone does.
"""
import itertools
import random
import sys

PERSONS = ["ann", "bob", "cy", "dan", "eve"]
PLACES = ["hall", "garden", "kitchen", "office", "park"]

_PERMS = {}


def perms_for(n):
    if n not in _PERMS:
        _PERMS[n] = list(itertools.permutations(range(1, n + 1)))
    return _PERMS[n]


def last_place(events, pos, person, bound, strict):
    """Place of the person's last event at or before (or strictly before) bound."""
    best, best_pos = "none", 0
    for i, (pn, pl) in enumerate(events):
        if pn != person:
            continue
        p = pos[i]
        ok = p < bound if strict else p <= bound
        if ok and p > best_pos:
            best, best_pos = pl, p
    return best


def answer(spec, pos, events, persons):
    kind = spec[0]
    if kind == "where_at":
        return last_place(events, pos, spec[1], spec[2], False)
    if kind == "who_in":
        place, t = spec[1], spec[2]
        names = [p for p in persons if last_place(events, pos, p, t, False) == place]
        return ",".join(names) if names else "none"
    if kind == "where_before":
        return last_place(events, pos, spec[1], pos[spec[2]], True)
    return "yes" if pos[spec[1]] > pos[spec[2]] else "no"


def question_text(spec, events):
    kind = spec[0]
    if kind == "where_at":
        return f"where was {spec[1]} at time {spec[2]}?"
    if kind == "who_in":
        return f"who was in the {spec[1]} at time {spec[2]}?"
    if kind == "where_before":
        pn, pl = events[spec[2]]
        return f"where was {spec[1]} before {pn} went to the {pl}?"
    pa, la = events[spec[1]]
    pb, lb = events[spec[2]]
    return f"did {pa} go to the {la} after {pb} went to the {lb}?"


def random_spec(rng, events, persons, ne):
    kind = rng.choice(["where_at", "who_in", "where_before", "did"])
    if kind == "where_at":
        return ("where_at", rng.choice(persons), rng.randint(1, ne))
    if kind == "who_in":
        return ("who_in", rng.choice(PLACES), rng.randint(1, ne))
    if kind == "where_before":
        e = rng.randrange(len(events))
        others = [p for p in persons if p != events[e][0]]
        if not others:
            return None
        return ("where_before", rng.choice(others), e)
    a, b = rng.sample(range(len(events)), 2)
    return ("did", a, b)


def sat(st, pos):
    if st[0] == "t":
        return pos[st[1]] == st[2]
    return pos[st[1]] > pos[st[2]]


def try_story(rng):
    ne = rng.randint(2, 5)
    events = rng.sample([(p, q) for p in PERSONS for q in PLACES], ne)
    persons = sorted({p for p, _ in events})
    slots = list(range(1, ne + 1))
    rng.shuffle(slots)
    pos = tuple(slots)  # pos[i] = time slot of event i

    # Candidate statements that are true in the hidden timeline.
    pool = [("t", i, pos[i]) for i in range(ne)]
    for i in range(ne):
        for j in range(i + 1, ne):
            pool.append(("r", i, j) if pos[i] > pos[j] else ("r", j, i))
    if len(pool) < 3:
        return None
    n = rng.randint(3, min(12, len(pool)))
    chosen = rng.sample(pool, n)
    mentioned = set()
    for st in chosen:
        mentioned.add(st[1])
        if st[0] == "r":
            mentioned.add(st[2])
    if len(mentioned) < ne or all(st[0] == "t" for st in chosen):
        return None
    rng.shuffle(chosen)

    stmts = []
    for st in chosen:
        if st[0] == "t":
            pn, pl = events[st[1]]
            text = f"at time {st[2]} {pn} went to the {pl}."
        else:
            (pa, la), (pb, lb) = events[st[1]], events[st[2]]
            if rng.random() < 0.5:
                text = f"{pa} went to the {la} after {pb} went to the {lb}."
            else:
                text = f"{pb} went to the {lb} before {pa} went to the {la}."
        stmts.append((st[0], st[1], st[2], text))

    # Consistent timelines after the first k statements.
    perms = perms_for(ne)
    satvec = [[sat(st, p) for p in perms] for st in stmts]
    cons = [list(range(len(perms)))]
    for sv in satvec:
        cons.append([i for i in cons[-1] if sv[i]])

    seen_ev = set()
    cover = None
    for k, st in enumerate(stmts, 1):
        seen_ev.add(st[1])
        if st[0] == "r":
            seen_ev.add(st[2])
        if len(seen_ev) == ne:
            cover = k
            break
    if cover is None:
        return None

    # Questions: only after every event has been introduced.
    target = rng.randint(1, 5)
    questions = []
    texts = set()
    for _ in range(15):
        if len(questions) == target:
            break
        idx = rng.randint(cover, n)
        spec = random_spec(rng, events, persons, ne)
        if spec is None:
            continue
        text = question_text(spec, events)
        if text in texts:
            continue
        vals = [answer(spec, perms[i], events, persons) for i in range(len(perms))]
        ok_set = {vals[i] for i in cons[idx]}
        if len(ok_set) != 1:
            continue
        # No single statement above may already settle the answer.
        if any(len({vals[i] for i in range(len(perms)) if satvec[s][i]}) == 1
               for s in range(idx)):
            continue
        ans = answer(spec, pos, events, persons)
        assert ans in ok_set
        texts.add(text)
        questions.append((idx, text, ans))
    if not questions:
        return None

    lines = []
    for k in range(n + 1):
        for idx, text, ans in questions:
            if idx == k:
                lines.append(f"{text}\t{ans}")
        if k < n:
            lines.append(stmts[k][3])
    return [f"{i} {line}" for i, line in enumerate(lines, 1)]


def make_story(rng):
    while True:
        story = try_story(rng)
        if story is not None:
            return story


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n_stories):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
