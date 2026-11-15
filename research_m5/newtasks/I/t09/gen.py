"""Synthetic order-in-time reading task generator (standard library only).

Usage: python gen.py N SEED

Each story: people visit places. Each visit (event) is told only by relative
order ("X went to the l after Y went to the m") or by the start place of a
person, in shuffled order. Questions at the end ask where someone was before
or after an event, who was in a place after an event, or whether one event
came before another. The true order is simulated; a question is kept only if
its answer is the same for every ordering consistent with the statements.
"""
import itertools
import random
import sys

NAMES = ["ann", "bob", "cal", "dee", "eve"]
PLACES = ["hall", "kitchen", "garden", "park", "office"]


def ev(e):
    """Sentence fragment for event e = (person, place)."""
    return f"{e[0]} went to the {e[1]}"


def place_at(pos, events, starts, person, x, inclusive):
    """Place of person just before event x (just after it if inclusive)."""
    best_pos, best_place = -1, starts[person]
    for i, (who, place) in enumerate(events):
        if who != person:
            continue
        if pos[i] < pos[x] or (inclusive and pos[i] == pos[x]):
            if pos[i] > best_pos:
                best_pos, best_place = pos[i], place
    return best_place


def answer(q, pos, events, starts, people):
    kind = q[0]
    if kind == "where":
        _, person, x, inclusive = q
        return place_at(pos, events, starts, person, x, inclusive)
    if kind == "who":
        _, place, x = q
        found = sorted(p for p in people
                       if place_at(pos, events, starts, p, x, True) == place)
        return ",".join(found) if found else "nobody"
    _, i, x = q  # "did"
    return "yes" if pos[i] < pos[x] else "no"


def question_text(q, events):
    kind = q[0]
    if kind == "where":
        _, person, x, inclusive = q
        word = "after" if inclusive else "before"
        return f"where was {person} {word} {ev(events[x])}?"
    if kind == "who":
        _, place, x = q
        return f"who was in the {place} after {ev(events[x])}?"
    _, i, x = q
    p, l = events[i]
    return f"did {p} go to the {l} before {ev(events[x])}?"


def try_story(rng):
    """Return the lines of one story, or None if this draw does not work."""
    k = rng.randint(2, 4)
    people = rng.sample(NAMES, k)
    n = rng.randint(2, 5)
    events = []
    while len(events) < n:
        e = (rng.choice(people), rng.choice(PLACES))
        if e not in events:
            events.append(e)
    true_pos = list(range(n))
    rng.shuffle(true_pos)  # true_pos[i] = true time slot of event i
    starts = {p: rng.choice(PLACES) for p in people}

    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(pairs)
    r = min(len(pairs), rng.randint((n + 1) // 2, n + 2))
    chosen = pairs[:r]
    if len({i for pr in chosen for i in pr}) < n:
        return None  # every event must be mentioned

    stmts = [f"{p} started in the {starts[p]}." for p in people]
    rels = []  # (earlier, later) pairs of event indices, true order
    for a, b in chosen:
        e, l = (a, b) if true_pos[a] < true_pos[b] else (b, a)
        if rng.random() < 0.5:
            stmts.append(f"{ev(events[l])} after {ev(events[e])}.")
        else:
            stmts.append(f"{ev(events[e])} before {ev(events[l])}.")
        rels.append((e, l))
    if not 3 <= len(stmts) <= 12:
        return None
    rng.shuffle(stmts)

    # All orderings of the events consistent with the statements.
    consistent = [perm for perm in itertools.permutations(range(n))
                  if all(perm[e] < perm[l] for e, l in rels)]

    cands = []
    for x in range(n):
        for p in people:
            cands.append(("where", p, x, False))
            cands.append(("where", p, x, True))
        for place in PLACES:
            cands.append(("who", place, x))
        for i in range(n):
            if i != x:
                cands.append(("did", i, x))

    valid = []
    for q in cands:
        answers = {answer(q, perm, events, starts, people) for perm in consistent}
        if len(answers) == 1:
            valid.append((q, answers.pop()))
    if not valid:
        return None

    # The answers are computed from the true simulation (true_pos); the
    # uniqueness check above shows they are the only possible answers.
    picked = rng.sample(valid, rng.randint(1, min(5, len(valid))))
    lines = [f"{i} {s}" for i, s in enumerate(stmts, 1)]
    m = len(stmts)
    for j, (q, _) in enumerate(picked, m + 1):
        ans = answer(q, true_pos, events, starts, people)
        lines.append(f"{j} {question_text(q, events)}\t{ans}")
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 2
    count = int(argv[1])
    rng = random.Random(int(argv[2]))
    out = []
    for _ in range(count):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
