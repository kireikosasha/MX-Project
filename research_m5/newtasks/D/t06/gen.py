"""Synthetic order-in-time reading comprehension generator.

Usage: python gen.py N SEED

Events are (person, place) pairs. The story hides a true timeline of its
events and only reveals ordering facts ("X went to the P before Y went to
the Q"), so the reader must chain facts to place events in time. Every
question is checked by enumerating all event orders consistent with the
statements above it; a question is kept only if all consistent orders give
the same answer and the answer would differ without the ordering facts.
"""
import random
import sys
from itertools import permutations

NAMES = ["ann", "bob", "cat", "dan", "eve"]
PLACES = ["kitchen", "garden", "hall", "office", "park"]


def sentence(text):
    return text[0].upper() + text[1:]


def intro_fact(e):
    return ("intro", e, None, f"{e[0]} went to the {e[1]}.")


def rel_fact(rng, a, b):
    """Fact saying event a happened before event b."""
    if rng.random() < 0.5:
        text = f"{a[0]} went to the {a[1]} before {b[0]} went to the {b[1]}."
    else:
        text = f"{b[0]} went to the {b[1]} after {a[0]} went to the {a[1]}."
    return ("rel", a, b, text)


def ordered_rel(rng, tpos, x, y):
    a, b = (x, y) if tpos[x] < tpos[y] else (y, x)
    return rel_fact(rng, a, b)


def consistent(order, rels):
    pos = {e: i for i, e in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in rels)


def answer(order, q):
    """Answer to question q given a total order of events (earliest first)."""
    pos = {e: i for i, e in enumerate(order)}
    kind = q[0]
    if kind == "when":
        p, e = q[1], q[2]
        earlier = [x for x in order if x[0] == p and pos[x] < pos[e]]
        return earlier[-1][1] if earlier else None
    if kind == "first":
        mine = [x for x in order if x[0] == q[1]]
        return mine[0][1] if mine else None
    if kind == "whofirst":
        mine = [x for x in order if x[1] == q[1]]
        return mine[0][0] if mine else None
    if kind == "wholast":
        mine = [x for x in order if x[1] == q[1]]
        return mine[-1][0] if mine else None
    # kind == "order": did x happen before y
    return "yes" if pos[q[1]] < pos[q[2]] else "no"


def question_text(q):
    kind = q[0]
    if kind == "when":
        return f"Where was {q[1]} when {q[2][0]} went to the {q[2][1]}?"
    if kind == "first":
        return f"Where did {q[1]} go first?"
    if kind == "whofirst":
        return f"Who went to the {q[1]} first?"
    if kind == "wholast":
        return f"Who went to the {q[1]} last?"
    x, y = q[1], q[2]
    return f"Did {x[0]} go to the {x[1]} before {y[0]} went to the {y[1]}?"


def random_question(rng, visible, persons, places):
    kind = rng.randrange(5)
    if kind == 0:
        return ("when", rng.choice(persons), rng.choice(visible))
    if kind == 1:
        return ("first", rng.choice(persons))
    if kind == 2:
        return ("whofirst", rng.choice(places))
    if kind == 3:
        return ("wholast", rng.choice(places))
    if len(visible) < 2:
        return None
    x, y = rng.sample(visible, 2)
    return ("order", x, y)


def make_question(rng, prefix):
    visible = sorted({x for f in prefix for x in (f[1], f[2]) if x is not None})
    rels = [(f[1], f[2]) for f in prefix if f[0] == "rel"]
    orders = list(permutations(visible))
    valid = [o for o in orders if consistent(o, rels)]
    persons = sorted({e[0] for e in visible})
    places = sorted({e[1] for e in visible})
    for _ in range(12):
        q = random_question(rng, visible, persons, places)
        if q is None:
            continue
        cons = {answer(o, q) for o in valid}
        if len(cons) != 1 or None in cons:
            continue
        # The ordering facts must matter: without them the answer is open.
        if len({answer(o, q) for o in orders}) < 2:
            continue
        return (question_text(q), cons.pop())
    return None


def make_story(rng):
    while True:
        k = rng.randint(3, 5)
        pairs = set()
        while len(pairs) < k:
            pairs.add((rng.choice(NAMES), rng.choice(PLACES)))
        truth = sorted(pairs)
        rng.shuffle(truth)  # truth[i] is the i-th event in time
        tpos = {e: i for i, e in enumerate(truth)}

        n_target = rng.randint(3, 12)
        facts, covered = [], set()
        for e in truth:
            if e not in covered:
                other = rng.choice([x for x in truth if x != e])
                facts.append(ordered_rel(rng, tpos, e, other))
                covered.update((e, other))
        while len(facts) < n_target:
            if rng.random() < 0.3:
                e = rng.choice(truth)
                facts.append(intro_fact(e))
                covered.add(e)
            else:
                x, y = rng.sample(truth, 2)
                facts.append(ordered_rel(rng, tpos, x, y))
                covered.update((x, y))
        rng.shuffle(facts)

        target_q = rng.randint(1, 5)
        lines, qn = [], 0
        for i, f in enumerate(facts):
            lines.append((f[3], None))
            if qn < target_q and rng.random() < 0.5:
                q = make_question(rng, facts[: i + 1])
                if q:
                    lines.append(q)
                    qn += 1
        tries = 0
        while qn < target_q and tries < 20:
            tries += 1
            q = make_question(rng, facts)
            if q:
                lines.append(q)
                qn += 1
        if qn:
            return lines


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                out.append(f"{i} {sentence(text)}")
            else:
                out.append(f"{i} {sentence(text)}\t{ans}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
