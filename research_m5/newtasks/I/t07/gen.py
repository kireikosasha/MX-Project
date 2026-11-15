"""Synthetic order-in-time reading task (standard library only).

Usage: python gen.py N SEED

Each story: people start in places, then make moves to places. The moves are
told in shuffled order; "before" / "after" statements reveal part of the true
timeline. Questions ask where someone was just before a move, whether someone
was in a place at that moment, whether one move happened before another, or
how many times someone moved before a move.

A question is only asked when every ordering of the moves that agrees with the
statements above it gives the same answer; the answer is computed by replaying
those orderings, never written by hand.
"""

import itertools
import random
import sys

NAMES = ["Sam", "Mary", "Anna", "Tom", "Jack", "Lily"]
PLACES = ["kitchen", "garden", "hall", "office", "park"]
MAX_STATEMENTS = 12
MAX_QUESTIONS = 5
QUESTION_TRIES = 60

# All orderings of 2..5 moves (move indices are positions in the true timeline).
PERMS = {n: list(itertools.permutations(range(n))) for n in range(2, 6)}


def make_world(rng):
    people = rng.sample(NAMES, rng.randint(2, 4))
    start = {p: rng.choice(PLACES) for p in people}
    cur = dict(start)
    moves = []  # true timeline, in order: (person, place)
    for _ in range(rng.randint(2, 5)):
        options = [(p, pl) for p in people for pl in PLACES
                   if pl != cur[p] and (p, pl) not in moves]
        if not options:
            break
        p, pl = rng.choice(options)
        moves.append((p, pl))
        cur[p] = pl
    if len(moves) < 2:
        return None
    return people, start, moves


def simulate(perm, start, moves):
    """Replay the moves in the given order.

    Returns (pos, snap): pos[e] is the position of move e in the order, and
    snap[e] = (locations, move counts) just before move e happens.
    """
    loc = dict(start)
    cnt = {p: 0 for p in start}
    snap = {}
    for e in perm:
        snap[e] = (dict(loc), dict(cnt))
        p, pl = moves[e]
        loc[p] = pl
        cnt[p] += 1
    pos = {e: i for i, e in enumerate(perm)}
    return pos, snap


def consistent_sims(start, moves, cons):
    sims = []
    for perm in PERMS[len(moves)]:
        pos = {e: i for i, e in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b in cons):
            sims.append(simulate(perm, start, moves))
    return sims


def random_spec(rng, people, moves):
    n = len(moves)
    kind = rng.choice(["where", "in", "did", "count"])
    qe = rng.randrange(n)
    if kind == "where":
        return ("where", rng.choice(people), qe)
    if kind == "in":
        return ("in", rng.choice(people), rng.choice(PLACES), qe)
    if kind == "count":
        return ("count", rng.choice(people), qe)
    pe = rng.randrange(n)
    if pe == qe:
        return None
    return ("did", pe, qe)


def answer(spec, sim):
    pos, snap = sim
    kind = spec[0]
    if kind == "where":
        _, p, qe = spec
        return snap[qe][0][p]
    if kind == "in":
        _, p, pl, qe = spec
        return "yes" if snap[qe][0][p] == pl else "no"
    if kind == "did":
        _, pe, qe = spec
        return "yes" if pos[pe] < pos[qe] else "no"
    _, p, qe = spec
    return str(snap[qe][1][p])


def question_text(spec, moves):
    kind = spec[0]
    if kind == "where":
        _, p, qe = spec
        q, pl = moves[qe]
        return f"Where was {p} just before {q} went to the {pl}"
    if kind == "in":
        _, p, place, qe = spec
        q, pl = moves[qe]
        return f"Was {p} in the {place} just before {q} went to the {pl}"
    if kind == "did":
        _, pe, qe = spec
        p, place = moves[pe]
        q, pl = moves[qe]
        return f"Did {p} go to the {place} before {q} went to the {pl}"
    _, p, qe = spec
    q, pl = moves[qe]
    return f"How many times did {p} move before {q} went to the {pl}"


def order_statement(rng, moves):
    # moves are in true timeline order, so a < b means a really happened first.
    a, b = sorted(rng.sample(range(len(moves)), 2))
    pa, la = moves[a]
    pb, lb = moves[b]
    if rng.random() < 0.5:
        text = f"{pa} went to the {la} before {pb} went to the {lb}"
    else:
        text = f"{pb} went to the {lb} after {pa} went to the {la}"
    return text, (a, b)


def build_story(rng):
    world = make_world(rng)
    if world is None:
        return None
    people, start, moves = world

    setup = [f"{p} started in the {start[p]}" for p in people]
    setup += [f"{p} went to the {pl}" for p, pl in moves]
    rng.shuffle(setup)

    cap = MAX_STATEMENTS - len(setup)
    want_q = rng.randint(1, MAX_QUESTIONS)
    want_o = rng.randint(0, cap)

    seq = [("s", t) for t in setup]  # ("s", text) or ("q", text, answer)
    texts = set(setup)
    cons = []
    sims = consistent_sims(start, moves, cons)
    n_q = 0
    n_o = 0

    while n_q < want_q or n_o < want_o:
        if n_q < want_q:
            found = None
            for _ in range(QUESTION_TRIES):
                spec = random_spec(rng, people, moves)
                if spec is None:
                    continue
                text = question_text(spec, moves)
                if text in texts:
                    continue
                answers = {answer(spec, s) for s in sims}
                if len(answers) == 1:
                    found = (text, answers.pop())
                    break
            if found is not None:
                seq.append(("q", found[0], found[1]))
                texts.add(found[0])
                n_q += 1
                continue
        if n_o >= cap:
            return None
        text, con = order_statement(rng, moves)
        for _ in range(10):
            if text not in texts:
                break
            text, con = order_statement(rng, moves)
        if text in texts:
            return None
        seq.append(("s", text))
        texts.add(text)
        cons.append(con)
        sims = consistent_sims(start, moves, cons)
        n_o += 1

    lines = []
    for i, item in enumerate(seq, 1):
        if item[0] == "s":
            lines.append(f"{i} {item[1]}.")
        else:
            lines.append(f"{i} {item[1]}?\t{item[2]}")
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n):
        lines = None
        while lines is None:
            lines = build_story(rng)
        out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
