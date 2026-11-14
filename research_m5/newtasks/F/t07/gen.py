import random
import sys

NAMES = ["alice", "bob", "carol", "dan", "eve"]
THINGS = ["apples", "coins", "pens"]
MAX_AMOUNT = 9


def cap(word):
    return word[0].upper() + word[1:]


def statement(rng, counts, changed, allow_has):
    """Pick a feasible action, update the world state, return the sentence.

    counts maps (person, thing) to the known number of that thing the person
    holds. A pair becomes known only through a 'has' statement, so every
    later action and every question refers to a known amount. changed holds
    the pairs whose amount was changed by an action since their last 'has'.
    """
    options = []
    if allow_has:
        options.append(("has",))
    for (p, t), c in counts.items():
        options.append(("gets", p, t))
        if c >= 2:
            options.append(("loses", p, t))
        for q in NAMES:
            if q == p or (q, t) not in counts:
                continue
            if c >= 2:
                options.append(("gives", p, q, t))
            if counts[(q, t)] >= 2:
                options.append(("takes", p, q, t))
    if not options:
        options.append(("has",))
    choice = rng.choice(options)
    action = choice[0]

    if action == "has":
        p = rng.choice(NAMES)
        t = rng.choice(THINGS)
        a = rng.randint(2, MAX_AMOUNT)
        counts[(p, t)] = a
        return f"{cap(p)} has {a} {t}."

    if action == "gets":
        _, p, t = choice
        a = rng.randint(2, MAX_AMOUNT)
        counts[(p, t)] += a
        changed.add((p, t))
        return f"{cap(p)} gets {a} {t}."

    if action == "loses":
        _, p, t = choice
        a = rng.randint(2, min(counts[(p, t)], MAX_AMOUNT))
        counts[(p, t)] -= a
        changed.add((p, t))
        return f"{cap(p)} loses {a} {t}."

    if action == "gives":
        _, p, q, t = choice
        a = rng.randint(2, min(counts[(p, t)], MAX_AMOUNT))
        counts[(p, t)] -= a
        counts[(q, t)] += a
        changed.add((p, t))
        changed.add((q, t))
        return f"{cap(p)} gives {a} {t} to {cap(q)}."

    # takes: p takes some things from q
    _, p, q, t = choice
    a = rng.randint(2, min(counts[(q, t)], MAX_AMOUNT))
    counts[(q, t)] -= a
    counts[(p, t)] += a
    changed.add((p, t))
    changed.add((q, t))
    return f"{cap(p)} takes {a} {t} from {cap(q)}."


def question(rng, counts, changed):
    """Return (question text without '?', answer) about the current state.

    Only pairs changed by an action are asked about, so the answer needs
    the statements that changed them, not a single lookup.
    """
    options = []
    for (p, t) in changed:
        options.append(("count", p, t))
        for q in NAMES:
            if q != p and (q, t) in counts:
                options.append(("more", p, q, t))
                options.append(("more", q, p, t))
    choice = rng.choice(options)

    if choice[0] == "count":
        _, p, t = choice
        return f"How many {t} does {cap(p)} have", str(counts[(p, t)])

    _, p, q, t = choice
    answer = "yes" if counts[(p, t)] > counts[(q, t)] else "no"
    return f"Does {cap(p)} have more {t} than {cap(q)}", answer


def story(rng):
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    # Line 1 sets an amount, line 2 changes it, so questions from line 3 on
    # can always be about a changed amount.
    rest = ["S"] * (n_stmt - 2) + ["Q"] * n_q
    rng.shuffle(rest)
    kinds = ["S", "S"] + rest

    counts = {}
    changed = set()
    lines = []
    for number, kind in enumerate(kinds, 1):
        if kind == "S":
            text = statement(rng, counts, changed, allow_has=(number != 2))
            lines.append(f"{number} {text}")
        else:
            q, a = question(rng, counts, changed)
            lines.append(f"{number} {q}?\t{a}")
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in story(rng):
            print(line)


if __name__ == "__main__":
    main()
