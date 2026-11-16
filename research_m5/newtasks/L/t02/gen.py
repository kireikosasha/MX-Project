"""Generate synthetic exchange-of-things reading tasks.

Usage: python gen.py N SEED

Each story: people hold at most one thing each; things change hands by
"has" (first acquisition), "gives" (one person hands a thing to an empty-handed
person) and "swaps" (two people exchange what they hold). Questions are
answered by simulating the story, and only questions about things or people
that were moved at least once are asked, so one sentence is never enough.
"""

import random
import sys

NAMES = ["ann", "ben", "cal", "dan", "eve", "fay", "gus"]
THINGS = ["apple", "ball", "book", "coin", "hat", "key", "pen"]


def cap(text):
    return text[0].upper() + text[1:]


def held(holder, person):
    for thing, owner in holder.items():
        if owner == person:
            return thing
    return None


def statement(rng, names, things, holder, seen_thing, seen_person):
    options = []
    empty = [p for p in names if held(holder, p) is None]
    for thing in things:
        if holder[thing] is None:
            for p in empty:
                options.append(("has", p, thing))
        else:
            for q in empty:
                options.append(("gives", holder[thing], thing, q))
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            if held(holder, p) or held(holder, q):
                options.append(("swaps", p, q))

    choice = rng.choice(options)
    kind = choice[0]
    if kind == "has":
        _, p, t = choice
        holder[t] = p
        seen_thing[t] += 1
        seen_person[p] += 1
        return f"{p} has the {t}."
    if kind == "gives":
        _, p, t, q = choice
        holder[t] = q
        seen_thing[t] += 1
        seen_person[p] += 1
        seen_person[q] += 1
        return f"{p} gives the {t} to {q}."
    _, p, q = choice
    tp, tq = held(holder, p), held(holder, q)
    if tp:
        holder[tp] = q
    if tq:
        holder[tq] = p
    seen_person[p] += 1
    seen_person[q] += 1
    return f"{p} swaps with {q}."


def render(option, holder):
    kind = option[0]
    if kind == "who":
        t = option[1]
        return f"Who has the {t}?", holder[t] or "nobody"
    if kind == "does":
        _, p, t = option
        return f"Does {p} have the {t}?", ("yes" if holder[t] == p else "no")
    p = option[1]
    return f"What does {p} have?", held(holder, p) or "nothing"


def question(names, things, holder, seen_thing, seen_person, asked):
    options = []
    for t in things:
        if seen_thing[t] >= 2:
            options.append(("who", t))
            for p in names:
                options.append(("does", p, t))
    for p in names:
        if seen_person[p] >= 2:
            options.append(("what", p))
    options = [o for o in options if render(o, holder)[0] not in asked]
    if not options:
        return None
    return options


def attempt(rng):
    n_statements = rng.randint(3, 12)
    n_questions = rng.randint(1, 5)
    names = rng.sample(NAMES, rng.randint(3, 5))
    things = rng.sample(THINGS, rng.randint(2, 4))
    holder = {t: None for t in things}
    seen_thing = {t: 0 for t in things}
    seen_person = {p: 0 for p in names}

    kinds = ["s"] * n_statements + ["q"] * n_questions
    rng.shuffle(kinds)
    while kinds[0] == "q":
        rng.shuffle(kinds)

    asked = set()
    lines = []
    for kind in kinds:
        if kind == "s":
            text = statement(rng, names, things, holder, seen_thing, seen_person)
            lines.append((cap(text), None))
            continue
        options = question(names, things, holder, seen_thing, seen_person, asked)
        if options is None:
            return None
        text, answer = render(rng.choice(options), holder)
        asked.add(text)
        lines.append((cap(text), answer))
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lines = None
        while lines is None:
            lines = attempt(rng)
        for number, (text, answer) in enumerate(lines, 1):
            if answer is None:
                out.append(f"{number} {text}")
            else:
                out.append(f"{number} {text}\t{answer}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
