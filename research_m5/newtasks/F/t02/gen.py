#!/usr/bin/env python3
"""Generate synthetic exchange-of-things reading-comprehension stories.

Usage: python gen.py N SEED

People hold at most one thing each. Facts are: "X has the T.",
"X gives the T to Y.", "X takes the T from Y.", "X swaps with Y."
Questions are: "Who has the T?", "What does X have?", "Does X have the T?".
All answers come from simulating the statements above the question.
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cal", "Dan", "Eve", "Fay", "Gus", "Hal"]
THINGS = ["ball", "book", "cup", "hat", "key", "pen"]


def statement(rng, people, things, has, holder):
    """Pick a valid action, apply it to the world, return (sentence, is_move)."""
    opts = []
    empty = [p for p in people if has[p] is None]
    for t in things:
        if holder[t] is None:
            for p in empty:
                opts.append(("has", p, t))
        else:
            src = holder[t]
            for p in empty:
                if p != src:
                    opts.append(("gives", src, t, p))
                    opts.append(("takes", p, t, src))
    holders = [p for p in people if has[p] is not None]
    for i, a in enumerate(holders):
        for b in holders[i + 1:]:
            opts.append(("swaps", a, b))

    kind, *args = rng.choice(opts)
    if kind == "has":
        p, t = args
        has[p] = t
        holder[t] = p
        return f"{p} has the {t}.", False
    if kind == "gives":
        src, t, dst = args
        text = f"{src} gives the {t} to {dst}."
    elif kind == "takes":
        dst, t, src = args
        text = f"{dst} takes the {t} from {src}."
    if kind in ("gives", "takes"):
        has[src] = None
        has[dst] = t
        holder[t] = dst
        return text, True
    a, b = args
    ta, tb = has[a], has[b]
    has[a], has[b] = tb, ta
    holder[tb], holder[ta] = a, b
    return f"{a} swaps with {b}.", True


def question(rng, people, things, has, holder):
    """Pick a question about the current world; return (text, answer)."""
    opts = []
    for t in things:
        opts.append(("who", t))
    for p in people:
        opts.append(("what", p))
    for p in people:
        for t in things:
            opts.append(("does", p, t))

    kind, *args = rng.choice(opts)
    if kind == "who":
        (t,) = args
        return f"Who has the {t}?", (holder[t] or "nobody").lower()
    if kind == "what":
        (p,) = args
        return f"What does {p} have?", (has[p] or "nothing").lower()
    p, t = args
    return f"Does {p} have the {t}?", ("yes" if has[p] == t else "no")


def story(rng):
    """Return the numbered-line strings (without numbers) of one story."""
    while True:
        people = rng.sample(NAMES, rng.randint(3, 5))
        things = rng.sample(THINGS, rng.randint(2, 4))
        has = {p: None for p in people}
        holder = {t: None for t in things}

        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        rest = ["S"] * (n_st - 1) + ["Q"] * n_q
        rng.shuffle(rest)
        kinds = ["S"] + rest  # the first line is always a statement

        lines, seen, moves = [], set(), 0
        for k in kinds:
            if k == "S":
                text, is_move = statement(rng, people, things, has, holder)
                moves += is_move
                lines.append(text)
            else:
                while True:
                    qtext, ans = question(rng, people, things, has, holder)
                    if qtext not in seen:
                        break
                seen.add(qtext)
                lines.append(f"{qtext}\t{ans}")

        if moves >= 2:  # each story needs several exchanges to answer
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(story(rng), 1):
            out.append(f"{i} {line}")
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
