#!/usr/bin/env python3
"""Synthetic chain-of-relations reading task generator.

Relation: "boss of" (each person has at most one boss; bosses form chains).
Questions ask about the boss of the boss of a person, so two links must be
followed one after another. Answers come from a simulation of the facts stated
so far, and a question is only asked once its chain is fully known.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "bob", "cole", "dana", "eli", "fay", "gus", "hana",
         "ivan", "jade", "kit", "lena", "max", "nina"]


def cap(word):
    return word[:1].upper() + word[1:]


def question_options(people, boss):
    """Return unasked-independent candidate questions as (text, answer, key)."""
    opts = []
    for p in people:
        if p in boss and boss[p] in boss:
            z = boss[boss[p]]  # two links up from p
            opts.append(("who", "Who is the boss of the boss of %s?" % cap(p), z, (p,)))
            for a in people:
                if a == p:
                    continue
                if a == z:
                    opts.append(("yes", "Is %s the boss of the boss of %s?" % (cap(a), cap(p)),
                                 "yes", ("yes", p, a)))
                else:
                    opts.append(("no", "Is %s the boss of the boss of %s?" % (cap(a), cap(p)),
                                 "no", ("no", p, a)))
    return opts


def try_story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    people = rng.sample(NAMES, 13)  # people[i] may only have a boss among people[:i]
    boss = {}
    asked = set()
    lines = []  # (text, answer or None)
    st_left, q_left = n_st, n_q
    while st_left > 0 or q_left > 0:
        opts = [o for o in question_options(people, boss) if o[3] not in asked]
        can_q = q_left > 0 and bool(opts)
        can_s = st_left > 0
        if not can_q and not can_s:
            return None
        if can_q and (not can_s or rng.random() < 0.4):
            kinds = sorted(set(o[0] for o in opts))
            kind = rng.choice(kinds)
            choice = rng.choice([o for o in opts if o[0] == kind])
            asked.add(choice[3])
            lines.append((choice[1], choice[2]))
            q_left -= 1
        else:
            free = [p for p in people[1:] if p not in boss]
            y = rng.choice(free)
            x = rng.choice(people[:people.index(y)])
            boss[y] = x
            if rng.random() < 0.5:
                lines.append(("%s is the boss of %s." % (cap(x), cap(y)), None))
            else:
                lines.append(("%s reports to %s." % (cap(y), cap(x)), None))
            st_left -= 1
    return lines


def gen_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(gen_story(rng), start=1):
            if ans is None:
                out.append("%d %s" % (i, text))
            else:
                out.append("%d %s\t%s" % (i, text, ans))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
