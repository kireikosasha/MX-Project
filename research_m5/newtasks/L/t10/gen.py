"""Synthetic reading task about groups: people join, leave and move between groups.

Usage: python gen.py N SEED
Prints N stories. All randomness comes from random.Random(SEED).
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eve", "finn", "gus", "hana"]
GROUPS = ["red", "blue", "green", "gold"]


class World:
    def __init__(self):
        self.group = {p: None for p in NAMES}    # person -> current group or None
        self.mention_p = {p: 0 for p in NAMES}   # statements about each person
        self.mention_g = {g: 0 for g in GROUPS}  # statements about each group


def statement(rng, w):
    p = rng.choice(NAMES)
    cur = w.group[p]
    if cur is None:
        new = rng.choice(GROUPS)
        text = f"{p} joins the {new} group."
        w.group[p] = new
        touched = [new]
    elif rng.random() < 0.4:
        text = f"{p} leaves the {cur} group."
        w.group[p] = None
        touched = [cur]
    else:
        new = rng.choice([g for g in GROUPS if g != cur])
        text = f"{p} moves from the {cur} group to the {new} group."
        w.group[p] = new
        touched = [cur, new]
    w.mention_p[p] += 1
    for g in touched:
        w.mention_g[g] += 1
    return text, None


def question(rng, w):
    # Only ask about people / groups that appear in at least two statements,
    # so the answer needs reasoning over several sentences.
    options = []
    for p in NAMES:
        if w.mention_p[p] >= 2:
            options.append(("which", p))
            options.append(("is", p))
    for g in GROUPS:
        if w.mention_g[g] >= 2:
            options.append(("who", g))
            options.append(("count", g))
    if not options:
        return None
    kind, x = rng.choice(options)
    if kind == "which":
        return f"which group is {x} in?", (w.group[x] or "none")
    if kind == "is":
        g = rng.choice(GROUPS)
        return f"is {x} in the {g} group?", ("yes" if w.group[x] == g else "no")
    if kind == "who":
        members = sorted(p for p in NAMES if w.group[p] == x)
        return f"who is in the {x} group?", (",".join(members) or "none")
    n = sum(1 for p in NAMES if w.group[p] == x)
    return f"how many are in the {x} group?", str(n)


def story(rng):
    while True:
        w = World()
        lines = []
        target_st = rng.randint(3, 12)
        target_q = rng.randint(1, 5)
        n_st = n_q = 0
        while n_st < target_st or n_q < target_q:
            if n_q < target_q and n_st >= 2 and (n_st == target_st or rng.random() < 0.35):
                q = question(rng, w)
                if q is not None:
                    lines.append(q)
                    n_q += 1
                    continue
            if n_st == target_st:
                break  # remaining questions could not be placed: retry the story
            lines.append(statement(rng, w))
            n_st += 1
        else:
            return lines


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(story(rng), start=1):
            out.append(f"{i} {text}" if ans is None else f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
