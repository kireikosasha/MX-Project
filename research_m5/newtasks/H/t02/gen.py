#!/usr/bin/env python3
"""Synthetic exchange-of-things reading comprehension stories.

Usage: python gen.py N SEED

People hold things. Statements say who gets a thing, who passes a thing
to whom, or who swaps a thing for another person's thing. Questions ask who
holds a thing, what a person holds, how many things a person holds, or
whether a person holds a thing. Answers come from simulating the statements
above each question. Standard library only.
"""
import random
import sys

NAMES = ["Alice", "Bob", "Carol", "Dan", "Eve"]
THINGS = ["ball", "apple", "book", "cup"]
MAX_QUESTIONS = 5

HAS_PATTERNS = ["{p} has the {t}.", "{p} holds the {t}."]
GIVE_PATTERNS = ["{p} gives the {t} to {q}.", "{p} passes the {t} to {q}."]
SWAP_PATTERNS = ["{p} swaps the {t1} for the {t2}."]


def holdings(owner, person):
    return [t for t in THINGS if owner[t] == person]


def fact_options(owner):
    """All facts that are valid in the current world state.

    ("has", p, t): t is free, p receives it.
    ("give", p, t, q): p holds t and gives it to q.
    ("swap", p, t1, q, t2): p holds t1, q holds t2; they trade.
    """
    opts = []
    for t in THINGS:
        if owner[t] is None:
            for p in NAMES:
                opts.append(("has", p, t))
    for p in NAMES:
        for t in holdings(owner, p):
            for q in NAMES:
                if q != p:
                    opts.append(("give", p, t, q))
    for p in NAMES:
        for q in NAMES:
            if q == p:
                continue
            for t1 in holdings(owner, p):
                for t2 in holdings(owner, q):
                    opts.append(("swap", p, t1, q, t2))
    return opts


def apply_fact(owner, opt):
    kind = opt[0]
    if kind == "has":
        _, p, t = opt
        owner[t] = p
    elif kind == "give":
        _, p, t, q = opt
        owner[t] = q
    else:
        _, p, t1, q, t2 = opt
        owner[t1] = q
        owner[t2] = p


def render_fact(rng, opt):
    kind = opt[0]
    if kind == "has":
        _, p, t = opt
        return rng.choice(HAS_PATTERNS).format(p=p, t=t)
    if kind == "give":
        _, p, t, q = opt
        return rng.choice(GIVE_PATTERNS).format(p=p, t=t, q=q)
    _, p, t1, q, t2 = opt
    return rng.choice(SWAP_PATTERNS).format(p=p, t1=t1, t2=t2)


def named(opt):
    """Names and things that appear in the sentence text of a fact."""
    kind = opt[0]
    if kind == "has":
        return {opt[1], opt[2]}
    if kind == "give":
        return {opt[1], opt[2], opt[3]}
    return {opt[1], opt[2], opt[4]}


def make_question(rng, owner, history):
    """Return (question_text, answer) or None if no valid question exists."""

    def count(e):
        return sum(1 for o in history if e in named(o))

    cands = []
    for t in THINGS:
        if count(t) >= 2:
            cands.append(("who", t))
    for p in NAMES:
        if count(p) >= 2:
            cands.append(("what", p))
            cands.append(("many", p))
            for t in THINGS:
                cands.append(("does", p, t))
    if not cands:
        return None

    c = rng.choice(cands)
    if c[0] == "who":
        t = c[1]
        ans = owner[t].lower() if owner[t] is not None else "nobody"
        return "Who has the %s?" % t, ans
    if c[0] == "what":
        p = c[1]
        items = holdings(owner, p)
        ans = ",".join(items) if items else "nothing"
        return "What does %s have?" % p, ans
    if c[0] == "many":
        p = c[1]
        return "How many things does %s have?" % p, str(len(holdings(owner, p)))
    _, p, t = c
    ans = "yes" if owner[t] == p else "no"
    return "Does %s have the %s?" % (p, t), ans


def make_story(rng):
    """Return a list of (text, answer) pairs; answer is None for statements."""
    while True:
        owner = {t: None for t in THINGS}
        n = rng.randint(3, 12)
        target_q = rng.randint(1, MAX_QUESTIONS)
        # A question slot k means the question follows statement k.
        slots = [rng.randint(2, n) for _ in range(target_q)]
        history = []
        lines = []
        ok = True
        for k in range(1, n + 1):
            opts = fact_options(owner)
            if not opts:
                ok = False
                break
            opt = rng.choice(opts)
            apply_fact(owner, opt)
            history.append(opt)
            lines.append((render_fact(rng, opt), None))
            for _ in range(slots.count(k)):
                q = make_question(rng, owner, history)
                if q is not None:
                    lines.append(q)
        nq = sum(1 for _, a in lines if a is not None)
        if ok and 1 <= nq <= MAX_QUESTIONS:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n_stories):
        story = make_story(rng)
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                print("%d %s" % (i, text))
            else:
                print("%d %s\t%s" % (i, text, ans))


if __name__ == "__main__":
    main()
