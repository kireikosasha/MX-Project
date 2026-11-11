#!/usr/bin/env python3
"""Synthetic rule-based reasoning stories.

Usage: python gen.py N SEED

Each story mixes fact sentences ("Ann is a fox." / "Ann is loud.") with
rule sentences stated in the story itself ("Every fox is loud." /
"Every loud thing is quick."). Questions ask whether an individual has a
property, or which individuals have an adjective property. Answers are
computed by forward chaining over the statements above each question
(closed world: a property that is neither stated nor derivable is "no").
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cy", "Dana", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo"]
NOUNS = ["fox", "robot", "pilot", "tiger", "dragon", "sailor"]
ADJS = ["loud", "quick", "round", "green", "heavy", "sleepy", "shiny", "gentle"]
TYPE = {p: "n" for p in NOUNS}
TYPE.update({p: "a" for p in ADJS})


def fact_text(name, p):
    if TYPE[p] == "n":
        return f"{name} is a {p}."
    return f"{name} is {p}."


def rule_text(ant, cons):
    head = f"Every {ant}" if TYPE[ant] == "n" else f"Every {ant} thing"
    tail = f"is a {cons}." if TYPE[cons] == "n" else f"is {cons}."
    return f"{head} {tail}"


def question_text(name, p):
    if TYPE[p] == "n":
        return f"Is {name} a {p}?"
    return f"Is {name} {p}?"


def closure(statements):
    """Forward-chain the facts and rules of the given statements."""
    state = set()
    rules = []
    for kind, data in statements:
        if kind == "F":
            state.add(data)
        else:
            rules.append(data)
    changed = True
    while changed:
        changed = False
        for (n, p) in sorted(state):
            for (a, c) in rules:
                if a == p and (n, c) not in state:
                    state.add((n, c))
                    changed = True
    return state


def make_question(rng, above, preds, used):
    facts_above = {d for k, d in above if k == "F"}
    state = closure(above)
    subjects = sorted({n for n, _ in facts_above})
    if not subjects:
        return None
    for _ in range(10):
        if rng.random() < 0.2:
            adjs = [p for p in preds if TYPE[p] == "a"]
            if not adjs:
                continue
            p = rng.choice(adjs)
            who = [n for n in subjects if (n, p) in state]
            if not who:
                continue
            text = f"Which things are {p}?"
            ans = ",".join(n.lower() for n in who)
        else:
            yes = sorted((n, p) for (n, p) in state
                         if n in subjects and (n, p) not in facts_above)
            no = [(n, p) for n in subjects for p in preds
                  if (n, p) not in state]
            if yes and (not no or rng.random() < 0.7):
                pool = yes
            else:
                pool = no
            if not pool:
                continue
            n, p = rng.choice(pool)
            text = question_text(n, p)
            ans = "yes" if (n, p) in state else "no"
        if text in used:
            continue
        used.add(text)
        return text, ans
    return None


def try_story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    names = rng.sample(NAMES, rng.randint(2, 4))
    preds = rng.sample(NOUNS, rng.randint(2, 3)) + rng.sample(ADJS, rng.randint(2, 4))

    n_rules = rng.randint(1, min(4, n_st - 1))
    n_facts = n_st - n_rules
    combos = [(n, p) for n in names for p in preds]
    if n_facts > len(combos):
        return None
    facts = rng.sample(combos, n_facts)
    pairs = [(a, c) for a in preds for c in preds if a != c]
    rules = rng.sample(pairs, n_rules)

    stmts = [("F", f) for f in facts] + [("R", r) for r in rules]
    rng.shuffle(stmts)

    positions = sorted(rng.randint(1, n_st) for _ in range(n_q))
    items = []  # (text, answer or None)
    used = set()
    qi = 0
    for j in range(n_st + 1):
        while qi < len(positions) and positions[qi] == j:
            q = make_question(rng, stmts[:j], preds, used)
            if q is None:
                return None
            items.append(q)
            qi += 1
        if j < n_st:
            kind, data = stmts[j]
            if kind == "F":
                items.append((fact_text(*data), None))
            else:
                items.append((rule_text(*data), None))
    return items


def make_story(rng):
    while True:
        items = try_story(rng)
        if items is not None:
            return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        items = make_story(rng)
        for i, (text, ans) in enumerate(items, 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
