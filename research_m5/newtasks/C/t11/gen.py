#!/usr/bin/env python3
"""Synthetic reading-comprehension task: chained rules over named individuals.

Usage: python gen.py N SEED

Each story mixes facts about named individuals ("Sam is a frog.", "Sam is big.")
with general rules ("Every frog is green.", "No big thing is fast.").
Questions ask whether an individual has a property, or who has a property.
Answers come from a forward-chaining simulation of the statements above each
question. A question is only asked when its answer is derived through at least
one rule, so the answer needs reasoning over several sentences.
"""

import random
import sys

NAMES = ["Sam", "Mia", "Ben", "Zoe", "Leo", "Ada"]
KINDS = ["frog", "bird", "fish", "cat"]
PROPS = ["green", "blue", "red", "big", "small", "fast"]

POS = 1
NEG = -1


def closure(facts, rules):
    """Forward chaining. Returns {(name, prop, sign): min_rule_depth}.

    facts: list of (name, prop) stated directly (depth 0).
    rules: list of (premise, conclusion, sign); sign POS means "Every premise
           is conclusion", NEG means "No premise is conclusion".
    """
    best = {}
    for name, prop in facts:
        best[(name, prop, POS)] = 0
    changed = True
    while changed:
        changed = False
        for a, b, s in rules:
            for (e, p, sg), d in list(best.items()):
                if sg == POS and p == a:
                    key = (e, b, s)
                    if key not in best or best[key] > d + 1:
                        best[key] = d + 1
                        changed = True
    return best


def conflicts(best):
    for (e, p, s) in best:
        if s == POS and (e, p, NEG) in best:
            return True
    return False


def fact_text(name, prop):
    if prop in KINDS:
        return "%s is a %s." % (name, prop)
    return "%s is %s." % (name, prop)


def rule_text(a, b, s):
    if s == POS:
        if a in KINDS:
            return "Every %s is %s." % (a, b)
        return "Every %s thing is %s." % (a, b)
    if a in KINDS:
        return "No %s is %s." % (a, b)
    return "No %s thing is %s." % (a, b)


def pick_statement(rng, facts, rules, names, used):
    base_keys = set(closure(facts, rules))
    for _ in range(60):
        if rng.random() < 0.4:
            name = rng.choice(names)
            prop = rng.choice(KINDS + PROPS)
            new_facts = facts + [(name, prop)]
            new_rules = rules
            text = fact_text(name, prop)
        else:
            a = rng.choice(KINDS + PROPS)
            b = rng.choice(PROPS)
            if a == b:
                continue
            s = POS if rng.random() < 0.7 else NEG
            new_facts = facts
            new_rules = rules + [(a, b, s)]
            text = rule_text(a, b, s)
        if text in used:
            continue
        new = closure(new_facts, new_rules)
        if conflicts(new):
            continue
        if not (set(new) - base_keys):
            continue  # statement must add something to the world state
        return new_facts, new_rules, text
    return None


def pick_question(rng, best, names, asked):
    cands = []
    for n in names:
        for p in PROPS:
            dpos = best.get((n, p, POS))
            dneg = best.get((n, p, NEG))
            ans = None
            if dpos is not None and dpos >= 1:
                ans = "yes"
            elif dneg is not None and dneg >= 1:
                ans = "no"
            if ans is not None:
                text = "Is %s %s?" % (n, p)
                if text not in asked:
                    cands.append((text, ans))
    for p in PROPS:
        holders = [n for n in names if (n, p, POS) in best]
        if holders and any(best[(n, p, POS)] >= 1 for n in holders):
            text = "Who is %s?" % p
            if text not in asked:
                ans = ",".join(sorted(n.lower() for n in holders))
                cands.append((text, ans))
    if not cands:
        return None
    return rng.choice(cands)


def build_story(rng):
    """Returns a list of (text, answer_or_None), or None if this attempt fails."""
    S = rng.randint(3, 12)  # number of statements
    Q = rng.randint(1, 5)   # number of questions
    names = rng.sample(NAMES, rng.randint(3, 4))
    facts, rules = [], []
    lines = []
    used = set()
    asked = set()
    nst = 0
    nq = 0
    steps = 0
    while nst < S or nq < Q:
        steps += 1
        if steps > 300:
            return None
        if nq < Q and nst >= 1 and (nst == S or rng.random() < 0.4):
            best = closure(facts, rules)
            q = pick_question(rng, best, names, asked)
            if q is not None:
                asked.add(q[0])
                lines.append(q)
                nq += 1
                continue
            if nst == S:
                return None
        if nst >= S:
            return None
        res = pick_statement(rng, facts, rules, names, used)
        if res is None:
            return None
        facts, rules, text = res
        used.add(text)
        lines.append((text, None))
        nst += 1
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = build_story(rng)
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                out.append("%d %s" % (i, text))
            else:
                out.append("%d %s\t%s" % (i, text, ans))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
