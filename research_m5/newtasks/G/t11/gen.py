#!/usr/bin/env python3
"""Synthetic reading-comprehension stories where general rules are stated in the text.

Usage: python gen.py N SEED

Each story first states 1-3 general rules (Horn clauses) and then facts about
people, objects and places. Questions are answered by a simulation of the
world plus forward chaining over the stated rules (closed world: a trait or an
entry permission holds only if it can be derived from the story).

Rule kinds:
  H: Whoever has a OBJ is TRAIT.
  T: Every TRAIT1 person is TRAIT2.
  E: Every TRAIT person can enter the PLACE.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan"]
OBJECTS = ["key", "lamp", "cup"]
PLACES = ["kitchen", "garden", "cellar", "hall"]
TRAITS = ["brave", "calm", "tidy"]


class World:
    def __init__(self):
        self.loc = {}        # person -> place (set once, at introduction)
        self.holder = {}     # object -> holding person, or None (set once, at introduction)
        self.obj_place = {}  # object -> place, used while nobody holds it

    def obj_loc(self, o):
        h = self.holder[o]
        return self.loc[h] if h is not None else self.obj_place[o]


def traits_of(w, p, rules):
    """Traits that person p has, by forward chaining over the stated rules."""
    ts = set()
    for r in rules:
        if r[0] == "H" and w.holder.get(r[1]) == p:
            ts.add(r[2])
    changed = True
    while changed:
        changed = False
        for r in rules:
            if r[0] == "T" and r[1] in ts and r[2] not in ts:
                ts.add(r[2])
                changed = True
    return ts


def can_enter(w, p, place, rules):
    ts = traits_of(w, p, rules)
    return any(r[0] == "E" and r[2] == place and r[1] in ts for r in rules)


def rule_text(r, rng):
    if r[0] == "H":
        return f"Whoever has a {r[1]} is {r[2]}."
    if r[0] == "T":
        if rng.random() < 0.5:
            return f"Every {r[1]} person is {r[2]}."
        return f"All {r[1]} people are {r[2]}."
    if rng.random() < 0.5:
        return f"Every {r[1]} person can enter the {r[2]}."
    return f"All {r[1]} people can enter the {r[2]}."


def statement_options(w, names, objs):
    placements, actions = [], []
    for p in names:
        if p not in w.loc:
            placements.append(("person", p))
    for o in objs:
        if o not in w.holder:
            placements.append(("object", o))
    for p in names:
        if p not in w.loc:
            continue
        for pl in _PLACE_POOL:
            if pl != w.loc[p]:
                actions.append(("move", p, pl))
        for o in objs:
            if o not in w.holder:
                continue
            h = w.holder[o]
            if h is None:
                if w.obj_place[o] == w.loc[p]:
                    actions.append(("take", p, o))
            elif h == p:
                actions.append(("drop", p, o))
                for q in names:
                    if q != p and q in w.loc and w.loc[q] == w.loc[p]:
                        actions.append(("give", p, o, q))
    return placements, actions


_PLACE_POOL = []  # places chosen for the current story (set in try_story)


def apply_statement(w, act, rng):
    kind = act[0]
    if kind == "person":
        p = act[1]
        pl = rng.choice(_PLACE_POOL)
        w.loc[p] = pl
        return f"{p} is in the {pl}."
    if kind == "object":
        o = act[1]
        pl = rng.choice(_PLACE_POOL)
        w.holder[o] = None
        w.obj_place[o] = pl
        return f"The {o} is in the {pl}."
    if kind == "move":
        _, p, pl = act
        w.loc[p] = pl
        return f"{p} went to the {pl}."
    if kind == "take":
        _, p, o = act
        w.holder[o] = p
        return f"{p} took the {o}."
    if kind == "drop":
        _, p, o = act
        w.holder[o] = None
        w.obj_place[o] = w.loc[p]
        return f"{p} dropped the {o}."
    _, p, o, q = act  # give
    w.holder[o] = q
    return f"{p} gave the {o} to {q}."


def make_statement(w, names, objs, rng):
    placements, actions = statement_options(w, names, objs)
    if not placements and not actions:
        return None
    if placements and (not actions or rng.random() < 0.4):
        act = rng.choice(placements)
    else:
        takes = [a for a in actions if a[0] == "take"]
        if takes and rng.random() < 0.5:
            act = rng.choice(takes)
        else:
            act = rng.choice(actions)
    return apply_statement(w, act, rng)


def mentioned_traits(rules):
    out = []
    for r in rules:
        ts = [r[1], r[2]] if r[0] == "T" else [r[2]] if r[0] == "H" else [r[1]]
        for t in ts:
            if t not in out:
                out.append(t)
    return out


def enter_places(rules):
    out = []
    for r in rules:
        if r[0] == "E" and r[2] not in out:
            out.append(r[2])
    return out


def question_candidates(w, rules, rng):
    cands = []  # (text, answer)
    traits = mentioned_traits(rules)
    places = enter_places(rules)
    for p in w.loc:
        if rng.random() < 0.5:
            text = f"Where is {p}?"
        else:
            text = f"In which place is {p}?"
        cands.append((text, w.loc[p]))
        for t in traits:
            ans = "yes" if t in traits_of(w, p, rules) else "no"
            cands.append((f"Is {p} {t}?", ans))
        for pl in places:
            ans = "yes" if can_enter(w, p, pl, rules) else "no"
            cands.append((f"Can {p} enter the {pl}?", ans))
    for o in w.holder:
        if rng.random() < 0.5:
            text = f"Where is the {o}?"
        else:
            text = f"In which place is the {o}?"
        cands.append((text, w.obj_loc(o)))
        if rng.random() < 0.5:
            text = f"Who has the {o}?"
        else:
            text = f"Which person has the {o}?"
        h = w.holder[o]
        cands.append((text, h.lower() if h is not None else "nobody"))
    return cands


def pick_question(w, rules, rng, asked):
    cands = [c for c in question_candidates(w, rules, rng) if c[0] not in asked]
    yes = [c for c in cands if c[1] == "yes"]
    if yes and rng.random() < 0.5:
        cands = yes
    if not cands:
        return None
    text, ans = rng.choice(cands)
    asked.add(text)
    return [text, ans]


def try_story(rng):
    global _PLACE_POOL
    names = rng.sample(NAMES, rng.randint(2, 4))
    objs = rng.sample(OBJECTS, rng.randint(2, 3))
    _PLACE_POOL = rng.sample(PLACES, rng.randint(2, 4))
    traits = rng.sample(TRAITS, rng.randint(2, 3))

    rules = []
    n_rules = rng.randint(1, 3)
    for _ in range(30):
        if len(rules) >= n_rules:
            break
        kind = rng.choice("HTE")
        if kind == "H":
            r = ("H", rng.choice(objs), rng.choice(traits))
        elif kind == "T":
            a, b = rng.sample(traits, 2)
            r = ("T", a, b)
        else:
            r = ("E", rng.choice(traits), rng.choice(_PLACE_POOL))
        if r not in rules:
            rules.append(r)

    lines = []  # [text, answer]; answer is None for statements
    for r in rules:
        lines.append([rule_text(r, rng), None])

    w = World()
    n_stmt = len(rules)
    target_stmt = rng.randint(max(3, n_stmt + 1), 12)
    target_q = rng.randint(1, 4)
    asked = set()
    n_q = 0
    while n_stmt < target_stmt:
        if n_q < target_q and rng.random() < 0.4:
            q = pick_question(w, rules, rng, asked)
            if q is not None:
                lines.append(q)
                n_q += 1
        text = make_statement(w, names, objs, rng)
        if text is None:
            break
        lines.append([text, None])
        n_stmt += 1
    while n_q < target_q:
        q = pick_question(w, rules, rng, asked)
        if q is None:
            break
        lines.append(q)
        n_q += 1

    if n_stmt < 3 or n_q < 1:
        return None
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            break
    out = []
    for i, (text, ans) in enumerate(lines, 1):
        if ans is None:
            out.append(f"{i} {text}")
        else:
            out.append(f"{i} {text}\t{ans}")
    return out


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
