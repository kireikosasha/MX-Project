#!/usr/bin/env python3
"""Synthetic reading-comprehension task: kinds and their properties.

Usage: python gen.py N SEED

World model: named individuals belong to kinds; kinds form a taxonomy
("Every X is a Y."); kinds carry properties ("Every X is P."); individuals
may also carry properties directly ("N is P."). Questions ask whether an
individual has a property or belongs to a kind, or list its properties.
Answers are derived by closure over the statements above the question.
Closed world: a property or kind that cannot be derived is answered "no".
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cal", "Dana", "Eli", "Fay", "Gus", "Hal"]
KINDS = ["bird", "fish", "frog", "cat", "creature"]
PROPS = ["loud", "wet", "big", "cold", "fast", "green"]
MAX_TRIES = 2000


class World:
    def __init__(self):
        self.parents = {k: set() for k in KINDS}      # kind -> direct parent kinds
        self.kind_props = {k: set() for k in KINDS}   # kind -> properties of all members
        self.member = {}                              # name -> direct kinds
        self.prop = {}                                # name -> directly stated properties

    def ancestors(self, k):
        seen = {k}
        stack = [k]
        while stack:
            x = stack.pop()
            for p in self.parents[x]:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def kinds_of(self, name):
        out = set()
        for k in self.member.get(name, ()):
            out |= self.ancestors(k)
        return out

    def props_of(self, name):
        out = set(self.prop.get(name, ()))
        for k in self.kinds_of(name):
            out |= self.kind_props[k]
        return out


def gen_statement(w, names, rng):
    """Apply one random new statement to the world and return its text, or None."""
    for _ in range(60):
        t = rng.choices(["kk", "kp", "mem", "np"], weights=[2, 3, 3, 2])[0]
        if t == "kk":
            k, p = rng.sample(KINDS, 2)
            if p in w.ancestors(k) or k in w.ancestors(p):
                continue  # already implied, or would create a cycle
            w.parents[k].add(p)
            return f"Every {k} is a {p}."
        if t == "kp":
            k = rng.choice(KINDS)
            p = rng.choice(PROPS)
            if p in w.kind_props[k]:
                continue
            w.kind_props[k].add(p)
            return f"Every {k} is {p}."
        n = rng.choice(names)
        if t == "mem":
            k = rng.choice(KINDS)
            if k in w.kinds_of(n):
                continue
            w.member.setdefault(n, set()).add(k)
            return f"{n} is a {k}."
        p = rng.choice(PROPS)
        if p in w.props_of(n):
            continue
        w.prop.setdefault(n, set()).add(p)
        return f"{n} is {p}."
    return None


def gen_question(w, asked, rng):
    """Return (question_text, answer) about the current world, or None."""
    named = sorted(set(w.member) | set(w.prop))
    if not named:
        return None
    for _ in range(60):
        n = rng.choice(named)
        t = rng.choice(["prop", "kind", "list"])
        if t == "prop":
            p = rng.choice(PROPS)
            text = f"Is {n} {p}?"
            yes = p in w.props_of(n)
            if yes and p in w.prop.get(n, ()):
                continue  # answer is stated directly, not reasoned out
            ans = "yes" if yes else "no"
        elif t == "kind":
            k = rng.choice(KINDS)
            text = f"Is {n} a {k}?"
            yes = k in w.kinds_of(n)
            if yes and k in w.member.get(n, ()):
                continue
            ans = "yes" if yes else "no"
        else:
            ps = sorted(w.props_of(n))
            if len(ps) < 2:
                continue
            text = f"Which properties does {n} have?"
            ans = ",".join(ps)
        if text in asked:
            continue
        asked.add(text)
        return text, ans
    return None


def try_story(rng):
    w = World()
    names = rng.sample(NAMES, rng.randint(2, 4))
    S = rng.randint(3, 12)
    Q = rng.randint(1, 5)
    qpos = sorted(rng.randint(1, S) for _ in range(Q))
    asked = set()
    lines = []          # (text, answer or None)
    qi = 0
    yes_count = 0
    for i in range(1, S + 1):
        text = gen_statement(w, names, rng)
        if text is None:
            return None
        lines.append((text, None))
        while qi < Q and qpos[qi] == i:
            q = gen_question(w, asked, rng)
            if q is None:
                return None
            if q[1] == "yes":
                yes_count += 1
            lines.append(q)
            qi += 1
    if yes_count == 0:
        return None
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        for _ in range(MAX_TRIES):
            story = try_story(rng)
            if story is not None:
                break
        if story is None:
            raise RuntimeError("could not generate a story")
        for i, (text, ans) in enumerate(story, 1):
            out.append(f"{i} {text}" if ans is None else f"{i} {text}\t{ans}")
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
