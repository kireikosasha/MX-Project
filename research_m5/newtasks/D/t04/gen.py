#!/usr/bin/env python3
"""Synthetic reasoning task: kinds and properties.

Individuals belong to kinds. Kinds may belong to the kind "creature", and
kinds carry positive or negative properties ("Every K is P", "No K is P").
Questions ask whether an individual has a property, whether it belongs to a
kind, or which kinds it belongs to; each answer follows from a rule plus a
membership statement, so it cannot be read from a single sentence.

Usage: python gen.py N SEED
"""
import copy
import random
import sys

NAMES = ["Tom", "Ann", "Max", "Sue", "Ben", "Kim"]
KINDS = ["wolf", "bear", "hawk", "cat"]
SUPER = "creature"
ALL_KINDS = KINDS + [SUPER]
PROPS = ["fierce", "lazy", "happy", "quiet", "hungry", "tall"]


class World:
    def __init__(self):
        self.members = {}  # name -> set of direct kinds
        self.sub = set()   # kinds K with "Every K is a creature."
        self.pos = {}      # kind -> set of props from "Every K is P."
        self.neg = {}      # kind -> set of props from "No K is P."

    def closure(self, name):
        kinds = set(self.members.get(name, ()))
        for k in list(kinds):
            if k in self.sub:
                kinds.add(SUPER)
        return kinds

    def props(self, name):
        pos, neg = set(), set()
        for k in self.closure(name):
            pos |= self.pos.get(k, set())
            neg |= self.neg.get(k, set())
        return pos, neg

    def consistent(self):
        for name in self.members:
            pos, neg = self.props(name)
            if pos & neg:
                return False
        return True


def propose_statement(world, rng):
    """Return (text, new_world) for a consistent new statement, or None."""
    w = copy.deepcopy(world)
    r = rng.random()
    if r < 0.35:
        name = rng.choice(NAMES)
        k = rng.choice(ALL_KINDS)
        if k in w.members.get(name, ()):
            return None
        w.members.setdefault(name, set()).add(k)
        text = f"{name} is a {k}."
    elif r < 0.50:
        k = rng.choice(KINDS)
        if k in w.sub:
            return None
        w.sub.add(k)
        text = f"Every {k} is a {SUPER}."
    elif r < 0.80:
        k = rng.choice(ALL_KINDS)
        p = rng.choice(PROPS)
        if p in w.pos.get(k, ()):
            return None
        w.pos.setdefault(k, set()).add(p)
        text = f"Every {k} is {p}."
    else:
        k = rng.choice(ALL_KINDS)
        p = rng.choice(PROPS)
        if p in w.neg.get(k, ()):
            return None
        w.neg.setdefault(k, set()).add(p)
        text = f"No {k} is {p}."
    if not w.consistent():
        return None
    return text, w


def propose_question(world, rng):
    """Return (text, answer) for a question decided by the current world, or None."""
    named = sorted(world.members)
    if not named:
        return None
    name = rng.choice(named)
    r = rng.random()
    if r < 0.4:
        p = rng.choice(PROPS)
        pos, neg = world.props(name)
        if p in pos:
            ans = "yes"
        elif p in neg:
            ans = "no"
        else:
            return None
        return f"Is {name} {p}?", ans
    if r < 0.7:
        k = rng.choice(ALL_KINDS)
        if k in world.members[name]:
            return None  # stated directly, a single sentence would do
        if k not in world.closure(name):
            return None
        return f"Is {name} a {k}?", "yes"
    cl = world.closure(name)
    if len(cl) <= len(world.members[name]):
        return None  # nothing was derived, so the answer is a single sentence
    return f"What is {name}?", ",".join(sorted(cl))


def make_story(rng):
    while True:
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        world = World()
        lines = []  # (text, answer or None)
        asked = set()
        st = qs = fails = 0
        ok = True
        while st < n_st or qs < n_q:
            want_q = qs < n_q and st >= 2 and (st >= n_st or rng.random() < 0.4)
            if want_q:
                cand = propose_question(world, rng)
                if cand and cand[0] not in asked:
                    asked.add(cand[0])
                    lines.append(cand)
                    qs += 1
                    continue
            elif st < n_st:
                res = None
                for _ in range(50):
                    res = propose_statement(world, rng)
                    if res:
                        break
                if res:
                    world = res[1]
                    lines.append((res[0], None))
                    st += 1
                    continue
            fails += 1
            if fails > 300:
                ok = False
                break
        if ok:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
