#!/usr/bin/env python3
"""Synthetic reading-comprehension stories with rule-based world states.

Usage: python gen.py N SEED
Prints N stories. Every story states two standing rules, then simple facts
about people moving and carrying things, with questions whose answers are
computed by simulating the world.
"""
import random
import sys

PEOPLE = ["ana", "ben", "cole", "dara"]
ROOMS = ["hall", "kitchen", "garden", "cellar"]
ITEMS = ["key", "lamp", "apple", "book"]
BOX = "box"
THINGS = ITEMS + [BOX]

RULE_LINES = [
    "Whatever someone holds goes wherever that someone goes.",
    "Whatever lies in the box goes wherever the box goes.",
]


def cap(word):
    return word[0].upper() + word[1:]


class World:
    def __init__(self):
        self.person_room = {}   # person -> room
        self.holder = {}        # thing -> person carrying it
        self.room_of = {}       # thing lying free in a room -> room
        self.inside = set()     # items lying inside the box

    def loc(self, thing):
        if thing in self.inside:
            return self.loc(BOX)
        if thing in self.holder:
            return self.person_room[self.holder[thing]]
        return self.room_of.get(thing)


def fact_options(w):
    opts = []
    for p in PEOPLE:
        cur = w.person_room.get(p)
        for r in ROOMS:
            if r != cur:
                opts.append(("move", p, r))
    for p in PEOPLE:
        cur = w.person_room.get(p)
        if cur is None:
            continue
        for x in THINGS:
            if x in w.holder or x in w.inside:
                continue
            if w.loc(x) == cur:
                opts.append(("take", p, x))
        for x in THINGS:
            if w.holder.get(x) == p:
                opts.append(("drop", p, x))
                if x != BOX and w.loc(BOX) == cur:
                    opts.append(("store", p, x))
    for x in THINGS:
        if w.loc(x) is None:
            for r in ROOMS:
                opts.append(("place", x, r))
            if x != BOX and w.loc(BOX) is not None:
                opts.append(("inbox", x, None))
    return opts


def apply_fact(w, opt):
    kind, a, b = opt
    if kind == "move":
        w.person_room[a] = b
        return "%s went to the %s." % (cap(a), b)
    if kind == "take":
        w.room_of.pop(b, None)
        w.holder[b] = a
        return "%s took the %s." % (cap(a), b)
    if kind == "drop":
        del w.holder[b]
        w.room_of[b] = w.person_room[a]
        return "%s put down the %s." % (cap(a), b)
    if kind == "store":
        del w.holder[b]
        w.room_of.pop(b, None)
        w.inside.add(b)
        return "%s put the %s in the box." % (cap(a), b)
    if kind == "place":
        w.room_of[a] = b
        return "The %s lies in the %s." % (a, b)
    if kind == "inbox":
        w.inside.add(a)
        return "The %s lies in the box." % a
    raise ValueError(kind)


def question_options(w):
    placed = [x for x in THINGS if w.loc(x) is not None]
    rule_opts, other_opts = [], []
    for x in placed:
        rule = x in w.holder or x in w.inside
        here = w.loc(x)
        target = rule_opts if rule else other_opts
        target.append(("Where is the %s?" % x, here))
        for r in ROOMS:
            target.append(("Is the %s in the %s?" % (x, r),
                           "yes" if here == r else "no"))
    for r in ROOMS:
        members = [x for x in placed if w.loc(x) == r]
        if len(members) >= 2:
            other_opts.append(("Which things are in the %s?" % r,
                               ",".join(sorted(members))))
        if len(members) <= 3:
            other_opts.append(("How many things are in the %s?" % r,
                               str(len(members))))
    for p in PEOPLE:
        if w.person_room.get(p) is not None:
            other_opts.append(("Where is %s?" % cap(p), w.person_room[p]))
    return rule_opts, other_opts


def pick_question(w, rng):
    rule_opts, other_opts = question_options(w)
    if rule_opts and (not other_opts or rng.random() < 0.7):
        pool = rule_opts
    elif other_opts:
        pool = other_opts
    else:
        return None
    return rng.choice(pool)


def build_story(rng):
    while True:
        w = World()
        lines = [(t, None) for t in RULE_LINES]
        f = rng.randint(2, 10)               # fact statements
        q = rng.randint(1, min(5, f))        # questions
        slots = [rng.randint(1, f) for _ in range(q)]
        ok = True
        for k in range(1, f + 1):
            opts = fact_options(w)
            lines.append((apply_fact(w, rng.choice(opts)), None))
            for _ in range(slots.count(k)):
                qa = pick_question(w, rng)
                if qa is None:
                    ok = False
                    break
                lines.append(qa)
            if not ok:
                break
        if ok:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(build_story(rng), 1):
            if ans is None:
                out.append("%d %s" % (i, text))
            else:
                out.append("%d %s\t%s" % (i, text, ans))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
