#!/usr/bin/env python3
"""Synthetic reading-comprehension stories about things whose states change.

Usage: python gen.py N SEED

Prints N stories. Each line is "<number> <sentence>"; a question line carries
its answer after one TAB. Numbers restart at 1 for every story. Everything is
produced by a small simulation of a house with people, rooms, a key, a ball,
a box with a lid, a lamp and a ball whose colour changes.
"""
import random
import sys

NAMES = ["ann", "ben", "cara"]
ROOMS = ["kitchen", "hall", "garden"]
THINGS = ["key", "ball", "box"]
CARRY_INTO_BOX = ["key", "ball"]
COLORS = ["red", "blue", "green"]
TRUE_WORDS = {"on": True, "off": False, "open": True, "closed": False}


def cap(s):
    return s[0].upper() + s[1:]


class World(object):
    """Simulated state of one story.

    loc[thing] = (kind, value, deps)
        kind "room":   value is a room name
        kind "person": value is the name of the holder
        kind "box":    the thing lies inside the box
    deps is a frozenset of the statement numbers that fixed this position.
    person[name] = (room, deps)
    state[key] = [value, count]   for keys "lamp", "box", "ball"
    """

    def __init__(self, rng):
        self.rng = rng
        self.line = 0
        self.person = {}
        self.loc = {}
        self.state = {}

    # ----- queries -----------------------------------------------------
    def where(self, thing):
        """Return (room, deps) for a thing, or None if its place is unknown."""
        e = self.loc.get(thing)
        if e is None:
            return None
        kind, val, deps = e
        if kind == "room":
            return val, deps
        if kind == "person":
            room, pdeps = self.person[val]
            return room, deps | pdeps
        r = self.where("box")
        if r is None:
            return None
        return r[0], deps | r[1]

    def at_hand(self, thing, name):
        """True if the thing is held by name, or lies loose in name's room."""
        e = self.loc.get(thing)
        p = self.person.get(name)
        if e is None or p is None:
            return False
        if e[0] == "person":
            return e[1] == name
        return e[0] == "room" and e[1] == p[0]

    def is_state(self, key, value):
        s = self.state.get(key)
        return s is not None and s[0] == value

    # ----- mutators (self.line is the number of the statement) ----------
    def go(self, name, room):
        self.person[name] = (room, frozenset([self.line]))

    def place(self, thing, room):
        self.loc[thing] = ("room", room, frozenset([self.line]))

    def take(self, name, thing):
        e = self.loc[thing]
        self.loc[thing] = ("person", name, e[2] | {self.line})

    def drop(self, name, thing):
        e = self.loc[thing]
        room, pdeps = self.person[name]
        self.loc[thing] = ("room", room, e[2] | pdeps | {self.line})

    def put_in_box(self, name, thing):
        e = self.loc[thing]
        self.loc[thing] = ("box", "box", e[2] | {self.line})

    def set_state(self, key, value):
        old = self.state.get(key)
        count = (old[1] if old else 0) + 1
        self.state[key] = [value, count]

    # ----- options -----------------------------------------------------
    def statement_options(self):
        """Return a list of categories; each is a list of (text, fn, args)."""
        moves, places, takes, drops = [], [], [], []
        boxins, boxstate, lamps, colors = [], [], [], []

        for p in NAMES:
            cur = self.person.get(p)
            for room in ROOMS:
                if cur is None or cur[0] != room:
                    moves.append(("%s went to the %s." % (p, room),
                                  self.go, (p, room)))

        for t in THINGS:
            if t not in self.loc:
                for room in ROOMS:
                    places.append(("the %s is in the %s." % (t, room),
                                   self.place, (t, room)))

        for p in NAMES:
            for t in THINGS:
                e = self.loc.get(t)
                if e is None:
                    continue
                if e[0] == "room" and self.at_hand(t, p):
                    takes.append(("%s took the %s." % (p, t),
                                  self.take, (p, t)))
                if e[0] == "person" and e[1] == p:
                    drops.append(("%s dropped the %s." % (p, t),
                                  self.drop, (p, t)))

        for p in NAMES:
            if self.is_state("box", True) and self.at_hand("box", p):
                for t in CARRY_INTO_BOX:
                    if self.at_hand(t, p):
                        boxins.append(("%s put the %s in the box." % (p, t),
                                       self.put_in_box, (p, t)))

        for word in ("open", "closed"):
            v = TRUE_WORDS[word]
            if not self.is_state("box", v):
                boxstate.append(("the box is %s." % word,
                                 self.set_state, ("box", v)))
        for p in NAMES:
            if self.at_hand("box", p):
                if self.is_state("box", False):
                    boxstate.append(("%s opened the box." % p,
                                     self.set_state, ("box", True)))
                if self.is_state("box", True):
                    boxstate.append(("%s closed the box." % p,
                                     self.set_state, ("box", False)))

        for word in ("on", "off"):
            v = TRUE_WORDS[word]
            if not self.is_state("lamp", v):
                lamps.append(("the lamp is %s." % word,
                              self.set_state, ("lamp", v)))
                for p in NAMES:
                    lamps.append(("%s switched the lamp %s." % (p, word),
                                  self.set_state, ("lamp", v)))

        for c in COLORS:
            if not self.is_state("ball", c):
                colors.append(("the ball is %s." % c,
                               self.set_state, ("ball", c)))

        return [moves, places, takes, drops, boxins, boxstate, lamps, colors]

    def question_options(self):
        """Return a list of categories; each is a list of (text, answer).

        Location questions need a position that rests on at least two
        statements; state and colour questions need a value that has been
        set at least twice, so the reader must follow the changes.
        """
        where, isin, state, color = [], [], [], []

        for t in THINGS:
            r = self.where(t)
            if r is not None and len(r[1]) >= 2:
                where.append(("where is the %s?" % t, r[0]))
                for room in ROOMS:
                    ans = "yes" if room == r[0] else "no"
                    isin.append(("is the %s in the %s?" % (t, room), ans))

        for key, words in (("lamp", ("on", "off")),
                           ("box", ("open", "closed"))):
            s = self.state.get(key)
            if s is not None and s[1] >= 2:
                for word in words:
                    ans = "yes" if s[0] == TRUE_WORDS[word] else "no"
                    state.append(("is the %s %s?" % (key, word), ans))

        s = self.state.get("ball")
        if s is not None and s[1] >= 2:
            color.append(("what color is the ball?", s[0]))

        return [where, isin, state, color]


STMT_WEIGHTS = [2, 2, 3, 3, 2, 1, 1, 1]   # moves, places, takes, drops,
                                          # box in, box state, lamp, colour
QUES_WEIGHTS = [3, 3, 1, 1]               # where, is-in, state, colour


def pick(rng, cats, weights):
    """Choose a non-empty category by weight, then one option within it."""
    pairs = [(c, w) for c, w in zip(cats, weights) if c]
    if not pairs:
        return None
    chosen = rng.choices([c for c, _ in pairs], weights=[w for _, w in pairs])[0]
    return rng.choice(chosen)


def gen_story(rng):
    """Return a list of (text, answer-or-None) lines for one story."""
    while True:
        w = World(rng)
        items = []
        asked = set()
        n_stmt_goal = rng.randint(3, 12)
        n_q_goal = rng.randint(1, 5)
        n_stmt = 0
        n_q = 0
        n_loc = 0
        ok = True

        def ask():
            """Ask one new question; return None if none was asked."""
            q = pick(rng, w.question_options(), QUES_WEIGHTS)
            if q is None or q[0] in asked:
                return None
            asked.add(q[0])
            items.append((cap(q[0]), q[1]))
            return q[0].startswith("where") or " in the " in q[0]

        while n_stmt < n_stmt_goal:
            choice = pick(rng, w.statement_options(), STMT_WEIGHTS)
            if choice is None:
                ok = False
                break
            text, fn, args = choice
            n_stmt += 1
            w.line = len(items) + 1
            fn(*args)
            items.append((cap(text), None))
            if n_stmt >= 2 and n_q < n_q_goal and rng.random() < 0.4:
                loc = ask()
                if loc is not None:
                    n_q += 1
                    n_loc += loc

        tries = 0
        while ok and n_q < n_q_goal and tries < 20:
            tries += 1
            loc = ask()
            if loc is not None:
                n_q += 1
                n_loc += loc

        # Every story must contain at least one location question, which
        # can only be answered by following a chain of several statements.
        if ok and n_q >= 1 and n_loc >= 1:
            return items


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: gen.py N SEED\n")
        return 1
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(gen_story(rng), 1):
            line = "%d %s" % (i, text)
            if ans is not None:
                line += "\t" + ans
            out.append(line)
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
