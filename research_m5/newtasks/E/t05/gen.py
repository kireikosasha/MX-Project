"""Synthetic reading task about changing states.

People move around, pick up and drop things, and switch a door and a lamp
open/closed and on/off. Questions ask the state of things now.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "bob", "cal", "dan"]
OBJECTS = ["key", "ball", "book", "cup"]
PLACES = ["kitchen", "garden", "hall", "office"]


class World:
    def __init__(self):
        self.person = {}  # name -> (place, deps)
        self.thing = {}   # object -> [holder or None, place or None, deps]
        self.door = None  # (state, deps)
        self.lamp = None  # (state, deps)

    def where_thing(self, o):
        holder, place, deps = self.thing[o]
        if holder is not None:
            p, d = self.person[holder]
            return p, deps | d
        return place, deps


def put_person(w, n, p):
    def f(i):
        w.person[n] = (p, {i})
    return f


def put_thing(w, o, p):
    def f(i):
        w.thing[o] = [None, p, {i}]
    return f


def set_state(w, attr, s):
    def f(i):
        old = getattr(w, attr)
        deps = {i} | (old[1] if old else set())
        setattr(w, attr, (s, deps))
    return f


def move(w, n, p):
    def f(i):
        w.person[n] = (p, {i})
    return f


def take(w, n, o):
    def f(i):
        _, _, deps = w.thing[o]
        w.thing[o] = [n, None, deps | w.person[n][1] | {i}]
    return f


def drop(w, n, o):
    def f(i):
        p, d = w.person[n]
        w.thing[o] = [None, p, d | {i}]
    return f


def statements(w):
    """All valid next statements as (sentence, apply(line_no))."""
    out = []
    for n in NAMES:
        if n not in w.person:
            for p in PLACES:
                out.append((f"{n} is in the {p}.", put_person(w, n, p)))
    for o in OBJECTS:
        if o not in w.thing:
            for p in PLACES:
                out.append((f"The {o} is in the {p}.", put_thing(w, o, p)))
    if w.door is None:
        for s in ("open", "closed"):
            out.append((f"The door is {s}.", set_state(w, "door", s)))
    if w.lamp is None:
        for s in ("on", "off"):
            out.append((f"The lamp is {s}.", set_state(w, "lamp", s)))
    for n, (cur, _) in w.person.items():
        for p in PLACES:
            if p != cur:
                out.append((f"{n} went to the {p}.", move(w, n, p)))
        for o, (holder, place, _) in w.thing.items():
            if holder is None and place == cur:
                out.append((f"{n} took the {o}.", take(w, n, o)))
            if holder == n:
                out.append((f"{n} dropped the {o}.", drop(w, n, o)))
        if w.door is not None:
            s = "closed" if w.door[0] == "open" else "open"
            verb = "opened" if s == "open" else "closed"
            out.append((f"{n} {verb} the door.", set_state(w, "door", s)))
        if w.lamp is not None:
            s = "off" if w.lamp[0] == "on" else "on"
            out.append((f"{n} switched the lamp {s}.", set_state(w, "lamp", s)))
    return out


def questions(w):
    """Answerable questions (question, answer) whose answer rests on at
    least two statements."""
    out = []
    for o in w.thing:
        place, deps = w.where_thing(o)
        out.append((f"Where is the {o}?", place, deps))
        holder, _, hdeps = w.thing[o]
        out.append((f"Who has the {o}?", holder or "nobody", hdeps))
    for n, (place, deps) in w.person.items():
        out.append((f"Where is {n}?", place, deps))
    if w.door is not None:
        out.append(("Is the door open?",
                    "yes" if w.door[0] == "open" else "no", w.door[1]))
    if w.lamp is not None:
        out.append(("Is the lamp on?",
                    "yes" if w.lamp[0] == "on" else "no", w.lamp[1]))
    return [(q, a) for q, a, d in out if len(d) >= 2]


def fresh(w, asked):
    """Answerable questions not already asked in this story."""
    return [(q, a) for q, a in questions(w) if q not in asked]


def make_story(rng):
    while True:
        w = World()
        lines = []
        asked = set()
        nstmt = 0
        nq = 0
        target = rng.randint(3, 12)
        want_q = rng.randint(1, 5)
        while nstmt < target:
            if nq < want_q and nstmt >= 2 and rng.random() < 0.4:
                valid = fresh(w, asked)
                if valid:
                    q, a = rng.choice(valid)
                    asked.add(q)
                    lines.append(f"{len(lines) + 1} {q}\t{a}")
                    nq += 1
                    continue
            text, apply = rng.choice(statements(w))
            num = len(lines) + 1
            apply(num)
            lines.append(f"{num} {text}")
            nstmt += 1
        while nq < want_q:
            valid = fresh(w, asked)
            if not valid:
                break
            q, a = rng.choice(valid)
            asked.add(q)
            lines.append(f"{len(lines) + 1} {q}\t{a}")
            nq += 1
        if nq >= 1:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
