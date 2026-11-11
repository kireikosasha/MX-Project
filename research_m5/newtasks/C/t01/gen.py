#!/usr/bin/env python3
"""Synthetic container-tracking reading comprehension stories.

Usage: python gen.py N SEED

People carry objects, put them into or take them out of containers, and
move containers between places. Questions ask where things are now, which
requires following the chain of moves through the statements above.
"""
import random
import sys

PEOPLE = ["Ann", "Ben", "Cal", "Dan"]
OBJECTS = ["key", "coin", "ball"]
CONTAINERS = ["box", "bag"]
PLACES = ["kitchen", "garden", "hall", "office"]


class World:
    """Simulated state. An entity is 'introduced' once its first statement is out."""

    def __init__(self):
        self.person_place = {}   # person -> place
        self.cont_place = {}     # container -> place
        self.obj = {}            # object -> {"container", "holder", "place"}

    def clone(self):
        w = World()
        w.person_place = dict(self.person_place)
        w.cont_place = dict(self.cont_place)
        w.obj = {k: dict(v) for k, v in self.obj.items()}
        return w

    def obj_loc(self, o):
        d = self.obj[o]
        if d["container"] is not None:
            return self.cont_place[d["container"]]
        if d["holder"] is not None:
            return self.person_place[d["holder"]]
        return d["place"]


def do_init(rng, w, kind, name):
    n = w.clone()
    if kind == "person":
        pl = rng.choice(PLACES)
        n.person_place[name] = pl
        return f"{name} is in the {pl}.", n
    if kind == "cont":
        pl = rng.choice(PLACES)
        n.cont_place[name] = pl
        return f"The {name} is in the {pl}.", n
    options = [("place", p) for p in PLACES] + [("cont", c) for c in w.cont_place]
    dk, dest = rng.choice(options)
    if dk == "cont":
        n.obj[name] = {"container": dest, "holder": None, "place": None}
    else:
        n.obj[name] = {"container": None, "holder": None, "place": dest}
    return f"The {name} is in the {dest}.", n


def actions(w):
    """All legal (text, new_world) actions in the current world."""
    out = []
    for p, here in w.person_place.items():
        for c, cpl in w.cont_place.items():
            if cpl != here:
                continue
            for o, d in w.obj.items():
                if d["container"] != c:
                    n = w.clone()
                    n.obj[o] = {"container": c, "holder": None, "place": None}
                    out.append((f"{p} put the {o} in the {c}.", n))
                else:
                    n = w.clone()
                    n.obj[o] = {"container": None, "holder": p, "place": None}
                    out.append((f"{p} took the {o} out of the {c}.", n))
            for pl in PLACES:
                if pl != cpl:
                    n = w.clone()
                    n.cont_place[c] = pl
                    out.append((f"{p} moved the {c} to the {pl}.", n))
        for pl in PLACES:
            if pl != here:
                n = w.clone()
                n.person_place[p] = pl
                out.append((f"{p} went to the {pl}.", n))
    return out


def questions(w):
    """All (question, answer) pairs that the current world determines."""
    out = []
    for p, pl in w.person_place.items():
        out.append((f"Where is {p}?", pl))
        for x in PLACES:
            out.append((f"Is {p} in the {x}?", "yes" if pl == x else "no"))
    for c, cpl in w.cont_place.items():
        out.append((f"Where is the {c}?", cpl))
        for x in PLACES:
            out.append((f"Is the {c} in the {x}?", "yes" if cpl == x else "no"))
        contents = sorted(o for o in w.obj if w.obj[o]["container"] == c)
        if contents:
            out.append((f"What is in the {c}?", ",".join(contents)))
    for o, d in w.obj.items():
        out.append((f"Where is the {o}?", w.obj_loc(o)))
        for c in w.cont_place:
            out.append((f"Is the {o} in the {c}?", "yes" if d["container"] == c else "no"))
        if d["holder"] is not None:
            out.append((f"Who has the {o}?", d["holder"].lower()))
    return out


def try_story(rng):
    people = rng.sample(PEOPLE, rng.randint(2, 3))
    conts = rng.sample(CONTAINERS, rng.randint(1, 2))
    objs = rng.sample(OBJECTS, rng.randint(1, 3))
    pending = ([("person", p) for p in people]
               + [("cont", c) for c in conts]
               + [("obj", o) for o in objs])
    rng.shuffle(pending)
    n_init = len(pending)
    target = rng.randint(max(n_init, 4), 12)

    w = World()
    stmts = []          # (text, world after this statement)
    n_actions = 0
    while len(stmts) < target:
        acts = actions(w) if w.person_place else []
        if pending and (not acts or rng.random() < 0.5):
            kind, name = pending.pop(0)
            text, w = do_init(rng, w, kind, name)
        elif acts:
            text, w = rng.choice(acts)
            n_actions += 1
        else:
            break
        stmts.append((text, w))

    if len(stmts) < 3 or n_actions < 2:
        return None

    cands = []
    for p in range(2, len(stmts) + 1):
        for q, a in questions(stmts[p - 1][1]):
            cands.append((p, q, a))
    if not cands:
        return None

    k = rng.randint(1, 5)
    chosen = sorted(rng.sample(cands, min(k, len(cands))), key=lambda t: t[0])
    final = []
    seen = set()
    for p, q, a in chosen:
        if q not in seen:
            seen.add(q)
            final.append((p, q, a))

    at = {}
    for p, q, a in final:
        at.setdefault(p, []).append((q, a))

    lines = []
    num = 0
    for idx, (text, _) in enumerate(stmts, start=1):
        num += 1
        lines.append(f"{num} {text}")
        for q, a in at.get(idx, []):
            num += 1
            lines.append(f"{num} {q}\t{a}")
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
