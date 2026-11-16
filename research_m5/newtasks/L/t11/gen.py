#!/usr/bin/env python3
"""Generate synthetic reading-comprehension stories.

People carry things between places. The story states two general rules
(held things move with their holder; only someone holding the lamp can open
the shed) and simple facts. Answers are computed by simulating the facts in
order, so a question often needs several statements and a rule to answer.

Usage: python gen.py N SEED
Prints N stories. Numbering restarts at 1 for each story.
"""
import random
import sys

NAMES = ["ann", "bob", "cal", "dee"]
OBJECTS = ["key", "box", "lamp"]
PLACES = ["hall", "shed", "garden"]

RULE_LINES = [
    "anything held moves with its holder.",
    "only someone who has the lamp can open the shed.",
]


def act_text(a):
    kind = a[0]
    if kind == "goes":
        return f"{a[1]} goes to the {a[2]}."
    if kind == "is":
        return f"the {a[1]} is in the {a[2]}."
    if kind == "takes":
        return f"{a[1]} takes the {a[2]}."
    if kind == "drops":
        return f"{a[1]} drops the {a[2]}."
    if kind == "gives":
        return f"{a[1]} gives the {a[2]} to {a[3]}."
    raise ValueError(kind)


class World:
    def __init__(self):
        self.loc = {}     # person -> place (known once the person has gone somewhere)
        self.where = {}   # object -> place, used only when nobody holds it
        self.holder = {}  # object -> person holding it, or None

    def obj_loc(self, o):
        h = self.holder.get(o)
        if h is not None:
            return self.loc.get(h)  # rule: a held thing is wherever its holder is
        return self.where.get(o)

    def apply(self, a):
        kind = a[0]
        if kind == "goes":
            self.loc[a[1]] = a[2]
        elif kind == "is":
            self.where[a[1]] = a[2]
        elif kind == "takes":
            self.holder[a[2]] = a[1]
        elif kind == "drops":
            self.where[a[2]] = self.loc[a[1]]
            self.holder[a[2]] = None
        elif kind == "gives":
            self.holder[a[2]] = a[3]


def candidates(w):
    """All valid next actions in the current world state."""
    out = []
    for p in NAMES:
        for r in PLACES:
            if w.loc.get(p) != r:
                out.append(("goes", p, r))
    for o in OBJECTS:
        h = w.holder.get(o)
        if h is None:
            for r in PLACES:
                out.append(("is", o, r))
            here = w.obj_loc(o)
            if here is not None:
                for p in NAMES:
                    if w.loc.get(p) == here:
                        out.append(("takes", p, o))
        else:
            out.append(("drops", h, o))
            if w.loc.get(h) is not None:
                for q in NAMES:
                    if q != h and w.loc.get(q) == w.loc[h]:
                        out.append(("gives", h, o, q))
    return out


def questions(w, rng):
    """Answerable questions in the current state as (text, answer, needs_rule)."""
    qs = []
    for o in OBJECTS:
        loc = w.obj_loc(o)
        held = w.holder.get(o) is not None
        if loc is not None:
            qs.append((f"where is the {o}?", loc, held))
            r = rng.choice(PLACES)
            qs.append((f"is the {o} in the {r}?", "yes" if loc == r else "no", held))
        if held:
            qs.append((f"who has the {o}?", w.holder[o], False))
    for p in NAMES:
        if p in w.loc:
            qs.append((f"where is {p}?", w.loc[p], False))
        qs.append((f"can {p} open the shed?",
                   "yes" if w.holder.get("lamp") == p else "no", False))
    return qs


def make_story(rng):
    while True:
        w = World()
        init = [("goes", p, rng.choice(PLACES)) for p in rng.sample(NAMES, 2)]
        init += [("is", o, rng.choice(PLACES))
                 for o in rng.sample(OBJECTS, rng.randint(1, 2))]
        n_facts = rng.randint(len(init) + 1, 10)  # 3 to 12 statements in total
        pending = list(init)
        lines = list(RULE_LINES)
        q_target = rng.randint(1, 5)
        q_done = 0
        rule_q = False

        for i in range(n_facts):
            act = pending.pop(0) if pending else rng.choice(candidates(w))
            w.apply(act)
            lines.append(act_text(act))
            if i + 1 >= len(init) and q_done < q_target and rng.random() < 0.5:
                q = rng.choice(questions(w, rng))
                lines.append(f"{q[0]}\t{q[1]}")
                q_done += 1
                rule_q = rule_q or q[2]

        # Make sure the story ends with a question that needs a rule.
        if not rule_q and q_done < 5:
            rq = [q for q in questions(w, rng) if q[2]]
            if rq:
                q = rng.choice(rq)
                lines.append(f"{q[0]}\t{q[1]}")
                q_done += 1
                rule_q = True

        if q_done == 0 or not rule_q:
            continue  # rejected: try another story
        return lines


def main():
    if len(sys.argv) != 3:
        print("usage: python gen.py N SEED", file=sys.stderr)
        sys.exit(1)
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            print(f"{i} {line}")


if __name__ == "__main__":
    main()
