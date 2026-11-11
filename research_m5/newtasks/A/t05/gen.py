#!/usr/bin/env python3
"""Synthetic state-tracking reading task (switches and hatches that change state).

Usage: python gen.py N SEED
Prints N stories to standard output. Numbers restart at 1 in every story.
A statement line ends with '.'. A question line is 'question?<TAB>answer'.
All randomness comes from random.Random(SEED).
"""
import random
import sys

NAMES = ["ann", "ben", "cy", "dora"]
KIND_OF = {"lamp": "switch", "fan": "switch", "door": "hatch", "gate": "hatch"}
STATES = {"switch": ["on", "off"], "hatch": ["open", "closed"]}
VERB = {"on": "turns on", "off": "turns off", "open": "opens", "closed": "closes"}


def other(kind, state):
    a, b = STATES[kind]
    return b if state == a else a


class World:
    def __init__(self):
        self.state = {d: None for d in KIND_OF}   # None = never set in this story
        self.last = {}                            # device -> actor of last change
        self.copies = {}                          # source -> set of targets that follow it
        self.mentioned = set()                    # devices named in statements so far

    def set_state(self, dev, new, actor, seen=None):
        """Change a device, then let every device that copies it follow."""
        if seen is None:
            seen = set()
        if dev in seen:
            return
        seen.add(dev)
        self.state[dev] = new
        self.last[dev] = actor
        for tgt in sorted(self.copies.get(dev, ())):
            if self.state[tgt] != new:
                self.set_state(tgt, new, actor, seen)


def make_statement(rng, w, devs, names):
    pairs = [(s, t) for s in devs for t in devs
             if s != t and KIND_OF[s] == KIND_OF[t] and t not in w.copies.get(s, ())]
    if pairs and rng.random() < 0.3:
        s, t = rng.choice(pairs)
        w.copies.setdefault(s, set()).add(t)
        w.mentioned.update([s, t])
        return f"The {t} copies the {s}."
    actor = rng.choice(names)
    dev = rng.choice(devs)
    kind = KIND_OF[dev]
    cur = w.state[dev]
    new = rng.choice(STATES[kind]) if cur is None else other(kind, cur)
    w.set_state(dev, new, actor)
    w.mentioned.add(dev)
    return f"{actor.capitalize()} {VERB[new]} the {dev}."


def make_question(rng, w):
    """Return (question_text_with_mark, answer) or None if nothing is answerable."""
    known = sorted(d for d in w.mentioned if w.state[d] is not None)
    cands = []
    if known:
        cands += ["yesno", "who"]
    cands.append("which")
    rng.shuffle(cands)
    for c in cands:
        if c == "yesno":
            d = rng.choice(known)
            s = rng.choice(STATES[KIND_OF[d]])
            ans = "yes" if w.state[d] == s else "no"
            return f"Is the {d} {s}?", ans
        if c == "who":
            d = rng.choice(known)
            return f"Who last changed the {d}?", w.last[d]
        kinds = sorted({KIND_OF[d] for d in w.mentioned})
        if not kinds:
            continue
        k = rng.choice(kinds)
        members = sorted(d for d in w.mentioned if KIND_OF[d] == k)
        if any(w.state[d] is None for d in members):
            continue  # an unknown device would make the list ambiguous
        s = rng.choice(STATES[k])
        lst = [d for d in members if w.state[d] == s]
        if lst:
            return f"Which devices are {s}?", ",".join(lst)
    return None


def try_story(rng):
    n_s = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    devs = rng.sample(sorted(KIND_OF), rng.randint(2, 4))
    names = rng.sample(NAMES, rng.randint(2, 4))
    w = World()
    lines = []
    asked = set()
    s_left, q_left = n_s, n_q
    while s_left or q_left:
        if q_left and (not s_left or rng.random() < q_left / (q_left + s_left)):
            q = make_question(rng, w)
            if q is not None and q[0] not in asked:
                asked.add(q[0])
                lines.append(f"{q[0]}\t{q[1]}")
                q_left -= 1
                continue
            if not s_left:
                return None  # cannot place the remaining questions; retry story
        lines.append(make_statement(rng, w, devs, names))
        s_left -= 1
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
