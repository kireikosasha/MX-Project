#!/usr/bin/env python3
"""Synthetic reading-comprehension stories with rule-based answers.

Usage: python gen.py N SEED

World: people (Ann, Bob, ...) are of a kind (cat, bird, dog), items (key, coin,
hat) pass between people, and people go to rooms (hall, attic, shed). Each story
states general rules ("Every cat is a dog.", "Anyone with the key can enter the
attic.") together with facts ("Ann has the key.", "Ann gives the key to Bob.").
Answers are computed by simulating the statements in order: a question only sees
the statements above it, and rules are applied by forward chaining over the rules
stated so far. The stated rules are the only rules of the world.
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cal", "Dan", "Eve", "Fay"]
ITEMS = ["key", "coin", "hat"]
ROOMS = ["hall", "attic", "shed"]
KINDS = ["cat", "bird", "dog"]
CLASS_EDGES = [("cat", "dog"), ("cat", "pet"), ("bird", "dog"),
               ("bird", "pet"), ("dog", "pet")]
QUESTION_KINDS = ["cat", "bird", "dog", "pet"]


def closure(kind, edges):
    """All classes reachable from kind through the stated 'Every a is a b' rules."""
    out = set()
    frontier = [kind]
    while frontier:
        k = frontier.pop()
        for a, b in edges:
            if a == k and b not in out:
                out.add(b)
                frontier.append(b)
    return out


class World:
    def __init__(self):
        self.kind = {}      # name -> kind, from "X is a Y." facts
        self.edges = set()  # stated class rules (a, b)
        self.holder = {}    # item -> name who has it
        self.moved = set()  # items that have been given at least once
        self.where = {}     # name -> room, from "X goes to the R." facts
        self.rules = set()  # stated entry rules (item, room)


def statement(rng, w):
    """Apply one random statement to w and return its text, or None if invalid."""
    kind = rng.choice(["classfact", "classrule", "has", "gives", "goes", "entryrule"])
    if kind == "classfact":
        free = [n for n in NAMES if n not in w.kind]
        if not free:
            return None
        name, k = rng.choice(free), rng.choice(KINDS)
        w.kind[name] = k
        return f"{name} is a {k}."
    if kind == "classrule":
        free = [e for e in CLASS_EDGES if e not in w.edges]
        if not free:
            return None
        a, b = rng.choice(free)
        w.edges.add((a, b))
        return f"Every {a} is a {b}."
    if kind == "has":
        free = [i for i in ITEMS if i not in w.holder]
        if not free:
            return None
        item, name = rng.choice(free), rng.choice(NAMES)
        w.holder[item] = name
        return f"{name} has the {item}."
    if kind == "gives":
        held = sorted(w.holder)
        if not held:
            return None
        item = rng.choice(held)
        old = w.holder[item]
        new = rng.choice([n for n in NAMES if n != old])
        w.holder[item] = new
        w.moved.add(item)
        return f"{old} gives the {item} to {new}."
    if kind == "goes":
        name, room = rng.choice(NAMES), rng.choice(ROOMS)
        w.where[name] = room
        return f"{name} goes to the {room}."
    # entryrule
    free = [(i, r) for i in ITEMS for r in ROOMS if (i, r) not in w.rules]
    if not free:
        return None
    item, room = rng.choice(free)
    w.rules.add((item, room))
    return f"Anyone with the {item} can enter the {room}."


def questions(w):
    """Every (text, answer) that the current world determines."""
    cands = []
    for name, k in w.kind.items():
        for x in QUESTION_KINDS:
            if x == k:
                continue  # trivial: just repeats the fact
            ans = "yes" if x in closure(k, w.edges) else "no"
            cands.append((f"Is {name} a {x}?", ans))
    for item in sorted(w.moved):
        name = w.holder[item]
        cands.append((f"Who has the {item}?", name.lower()))
    for item, name in sorted(w.holder.items()):
        if name in w.where:
            cands.append((f"Where is the {item}?", w.where[name]))
    people = set(w.kind) | set(w.holder.values()) | set(w.where)
    for room in ROOMS:
        items = sorted(i for (i, r) in w.rules if r == room)
        if not items:
            continue
        for name in sorted(people):
            ok = any(w.holder.get(i) == name for i in items)
            cands.append((f"Can {name} enter the {room}?", "yes" if ok else "no"))
        who = sorted({w.holder[i].lower() for i in items if i in w.holder})
        if who:
            cands.append((f"Who can enter the {room}?", ",".join(who)))
    return cands


def pick_question(rng, w, asked):
    cands = [c for c in questions(w) if c[0] not in asked]
    if not cands:
        return None
    text, ans = rng.choice(cands)
    asked.add(text)
    return (text, ans)


def make_story(rng):
    """Return a list of (text, answer-or-None) lines for one story."""
    for _ in range(1000):
        w = World()
        lines = []
        asked = set()
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        st = qn = 0
        ok = True
        while st < n_st:
            if qn < n_q and st > 0 and rng.random() < 0.4:
                q = pick_question(rng, w, asked)
                if q:
                    lines.append(q)
                    qn += 1
                    continue
            text = None
            for _ in range(20):
                text = statement(rng, w)
                if text:
                    break
            if not text:
                ok = False
                break
            lines.append((text, None))
            st += 1
        while ok and qn < n_q:
            q = pick_question(rng, w, asked)
            if not q:
                ok = False
                break
            lines.append(q)
            qn += 1
        if ok:
            return lines
    raise RuntimeError("could not build a story")


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            out.append(f"{i} {text}" if ans is None else f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
