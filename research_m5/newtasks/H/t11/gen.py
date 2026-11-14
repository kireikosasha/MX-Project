"""Synthetic rule-based reasoning task generator.

Usage: python gen.py N SEED

Each story states a few general rules ("Every cat is a pet.", "Whoever has the key
can open the box.", "Every pet can open the box.") together with facts (class
membership, taking and giving items). Questions are answered by a simulation of
the stated facts and rules, applying the rules transitively. Answers follow the
closed-world convention: anything that cannot be derived from the statements above
the question is "no".

Vocabulary (35 words): ann ben cara dan eli fay cat dog fish pet thing key coin map
box every whoever is a has the takes gives to can open who what how many does have
items yes no
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eli", "fay"]
DIRECT_CLASSES = ["cat", "dog", "fish", "pet"]
CLASSES = ["cat", "dog", "fish", "pet", "thing"]
CLASS_RULE_POOL = [("cat", "pet"), ("dog", "pet"), ("fish", "pet"), ("pet", "thing")]
OPEN_POOL = ["cat", "dog", "fish", "pet", "thing"]
ITEMS = ["key", "coin", "map"]


class World:
    def __init__(self, names):
        self.names = names
        self.direct = {n: set() for n in names}   # classes stated directly for a name
        self.class_rules = set()                   # (sub, sup) pairs
        self.open_classes = set()                  # classes whose members can open the box
        self.poss = set()                          # items whose holder can open the box
        self.holder = {i: None for i in ITEMS}     # current holder of each item


def closure(w, name):
    """All classes a name belongs to, following class rules transitively."""
    s = set(w.direct[name])
    changed = True
    while changed:
        changed = False
        for a, b in w.class_rules:
            if a in s and b not in s:
                s.add(b)
                changed = True
    return s


def can_open(w, name):
    if any(w.holder[i] == name for i in w.poss):
        return True
    return bool(closure(w, name) & w.open_classes)


# ---- statements (each returns a sentence and updates the world, or None) ----

def stmt_fact(rng, w):
    n = rng.choice(w.names)
    c = rng.choice(DIRECT_CLASSES)
    w.direct[n].add(c)
    return f"{n} is a {c}."


def stmt_class_rule(rng, w):
    avail = [r for r in CLASS_RULE_POOL if r not in w.class_rules]
    if not avail:
        return None
    a, b = rng.choice(avail)
    w.class_rules.add((a, b))
    return f"Every {a} is a {b}."


def stmt_open_rule(rng, w):
    avail = [c for c in OPEN_POOL if c not in w.open_classes]
    if not avail:
        return None
    c = rng.choice(avail)
    w.open_classes.add(c)
    return f"Every {c} can open the box."


def stmt_poss_rule(rng, w):
    avail = [i for i in ITEMS if i not in w.poss]
    if not avail:
        return None
    i = rng.choice(avail)
    w.poss.add(i)
    return f"Whoever has the {i} can open the box."


def stmt_takes(rng, w):
    n = rng.choice(w.names)
    i = rng.choice(ITEMS)
    w.holder[i] = n
    return f"{n} takes the {i}."


def stmt_gives(rng, w):
    held = [i for i in ITEMS if w.holder[i] is not None]
    if not held:
        return None
    i = rng.choice(held)
    giver = w.holder[i]
    others = [n for n in w.names if n != giver]
    if not others:
        return None
    receiver = rng.choice(others)
    w.holder[i] = receiver
    return f"{giver} gives the {i} to {receiver}."


STMT_FUNCS = [stmt_fact, stmt_fact, stmt_class_rule, stmt_open_rule,
              stmt_poss_rule, stmt_takes, stmt_takes, stmt_gives]


# ---- questions (each returns (question, answer) or None) ----

def q_class(rng, w):
    cands = [n for n in w.names if w.direct[n]]
    if not cands:
        return None
    n = rng.choice(cands)
    c = rng.choice(CLASSES)
    if c in w.direct[n]:
        return None  # would be answered by a single sentence; skip
    ans = "yes" if c in closure(w, n) else "no"
    return (f"Is {n} a {c}?", ans)


def q_what(rng, w):
    cands = [n for n in w.names if w.direct[n]]
    if not cands:
        return None
    n = rng.choice(cands)
    return (f"What is {n}?", ",".join(sorted(closure(w, n))))


def q_who_has(rng, w):
    held = [i for i in ITEMS if w.holder[i] is not None]
    if not held:
        return None
    i = rng.choice(held)
    return (f"Who has the {i}?", w.holder[i])


def q_can(rng, w):
    if not w.poss and not w.open_classes:
        return None
    n = rng.choice(w.names)
    return (f"Can {n} open the box?", "yes" if can_open(w, n) else "no")


def q_who_can(rng, w):
    if not w.poss and not w.open_classes:
        return None
    ok = sorted(n for n in w.names if can_open(w, n))
    if not ok:
        return None
    return ("Who can open the box?", ",".join(ok))


def q_count(rng, w):
    n = rng.choice(w.names)
    k = sum(1 for i in ITEMS if w.holder[i] == n)
    return (f"How many items does {n} have?", str(k))


QUESTION_FUNCS = [q_class, q_what, q_who_has, q_can, q_who_can, q_count]


def try_question(rng, w):
    funcs = QUESTION_FUNCS[:]
    rng.shuffle(funcs)
    for f in funcs:
        r = f(rng, w)
        if r:
            return r
    return None


def make_story(rng):
    while True:
        names = rng.sample(NAMES, rng.randint(3, 5))
        w = World(names)
        target_s = rng.randint(3, 12)
        target_q = rng.randint(1, 5)
        lines = []
        s = q = 0
        ok = True
        while s < target_s or q < target_q:
            if q < target_q and s >= 2 and (s >= target_s or rng.random() < 0.4):
                r = try_question(rng, w)
                if r:
                    lines.append(r)
                    q += 1
                    continue
                if s >= target_s:
                    ok = False
                    break
            sent = None
            while sent is None:
                sent = rng.choice(STMT_FUNCS)(rng, w)
            lines.append((sent, None))
            s += 1
        if ok:
            return lines


def format_story(lines):
    out = []
    for idx, (text, ans) in enumerate(lines, 1):
        text = text[0].upper() + text[1:]
        if ans is None:
            out.append(f"{idx} {text}")
        else:
            out.append(f"{idx} {text}\t{ans}")
    return out


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(format_story(make_story(rng)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
