"""Synthetic rule-reasoning reading-comprehension task generator.

Usage: python gen.py N SEED

Each story states general rules (kind taxonomy, a two-step ability chain,
and a transitive "beats" rule) together with facts (kinds, taking and giving
objects, beating pairs). Questions are answered by simulating the statements
written above them (closed world: a yes/no question is yes only if it can be
derived from those statements). Answers are never written by hand.
"""
import random
import sys

NAMES = ["anna", "ben", "carla", "dev", "eli"]
OBJECTS = ["key", "map"]
ROOMS = ["garden", "hall", "attic"]
KINDS = ["kitten", "cat", "puppy", "dog", "animal"]
LEAF_KINDS = ["kitten", "cat", "puppy", "dog"]
ANIMALS = ["cat", "dog", "fox", "owl", "mouse"]

# general rules: (sub kind, super kind)
TAX_RULES = [("kitten", "cat"), ("cat", "animal"), ("puppy", "dog"), ("dog", "animal")]
# whoever has OBJ can open DOOR
OPEN_RULES = [("key", "gate"), ("map", "door")]
# whoever can open DOOR can enter ROOM
ENTER_RULES = [("gate", "garden"), ("gate", "hall"), ("door", "attic"), ("door", "hall")]


def art(word):
    return "an" if word[0] in "aeiou" else "a"


def fresh():
    return {
        "kind": {},      # name -> direct kind (one fact per name)
        "held": {},      # object -> current holder
        "events": {},    # object -> number of took/gave statements
        "tax": [],       # stated taxonomy rules
        "open": [],      # stated object -> door rules
        "enter": [],     # stated door -> room rules
        "trans": False,  # transitive beats rule stated
        "beats": [],     # stated (a, b) pairs
    }


def stmt_options(st):
    opts = []
    for n in NAMES:
        if n not in st["kind"]:
            for k in LEAF_KINDS:
                opts.append((f"{n} is {art(k)} {k}.", ("kind", n, k)))
    for n in NAMES:
        for o in OBJECTS:
            opts.append((f"{n} took the {o}.", ("took", n, o)))
    for o, holder in st["held"].items():
        for m in NAMES:
            if m != holder:
                opts.append((f"{holder} gave the {o} to {m}.", ("gave", holder, m, o)))
    for pair in TAX_RULES:
        if pair not in st["tax"]:
            opts.append((f"every {pair[0]} is {art(pair[1])} {pair[1]}.", ("tax", pair)))
    for pair in OPEN_RULES:
        if pair not in st["open"]:
            opts.append((f"whoever has the {pair[0]} can open the {pair[1]}.", ("open", pair)))
    for pair in ENTER_RULES:
        if pair not in st["enter"]:
            opts.append((f"whoever can open the {pair[0]} can enter the {pair[1]}.", ("enter", pair)))
    if not st["trans"]:
        opts.append(("whatever beats a thing beats what it beats.", ("trans",)))
    for a in ANIMALS:
        for b in ANIMALS:
            if a != b and (a, b) not in st["beats"]:
                opts.append((f"the {a} beats the {b}.", ("beats", a, b)))
    return opts


def apply_act(st, act):
    t = act[0]
    if t == "kind":
        st["kind"][act[1]] = act[2]
    elif t == "took":
        holder, obj = act[1], act[2]
        st["held"][obj] = holder
        st["events"][obj] = st["events"].get(obj, 0) + 1
    elif t == "gave":
        obj, receiver = act[3], act[2]
        st["held"][obj] = receiver
        st["events"][obj] = st["events"].get(obj, 0) + 1
    elif t == "tax":
        st["tax"].append(act[1])
    elif t == "open":
        st["open"].append(act[1])
    elif t == "enter":
        st["enter"].append(act[1])
    elif t == "trans":
        st["trans"] = True
    elif t == "beats":
        st["beats"].append((act[1], act[2]))


def supers(st, name):
    if name not in st["kind"]:
        return set()
    res = {st["kind"][name]}
    grew = True
    while grew:
        grew = False
        for sub, sup in st["tax"]:
            if sub in res and sup not in res:
                res.add(sup)
                grew = True
    return res


def can_enter(st, name, room):
    for obj, holder in st["held"].items():
        if holder != name:
            continue
        for rule_obj, door in st["open"]:
            if rule_obj == obj and (door, room) in st["enter"]:
                return True
    return False


def reach(st, a):
    seen = []
    stack = [a]
    while stack:
        x = stack.pop()
        for p, q in st["beats"]:
            if p == x and q not in seen:
                seen.append(q)
                stack.append(q)
    return seen


def question_options(st, asked):
    opts = []
    for n, direct in st["kind"].items():
        sup = supers(st, n)
        for k in KINDS:
            yes = k in sup
            if yes and k == direct:
                continue  # answered by one sentence, not reasoning
            if not yes and not st["tax"]:
                continue
            opts.append((f"is {n} {art(k)} {k}?", "yes" if yes else "no"))
    if st["open"] and st["enter"]:
        for n in NAMES:
            for r in ROOMS:
                opts.append((f"can {n} enter the {r}?", "yes" if can_enter(st, n, r) else "no"))
        for r in ROOMS:
            who = sorted(n for n in NAMES if can_enter(st, n, r))
            if who:
                opts.append((f"who can enter the {r}?", ",".join(who)))
    for obj, count in st["events"].items():
        if count >= 2:
            opts.append((f"who has the {obj}?", st["held"][obj]))
    if st["trans"] and st["beats"]:
        for a in ANIMALS:
            r = reach(st, a)
            for b in ANIMALS:
                if b != a:
                    opts.append((f"does the {a} beat the {b}?", "yes" if b in r else "no"))
            if r:
                opts.append((f"what does the {a} beat?", ",".join(sorted(r))))
    return [o for o in opts if o[0] not in asked]


def make_story(rng):
    for _ in range(1000):
        st = fresh()
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        lines = []
        asked = set()
        stmts = 0
        qs = 0
        failed = False
        while stmts < n_st or qs < n_q:
            if qs < n_q and stmts >= 1 and (stmts == n_st or rng.random() < 0.4):
                opts = question_options(st, asked)
                if opts:
                    text, ans = rng.choice(opts)
                    asked.add(text)
                    lines.append(f"{text}\t{ans}")
                    qs += 1
                    continue
                if stmts == n_st:
                    failed = True
                    break
            if stmts >= n_st:
                failed = True
                break
            text, act = rng.choice(stmt_options(st))
            apply_act(st, act)
            lines.append(text)
            stmts += 1
        if not failed:
            return lines
    raise RuntimeError("could not generate a story")


def main():
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
