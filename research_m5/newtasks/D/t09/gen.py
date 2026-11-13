import random
import sys

# Task: chains of relations. People are linked by "parent of" and "boss of";
# things and places are linked by "in". Questions need two links in a row.

NAMES = ["ann", "bob", "cara", "dan", "eli", "fay", "gus", "hal"]
THINGS = ["key", "box", "ball", "coin"]
PLACES = ["shed", "barn", "hall", "attic"]
ITEMS = THINGS + PLACES
PERSON_RELS = ["parent", "boss"]
MAX_STATEMENTS = 12


def make_forest(rng, nodes, p):
    # Each node gets at most one "up" link, to an earlier node in a random
    # order, so there are no cycles and each child has a single value.
    order = list(nodes)
    rng.shuffle(order)
    up = {}
    for i, node in enumerate(order):
        if i > 0 and rng.random() < p:
            up[node] = rng.choice(order[:i])
    return up


def make_containers(rng, items, p):
    # A thing may be in any item; a place may only be in a place.
    order = list(items)
    rng.shuffle(order)
    up = {}
    for i, node in enumerate(order):
        cands = [o for o in order[:i] if node in THINGS or o in PLACES]
        if cands and rng.random() < p:
            up[node] = rng.choice(cands)
    return up


def make_world(rng):
    names = rng.sample(NAMES, rng.randint(4, 6))
    items = rng.sample(ITEMS, rng.randint(3, 6))
    facts = []
    for child, up in make_forest(rng, names, 0.8).items():
        facts.append(("parent", child, up))
    for child, up in make_forest(rng, names, 0.8).items():
        facts.append(("boss", child, up))
    for child, up in make_containers(rng, items, 0.7).items():
        facts.append(("in", child, up))
    rng.shuffle(facts)
    return names, facts


def sentence(fact):
    rel, child, up = fact
    if rel == "parent":
        return f"{up.capitalize()} is the parent of {child}."
    if rel == "boss":
        return f"{up.capitalize()} is the boss of {child}."
    return f"The {child} is in the {up}."


def index(known):
    m = {"parent": {}, "boss": {}, "in": {}}
    for rel, child, up in known:
        m[rel][child] = up
    return m


def candidates(known, names, rng):
    # Every candidate is derived only from the stated facts in `known`.
    m = index(known)
    out = []
    for f1 in PERSON_RELS:
        for f2 in PERSON_RELS:
            for x in names:
                v = m[f2].get(x)
                if v is None:
                    continue
                w = m[f1].get(v)
                if w is None:
                    continue
                # w is the f1 of (the f2 of x); each relation is single-valued,
                # so a different name in a yes/no question gives "no".
                out.append((f"Who is the {f1} of the {f2} of {x}?", w))
                if rng.random() < 0.5:
                    a = w
                else:
                    a = rng.choice([n for n in names if n != w])
                ans = "yes" if a == w else "no"
                out.append((f"Is {a} the {f1} of the {f2} of {x}?", ans))
    for t in ITEMS:
        c1 = m["in"].get(t)
        if c1 is None:
            continue
        c2 = m["in"].get(c1)
        if c2 is None:
            continue
        out.append((f"Is the {t} in the {c2}?", "yes"))
    return out


def make_story(rng):
    while True:
        names, facts = make_world(rng)
        if len(facts) < 3:
            continue
        n = rng.randint(3, min(MAX_STATEMENTS, len(facts)))
        stmts = facts[:n]
        q_count = rng.randint(1, 5)
        cuts = set(rng.sample(range(1, n + 1), min(q_count, n)))
        lines = []
        known = []
        asked = set()
        for k, fact in enumerate(stmts, 1):
            known.append(fact)
            lines.append((sentence(fact), None))
            if k in cuts:
                cands = [c for c in candidates(known, names, rng)
                         if c[0] not in asked]
                if cands:
                    text, ans = rng.choice(cands)
                    asked.add(text)
                    lines.append((text, ans))
        if asked:
            return lines


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n_stories):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                print(f"{i} {text}")
            else:
                print(f"{i} {text}\t{ans}")


if __name__ == "__main__":
    main()
