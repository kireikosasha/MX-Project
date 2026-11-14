import random
import sys

# Kinds form a tree: each kind has one parent kind, the root is "creature".
# Individuals belong to one kind. Kinds have properties (can swim, can fly, ...)
# stated as "Every K can P." or "No K can P.". A property of an individual
# follows from its kind chain (kind -> parent -> ... -> creature).

ROOT = "creature"
PARENT = {"cat": "mammal", "dog": "mammal", "mammal": "creature",
          "bird": "creature", "fish": "creature"}
KINDS = ["cat", "dog", "mammal", "bird", "fish", "creature"]
IND_KINDS = ["cat", "dog", "mammal", "bird", "fish"]
NAMES = ["Ren", "Tam", "Oz", "Pim", "Yul", "Bea"]
PROPS = ["swim", "fly", "climb", "run"]


def sentence(f):
    if f[0] == "ind":
        return "%s is a %s." % (f[1], f[2])
    if f[0] == "edge":
        return "Every %s is a %s." % (f[1], f[2])
    kind, prop, pol = f[1], f[2], f[3]
    if pol:
        return "Every %s can %s." % (kind, prop)
    return "No %s can %s." % (kind, prop)


def known(prefix):
    kind_of, parent, rules = {}, {}, {}
    for f in prefix:
        if f[0] == "ind":
            kind_of[f[1]] = f[2]
        elif f[0] == "edge":
            parent[f[1]] = f[2]
        else:
            rules[(f[1], f[2])] = f[3]
    return kind_of, parent, rules


def walk_up(kind, parent):
    # full ancestor path if every parent link is stated above, else None
    path = [kind]
    cur = kind
    while cur != ROOT:
        if cur not in parent:
            return None
        cur = parent[cur]
        path.append(cur)
    return path


def prop_answer(kind, prop, parent, rules):
    # nearest stated rule up the kind chain; None if not decidable yet
    cur = kind
    while True:
        if (cur, prop) in rules:
            return rules[(cur, prop)]
        if cur == ROOT or cur not in parent:
            return None
        cur = parent[cur]


def pick_question(prefix, asked, rng):
    kind_of, parent, rules = known(prefix)
    cands = []
    for name, kind in kind_of.items():
        for prop in PROPS:
            pol = prop_answer(kind, prop, parent, rules)
            if pol is not None:
                cands.append(("Can %s %s?" % (name, prop), "yes" if pol else "no"))
        path = walk_up(kind, parent)
        if path is not None:
            for k in KINDS:
                if k == kind:
                    continue  # direct kind is a plain lookup, skip
                cands.append(("Is %s a %s?" % (name, k), "yes" if k in path else "no"))
    cands = [c for c in cands if c[0] not in asked]
    if not cands:
        return None
    return rng.choice(cands)


def make_story(rng):
    n = rng.randint(1, 3)
    names = rng.sample(NAMES, n)
    kinds = [rng.choice(IND_KINDS) for _ in range(n)]
    facts = [("ind", nm, kd) for nm, kd in zip(names, kinds)]

    edges, chain_nodes = [], set()
    for kd in kinds:
        cur = kd
        chain_nodes.add(cur)
        while cur != ROOT:
            e = ("edge", cur, PARENT[cur])
            if e not in edges:
                edges.append(e)
            cur = PARENT[cur]
            chain_nodes.add(cur)
    facts += edges

    pol = {p: rng.random() < 0.5 for p in PROPS}
    pairs = [(k, p) for k in sorted(chain_nodes) for p in PROPS]
    cap = 12 - len(facts)
    r = rng.randint(1, min(3, cap))
    for k, p in rng.sample(pairs, r):
        facts.append(("rule", k, p, pol[p]))
    rng.shuffle(facts)

    target = rng.randint(1, 4)
    prefix, seq, asked, nq = [], [], set(), 0
    for i, f in enumerate(facts):
        prefix.append(f)
        seq.append(("S", f))
        if nq < target and i >= 1 and rng.random() < 0.5:
            q = pick_question(prefix, asked, rng)
            if q is not None:
                asked.add(q[0])
                seq.append(("Q", q))
                nq += 1
    while nq < target:
        q = pick_question(prefix, asked, rng)
        if q is None:
            break
        asked.add(q[0])
        seq.append(("Q", q))
        nq += 1

    lines = []
    for num, (typ, item) in enumerate(seq, 1):
        if typ == "S":
            lines.append("%d %s" % (num, sentence(item)))
        else:
            lines.append("%d %s\t%s" % (num, item[0], item[1]))
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
