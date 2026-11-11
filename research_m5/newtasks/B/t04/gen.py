#!/usr/bin/env python3
"""Synthetic reading-comprehension task: kinds and their properties.

Kinds form a fixed taxonomy and carry properties ("Every dog is loud.",
"No cat is shy."). Named individuals are told only which kind they belong
to. Questions ask whether an individual has a property or belongs to a kind;
the answer has to be derived by chaining membership, taxonomy and kind rules.

Usage: python gen.py N SEED
"""
import random
import sys

KINDS = ["animal", "mammal", "bird", "fish", "dog", "cat", "cow", "owl", "duck"]
PARENT = {"mammal": "animal", "bird": "animal", "fish": "animal",
          "dog": "mammal", "cat": "mammal", "cow": "mammal",
          "owl": "bird", "duck": "bird"}
EDGES = [(k, PARENT[k]) for k in KINDS if k in PARENT]
PROPS = ["loud", "quiet", "fast", "shy", "warm"]
NAMES = ["Sam", "Lily", "Max", "Ana", "Tom", "Mia"]

# Fact tuples:
#   ("mem", name, kind)          Sam is a dog.
#   ("edge", child, parent)      Every dog is a mammal.
#   ("rule", kind, prop, pol)    Every dog is loud. / No dog is loud.  (pol: "Every" or "No")
#   ("dis", a, b)                No dog is a cat.


def art(word):
    return "an" if word[0] in "aeiou" else "a"


def ancestors(kind):
    out = [kind]
    while kind in PARENT:
        kind = PARENT[kind]
        out.append(kind)
    return out


def nested(a, b):
    return a in ancestors(b) or b in ancestors(a)


def text(f):
    if f[0] == "mem":
        return "%s is %s %s." % (f[1], art(f[2]), f[2])
    if f[0] == "edge":
        return "Every %s is %s %s." % (f[1], art(f[2]), f[2])
    if f[0] == "rule":
        return "%s %s is %s." % (f[3], f[1], f[2])
    return "No %s is %s %s." % (f[1], art(f[2]), f[2])


def question_text(q):
    typ, x, val = q
    if typ == "prop":
        return "Is %s %s?" % (x, val)
    return "Is %s %s %s?" % (x, art(val), val)


def consistent(rules):
    """No kind may fall under both a positive and a negative rule for one property."""
    for k in KINDS:
        anc = ancestors(k)
        for p in PROPS:
            pols = {r[3] for r in rules if r[2] == p and r[1] in anc}
            if len(pols) == 2:
                return False
    return True


def make_world(rng):
    names = rng.sample(NAMES, rng.randint(2, 4))
    mem = [("mem", n, rng.choice(KINDS)) for n in names]
    rules = []
    for p in PROPS:
        if rng.random() < 0.6:
            for _ in range(rng.randint(1, 2)):
                cand = rules + [("rule", rng.choice(KINDS), p, rng.choice(["Every", "No"]))]
                if consistent(cand):
                    rules = cand
    pairs = [(a, b) for i, a in enumerate(KINDS) for b in KINDS[i + 1:] if not nested(a, b)]
    dis = []
    for a, b in rng.sample(pairs, rng.randint(0, 3)):
        dis.append(("dis", a, b) if rng.random() < 0.5 else ("dis", b, a))
    edges = [("edge", c, p) for c, p in EDGES]
    pool = mem + edges + rules + dis
    kind_of = {f[1]: f[2] for f in mem}
    return pool, kind_of, rules


def world_answer(kind_of, rules, query):
    """Answer from the simulated world (full taxonomy and all rules)."""
    typ, x, val = query
    k = kind_of[x]
    if typ == "prop":
        anc = ancestors(k)
        pols = {r[3] for r in rules if r[2] == val and r[1] in anc}
        if not pols:
            return None
        return "yes" if "Every" in pols else "no"
    return "yes" if val in ancestors(k) else "no"


def derive(prefix, query):
    """Answer a query from the statements in prefix only.

    Returns (answer, support) where support is the set of prefix indices used,
    or None when the prefix does not settle the question.
    """
    typ, x, val = query
    known = {}  # kind -> support, for the kinds x is known to belong to
    for i, f in enumerate(prefix):
        if f[0] == "mem" and f[1] == x and f[2] not in known:
            known[f[2]] = frozenset([i])
    if not known:
        return None
    changed = True
    while changed:
        changed = False
        for i, f in enumerate(prefix):
            if f[0] == "edge" and f[1] in known and f[2] not in known:
                known[f[2]] = known[f[1]] | {i}
                changed = True
    if typ == "prop":
        found = {}
        for i, f in enumerate(prefix):
            if f[0] == "rule" and f[2] == val and f[1] in known:
                found.setdefault(f[3], []).append(known[f[1]] | {i})
        if not found:
            return None
        assert len(found) == 1, "contradictory prefix"
        pol, sups = next(iter(found.items()))
        return ("yes" if pol == "Every" else "no"), min(sups, key=len)
    if val in known:
        return "yes", known[val]
    up = {val: frozenset()}  # kinds that contain val, with support
    changed = True
    while changed:
        changed = False
        for i, f in enumerate(prefix):
            if f[0] == "edge" and f[1] in up and f[2] not in up:
                up[f[2]] = up[f[1]] | {i}
                changed = True
    for i, f in enumerate(prefix):
        if f[0] == "dis":
            for a, c in ((f[1], f[2]), (f[2], f[1])):
                if a in known and c in up:
                    return "no", known[a] | up[c] | {i}
    return None


def queries_from(prefix):
    names = []
    for f in prefix:
        if f[0] == "mem" and f[1] not in names:
            names.append(f[1])
    out = []
    for x in names:
        for p in PROPS:
            out.append(("prop", x, p))
        for b in KINDS:
            out.append(("kind", x, b))
    return out


def build(rng, stmts, nq, kind_of, rules):
    prefix, lines, asked, qs = [], [], set(), []

    def ask():
        opts = []
        for q in queries_from(prefix):
            if q in asked:
                continue
            r = derive(prefix, q)
            if r is None or len(r[1]) < 2:  # unsettled, or answered by one sentence
                continue
            opts.append((q, r[0]))
        if not opts:
            return False
        q, ans = rng.choice(opts)
        assert world_answer(kind_of, rules, q) == ans, "prefix answer disagrees with world"
        asked.add(q)
        qs.append(q)
        lines.append(("q", question_text(q), ans))
        return True

    for f in stmts:
        prefix.append(f)
        lines.append(("s", text(f), None))
        if len(qs) < nq and rng.random() < 0.4:
            ask()
    while len(qs) < nq:
        if not ask():
            return None
    return lines


def make_story(rng):
    for _ in range(10000):
        pool, kind_of, rules = make_world(rng)
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        if n_stmt > len(pool):
            continue
        stmts = rng.sample(pool, n_stmt)
        lines = build(rng, stmts, n_q, kind_of, rules)
        if lines is not None:
            return lines
    raise RuntimeError("could not generate a story")


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for i, (kind, t, ans) in enumerate(make_story(rng), 1):
            if kind == "s":
                print("%d %s" % (i, t))
            else:
                print("%d %s\t%s" % (i, t, ans))


if __name__ == "__main__":
    main()
