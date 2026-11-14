import random
import sys

# Chains of relations: family trees (parent of) and nested places (is in).
# Questions need two or more links, e.g. a grandparent, or the outermost
# place of an item that sits in a box that sits in a room.

NAMES = ["ann", "bob", "cy", "dan", "eve", "fay", "gus", "hal", "ivy", "jon"]
ITEMS = ["key", "coin", "ball", "hat"]
PLACES = ["box", "bag", "shelf", "hall", "attic", "room"]


def cap(word):
    return word[:1].upper() + word[1:]


def ancestors(links, x):
    # links maps a node to the node it hangs under; returns the chain upwards
    out = []
    while x in links:
        x = links[x]
        out.append(x)
    return out


# ---- family domain: each person has at most one parent ----

def family_fact(rng, names, par):
    pairs = []
    for c in names:
        if c in par:
            continue
        for p in names:
            if p != c and c not in ancestors(par, p):
                pairs.append((p, c))
    if not pairs:
        return None
    p, c = rng.choice(pairs)
    par[c] = p
    return "%s is the parent of %s." % (cap(p), cap(c))


def family_questions(names, par):
    out = []
    grand = {}
    for x in names:
        p = par.get(x)
        g = par.get(p) if p is not None else None
        if g is not None:
            grand[x] = g
            out.append(("who", "Who is the grandparent of %s?" % cap(x), g))
    for b, g in grand.items():
        out.append(("yes", "Is %s the grandparent of %s?" % (cap(g), cap(b)), "yes"))
        for a in names:
            if a != b and a != g:
                out.append(("no", "Is %s the grandparent of %s?" % (cap(a), cap(b)), "no"))
    return out


# ---- place domain: things sit in places, places sit in places ----

def place_fact(rng, loc):
    pairs = []
    for x in ITEMS + PLACES:
        for y in PLACES:
            if y == x or y == loc.get(x):
                continue
            if x in ancestors(loc, y):
                continue
            pairs.append((x, y))
    if not pairs:
        return None
    x, y = rng.choice(pairs)
    loc[x] = y
    return "The %s is in the %s." % (x, y)


def place_questions(loc):
    out = []
    for x in ITEMS:
        chain = ancestors(loc, x)
        if len(chain) >= 2:
            top = chain[-1]
            out.append(("where", "Where is the %s finally?" % x, top))
            for p in PLACES:
                if p == top:
                    out.append(("yes", "Is the %s finally in the %s?" % (x, p), "yes"))
                else:
                    out.append(("no", "Is the %s finally in the %s?" % (x, p), "no"))
    return out


# ---- story assembly ----

def pick_question(rng, cands):
    # Choose the question kind first so yes and no answers stay balanced.
    kinds = sorted(set(kind for kind, _, _ in cands))
    kind = rng.choice(kinds)
    pool = [(q, a) for k, q, a in cands if k == kind]
    return rng.choice(pool)


def make_story(rng):
    while True:
        domain = rng.choice(["family", "place"])
        target_s = rng.randint(3, 12)
        target_q = rng.randint(1, 5)
        lines = []
        stmts = 0
        qs = 0
        asked = set()
        if domain == "family":
            names = rng.sample(NAMES, rng.randint(4, 7))
            par = {}
        else:
            loc = {}
        while True:
            if domain == "family":
                cands = family_questions(names, par)
            else:
                cands = place_questions(loc)
            cands = [c for c in cands if c[1] not in asked]
            can_ask = qs < target_q and len(cands) > 0
            if can_ask and stmts >= 1 and (stmts >= target_s or rng.random() < 0.5):
                q, a = pick_question(rng, cands)
                asked.add(q)
                lines.append(q + "\t" + a)
                qs += 1
                continue
            if stmts < target_s:
                if domain == "family":
                    f = family_fact(rng, names, par)
                else:
                    f = place_fact(rng, loc)
                if f is not None:
                    lines.append(f)
                    stmts += 1
                    continue
            if can_ask and stmts >= 1:
                q, a = pick_question(rng, cands)
                asked.add(q)
                lines.append(q + "\t" + a)
                qs += 1
                continue
            break
        if qs >= 1 and 3 <= stmts <= 12:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, text in enumerate(make_story(rng), 1):
            out.append("%d %s" % (i, text))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
