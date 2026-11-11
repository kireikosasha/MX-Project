import random
import sys

NAMES = ["Ruth", "Omar", "Lena", "Tomas", "Mira", "Hugo"]
KINDS = ["creature", "bird", "fish", "frog", "toad", "lizard"]
PROPS = ["loud", "quiet", "tall", "small", "sleepy"]

# Statement tuples:
#   ("fact", name, kind)     "Ruth is a frog."
#   ("isa", sub, sup)        "Every frog is a creature."
#   ("prop", kind, prop)     "Every frog is loud."
#   ("neg", kind, prop)      "No frog is loud."


def derive(stmts, name):
    """Return (positive props, negative props) that hold for name, or None."""
    kinds = set(s[2] for s in stmts if s[0] == "fact" and s[1] == name)
    if not kinds:
        return None
    changed = True
    while changed:
        changed = False
        for s in stmts:
            if s[0] == "isa" and s[1] in kinds and s[2] not in kinds:
                kinds.add(s[2])
                changed = True
    pos = set(s[2] for s in stmts if s[0] == "prop" and s[1] in kinds)
    neg = set(s[2] for s in stmts if s[0] == "neg" and s[1] in kinds)
    return pos, neg


def text(s):
    if s[0] == "fact":
        return "%s is a %s." % (s[1], s[2])
    if s[0] == "isa":
        return "Every %s is a %s." % (s[1], s[2])
    if s[0] == "prop":
        return "Every %s is %s." % (s[1], s[2])
    return "No %s is %s." % (s[1], s[2])


def candidates(prefix):
    """Yes/no questions whose answer is derivable from the prefix alone."""
    out = []
    for name in sorted(set(s[1] for s in prefix if s[0] == "fact")):
        pos, neg = derive(prefix, name)
        for p in PROPS:
            if p in pos and p not in neg:
                out.append((name, p, "yes"))
            elif p in neg and p not in pos:
                out.append((name, p, "no"))
    return out


def make_story(rng):
    while True:
        n = rng.randint(3, 12)
        names = rng.sample(NAMES, rng.randint(1, 3))
        order = KINDS[:]
        rng.shuffle(order)
        rank = {k: i for i, k in enumerate(order)}
        stmts = []
        seen = set()
        tries = 0
        while len(stmts) < n and tries < 500:
            tries += 1
            r = rng.random()
            if r < 0.4:
                s = ("fact", rng.choice(names), rng.choice(KINDS))
            elif r < 0.6:
                a, b = rng.sample(KINDS, 2)
                if rank[a] < rank[b]:
                    a, b = b, a
                s = ("isa", a, b)  # rank strictly decreases along isa: no cycles
            elif r < 0.85:
                s = ("prop", rng.choice(KINDS), rng.choice(PROPS))
            else:
                s = ("neg", rng.choice(KINDS), rng.choice(PROPS))
            if s in seen:
                continue
            seen.add(s)
            stmts.append(s)
        if len(stmts) < n:
            continue
        # Global consistency: no individual may get a property and its negation.
        ok = True
        for name in names:
            d = derive(stmts, name)
            if d is not None and (d[0] & d[1]):
                ok = False
        if not ok:
            continue
        nq = rng.randint(1, min(5, n - 1))
        slots = sorted(rng.sample(range(2, n + 1), nq))
        qs = []
        asked = set()
        for k in slots:
            cands = [c for c in candidates(stmts[:k]) if (c[0], c[1]) not in asked]
            if not cands:
                break
            q = rng.choice(cands)
            asked.add((q[0], q[1]))
            qs.append((k, q))
        if len(qs) != nq:
            continue
        lines = []
        qi = 0
        for i in range(n):
            lines.append(text(stmts[i]))
            if qi < len(qs) and qs[qi][0] == i + 1:
                name, p, ans = qs[qi][1]
                lines.append("Is %s %s?\t%s" % (name, p, ans))
                qi += 1
        return lines


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n_stories):
        for i, line in enumerate(make_story(rng), 1):
            out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
