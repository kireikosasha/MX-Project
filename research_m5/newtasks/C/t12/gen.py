import random
import sys

# Reasoning kind: tracking who holds which object, what was handed over,
# what was put into which container, and where that container sits.
# A question about a container's place needs a chain of several statements.

NAMES = ["ann", "ben", "cara", "dan"]
OBJECTS = ["book", "pen", "cup", "hat"]
CONTAINERS = ["box", "bag"]
PLACES = ["shelf", "desk"]


def cap(s):
    return s[0].upper() + s[1:]


class World:
    def __init__(self):
        # location of each object: None (unknown), ("p", name) or ("c", container)
        self.loc = {o: None for o in OBJECTS}
        # place of each container: None (unknown) or one of PLACES
        self.place = {c: None for c in CONTAINERS}

    def holder(self, o):
        l = self.loc[o]
        if l is not None and l[0] == "p":
            return l[1]
        return None


def make_statement(rng, w):
    held = [o for o in OBJECTS if w.holder(o) is not None]
    kind = rng.choice(["has", "gave", "put", "placed"])
    if kind in ("gave", "put") and not held:
        kind = "has"
    if kind == "has":
        p = rng.choice(NAMES)
        o = rng.choice(OBJECTS)
        w.loc[o] = ("p", p)
        return "%s has the %s." % (cap(p), o)
    if kind == "placed":
        c = rng.choice(CONTAINERS)
        pl = rng.choice(PLACES)
        w.place[c] = pl
        return "The %s is on the %s." % (c, pl)
    o = rng.choice(held)
    p = w.holder(o)
    if kind == "gave":
        q = rng.choice([n for n in NAMES if n != p])
        w.loc[o] = ("p", q)
        return "%s gave the %s to %s." % (cap(p), o, q)
    c = rng.choice(CONTAINERS)
    w.loc[o] = ("c", c)
    return "%s put the %s in the %s." % (cap(p), o, c)


def make_question(rng, w):
    opts = []
    for o in OBJECTS:
        l = w.loc[o]
        if l is None:
            continue
        opts.append(("Who has the %s?" % o, l[1] if l[0] == "p" else "nobody"))
        if l[0] == "c" and w.place[l[1]] is not None:
            opts.append(("Where is the %s?" % o, w.place[l[1]]))
        for c in CONTAINERS:
            ans = "yes" if l == ("c", c) else "no"
            opts.append(("Is the %s in the %s?" % (o, c), ans))
    for c in CONTAINERS:
        if w.place[c] is None:
            continue
        for pl in PLACES:
            ans = "yes" if w.place[c] == pl else "no"
            opts.append(("Is the %s on the %s?" % (c, pl), ans))
    if all(w.loc[o] is not None for o in OBJECTS):
        for p in NAMES:
            held = sorted(o for o in OBJECTS if w.loc[o] == ("p", p))
            ans = ",".join(held) if held else "nothing"
            opts.append(("What does %s have?" % p, ans))
    if not opts:
        return None
    return rng.choice(opts)


def make_story(rng):
    while True:
        w = World()
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        lines = []
        st = qs = 0
        failed = False
        while st < n_st or qs < n_q:
            if qs < n_q and st > 0 and (st >= n_st or rng.random() < 0.4):
                q = make_question(rng, w)
                if q is not None:
                    text, ans = q
                    lines.append("%d %s\t%s" % (len(lines) + 1, text, ans))
                    qs += 1
                    continue
                if st >= n_st:
                    failed = True
                    break
            s = make_statement(rng, w)
            lines.append("%d %s" % (len(lines) + 1, s))
            st += 1
        if not failed:
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
