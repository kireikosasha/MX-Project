"""Synthetic containment task: things sit inside containers, containers sit
inside places, and people carry things between places.

Usage: python gen.py N SEED
Prints N stories. Numbers restart at 1 for each story. A question line is
"<n> <question>?<TAB><answer>".
"""

import random
import sys

PERSONS = ["Ann", "Ben", "Cara"]
PLACES = ["kitchen", "garden", "hall", "office"]
CONTAINERS = ["box", "bag", "chest"]
THINGS = ["key", "ball", "apple", "coin"]
MOVABLE = CONTAINERS + THINGS


class World:
    def __init__(self):
        # thing or container -> its holder: a container, a place, a person, or None (unknown)
        self.parent = {e: None for e in MOVABLE}
        # person -> place, or None (unknown)
        self.loc = {p: None for p in PERSONS}

    def where(self, e):
        """The place an entity is in, or None if not known."""
        if e in PLACES:
            return e
        if e in PERSONS:
            return self.loc[e]
        p = self.parent[e]
        if p is None:
            return None
        return self.where(p)

    def steps(self, e):
        """Number of links from e up to its place (a thing directly in a place has 1)."""
        if e in PLACES:
            return 0
        if e in PERSONS:
            return 1
        p = self.parent[e]
        if p is None:
            return 0
        return 1 + self.steps(p)

    def ancestors(self, e):
        """Holders of e, from nearest to farthest, stopping at a place or person."""
        out = []
        while e in self.parent and self.parent[e] is not None:
            e = self.parent[e]
            out.append(e)
        return out

    def held(self, e):
        return any(a in PERSONS for a in self.ancestors(e))

    # ---- statements ----

    def statement_options(self):
        opts = {"place": [], "goes": [], "takes": [], "puts": []}
        for x in MOVABLE:
            if self.held(x):
                continue
            for y in CONTAINERS + PLACES:
                if y == x or x in self.ancestors(y) or self.parent[x] == y:
                    continue
                opts["place"].append((x, y))
        for p in PERSONS:
            for z in PLACES:
                if self.loc[p] != z:
                    opts["goes"].append((p, z))
        for p in PERSONS:
            if self.loc[p] is None:
                continue
            for x in MOVABLE:
                if not self.held(x) and self.where(x) == self.loc[p]:
                    opts["takes"].append((p, x))
        for p in PERSONS:
            if self.loc[p] is None:
                continue
            for x in MOVABLE:
                if self.parent[x] != p:
                    continue
                for y in CONTAINERS + PLACES:
                    if y == x or x in self.ancestors(y):
                        continue
                    if self.where(y) != self.loc[p]:
                        continue
                    opts["puts"].append((p, x, y))
        return opts

    def apply(self, kind, c):
        if kind == "place":
            x, y = c
            self.parent[x] = y
            return "The %s is in the %s." % (x, y)
        if kind == "goes":
            p, z = c
            self.loc[p] = z
            return "%s goes to the %s." % (p, z)
        if kind == "takes":
            p, x = c
            self.parent[x] = p
            return "%s takes the %s." % (p, x)
        p, x, y = c
        self.parent[x] = y
        return "%s puts the %s in the %s." % (p, x, y)

    # ---- questions: each returns (text, answer) ----

    def question_options(self):
        qs = []
        for x in MOVABLE:
            if self.where(x) is not None and self.steps(x) >= 2:
                qs.append(("Where is the %s?" % x, self.where(x)))
        for x in MOVABLE:
            if self.where(x) is None:
                continue
            for y in CONTAINERS + PLACES:
                if x == y or self.where(y) is None:
                    continue
                inside = y in self.ancestors(x) or self.where(x) == y
                qs.append(("Is the %s in the %s?" % (x, y), "yes" if inside else "no"))
        for y in CONTAINERS:
            if self.where(y) is None:
                continue
            inside = sorted(e for e in MOVABLE if y in self.ancestors(e))
            ans = ",".join(inside) if inside else "nothing"
            qs.append(("What is in the %s?" % y, ans))
        for p in PERSONS:
            if self.loc[p] is not None:
                qs.append(("Where is %s?" % p, self.loc[p]))
        return qs


def make_story(rng):
    for _ in range(10000):
        w = World()
        s_left = rng.randint(3, 12)
        q_left = rng.randint(1, 5)
        items = []  # (is_question, text, answer)
        ok = True
        while s_left or q_left:
            if q_left and items and (s_left == 0 or rng.random() < q_left / float(s_left + q_left)):
                qs = w.question_options()
                if qs:
                    text, ans = rng.choice(qs)
                    items.append((True, text, ans))
                    q_left -= 1
                    continue
            if s_left == 0:
                ok = False
                break
            opts = w.statement_options()
            kinds = [k for k in ("place", "goes", "takes", "puts") if opts[k]]
            if not kinds:
                ok = False
                break
            kind = rng.choice(kinds)
            text = w.apply(kind, rng.choice(opts[kind]))
            items.append((False, text, None))
            s_left -= 1
        if ok:
            return items
    raise RuntimeError("could not build a story")


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for i, (is_q, text, ans) in enumerate(make_story(rng), 1):
            if is_q:
                print("%d %s\t%s" % (i, text, ans))
            else:
                print("%d %s" % (i, text))


if __name__ == "__main__":
    main()
