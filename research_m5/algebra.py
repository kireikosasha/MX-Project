#!/usr/bin/env python3
"""
An experiment, not part of the learner: can the questions be COMPOSED from a small fixed set of operators, instead of
being picked from the list of hand-written ends of records.py?

The state is the one of records.py. Every store is given out as a relation (a thing -> things), and the same operators
apply to any of them:

    relations   F_k (the value), W_k (the past values), E_k (what it leads to), L_k[d] (what lies from it in the direction
                d), PAIR (a thing -> the pairs it is part of), PART (a pair -> its parts);  inv R, R+ (through others as
                well), R;R'
    sets        x.R, the things named in the question, all the things of a kind, S and S', S less S', those of S that
                have an amount above (below, equal to) a number
    numbers     how many in S, the amount of x, the sum of the amounts of S, how many times x was given a value, a number
                of the question
    a thing     the utmost of S by amount or by an order (the one all the others come to)
    yes/no      x in S, S is not empty, a number against a number
    a look      the same, asked of the state as it was at a time
    a word      one parameter (the direction, or which utmost) may be said by a word of the question: which word says
                what is learnt from the answers

The readings of the sentences are GIVEN here (the ones the learner finds, written by hand below): only the questions
are searched. For every kind of question all the programs up to a depth are listed, scored on the stories learnt from
(20 and the 20 next), and the best is checked on the test.

    python algebra.py --data new --depth 4
"""
import argparse
import copy
import glob
import itertools
import os
import re
import time
from collections import Counter

import records
import rows
from records import MOVES, State, both, kind, taken

B = both


def reading(name, op, kinds):
    """the readings of the sentences, by hand: what the learner itself finds for these tasks"""
    nt = len([k for k in kinds if k != "#"])
    if name in ("F3", "G3", "H3", "F8"):                         # a is d of b: a link with a direction
        return (("link", 0, (0, nt - 1, nt - 2), None),)
    if name == "F6":                                             # where one is, by the time; the moves apart from the first place
        here = (("setat", 0, (nt - 2, nt - 1), 0),)
        return here if "was in" in op else here + (("setat", 1, (nt - 2, nt - 1), 0),)
    if name == "H7":                                             # how many of a thing one has: the amount of the two together
        if " has " in " " + op + " ":
            return (("qset", 0, (B(0, 1),), 0),)
        if "gives" in op:
            return (("qpass", 0, (B(0, 1), B(2, 1)), 0),)
        if "takes" in op:
            return (("qpass", 0, (B(2, 1), B(0, 1)), 0),)
        return ((("qadd" if "gets" in op else "qsub"), 0, (B(0, 1),), 0),)
    raise KeyError(name)


def universe(s):
    out = set()
    for f, h in zip(s.F, s.H):
        out.update(f)
        out.update(f.values())
        out.update(v for hist in h.values() for _, v in hist if v is not None)
    for e in s.E:
        out.update(x for pair in e for x in pair if x.__class__ is str)
    for links in s.L:
        out.update(x for a, b, _, _ in links for x in (a, b))
    for q in s.Q:
        out.update(q)
    return out


class Look:
    """a state at a question, with what the operators need worked out once"""

    def __init__(self, s, t, nums, opp):
        self.s, self.t, self.nums, self.opp = s, t, nums, opp
        self.all = universe(s) | {x for x in t[:-2]}
        self.memo = {}

    def image(self, r, S, d=None):
        """the things the relation r leads to from the set S"""
        key = (r, frozenset(S), d)
        if key not in self.memo:
            self.memo[key] = frozenset(self.go(r, S, d))
        return self.memo[key]

    def go(self, r, S, d):
        s, name = self.s, r[0]
        if name == "F":
            return {s.F[r[1]][x] for x in S if x in s.F[r[1]]}
        if name == "W":
            return {v for x in S for v in s.was(r[1], x)}
        if name == "E":
            return {b for a, b in s.E[r[1]] if a in S}
        if name == "L":
            way = d if r[2] == "?" else self.t[r[2]]
            return {a for a, b, lab, _ in s.L[r[1]] if lab == way and b in S} | {b for a, b, lab, _ in s.L[r[1]] if self.opp.get(lab) == way and a in S}
        if name == "PAIR":
            return {p for p in self.all if "+" in p and not S.isdisjoint(p.split("+"))}
        if name == "PART":
            return {x for p in S if "+" in p for x in p.split("+")}
        if name == "inv":
            return {y for y in self.all if not S.isdisjoint(self.image(r[1], {y}, d))}
        if name == "plus":
            out, new = set(), set(self.image(r[1], S, d))
            while new - out:
                out |= new
                new = set(self.image(r[1], new, d))
            return out
        return self.image(r[2], self.image(r[1], S, d), d)       # comp


def run(p, c, d=None):
    """what a program gives at a look: a set, a number, a thing, True or False, or None"""
    name = p[0]
    if name == "X":
        return c.t[p[1]]
    if name == "N":
        return c.nums[p[1]]
    if name == "0":
        return 0
    if name == "XX":                                             # two things of the row together
        return "+".join(sorted((c.t[p[1]], c.t[p[2]])))
    if name == "app":
        x = run(p[1], c, d)
        return None if x is None else c.image(p[2], frozenset([x]), d)
    if name == "named":
        return frozenset(x for x in c.t[:-2] if kind(x) == p[1])
    if name == "all":
        return frozenset(x for x in c.all if kind(x) == p[1])
    if name in ("and", "less"):
        a, b = run(p[1], c, d), run(p[2], c, d)
        return None if a is None or b is None else (a & b if name == "and" else a - b)
    if name == "keep":
        S, n = run(p[1], c, d), 0 if p[4] is None else run(p[4], c, d)
        q = c.s.Q[p[2]]
        return None if S is None else frozenset(x for x in S if (q.get(x, 0) > n if p[3] == ">" else q.get(x, 0) < n if p[3] == "<" else q.get(x, 0) == n))
    if name == "cnt":
        S = run(p[1], c, d)
        return None if S is None else len(S)
    if name == "q":
        x = run(p[2], c, d)
        return None if x is None else c.s.Q[p[1]].get(x, 0)
    if name == "sumq":
        S = run(p[2], c, d)
        return None if S is None else sum(c.s.Q[p[1]].get(x, 0) for x in S)
    if name == "times":
        x = run(p[2], c, d)
        return None if x is None else sum(1 for _, v in c.s.H[p[1]].get(x, ()) if v is not None)
    if name == "ext":                                            # the utmost of a set: by amount, or by an order
        S, most = run(p[1], c, d), (d if p[3] == "?" else p[3])
        if not S or len(S) < 2:
            return None
        if p[2][0] == "q":
            vs = {x: c.s.Q[p[2][1]].get(x, 0) for x in S}
            best = max(vs.values()) if most == "most" else min(vs.values())
            top = [x for x in S if vs[x] == best]
        else:
            reach = {x: c.image(("plus", p[2][1]), frozenset([x]), d) | {x} for x in S}
            top = [x for x in S if all((x in reach[y]) if most == "most" else (y in reach[x]) for y in S)]
        return top[0] if len(top) == 1 else None
    if name == "mem":
        x, S = run(p[1], c, d), run(p[2], c, d)
        return None if x is None or S is None else x in S
    if name == "some":
        S = run(p[1], c, d)
        return None if S is None else bool(S)
    if name == "cmp":
        a, b = run(p[1], c, d), run(p[3], c, d)
        return None if a is None or b is None else records.CMP[p[2]](a, b)
    if name == "which":                                          # the values of the parameter with which it is so
        return frozenset(v for v in sorted(c.opp) if run(p[1], c, v) is True)
    if name == "at":
        key = ("at", c.nums[p[1]], p[2])
        if key not in c.memo:
            c.memo[key] = Look(c.s.before(c.nums[p[1]], p[2]), c.t, c.nums, c.opp)
        return run(p[3], c.memo[key], d)
    raise KeyError(name)


def say(v, own, x=None):
    """the value as an answer is written"""
    if v is None or v.__class__ is bool:
        return v
    if v.__class__ is int:
        return str(v)
    if v.__class__ is str:
        v = [v]
    parts = {y for z in v for y in (z.split("+") if "+" in z else [z])} if any("+" in z for z in v) else set(v)
    out = sorted(y for y in parts if own is None or kind(y) == own)
    return ",".join(out) if out else None


def holes(p):
    """the parameters a program leaves to a word of the question: a direction, or which utmost"""
    out = set()
    for x in p if p.__class__ is tuple and p[0] != "which" else ():
        if p[0] == "L" and p[2] == "?":
            out.add("d")
        if p[0] == "ext" and p[3] == "?":
            out.add("m")
        out |= holes(x) if x.__class__ is tuple else set()
    return out


def used(p, tk):
    """the things of the row a program looks at"""
    if p[0] == "X" or p[0] == "L" and p[2].__class__ is int:
        return {p[-1]}
    if p[0] == "XX":
        return {p[1], p[2]}
    if p[0] == "named":
        return {i for i, k in enumerate(tk) if k == p[1]}
    return set().union(*[used(x, tk) for x in p if x.__class__ is tuple and x])


def listing(depth, things, tk, nn, own, sh, stores, dirs, marks, timed):
    """all the programs up to the depth, by what they give"""
    xs = [("X", i) for i in range(len(tk))]
    two = [("XX", i, j) for i in range(len(tk)) for j in range(i + 1, len(tk)) if tk[i] != tk[j]] if ("P", 0) in stores else []
    atoms = [(sort, k) for sort, k in stores if sort in "FE"] + [("W", k) for sort, k in stores if sort == "F"]
    atoms += [("L", k, j) for sort, k in stores if sort == "L" for j in dirs] + [("L", k, "?") for sort, k in stores if sort == "L"]
    atoms += [("PAIR",), ("PART",)] if ("P", 0) in stores else []
    rel = {1: list(atoms)}
    rel[2] = [("inv", r) for r in atoms] + [("plus", r) for r in atoms] + [("comp", a, b) for a in atoms for b in atoms if a != b or a[0] in "FEL"]
    rel[3] = [("plus", ("inv", r)) for r in atoms] + [("inv", ("plus", r)) for r in atoms]
    qs = [k for sort, k in stores if sort == "Q"]
    fs = [k for sort, k in stores if sort == "F"]
    kinds = sorted(set(things) | ({own} if own else set()))
    orders = [("ord", r) for r in atoms if r[0] in "EL"] + [("q", k) for k in qs]
    sets, nums, one, yes = {0: []}, {0: []}, {0: [], 1: xs + two}, {0: []}
    for d in range(1, depth + 1):
        new = [("app", x, r) for x in xs for dd in range(1, d) for r in rel.get(dd, ())]
        new += [("named", k) for k in kinds] + [("all", k) for k in kinds] if d == 1 else []
        new += [("app", x, r) for x in one.get(d - 1, ()) if x[0] != "X" for r in rel[1]] if d > 2 else []
        new += [("keep", S, k, c, n) for S in sets[d - 1] for k in qs for c in (">", "=", "<") for n in [None] + [("N", i) for i in range(nn)]]
        new += [(name, a, b) for name in ("and", "less") for a in sets[d - 1] for b in sets[d - 1] if a != b and a[0] != "named" and (name == "less" or str(a) < str(b))] if d <= 3 else []
        sets[d] = new
        one[d] = one.get(d, []) + [("ext", S, o, m) for S in sets[d - 1] for o in orders for m in ("most", "least", "?")]
        nums[d] = [("cnt", S) for S in sets[d - 1]] + [("sumq", k, S) for k in qs for S in sets[d - 1]]
        nums[d] += [("q", k, x) for k in qs for x in one.get(d - 1, ())] + [("times", k, x) for k in fs for x in one.get(d - 1, ())]
        yes[d] = [("mem", x, S) for x in xs for S in sets[d - 1]] + [("some", S) for S in sets[d - 1]]
        yes[d] += [("cmp", a, c, b) for a in nums[d - 1] for c in (">", "=") for b in nums[d - 1] + [("N", i) for i in range(nn)] + [("0",)] if a != b]
    pool = {"t": sets, "n": nums, "b": yes}[sh]
    out = [p for d in range(1, depth + 1) for p in pool[d]] + ([p for d in range(2, depth + 1) for p in one[d]] if sh == "t" else [])
    out += [("which", y) for d in range(1, depth + 1) for y in yes[d] if "d" in holes(y)] if sh == "t" and marks else []
    if nn and timed:
        out += [("at", n, strict, p) for p in out if p[0] not in ("cmp",) for n in range(nn) for strict in (True, False)]
    return list(dict.fromkeys(out))


def looks(episodes, rec, forms, opp):
    """the looks at the questions of the forms: play the sentences as they are read, stop at every question"""
    out = {f: [] for f in forms}
    for ep in episodes:
        s = State()
        for s.now, (op, quest, things, numbers, kinds, gold) in enumerate(ep):
            form, t = (op, quest, kinds), things + (op, records.ME)
            if quest:
                if form in out:
                    out[form].append((Look(copy.deepcopy(s), t, numbers, opp), gold))
            else:
                for x in things:
                    s.born.setdefault(x, s.now)
                for name, k, pick, n in rec.get(form, ()):
                    MOVES[name][0](s, k, taken(t, numbers, pick), (numbers[n],) if n is not None else ())
    return out


def answers(p, cs, own, hs, values, key):
    """what the program says at every look; with a parameter left to a word: for every value of it"""
    outs = []
    for c, gold in cs:
        try:
            if hs:
                opts = tuple((v, say(run(p, c, v), own)) for v in values)
                outs.append(("?", key(c), opts) if any(o is not None for _, o in opts) else None)
            else:
                outs.append(say(run(p, c), own))
        except (KeyError, IndexError, ValueError, TypeError):
            outs.append(None)
    return outs


def study(name, data, task, depth, stories=40):
    raw = rows.stories(glob.glob(os.path.join(data, f"qa{task}_*train.txt"))[0], stories)
    reader = rows.Reader(raw[:20], 0)
    eps = reader.rows(raw)
    test = reader.rows(rows.stories(glob.glob(os.path.join(data, f"qa{task}_*test.txt"))[0]))
    m = records.Learner(eps)
    rec = {f: reading(name, f[0], f[2]) for f in m.rec}
    types = m.types(rec)
    stores = sorted({(sort, k) for sort, k in types if sort in "FQEL"} | ({("P", 0)} if any("+" in str(v) for v in types.values()) else set()))
    timed = any(sort == "F@" for sort, k in types)
    dkinds = {ks[2] for (sort, k), v in types.items() if sort == "L" for ks in v}
    marks = sorted(w for dk in dkinds for w in m.names.get(dk, []))
    pairings = [dict(o) for o in records.pairings(marks)] if marks and len(marks) % 2 == 0 else [{}]
    print(f"{name}: stores {stores}, directions {len(marks)}, {len(pairings)} ways to pair them", flush=True)
    for f in sorted((f for f in m.forms if f[1]), key=lambda f: f[0]):
        golds = m.forms[f]
        tk = [k for k in f[2] if k != "#"]
        nn = len(f[2]) - len(tk)
        sh = Counter(records.shape(g) for g in golds).most_common(1)[0][0]
        own = m.owns.get(f, [None])[0]
        dirs = [j for j, k in enumerate(tk) if k in dkinds]
        progs = listing(depth, tk, tk, nn, own, sh, stores, dirs, marks, timed)
        yesno = all(g in ("yes", "no") for g in golds)
        best, seen, t0 = None, {}, time.time()
        for opp in pairings:
            cs = looks(eps, rec, [f], opp)[f]
            for p in progs:
                hs = holes(p)
                if len(hs) > 1:
                    continue
                values = marks if hs == {"d"} else ("most", "least")
                free = [i for i in range(len(tk)) if i not in used(p, tk)]
                outs = answers(p, cs, own, hs, values, lambda c: tuple(c.t[i] for i in free) or c.t[-2])
                sig = (tuple(sorted(opp.items())), tuple(outs))
                if sig in seen:
                    continue
                seen[sig] = p
                for flip, g in enumerate(records.Learner.grade(outs, golds, yesno)):
                    size = len(str(p))
                    if best is None or (g[1], -size) > (best[0], -best[1]):
                        best, tie = (g[1], size, p, bool(flip), g[2], opp, hs, values, free), 1
                    elif g[1] == best[0]:
                        tie += 1
        right, size, p, flip, default, opp, hs, values, free = best
        ts = looks(test, rec, [f] + [g for g in {(op, q, k) for ep in test for op, q, _, _, k, _ in ep} if g[:2] == f[:2] and records.size(g) == records.size(f)], opp)
        tcs = [x for v in ts.values() for x in v]
        table = {}
        if hs:                                                   # which word says what: from the stories learnt from
            for o, gold in zip(answers(p, looks(eps, rec, [f], opp)[f], own, hs, values, lambda c: tuple(c.t[i] for i in free) or c.t[-2]), golds):
                for word, v in records.pieces(o, gold) if o else ():
                    table.setdefault(word, Counter())[v] += 1
            table = {w: c.most_common(1)[0][0] for w, c in table.items()}
        good = 0
        for c, gold in tcs:
            try:
                key = tuple(c.t[i] for i in free) or c.t[-2]
                o = say(run(p, c, table.get(key)), own) if hs else say(run(p, c), own)
            except (KeyError, IndexError, ValueError, TypeError):
                o = None
            if o.__class__ is bool:
                o = "yes" if o != flip else "no"
            good += (default if o is None else o) == gold
        print(f"  ? {f[0]} {' '.join(f[2])}: {len(progs)} programs x {len(pairings)}, {len(seen)} that differ in what they say; best explains "
              f"{right}/{len(golds)} ({tie} as good), test {good}/{len(tcs)} = {100 * good / max(1, len(tcs)):.1f}%  ({time.time() - t0:.0f} s)\n      {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the folder with the sets (F, G, H inside)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for name in [x for x in a.only.split(",") if x] or ["F3", "G3", "H3", "F8", "F6", "H7"]:
        study(name, os.path.join(a.data, name[0]), int(name[1:]), a.depth)


if __name__ == "__main__":
    main()
