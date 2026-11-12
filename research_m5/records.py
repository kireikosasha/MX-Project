#!/usr/bin/env python3
"""
Learning tasks given as records (rows of codes and numbers: a log, a ledger, a table) from 20 episodes with answers.

The ground here is COMPOSED, not listed. There is one state:

    F[k]  what a thing has as its "k-th something" (where it is, who holds it, its kind, its boss): thing -> thing,
          with the history of the values
    Q[k]  how much a thing has: thing -> number
    E[k]  what leads to what (a is a b, a is above b, who has a and b has c): pairs
    L[k]  what lies where: a is n steps along a direction from b

A record row is one move over the state (set, unset, lift one step up, swap, move all, pair, tie, untie, add, take,
pass, ...). A query row is a walk and an end: the walk starts at a thing of the row and makes one or two steps (to the
value, back to those that have the value, along the whole chain, to its end, over the past values; over the links),
the end says what to do with what the walk has come to (name it, list it, count it, sum the amounts, is the other
thing of the row there, is anything there, compare). The kinds of things (the letters of the codes) say which steps
can follow which, so the walks worth trying are few.

The learner guesses the move of every record; for a guess, the best walk and end of every query is found by trying
them all on the questions of the episodes; a guess is kept if the queries are answered better with it. An answer that
is right by chance (always "no") is worth nothing. Nothing in the ground knows a code of any task.

    python records.py --data new/N --tasks 1,2,3 --also new/N3,new/N10 --show
"""
import argparse
import glob
import itertools
import os
import re
import time
from collections import Counter

K = {"F": 3, "Q": 2, "E": 2, "L": 1}
ME = "@"                                                  # the one who writes the log, when a row does not say who acts
KINDS = {ME: ME}
SPACE = ("dir", "dist", "route", "isdir")
WORK = 600000                                             # how much work one step of learning may take, in episodes played
PRICE = 0.25                                              # what a part of a guess costs, in questions answered beyond chance


def kind(t):
    if t not in KINDS:
        KINDS[t] = re.match(r"[a-z]*", t).group(0) if re.search(r"\d", t) else "=" + t
    return KINDS[t]


def read(path, limit=None):
    """a file of episodes -> [[(operation, is it a query, things, numbers, kinds, answer or None)]]"""
    out = []
    for line in open(path, encoding="utf-8"):
        num, rest = line.strip().split(" ", 1)
        if num == "1":
            if limit and len(out) >= limit:
                break
            out.append([])
        parts = rest.split("\t")
        toks = re.findall(r"[^\W_]+", parts[0].lower())
        things, numbers, kinds = [], [], []
        for t in toks[1:]:
            if t.isdigit():
                numbers.append(int(t))
                kinds.append("#")
            else:
                things.append(t)
                kinds.append(kind(t))
        out[-1].append((toks[0], len(parts) > 1, tuple(things), tuple(numbers), tuple(kinds), parts[1].strip().lower() if len(parts) > 1 else None))
    return out


class State:
    def __init__(self):
        self.F = [{} for _ in range(K["F"])]
        self.H = [{} for _ in range(K["F"])]              # thing -> ((when, value), ...) in the order of time
        self.Q = [{} for _ in range(K["Q"])]
        self.E = [set() for _ in range(K["E"])]           # (a, b): a leads to b; ((a, b), c): a and b together lead to c
        self.L = [[] for _ in range(K["L"])]              # (a, b, direction, n): a lies n steps along the direction from b
        self.W = [{} for _ in range(K["F"])]              # thing -> those who wait to be its value, in the order they came
        self.now = 0

    def put(self, k, a, b, when=None):
        h = self.H[k].get(a, ())
        h = h + ((self.now, b),) if when is None else tuple(sorted(h + ((when, b),), key=lambda x: x[0]))
        self.H[k][a] = h
        if h[-1][1] is None:
            self.F[k].pop(a, None)
        else:
            self.F[k][a] = h[-1][1]

    def chain(self, k, a):
        """a's value, the value of that, and so on"""
        out, seen, f = [], {a}, self.F[k]
        while True:
            a = f.get(a)
            if a is None or a in seen:
                return out
            seen.add(a)
            out.append(a)

    def closure(self, k, have):
        """all that the things lead to, they themselves among it"""
        have, e, grew = set(have), self.E[k], True
        while grew:
            grew = False
            for a, b in e:
                if b not in have and (a in have if a.__class__ is str else a[0] in have and a[1] in have):
                    have.add(b)
                    grew = True
        return have

    def was(self, k, a):
        return [v for _, v in self.H[k].get(a, ()) if v is not None]


# ---------------------------------------------------------------- the moves of a record
def m_set(s, k, a, n):
    s.put(k, a[0], a[1])


def m_unset(s, k, a, n):
    if a[0] in s.F[k]:
        line = s.W[k].get(a[0])
        s.put(k, a[0], line.pop(0) if line else None)


def m_claim(s, k, a, n):
    if a[0] not in s.F[k]:
        s.put(k, a[0], a[1])


def m_wait(s, k, a, n):
    if a[0] not in s.F[k]:
        s.put(k, a[0], a[1])
    elif s.F[k][a[0]] != a[1] and a[1] not in s.W[k].setdefault(a[0], []):
        s.W[k][a[0]].append(a[1])


def m_rise(s, k, a, n):
    f = s.F[k]
    if a[0] in f:
        s.put(k, a[0], f.get(f[a[0]]))


def m_swap(s, k, a, n):
    x, y = s.F[k].get(a[0]), s.F[k].get(a[1])
    s.put(k, a[0], y)
    s.put(k, a[1], x)


def m_moveall(s, k, a, n):
    for x in [x for x, v in s.F[k].items() if v == a[0]]:
        s.put(k, x, a[1])


def m_swapall(s, k, a, n):
    mine, yours = [x for x, v in s.F[k].items() if v == a[0]], [x for x, v in s.F[k].items() if v == a[1]]
    for x in mine:
        s.put(k, x, a[1])
    for x in yours:
        s.put(k, x, a[0])


def m_pair(s, k, a, n):
    f = s.F[k]
    for x in a[:2]:
        old = f.get(x)
        if old is not None and f.get(old) == x:
            s.put(k, old, None)
    s.put(k, a[0], a[1])
    s.put(k, a[1], a[0])


def m_unpair(s, k, a, n):
    f = s.F[k]
    old = f.get(a[0])
    if old is not None:
        if f.get(old) == a[0]:
            s.put(k, old, None)
        s.put(k, a[0], None)


def m_setat(s, k, a, n):
    s.put(k, a[0], a[1], n[0])


def m_tie(s, k, a, n):
    s.E[k].add((a[0], a[1]))


def m_untie(s, k, a, n):
    s.E[k].discard((a[0], a[1]))


def m_tieboth(s, k, a, n):
    s.E[k].add(((a[0], a[1]), a[2]))


def m_qset(s, k, a, n):
    s.Q[k][a[0]] = n[0]


def m_qadd(s, k, a, n):
    s.Q[k][a[0]] = s.Q[k].get(a[0], 0) + (n[0] if n else 1)


def m_qsub(s, k, a, n):
    s.Q[k][a[0]] = s.Q[k].get(a[0], 0) - (n[0] if n else 1)


def m_qpass(s, k, a, n):
    d = n[0] if n else 1
    s.Q[k][a[0]] = s.Q[k].get(a[0], 0) - d
    s.Q[k][a[1]] = s.Q[k].get(a[1], 0) + d


def m_qsame(s, k, a, n):
    for q in s.Q:
        q[a[0]] = q.get(a[1], 0)


def m_link(s, k, a, n):
    s.L[k].append((a[0], a[1], a[2], n[0] if n else 1))


# name: (the move, the sort of slot, things used, a number: 0 none, 1 needed, 2 taken if there is one)
MOVES = {"set": (m_set, "F", 2, 0), "unset": (m_unset, "F", 1, 0), "rise": (m_rise, "F", 1, 0), "swap": (m_swap, "F", 2, 0),
         "moveall": (m_moveall, "F", 2, 0), "swapall": (m_swapall, "F", 2, 0), "pair": (m_pair, "F", 2, 0), "unpair": (m_unpair, "F", 1, 0),
         "setat": (m_setat, "F", 2, 1), "claim": (m_claim, "F", 2, 0), "wait": (m_wait, "F", 2, 0), "tie": (m_tie, "E", 2, 0), "untie": (m_untie, "E", 2, 0), "tieboth": (m_tieboth, "E", 3, 0),
         "qset": (m_qset, "Q", 1, 1), "qadd": (m_qadd, "Q", 1, 2), "qsub": (m_qsub, "Q", 1, 2), "qpass": (m_qpass, "Q", 2, 2),
         "qsame": (m_qsame, "Q", 2, 0), "link": (m_link, "L", 3, 2)}
SAME_BOTH_WAYS = ("swap", "swapall", "pair", "tieboth")


# ---------------------------------------------------------------- the steps of a walk
F_STEPS = (">", "<", "^", ">>", "<<", "~", "<~")
E_STEPS = (">>", "<<", ">", "<")


def step(s, st, S):
    sort, k, op = st
    if sort == "W":
        return {y for x in S for y in s.W[k].get(x, ())}
    if sort == "F":
        f = s.F[k]
        if op == ">":
            return {f[x] for x in S if x in f}
        if op == "<":
            return {x for x, v in f.items() if v in S}
        if op == "^":
            return {c[-1] for c in (s.chain(k, x) for x in S) if c}
        if op == ">>":
            return {y for x in S for y in s.chain(k, x)}
        if op == "<<":
            return {x for x in f if not S.isdisjoint(s.chain(k, x))}
        if op == "~":
            return {v for x in S for v in s.was(k, x)}
        return {x for x in s.H[k] if not S.isdisjoint(s.was(k, x))}
    e = s.E[k]
    if op == ">":
        return {b for a, b in e if a in S}
    if op == "<":
        return {a for a, b in e if b in S and a.__class__ is str}
    if op == ">>":
        return s.closure(k, S) - S
    nodes = {x for a, _ in e for x in ((a,) if a.__class__ is str else a)} - S
    return {x for x in nodes if not S.isdisjoint(s.closure(k, {x}))}


def after(types, st, kinds):
    """of what kinds the things can be after the step"""
    sort, k, op = st
    pairs = types.get((sort, k), ())
    if op[0] == "<":
        pairs = {(b, a) for a, b in pairs}
    out = {b for a, b in pairs if a in kinds}
    if op in (">>", "<<", "^"):
        while True:
            more = {b for a, b in pairs if a in out} - out
            if not more:
                break
            out |= more
    return out


def walks(types, start):
    """the walks of one and two steps from a thing of the kind -> [(steps, the kinds they come to)]"""
    steps = [(sort, k, op) for (sort, k) in sorted(types) if sort in "FEW" for op in {"F": F_STEPS, "E": E_STEPS, "W": ("?",)}[sort]]
    one = [((st,), kinds) for st in steps for kinds in [after(types, st, {start})] if kinds]
    two = []
    for (a,), mid in one:
        for b in steps:
            if a[:2] == b[:2] and (a[2] == "^" and b[2][0] != "<" or a[2] == b[2] and a[2] in (">>", "<<", "~", "<~")):
                continue
            kinds = after(types, b, mid)
            if kinds:
                two.append(((a, b), kinds))
    return one + two


def walk(s, memo, x, path):
    key = (x, path)
    if key not in memo:
        memo[key] = step(s, path[-1], walk(s, memo, x, path[:-1])) if path else {x}
    return memo[key]


def amount(s, memo, e, x):
    """a number of a thing: how much it has, how many the walk comes to, the sum of what they have"""
    if e[0] == "q":
        return s.Q[e[1]].get(x, 0)
    S = walk(s, memo, x, e[1])
    if e[0] == "count":
        return len(S) if e[2] is None else sum(1 for y in S if kind(y) == e[2])
    q = s.Q[e[2]]
    return sum(q.get(y, 0) for y in S)


def where(s, k, origin, opposite):
    """the place of every thing joined to the origin by what lies where: thing -> {direction: steps}"""
    pos, todo = {origin: {}}, [origin]
    while todo:
        x = todo.pop()
        for a, b, d, n in s.L[k]:
            if d not in opposite or (a != x and b != x) or (a in pos and b in pos):
                continue
            axis, sign = (d, 1) if d < opposite[d] else (opposite[d], -1)
            new, sign = (a, sign) if b == x else (b, -sign)
            p = dict(pos[x])
            p[axis] = p.get(axis, 0) + sign * n
            pos[new] = p
            todo.append(new)
    return pos


def route(s, k, a, b, opposite):
    """the directions of the links on the way from a to b"""
    came, todo = {a: None}, [a]
    while todo and b not in came:
        x = todo.pop(0)
        for p, q, d, n in s.L[k]:
            if d in opposite:
                for y, z, turn in ((p, q, opposite[d]), (q, p, d)):
                    if y == x and z not in came:
                        came[z] = (x, turn)
                        todo.append(z)
    if b not in came or a == b:
        return None
    out = []
    while came[b]:
        b, d = came[b]
        out.append(d)
    return ",".join(reversed(out))


CMP = {"=": lambda a, b: a == b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}


def answer(s, p, t, nums, memo):
    """what the query gives: a string as it is written, True or False, or None if there is nothing to say"""
    name = p[0]
    if name in ("list", "seq"):
        _, i, path, c = p
        S = walk(s, memo, t[i], path)
        xs = [x for x in S if c is None or kind(x) == c]
        if not xs:
            return None
        if name == "list":
            return ",".join(sorted(xs))
        sort, k, op = path[-1]
        if op == "?":
            return ",".join(x for src in sorted(walk(s, memo, t[i], path[:-1])) for x in s.W[k].get(src, ()) if x in xs)
        if op == "~":
            src = walk(s, memo, t[i], path[:-1])
            order = [v for x in sorted(src) for v in s.was(k, x)]
            return ",".join(dict.fromkeys(v for v in order if v in xs))
        return ",".join(sorted(xs, key=lambda x: s.H[k][x][-1][0]))
    if name == "has":
        return t[p[3]] in walk(s, memo, t[p[1]], p[2])
    if name == "some":
        return bool(walk(s, memo, t[p[1]], p[2]))
    if name == "same":
        a = walk(s, memo, t[p[1]], p[3])
        return bool(a) and a == walk(s, memo, t[p[2]], p[3])
    if name == "num":
        return str(amount(s, memo, p[2], t[p[1]]))
    if name == "diff":
        return str(amount(s, memo, p[3], t[p[1]]) - amount(s, memo, p[3], t[p[2]]))
    if name == "plus":
        return str(sum(amount(s, memo, p[1], x) for x in t[:-1]))
    if name == "cmpn":
        return CMP[p[3]](amount(s, memo, p[2], t[p[1]]), nums[p[4]])
    if name == "cmp":
        return CMP[p[4]](amount(s, memo, p[3], t[p[1]]), amount(s, memo, p[3], t[p[2]]))
    if name in ("most", "least"):
        v = [amount(s, memo, p[1], x) for x in t[:-1]]
        top = max(v) if name == "most" else min(v)
        return t[v.index(top)] if v.count(top) == 1 else None
    if name in ("over", "under"):
        got = {x: s.closure(p[1], {x}) for x in t[:-1]}
        best = [x for x in t[:-1] if all(y in got[x] if name == "over" else x in got[y] for y in t[:-1])]
        return best[0] if len(best) == 1 else None
    if name in ("before", "after"):
        h, v = s.was(p[1], t[p[2]]), t[p[3]]
        if h.count(v) != 1:
            return None
        i = h.index(v) + (1 if name == "after" else -1)
        return h[i] if 0 <= i < len(h) else None
    if name == "span":
        own = [x for x in s.H[p[1]] if t[p[2]] in s.was(p[1], x) and t[p[3]] in s.was(p[1], x)]
        if len(own) != 1:
            return None
        h = s.was(p[1], own[0])
        if h.count(t[p[2]]) != 1 or h.count(t[p[3]]) != 1:
            return None
        i, j = h.index(t[p[2]]), h.index(t[p[3]])
        return ",".join(dict.fromkeys(h[i:j + 1] if i <= j else h[j:i + 1][::-1]))
    if name == "sooner":
        h = s.was(p[1], t[p[2]])
        if h.count(t[p[3]]) != 1 or h.count(t[p[4]]) != 1:
            return None
        return h.index(t[p[3]]) < h.index(t[p[4]])
    if name in ("first", "last"):
        v, when = t[p[4]], []
        for x in (t[p[2]], t[p[3]]):
            at = [w for w, y in s.H[p[1]].get(x, ()) if y == v]
            if len(at) != 1:
                return None
            when.append(at[0])
        if when[0] == when[1]:
            return None
        return t[p[2]] if (when[0] < when[1]) == (name == "first") else t[p[3]]
    opposite = p[-1]
    if name == "route":
        return route(s, p[1], t[p[2]], t[p[3]], opposite)
    pos = where(s, p[1], t[p[3]], opposite).get(t[p[2]])
    if pos is None:
        return None
    if name == "isdir":
        d = t[p[4]]
        if d not in opposite:
            return None
        return pos.get(d, 0) > 0 if d < opposite[d] else pos.get(opposite[d], 0) < 0
    far = [(d, n) for d, n in pos.items() if n]
    if len(far) != 1:
        return None
    return str(abs(far[0][1])) if name == "dist" else far[0][0] if far[0][1] > 0 else opposite[far[0][0]]


def shape(gold):
    return "b" if gold in ("yes", "no") else "n" if gold.isdigit() else "t"


def pairings(xs):
    """all the ways to put the things in pairs of opposites"""
    if not xs:
        yield {}
        return
    for i in range(1, len(xs)):
        for rest in pairings(xs[1:i] + xs[i + 1:]):
            yield {**rest, xs[0]: xs[i], xs[i]: xs[0]}


def size(form):
    return sum(k != "#" for k in form[2]), sum(k == "#" for k in form[2])


class Learner:
    def __init__(self, episodes):
        self.episodes = episodes
        self.forms = {}                                          # (operation, query, kinds) -> the answers of its rows
        names = {}
        for ep in episodes:
            for op, quest, things, numbers, kinds, gold in ep:
                self.forms.setdefault((op, quest, kinds), []).append(gold)
                for t in things:
                    names.setdefault(kind(t), set()).add(t)
        self.names = {k: sorted(v) for k, v in names.items()}
        self.own = {}                                            # of what kind the things of the answers are
        for f, golds in self.forms.items():
            if f[1]:
                c = Counter(kind(x) for g in golds if shape(g) == "t" for x in g.split(","))
                self.own[f] = c.most_common(1)[0][0] if c else None
        self.rec = {f: () for f in self.forms if not f[1]}       # nothing is understood at first
        self.ask = {}
        self.cache, self.options, self.runs, self.work = {}, {}, 0, 0

    # ------------------------------------------------------------ what a row may be
    def moves(self, nt, nn, top, sorts):
        out = []
        for name, (fn, sort, t, num) in MOVES.items():
            if sort not in sorts or (num == 1 and not nn) or t > nt + 1 or (t > nt and t == 1):
                continue
            idx = list(range(nt)) + ([-1] if t == nt + 1 else [])
            for k in range(min(K[sort], top[sort] + 1)):
                if name == "qsame" and k:
                    continue
                for pick in itertools.permutations(idx, t):
                    if name in SAME_BOTH_WAYS and pick[0] % (nt + 1) > pick[1] % (nt + 1):
                        continue
                    for n in (range(nn) if num and nn else [None]):
                        out.append((name, k, pick, n))
        return sorted(out, key=lambda m: -1 in m[2])             # the one who is not named comes last

    def types(self, rec):
        """of what kinds the things in every slot are, by the moves chosen"""
        types = {}
        for form, moves in rec.items():
            tk = [x for x in form[2] if x != "#"] + [ME]
            for name, k, pick, _ in moves:
                ks = [tk[i] for i in pick]
                sort = MOVES[name][1]
                if name in ("set", "setat", "tie", "claim", "wait"):
                    types.setdefault((sort, k), set()).add((ks[0], ks[1]))
                    if name == "wait":
                        types.setdefault(("W", k), set()).add((ks[0], ks[1]))
                elif name == "pair":
                    types.setdefault((sort, k), set()).update([(ks[0], ks[1]), (ks[1], ks[0])])
                elif name == "tieboth":
                    types.setdefault((sort, k), set()).update([(ks[0], ks[2]), (ks[1], ks[2])])
                elif name in ("qset", "qadd", "qsub", "qpass"):
                    types.setdefault((sort, k), set()).update(ks)
                elif name == "qsame":
                    for q in range(K["Q"]):
                        types.setdefault((sort, q), set()).update(ks)
                elif name == "link":
                    types.setdefault((sort, k), set()).add(tuple(ks))
        return types

    def queries(self, form, types):
        """all that a query of this form may be, the simple first"""
        op, _, kinds = form
        tk = [x for x in kinds if x != "#"]
        nt, nn = len(tk), len(kinds) - len(tk)
        sh = Counter(shape(g) for g in self.forms[form]).most_common(1)[0][0]
        own = self.own.get(form)
        starts = list(range(nt)) if nt else [-1]
        tk = tk + [ME]
        W = {i: walks(types, tk[i]) for i in starts}
        fs = [k for (sort, k) in sorted(types) if sort == "F"]
        qs = [k for (sort, k) in sorted(types) if sort == "Q"]
        alike = nt >= 2 and len(set(tk[:nt])) == 1

        def amounts(i, deep=True):
            out = [("q", k) for k in qs if tk[i] in types[("Q", k)]]
            for path, ends in W[i]:
                if deep or len(path) == 1:
                    out += [("count", path, c) for c in [None] + (sorted(ends) if len(ends) > 1 else [])]
                    out += [("sum", path, k) for k in qs if ends & types[("Q", k)]]
            return out
        out = []
        if sh == "t":
            for i in starts:
                for path, ends in W[i]:
                    if own in ends or own is None:
                        if path[-1][2] == "?":
                            out.append(("seq", i, path, own))
                        out.append(("list", i, path, own))
                        if path[-1][2] in ("<", "~") and path[-1][0] == "F":
                            out.append(("seq", i, path, own))
            if nt >= 2:
                for k in fs:
                    out += [(name, k, i, j) for name in ("before", "after") for i in range(nt) for j in range(nt)
                            if i != j and (tk[i], tk[j]) in types[("F", k)]]
                    out += [(name, k, i, j, v) for name in ("first", "last") for i in range(nt) for j in range(i + 1, nt) for v in range(nt)
                            if v not in (i, j) and tk[i] == tk[j] and (tk[i], tk[v]) in types[("F", k)]]
                    out += [("span", k, i, j) for i in range(nt) for j in range(nt)
                            if i != j and tk[i] == tk[j] == own and any(y == own for _, y in types[("F", k)])]
            if alike:
                out += [(name, e) for e in amounts(0, False) for name in ("most", "least")]
                out += [(name, k) for (sort, k) in sorted(types) if sort == "E" for name in ("over", "under")]
        elif sh == "n":
            for i in starts:
                out += [("num", i, e) for e in amounts(i)]
            if alike:
                out += [("plus", e) for e in amounts(0, False)]
                out += [("diff", i, j, e) for i in range(nt) for j in range(nt) if i != j for e in amounts(0, False)]
        else:
            for i in starts:
                for path, ends in W[i]:
                    out += [("has", i, path, j) for j in range(nt) if j != i and tk[j] in ends]
                    out.append(("some", i, path))
                for e in amounts(i, False) if nn else []:
                    out += [("cmpn", i, e, c, n) for c in CMP for n in range(nn)]
            for i in range(nt):
                for j in range(nt):
                    if i != j and tk[i] == tk[j]:
                        out += [("cmp", i, j, e, c) for e in amounts(i, False) for c in CMP if c != "=" or i < j]
                        out += [("same", i, j, path) for path, _ in W[i] if i < j and len(path) == 1]
            for k in fs:
                out += [("sooner", k, i, j, v) for i in range(nt) for j in range(nt) for v in range(nt)
                        if len({i, j, v}) == 3 and (tk[i], tk[j]) in types[("F", k)] and (tk[i], tk[v]) in types[("F", k)]]
        for (sort, k) in sorted(types):                          # what lies where
            if sort != "L":
                continue
            for ka, kb, kd in sorted(types[(sort, k)]):
                marks = self.names.get(kd, [])
                if len(marks) % 2 or not 2 <= len(marks) <= 6:
                    continue
                for opposite in pairings(marks):
                    opp = tuple(sorted(opposite.items()))
                    for i in range(nt):
                        for j in range(nt):
                            if i == j or {tk[i], tk[j]} - {ka, kb}:
                                continue
                            if sh == "t":
                                out += [("dir", k, i, j, opp), ("route", k, i, j, opp)]
                            elif sh == "n":
                                out.append(("dist", k, i, j, opp))
                            else:
                                out += [("isdir", k, i, j, d, opp) for d in range(nt) if tk[d] == kd]

        def steps(p):
            return max((len(x) for x in p if x.__class__ is tuple and x and x[0].__class__ is tuple), default=0)
        return sorted(dict.fromkeys(out), key=steps)

    # ------------------------------------------------------------ how good a guess is
    @staticmethod
    def grade(outs, golds, yesno):
        """-> for the query as it is and (yes/no) turned over: (right beyond chance, right, what to say when there is nothing to say)"""
        out = []
        for flip in ((False, True) if yesno else (False,)):
            o = [None if x is None else ("yes" if x != flip else "no") for x in outs] if yesno else outs
            miss = Counter(g for x, g in zip(o, golds) if x is None)
            default = miss.most_common(1)[0] if miss else (None, 0)
            default = default[0] if default[1] >= 2 and 3 * default[1] >= 2 * sum(miss.values()) else None
            o = [default if x is None else x for x in o]
            right = sum(x == g for x, g in zip(o, golds))
            co, cg = Counter(o), Counter(golds)
            chance = sum(co[v] * cg[v] for v in co) / (len(golds) + (not yesno))
            out.append((round(right - chance, 6), right, default))
        return out

    @staticmethod
    def price(p, flip=False):
        """how much there is in a query: its steps, and what is done over them"""
        c = 1 + flip
        for x in p[1:]:
            if x == -1:
                c += 1
            elif x.__class__ is tuple and x and x[0].__class__ is tuple and len(x[0]) == 3:
                c += len(x)
            elif x.__class__ is tuple and x and x[0] in ("count", "sum"):
                c += 1 + len(x[1])
        return c

    def play(self, episodes, rec, progs):
        """run the episodes -> for every query form, what every one of its programs says to every question"""
        outs = {f: [[] for _ in ps] for f, ps in progs.items()}
        for ep in episodes:
            s = State()
            for s.now, (op, quest, things, numbers, kinds, gold) in enumerate(ep):
                form = (op, quest, kinds)
                t = things + (ME,)
                if quest:
                    memo = {}
                    for p, o in zip(progs.get(form, ()), outs.get(form, ())):
                        try:
                            o.append(answer(s, p[:-1] + (dict(p[-1]),) if p[0] in SPACE else p, t, numbers, memo))
                        except (KeyError, IndexError, ValueError, TypeError):
                            o.append(None)
                else:
                    for name, k, pick, n in rec.get(form, ()):
                        MOVES[name][0](s, k, tuple(t[i] for i in pick), (numbers[n],) if n is not None else ())
        self.runs += 1
        self.work += len(episodes)
        return outs

    def fit(self, rec):
        """for the moves chosen, the best program of every query -> (right beyond chance less the price, right), the programs.
        One operation mostly means one thing: a program shared by the forms of an operation is paid for once."""
        key = tuple(sorted(rec.items()))
        if key in self.cache:
            return self.cache[key]
        types = self.types(rec)
        tkey = tuple(sorted((k, tuple(sorted(v))) for k, v in types.items()))
        progs = {}
        for f in self.forms:
            if f[1]:
                if (f, tkey) not in self.options:
                    self.options[(f, tkey)] = self.queries(f, types)
                progs[f] = self.options[(f, tkey)]
        outs = self.play(self.episodes, rec, progs)
        marks, ops = {}, {}                                  # form -> {(program, turned over): (beyond, right, default)}
        for f, ps in progs.items():
            golds = self.forms[f]
            yesno = all(g in ("yes", "no") for g in golds)
            marks[f] = {(p, flip): g for p, o in zip(ps, outs[f]) for flip, g in enumerate(self.grade(o, golds, yesno))}
            ops.setdefault(f[0], []).append(f)
        ask, score, right = {}, 0.0, 0
        for op, forms in ops.items():
            own = {}                                             # the best of a form by itself
            for f in forms:
                for c, g in marks[f].items():
                    v = (g[0] - PRICE * self.price(*c), g[1])
                    if g[0] > 0 and v[0] > 0 and (f not in own or v > own[f][0]):
                        own[f] = (v, c)
            best = (sum(v[0] for v, _ in own.values()), None)
            if len(forms) > 1:
                shared = Counter()
                for f in forms:
                    for c, g in marks[f].items():
                        if g[0] > 0:
                            shared[c] += g[0]
                for c, _ in shared.most_common(6):               # one program for the whole operation, the others where they are better
                    v = -PRICE * self.price(*c)
                    for f in forms:
                        v += max(marks[f][c][0] if c in marks[f] else 0, own[f][0][0] if f in own else 0)
                    if v > best[0] + 1e-9:
                        best = (v, c)
            score += best[0]
            for f in forms:
                c = own[f][1] if f in own else None
                if best[1] in marks[f] and (f not in own or marks[f][best[1]][0] >= own[f][0][0]):
                    c = best[1]
                if c:
                    ask[f] = (c[0], bool(c[1]), marks[f][c][2])
                    right += marks[f][c][1]
        for moves in {(f[0],) + size(f) + (m,) for f, m in rec.items() if m}:
            score -= PRICE * sum(1 + (-1 in m[2]) for m in moves[-1])
        self.cache[key] = ((round(score, 6), right), ask)
        return self.cache[key]

    # ------------------------------------------------------------ the search
    def top(self, rec, but=()):
        """how many slots of every sort the moves use: a new slot is taken only next to the used ones"""
        top = {sort: 0 for sort in K}
        for f, moves in rec.items():
            if f not in but:
                for name, k, _, _ in moves:
                    top[MOVES[name][1]] = max(top[MOVES[name][1]], k + 1)
        return top

    def climb(self, sorts, until, width=4, start=None):
        records = sorted(self.rec, key=lambda f: -len(self.forms[f]))
        groups = {}
        for f in records:                                        # one operation mostly means one thing, whatever the kinds of its things
            groups.setdefault((f[0],) + size(f), []).append(f)
        whole = [tuple(g) for g in groups.values()]
        units = whole + [(f,) for g in groups.values() if len(g) > 1 for f in g]
        trace = os.environ.get("TRACE")
        beam = [(self.fit(start or {f: () for f in self.rec})[0], 0, start or {f: () for f in self.rec})]
        while self.work < until and not start:                 # several guesses side by side: the first step is often a near tie
            pool, seen = [], {tuple(sorted(r.items())) for _, _, r in beam}
            for sc, _, rec in beam:
                n = 0
                for u in whole:
                    for m in self.moves(*size(u[0]), self.top(rec, u), sorts) + [None]:
                        n += 1
                        trial = {**rec, **{f: (m,) if m else () for f in u}}
                        key = tuple(sorted(trial.items()))
                        if key in seen or self.work > until:
                            continue
                        got = self.fit(trial)[0]
                        if got > sc:
                            seen.add(key)
                            pool.append((got, -n, trial))
            if not pool:
                break
            new = sorted(beam + pool, key=lambda x: x[:2], reverse=True)[:width]
            if [x[0] for x in new] == [x[0] for x in beam]:
                break
            beam = new
            if trace:
                print("  beam", [x[0] for x in beam], flush=True)
        best, _, rec = beam[0]
        while self.work < until:
            moved = False
            for u in units:                                      # one operation at a time
                found = simpler = None
                now = {rec[f] for f in u}
                for m in self.moves(*size(u[0]), self.top(rec, u), sorts) + [None]:
                    trial = {**rec, **{f: (m,) if m else () for f in u}}
                    if now == {(m,) if m else ()}:
                        now = None                               # what comes after the present guess is not simpler than it
                        continue
                    got = self.fit(trial)[0]
                    if got > best:
                        best, found = got, trial
                    elif got == best and now and len(now) == 1 and not found and not simpler:
                        simpler = trial
                    if self.work > until:
                        break
                if found:
                    rec, moved = found, True
                elif simpler:
                    rec = simpler
                    if trace:
                        print("  +", u[0][0], len(u), rec[u[0]], best, flush=True)
            if moved:
                continue
            idle = [u for u in units if not any(rec[f] for f in u)]
            weak = [u for u in units if u not in idle and best[0] - self.fit({**rec, **{f: () for f in u}})[0][0] < 2]
            for v in idle + weak:                                # two at a time, where one alone changes nothing (or little)
                for u in units:
                    if v == u or set(u) & set(v) or len({rec[f] for f in u}) > 1 or len(rec[u[0]]) > 1 or self.work > until:
                        continue
                    had, found = () if u in weak else rec[u[0]], None
                    for a in self.moves(*size(u[0]), self.top(rec, v), sorts):
                        if had and (MOVES[a[0]][1], a[1]) == (MOVES[had[0][0]][1], had[0][1]):
                            continue
                        mid = {**rec, **{f: had + (a,) for f in u}}
                        for b in self.moves(*size(v[0]), self.top(mid, v), sorts):
                            if MOVES[a[0]][1] != MOVES[b[0]][1] or (a[1] != b[1] and "qsame" not in (a[0], b[0])):
                                continue
                            trial = {**mid, **{f: (b,) for f in v}}
                            got = self.fit(trial)[0]
                            if got > best:
                                best, found = got, trial
                        if self.work > until:
                            break
                    if found:
                        rec, moved = found, True
                        if trace:
                            print("  ++", u[0][0], rec[u[0]], v[0][0], rec[v[0]], best, flush=True)
                        break
                if moved:
                    break
            if moved:
                continue
            for u in units:                                      # a second move of a row
                if all(len(rec[f]) == 1 for f in u) and len({rec[f] for f in u}) == 1 and self.work < until:
                    first, found = rec[u[0]][0], None
                    for m in self.moves(*size(u[0]), self.top(rec), sorts):
                        if (MOVES[m[0]][1], m[1]) == (MOVES[first[0]][1], first[1]):
                            continue
                        trial = {**rec, **{f: (first, m) for f in u}}
                        got = self.fit(trial)[0]
                        if got > best:
                            best, found = got, trial
                    if found:
                        rec, moved = found, True
            if not moved:
                break
        return best, rec

    def learn(self, limit=WORK, seed=None, wide=True):
        """limit: how much work may be done, in episodes played (not in seconds: the same on any machine);
        seed: what was learnt from fewer episodes, to go on from; wide: look for another reading from nothing as well"""
        t0 = self.work
        total = sum(len(g) for f, g in self.forms.items() if f[1])
        best = None
        if seed:
            start = {f: self.near(seed, f) or () for f in self.rec}
            best = self.climb("FQEL", t0 + limit / (4 if wide else 1), start=start)
        views = ("FQEL", "QEL", "FQL", "QL") if wide or not seed else ()     # the world as it comes; all in links; all in values; all in amounts
        for part, sorts in enumerate(views):
            if best and best[0][1] == total and all(best[1].values()):
                break
            got, rec = self.climb(sorts, t0 + limit * (part + 1 + bool(seed)) / (len(views) + 1 + bool(seed)))
            if best is None or got > best[0]:
                best = (got, rec)
        if views and not (best[0][1] == total and all(best[1].values())):
            best = self.climb("FQEL", t0 + limit, start=best[1])      # the finish: from the best view, with everything open
        self.rec = best[1]
        self.ask = self.fit(self.rec)[1]
        self.usual = {f: Counter(g).most_common(1)[0][0] for f, g in self.forms.items() if f[1]}
        return best[0][1], total

    # ------------------------------------------------------------ use
    def near(self, table, form):
        """a row of a form not met in learning is read as the same operation with as many things"""
        if form in table:
            return table[form]
        for f, v in table.items():
            if f[:2] == form[:2] and size(f) == size(form):
                return v
        for f, v in table.items():
            if f[:2] == form[:2] and v and form[1] and v[0][0] in ("most", "least", "over", "under", "plus"):
                return v
        return None

    def run(self, episodes):
        """-> right answers, all answers (self.gaps: how many rows were of the kinds that are not read)"""
        right = total = self.gaps = 0
        for ep in episodes:
            s = State()
            for s.now, (op, quest, things, numbers, kinds, gold) in enumerate(ep):
                form = (op, quest, kinds)
                t = things + (ME,)
                if quest:
                    total += 1
                    got = self.near(self.ask, form)
                    if got is None:                              # not understood: the usual answer
                        right += self.usual.get(form) == gold
                        self.gaps += 1
                        continue
                    p, flip, default = got
                    try:
                        out = answer(s, p[:-1] + (dict(p[-1]),) if p[0] in SPACE else p, t, numbers, {})
                    except (KeyError, IndexError, ValueError, TypeError):
                        out = None
                    if out.__class__ is bool:
                        out = "yes" if out != flip else "no"
                    right += (default if out is None else out) == gold
                else:
                    moves = self.near(self.rec, form) or ()
                    self.gaps += not moves
                    for name, k, pick, n in moves:
                        try:
                            MOVES[name][0](s, k, tuple(t[i] for i in pick), (numbers[n],) if n is not None else ())
                        except (KeyError, IndexError, ValueError, TypeError):
                            pass
        return right, total

    def describe(self):
        def tell(x):
            if x.__class__ is tuple and x and x[0].__class__ is tuple and len(x[0]) == 3 and x[0][0] in ("F", "E", "W"):
                return "[" + " ".join(f"{op}{sort}{k}" for sort, k, op in x) + "]"
            if x.__class__ is tuple and x and x[0] in ("q", "count", "sum"):
                return "(" + " ".join(tell(y) for y in x) + ")"
            if x.__class__ is tuple:
                return "{" + ",".join(a + "/" + b for a, b in x if a < b) + "}"
            return str(x)
        out = []
        for f in sorted(self.forms, key=lambda f: (f[1], f[0], f[2])):
            if f[1]:
                p = self.ask.get(f)
                what = "-" if p is None else " ".join(tell(x) for x in p[0]) + (" (turned over)" if p[1] else "") + (f" (else {p[2]})" if p[2] else "")
            else:
                what = "; ".join(f"{name} {MOVES[name][1]}{k} of {pick}" for name, k, pick, n in self.rec[f]) or "-"
            out.append(f"  {'?' if f[1] else ' '} {f[0]} {' '.join(f[2]):12} {what}")
        return "\n".join(out)


def study(path, stories=20, more=(), limit=WORK):
    """learn from the first episodes; check what is learnt on the next ones; where the check fails, learn from them too
    -> the learner, how many episodes it learnt from, what the checks said"""
    data = read(path, max((stories,) + tuple(more)))
    steps = (stories,) + tuple(n for n in more if n > stories)
    m = Learner(data[:stories])
    m.learn(limit)
    used, notes = stories, []
    for part, n in enumerate(steps[1:]):
        right, total = m.run(data[used:n])
        notes.append(f"{right}/{total} on the next {n - used}" + (f", {m.gaps} rows not read" if m.gaps else ""))
        if right == total and (not m.gaps or part):              # rows that stay unread with more episodes mean nothing for the answers
            break
        new = Learner(data[:n])
        new.learn(limit, seed=m.rec, wide=10 * right < 9 * total)
        m, used = new, n
    return m, used, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--stories", type=int, default=20)
    ap.add_argument("--more", default="", help="take more episodes where what is learnt fails on fresh ones: e.g. 60,200")
    ap.add_argument("--limit", type=float, default=WORK, help="how much work may be done for one step of learning, in episodes played")
    ap.add_argument("--also", default="", help="more folders with test files of the same tasks (bigger episodes), comma-separated")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    files = glob.glob(os.path.join(a.data, "qa*_train.txt"))
    tasks = [int(x) for x in a.tasks.split(",") if x] or sorted(int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1)) for f in files)
    right = total = 0
    for k in tasks:
        t0 = time.time()
        m, used, notes = study(glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0], a.stories, [int(x) for x in a.more.split(",") if x], a.limit)
        got, n = m.run(m.episodes)
        print(f"task {k:2d}: learnt from {used} episodes" + "".join(f" ({x})" for x in notes) + f", {got}/{n} of their questions right, "
              f"{sum(bool(p) for p in m.rec.values())} of {len(m.rec)} kinds of records and {len(m.ask)} of {len(m.forms) - len(m.rec)} kinds of queries read "
              f"({time.time() - t0:.0f} s)", flush=True)
        if a.show:
            print(m.describe(), flush=True)
        for folder in [a.data] + [x for x in a.also.split(",") if x]:
            r, t = m.run(read(glob.glob(os.path.join(folder, f"qa{k}_*test.txt"))[0]))
            if folder == a.data:
                right, total = right + r, total + t
                print(f"test {k:2d}: {r}/{t} = {100 * r / t:.1f}%", flush=True)
            else:
                print(f"test {k:2d} in {os.path.basename(folder.rstrip('/'))}: {r}/{t} = {100 * r / t:.1f}%", flush=True)
    if total:
        print(f"all: {right}/{total} = {100 * right / total:.2f}%", flush=True)


if __name__ == "__main__":
    main()
