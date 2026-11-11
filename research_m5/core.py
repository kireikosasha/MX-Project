#!/usr/bin/env python3
"""
The ground: what is there before anything is learnt. No word of any language, and no notion of any subject.

It is a small machine of general schemes of reasoning over nameless attributes and relations:

    an attribute     a thing has one value of an attribute at a time; the value stays until it is changed;
                     it may be unknown, or one of two
    attachment       a thing attached to another has the attribute values of its carrier, and keeps them when let go
    passing          a thing passes from one to another; who gave what to whom is remembered
    a kind           a thing is of a kind; what is told of the kind holds for it; if nothing is told, it is
                     like the others of its kind
    an order         a chain of "more than" (transitive)
    a map            links with a direction; directions add up as steps on a plane; a path is found
    time             everything is remembered in the order it happened, or in the order of named times

Attributes, orders and maps are numbered, not named: "where somebody is", "what colour something is" and "what
somebody feels" are the same scheme with different numbers. Which sentence changes what, which question asks
what, and all the words (pronouns, directions, times of day, forms of one word, how yes and no and numbers
are said) are learnt by acquire.py from stories with answers, by trying and checking.

    tell(kind, k, roles)          a statement changes the state
    ask(kind, k, roles) -> answer  ("e", thing) | ("s", symbol: yes no maybe nothing n0 n1 ..) | ("l", [things]) | ("p", [steps])
"""
from collections import deque

ROLES = ["O", "A1", "A2", "A3", "D", "T"]
# kind -> (the roles it needs, the roles it may have, what its number counts: "a" attribute, "o" order, "m" map, "" nothing)
STATEMENTS = {
    "SET": (("A1", "A2"), ("T",), "a"),          # the attribute of A1 becomes A2
    "UNSET": (("A1", "A2"), (), "a"),            # the attribute of A1 is not A2 any more
    "ANY": (("A1", "A2", "A3"), (), "a"),        # the attribute of A1 is A2 or A3
    "ATTACH": (("A1", "A2"), (), ""),            # A2 is attached to A1
    "DETACH": (("A1", "A2"), (), ""),            # A2 is let go by A1
    "PASS": (("A1", "A2", "A3"), (), ""),        # A2 passes from A1 to A3
    "ISA": (("A1", "A2"), (), ""),               # A1 is of the kind A2
    "ORDER": (("A1", "A2"), (), "o"),            # A1 is more than A2
    "LINK": (("A1", "A2", "D"), (), "m"),        # A1 is in the direction D from A2
}
QUERIES = {
    "VAL": (("A1",), ("A2",), "a"),              # the attribute of A1
    "BEFORE": (("A1", "A2"), (), "a"),           # its value before it became A2
    "IS": (("A1", "A2"), (), "a"),               # is the attribute of A1 equal to A2
    "COUNT": (("A1",), (), ""),                  # how many things are attached to A1
    "LIST": (("A1",), (), ""),                   # which
    "P_WHAT": (("A1", "A3"), (), ""),            # what passed from A1 to A3
    "P_FROM": (("A2",), ("A3",), ""),            # from whom A2 passed
    "P_TO": (("A2",), ("A1",), ""),              # to whom
    "NEXT": (("A1",), (), "aa"),                 # what, in another attribute, follows the value A1 has in this one
    "MORE": (("A1", "A2"), (), "o"),             # is A1 more than A2
    "L_A1": (("A2", "D"), (), "m"),              # what is in the direction D from A2
    "L_A2": (("A1", "D"), (), "m"),              # from what is A1 in the direction D
    "L_IS": (("A1", "A2", "D"), (), "m"),        # is A1 in the direction D from A2
    "PATH": (("A1", "A2"), (), "m"),             # the steps from A1 to A2
}
KINDS = {**STATEMENTS, **QUERIES}
# a question can only ask about what statements tell: which questions read what which statements write
READS = {"SET": {"VAL", "BEFORE", "IS", "NEXT"}, "UNSET": {"VAL", "IS"}, "ANY": {"VAL", "IS"},
         "ATTACH": {"VAL", "BEFORE", "COUNT", "LIST"}, "DETACH": {"VAL", "BEFORE", "COUNT", "LIST"},
         "PASS": {"VAL", "BEFORE", "COUNT", "LIST", "P_WHAT", "P_FROM", "P_TO"}, "ISA": {"VAL"},
         "ORDER": {"MORE"}, "LINK": {"L_A1", "L_A2", "L_IS", "PATH"}}
N_ATTR, N_ORDER, N_MAP = 5, 2, 2

INV = {"n": "s", "s": "n", "e": "w", "w": "e"}
STEP = {"n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0)}
CODES = "nesw"


class Lexicon:
    """what is learnt about single words (the sentence forms are in acquire.py)"""

    def __init__(self):
        self.pron = set()       # words that stand for whoever was spoken of last
        self.dirs = {}          # word -> n / e / s / w
        self.times = {}         # word -> its place in the order of the day
        self.alias = {}         # a form of a word -> the word
        self.kb = {}            # (attribute, attribute, value) -> what was seen to follow
        self.like = "kind"      # like the others of its kind: "kind" the one introduced last, "told" the one described last, "most" most of them
        self.gave = "last"      # what passed from A1 to A3: the last time or the first

    def copy(self):
        c = Lexicon()
        c.pron, c.dirs, c.times, c.alias, c.kb = set(self.pron), dict(self.dirs), dict(self.times), dict(self.alias), dict(self.kb)
        c.like, c.gave = self.like, self.gave
        return c

    def direction(self, w):
        """a word not learnt yet is some direction, always the same one"""
        return self.dirs.get(w) or CODES[sum(map(ord, w)) % 4]

    def time(self, w):
        return self.times.get(w, 100 + sum(map(ord, w)) % 50)


class World:
    def __init__(self, lex):
        self.lex = lex
        self.n = 0
        self.val = [{} for _ in range(N_ATTR)]       # attribute -> thing -> value (None: known not to be what it was)
        self.maybe = [{} for _ in range(N_ATTR)]     # thing -> the values it may have
        self.hist = [{} for _ in range(N_ATTR)]      # thing -> [(named time, order, value)]
        self.rest = [{} for _ in range(N_ATTR)]      # thing -> the value it was left with when let go
        self.track = [{} for _ in range(N_ATTR)]     # an attached thing -> the values it went through
        self.carrier, self.carried, self.was_held = {}, {}, set()
        self.passes = []                             # (from, what, to)
        self.kind, self.kind_at = {}, {}
        self.order = [set() for _ in range(N_ORDER)]
        self.links = [[] for _ in range(N_MAP)]      # (a, direction, b): a is in that direction from b
        self.last = []

    def name(self, w):
        return self.lex.alias.get(w, w)

    def who(self, names):
        if any(x in self.lex.pron for x in names):
            return list(self.last)
        self.last = list(names)
        return list(names)

    def _went(self, k, thing, value):
        if value is not None:
            t = self.track[k].setdefault(thing, [])
            if not t or t[-1] != value:
                t.append(value)

    def _take(self, who, thing):
        self.carrier[thing] = who
        self.was_held.add(thing)
        if thing not in self.carried.setdefault(who, []):
            self.carried[who].append(thing)
        for k in range(N_ATTR):
            self._went(k, thing, self.val[k].get(who))

    def _drop(self, who, thing):
        self.carrier[thing] = None
        for k in range(N_ATTR):
            self.rest[k][thing] = self.val[k].get(who)
        if thing in self.carried.get(who, []):
            self.carried[who].remove(thing)

    # ---- statements
    def tell(self, kind, k, a):
        self.n += 1
        g = lambda r: [self.name(x) for x in a.get(r, [])]              # noqa: E731
        if kind == "SET":
            rank = self.lex.time(g("T")[0]) if g("T") else None
            for p in self.who(g("A1")):
                self.val[k][p] = g("A2")[0]
                self.maybe[k].pop(p, None)
                self.hist[k].setdefault(p, []).append((rank, self.n, g("A2")[0]))
                for o in self.carried.get(p, []):
                    self._went(k, o, g("A2")[0])
        elif kind == "UNSET":
            for p in self.who(g("A1")):
                self.val[k][p] = None
                self.maybe[k][p] = set()
        elif kind == "ANY":
            for p in self.who(g("A1")):
                self.val[k][p] = None
                self.maybe[k][p] = {g("A2")[0], g("A3")[0]}
        elif kind == "ATTACH":
            for p in self.who(g("A1")):
                for o in g("A2"):
                    self._take(p, o)
        elif kind == "DETACH":
            for p in self.who(g("A1")):
                for o in g("A2"):
                    self._drop(p, o)
        elif kind == "PASS":
            p, q = self.who(g("A1"))[0], g("A3")[0]
            for o in g("A2"):
                if o in self.carried.get(p, []):
                    self.carried[p].remove(o)
                self._take(q, o)
                self.passes.append((p, o, q))
        elif kind == "ISA":
            self.kind[g("A1")[0]] = g("A2")[0]
            self.kind_at[g("A1")[0]] = self.n
        elif kind == "ORDER":
            self.order[k].add((g("A1")[0], g("A2")[0]))
        elif kind == "LINK":
            self.links[k].append((g("A1")[0], self.lex.direction(g("D")[0]), g("A2")[0]))
        else:
            raise ValueError(kind)

    # ---- questions
    def value(self, k, x):
        """the attribute of a thing: by its carrier, its own, by its kind, by the others of its kind"""
        if x in self.was_held:
            h = self.carrier.get(x)
            return self.val[k].get(h) if h else self.rest[k].get(x)
        v = self.val[k].get(x)
        if v is not None:
            return v
        kd = self.kind.get(x)
        if kd is None:
            return None
        if self.val[k].get(kd) is not None:
            return self.val[k][kd]
        same = [(self.kind_at[o], self.hist[k][o][-1][1], self.val[k][o]) for o, kk in self.kind.items()
                if kk == kd and o != x and self.val[k].get(o) is not None]
        if not same:
            return None
        if self.lex.like == "kind":
            return max(same)[2]
        if self.lex.like == "told":
            return max(same, key=lambda s: s[1])[2]
        vals = [s[2] for s in same]
        return max(vals, key=lambda v: (vals.count(v), -vals.index(v)))

    def _closure(self, k):
        big, grew = set(self.order[k]), True
        while grew:
            grew = False
            for a, b in list(big):
                for c, d in list(big):
                    if b == c and (a, d) not in big:
                        big.add((a, d))
                        grew = True
        return big

    def _edges(self, k):
        e = {}
        for a, d, b in self.links[k]:
            e.setdefault(b, []).append((d, a))
            e.setdefault(a, []).append((INV[d], b))
        return e

    def ask(self, kind, k, a):
        g = lambda r: [self.name(x) for x in a.get(r, [])]              # noqa: E731
        x = g("A1")[0] if g("A1") else None
        e = lambda v: None if v is None else ("e", v)                    # noqa: E731
        yes = lambda b: ("s", "yes" if b else "no")                      # noqa: E731
        if kind == "VAL":
            return e(self.value(k, x))
        if kind == "BEFORE":
            if x in self.track[k]:
                seq = self.track[k][x]
            else:
                mv = self.hist[k].get(x, [])
                timed = all(m[0] is not None for m in mv)
                seq = [m[2] for m in (sorted(mv, key=lambda m: (m[0], m[1])) if timed else mv)]
            for i in range(len(seq) - 1, 0, -1):
                if seq[i] == g("A2")[0]:
                    return e(seq[i - 1])
            return None
        if kind == "IS":
            if self.val[k].get(x) is not None:
                return yes(self.val[k][x] == g("A2")[0])
            return ("s", "maybe") if g("A2")[0] in self.maybe[k].get(x, ()) else ("s", "no")
        if kind == "COUNT":
            return ("s", f"n{len(self.carried.get(x, []))}")
        if kind == "LIST":
            have = list(self.carried.get(x, []))
            return ("l", have) if have else ("s", "nothing")
        if kind in ("P_WHAT", "P_FROM", "P_TO"):
            want, out = {"A1": 0, "A2": 1, "A3": 2}, {"P_WHAT": 1, "P_FROM": 0, "P_TO": 2}[kind]
            for ev in (reversed(self.passes) if self.lex.gave == "last" or kind != "P_WHAT" else self.passes):
                if all(ev[i] == g(r)[0] for r, i in want.items() if g(r) and i != out):
                    return e(ev[out])
            return None
        if kind == "NEXT":
            k1, k2 = k
            feel = self.val[k1].get(x)
            if feel is None:
                return None
            for o, vals in self.hist[k1].items():
                for _, n, v in vals:
                    later = [m for m in self.hist[k2].get(o, []) if m[1] > n]
                    if o != x and v == feel and later:
                        return e(later[0][2])
            return e(self.lex.kb.get((k1, k2, feel)))
        if kind == "MORE":
            return yes((x, g("A2")[0]) in self._closure(k))
        if kind in ("L_A1", "L_A2"):
            d = self.lex.direction(g("D")[0])
            for p, dd, q in self.links[k]:
                for s, ee, o in ((p, dd, q), (q, INV[dd], p)):
                    if ee == d and kind == "L_A1" and o == g("A2")[0]:
                        return e(s)
                    if ee == d and kind == "L_A2" and s == x:
                        return e(o)
            return None
        ed = self._edges(k)
        if kind == "L_IS":
            pos, todo = {x: (0, 0)}, deque([x])
            while todo:
                cur = todo.popleft()
                for d, nxt in ed.get(cur, []):
                    if nxt not in pos:
                        pos[nxt] = (pos[cur][0] + STEP[d][0], pos[cur][1] + STEP[d][1])
                        todo.append(nxt)
            y, d = g("A2")[0], self.lex.direction(g("D")[0])
            if y not in pos:
                return ("s", "no")
            dx, dy = pos[x][0] - pos[y][0], pos[x][1] - pos[y][1]
            return yes(dx * STEP[d][0] > 0 or dy * STEP[d][1] > 0)
        if kind == "PATH":
            goal, seen, todo = g("A2")[0], {x}, deque([(x, [])])
            while todo:
                cur, path = todo.popleft()
                if cur == goal:
                    return ("p", path)
                for d, nxt in ed.get(cur, []):
                    if nxt not in seen:
                        seen.add(nxt)
                        todo.append((nxt, path + [d]))
            return None
        raise ValueError(kind)
