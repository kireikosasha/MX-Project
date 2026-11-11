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

ROLES = ["O", "A1", "A2", "A3", "A4", "D", "T", "N"]
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
    "RISE": (("A1",), ("A2",), "a"),             # A1 leaves what it is in: its attribute becomes that of its value
    "SWAP": (("A1", "A3"), (), "a"),             # A1 and A3 exchange the attribute and what is attached to them
    "TRADE": (("A1", "A2", "A3", "A4"), (), ""), # A2 passes from A1 to A3, and A4 from A3 to A1
    "HAS": (("A1", "A2"), (), "h"),              # A1 is one of the A2 (a set it belongs to, a property it has)
    "EVERY": (("A1", "A2"), (), "h"),            # every A1 is one of the A2
    "NEVER": (("A1", "A2"), (), "h"),            # no A1 is one of the A2
    "QSET": (("A1", "N"), ("A2",), ""),          # A1 has N (of A2)
    "QADD": (("A1", "N"), ("A2",), ""),          # N more
    "QSUB": (("A1", "N"), ("A2",), ""),          # N less
    "QPASS": (("A1", "A3", "N"), ("A2",), ""),   # N pass from A1 to A3
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
    "TOP": (("A1",), (), "a"),                   # the attribute of the attribute ... of A1, to the end
    "V2": (("A1",), (), "a"),                    # the attribute of the attribute of A1
    "V2_IS": (("A1", "A2"), (), "a"),            # is it A2
    "IS_T": (("A1", "A2"), (), "a"),             # is A2 somewhere on that chain
    "INV": (("A2",), (), "a"),                   # the things whose attribute is A2
    "INV_N": (("A2",), (), "a"),                 # how many
    "SAME": (("A1", "A2"), (), "a"),             # have A1 and A2 the same attribute
    "WHO": (("A1",), (), "a"),                   # what stood by when the attribute of A1 was last set
    "HOLDER": (("A2",), (), ""),                 # to what A2 is attached
    "HELD": (("A1", "A2"), (), ""),              # is A2 attached to A1
    "H_IS": (("A1", "A2"), (), "h"),             # is A1 one of the A2
    "H_LIST": (("A1",), (), "h"),                # the sets of this number A1 belongs to
    "H_INV": (("A2",), (), "h"),                 # the single things that are among the A2
    "O_MAX": (("A1",), (), "o"),                 # of the things named, the one that is more than the others
    "O_MIN": (("A1",), (), "o"),                 # less than the others
    "QVAL": (("A1",), ("A2",), ""),              # how many (of A2) A1 has
    "QIS": (("A1", "N"), ("A2",), ""),           # has it N
    "QMORE": (("A1", "N"), ("A2",), ""),         # has it more than N
    "QLESS": (("A1", "N"), ("A2",), ""),         # less than N
    "QMAX": (("A2",), (), ""),                   # who has most of A2
    "L_LIST": (("A2", "D"), (), "m"),            # everything in the direction D from A2
    "O_TOP": ((), (), "o"),                      # the one that is more than all the others (less, if the word turns it)
    "L_END": (("D",), (), "m"),                  # the one that lies in the direction D from all the others
}
KINDS = {**STATEMENTS, **QUERIES}
# a question can only ask about what statements tell: which questions read what which statements write
_A = {"VAL", "BEFORE", "IS", "NEXT", "TOP", "V2", "V2_IS", "IS_T", "INV", "INV_N", "SAME", "WHO"}
_H = {"VAL", "BEFORE", "COUNT", "LIST", "HOLDER", "HELD"}
_Q = {"QVAL", "QIS", "QMAX", "QMORE", "QLESS"}
READS = {"SET": _A, "UNSET": {"VAL", "IS", "SAME", "INV", "INV_N"}, "ANY": {"VAL", "IS"}, "RISE": _A, "SWAP": _A | _H,
         "ATTACH": _H, "DETACH": _H, "PASS": _H | {"P_WHAT", "P_FROM", "P_TO"}, "TRADE": _H | {"P_WHAT", "P_FROM", "P_TO"},
         "ISA": {"VAL"}, "ORDER": {"MORE", "O_MAX", "O_MIN", "O_TOP"}, "LINK": {"L_A1", "L_A2", "L_IS", "PATH", "L_LIST", "L_END"},
         "HAS": {"H_IS", "H_LIST", "H_INV"}, "EVERY": {"H_IS", "H_LIST", "H_INV"}, "NEVER": {"H_IS", "H_LIST", "H_INV"},
         "QSET": _Q, "QADD": _Q, "QSUB": _Q, "QPASS": _Q, "NOP": set()}
# what a question of each kind answers with - t a thing, l several things, s a sign (yes, no, a step), n a number:
# (what it is asked for, what else it may answer: "nothing" where a thing was asked, a number said as a word)
GIVES = {"VAL": ("t", "s"), "BEFORE": ("t", ""), "IS": ("s", ""), "COUNT": ("ns", ""), "LIST": ("tl", "s"), "P_WHAT": ("t", ""),
         "P_FROM": ("t", ""), "P_TO": ("t", ""), "NEXT": ("t", ""), "MORE": ("s", ""), "L_A1": ("t", ""), "L_A2": ("t", ""),
         "L_IS": ("s", ""), "PATH": ("sl", ""), "TOP": ("t", ""), "V2": ("t", ""), "V2_IS": ("s", ""), "IS_T": ("s", ""),
         "INV": ("tl", "s"), "INV_N": ("ns", ""), "SAME": ("s", ""), "WHO": ("t", ""), "HOLDER": ("t", "s"), "HELD": ("s", ""),
         "H_IS": ("s", ""), "H_LIST": ("tl", "s"), "H_INV": ("tl", "s"), "O_MAX": ("t", ""), "O_MIN": ("t", ""),
         "QVAL": ("ns", ""), "QIS": ("s", ""), "QMORE": ("s", ""), "QLESS": ("s", ""), "QMAX": ("t", ""), "L_LIST": ("tl", "s"), "O_TOP": ("t", ""), "L_END": ("t", "")}
# of two readings that answer equally well the plainer one is taken: the first schemes are plain, the rest are built on them
PLAIN = ("SET", "UNSET", "ANY", "ATTACH", "DETACH", "PASS", "ISA", "ORDER", "LINK", "HAS", "EVERY", "NEVER", "QSET", "QADD", "QSUB", "QPASS",
         "VAL", "BEFORE", "IS", "COUNT", "LIST", "P_WHAT", "P_FROM", "P_TO", "NEXT", "MORE", "L_A1", "L_A2", "L_IS", "PATH",
         "H_IS", "H_LIST", "H_INV", "QVAL", "QIS", "QMORE", "QLESS", "HOLDER", "HELD")
COST = {k: 0 if k in PLAIN else 2 if k == "WHO" else 1 for k in KINDS}
# the first schemes are tried first; the others only when a task is not learnt with them
FIRST = ("SET", "UNSET", "ANY", "ATTACH", "DETACH", "PASS", "ISA", "ORDER", "LINK", "VAL", "BEFORE", "IS", "COUNT", "LIST",
         "P_WHAT", "P_FROM", "P_TO", "NEXT", "MORE", "L_A1", "L_A2", "L_IS", "PATH")
TIER = {k: 0 if k in FIRST else 1 for k in KINDS}
N_ATTR, N_ORDER, N_MAP, N_SET = 5, 2, 2, 2

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
        self.flip = set()       # words that turn a relation round ("shorter" where "taller" is the plain way)
        self.skip = set()       # words that are not part of a name of several words ("a" in "a cow", "every" in "every cow")
        self.map = "grid"       # directions: "grid" every step is one square of a grid, "axis" only more or less along two axes

    def copy(self):
        c = Lexicon()
        c.pron, c.dirs, c.times, c.alias, c.kb = set(self.pron), dict(self.dirs), dict(self.times), dict(self.alias), dict(self.kb)
        c.like, c.gave, c.map, c.flip, c.skip = self.like, self.gave, self.map, set(self.flip), set(self.skip)
        return c

    def norm(self, w):
        """a name with its words in their main form ("This morning" - "this morning")"""
        if w in self.alias:
            return self.alias[w]
        if " " in w:
            parts = [x for x in w.split() if x not in self.skip] or w.split()
            return " ".join(self.alias.get(x, x) for x in parts)
        return w

    def direction(self, w):
        """a word not learnt yet is some direction, always the same one"""
        w = self.norm(w)
        return self.dirs.get(w) or CODES[sum(map(ord, w)) % 4]

    def time(self, w):
        w = self.norm(w)
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
        self.by = [{} for _ in range(N_ATTR)]        # thing -> what stood by when its attribute was last set
        self.sup, self.single = {}, []               # a thing or a set -> {(a set it is in, number)}; the single things
        self.never = set()                           # (a set, a set): none of the first is of the second
        self.qty = {}                                # (thing, of what) -> how many
        self.last = []

    def name(self, w):
        return self.lex.norm(w)

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
    def turned(self, a):
        """a word that turns the relation round stands in the sentence: A1 and A2 change places"""
        if self.lex.flip and any(o in self.lex.flip for o in a.get("O", ())):
            a = dict(a)
            a["A1"], a["A2"] = a.get("A2", []), a.get("A1", [])
        return a

    def tell(self, kind, k, a):
        self.n += 1
        if kind == "NOP":                                        # a word of the sentence says that it tells nothing here
            return
        a = self.turned(a)
        g = lambda r: [self.name(x) for x in a.get(r, [])]              # noqa: E731
        if kind == "SET":
            rank = self.lex.time(g("T")[0]) if g("T") else None
            for p in self.who(g("A1")):
                self.val[k][p] = g("A2")[0]
                self.by[k][p] = g("O")[0] if g("O") else None
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
        elif kind == "RISE":
            for p in self.who(g("A1")):
                self.val[k][p] = self.val[k].get(self.val[k].get(p))
        elif kind == "SWAP":
            p, q = g("A1")[0], g("A3")[0]
            self.val[k][p], self.val[k][q] = self.val[k].get(q), self.val[k].get(p)
            mine, yours = list(self.carried.get(p, [])), list(self.carried.get(q, []))
            self.carried[p], self.carried[q] = [], []
            for o in mine:
                self._take(q, o)
            for o in yours:
                self._take(p, o)
        elif kind == "TRADE":
            p, q = self.who(g("A1"))[0], g("A3")[0]
            for o, frm, to in ((g("A2")[0], p, q), (g("A4")[0], q, p)):
                if o in self.carried.get(frm, []):
                    self.carried[frm].remove(o)
                self._take(to, o)
                self.passes.append((frm, o, to))
        elif kind == "NEVER":
            self.never.add((g("A1")[0], g("A2")[0]))
        elif kind in ("HAS", "EVERY"):
            self.sup.setdefault(g("A1")[0], set()).add((g("A2")[0], k))
            if kind == "HAS" and g("A1")[0] not in self.single:
                self.single.append(g("A1")[0])
        elif kind in ("QSET", "QADD", "QSUB", "QPASS"):
            n, of = int(g("N")[0]), (g("A2")[0] if g("A2") else "")
            for p in self.who(g("A1")):
                have = self.qty.get((p, of), 0)
                self.qty[(p, of)] = n if kind == "QSET" else have + n if kind == "QADD" else have - n
            if kind == "QPASS":
                self.qty[(g("A3")[0], of)] = self.qty.get((g("A3")[0], of), 0) + n
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

    def _chain(self, k, x):
        """the attribute of x, the attribute of that, and so on"""
        out, seen = [], {x}
        while True:
            x = self.value(k, x)
            if x is None or x in seen:
                return out
            out.append(x)
            seen.add(x)

    def _sets(self, x):
        """every set x is in, by the rules told: {(set, number)}"""
        out, todo = set(), [x]
        while todo:
            for y, k in self.sup.get(todo.pop(), ()):
                if not any(y == z for z, _ in out):
                    todo.append(y)
                out.add((y, k))
        return out

    def _towards(self, k, x, d, y):
        """is x in the direction d from y"""
        if self.lex.map == "axis":                               # only more or less along the axis of d
            up = {(a, b) if dd in "ne" else (b, a) for a, dd, b in self.links[k] if (dd in "ns") == (d in "ns")}
            grew = True
            while grew:
                grew = False
                for a, b in list(up):
                    for c, e in list(up):
                        if b == c and (a, e) not in up:
                            up.add((a, e))
                            grew = True
            return ((x, y) if d in "ne" else (y, x)) in up
        ed, pos, todo = self._edges(k), {x: (0, 0)}, deque([x])
        while todo:
            cur = todo.popleft()
            for dd, nxt in ed.get(cur, []):
                if nxt not in pos:
                    pos[nxt] = (pos[cur][0] + STEP[dd][0], pos[cur][1] + STEP[dd][1])
                    todo.append(nxt)
        if y not in pos:
            return False
        dx, dy = pos[x][0] - pos[y][0], pos[x][1] - pos[y][1]
        return dx * STEP[d][0] > 0 or dy * STEP[d][1] > 0

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
        a = self.turned(a)
        g = lambda r: [self.name(x) for x in a.get(r, [])]              # noqa: E731
        x = g("A1")[0] if g("A1") else None
        e = lambda v: None if v is None else ("e", v)                    # noqa: E731
        yes = lambda b: ("s", "yes" if b else "no")                      # noqa: E731
        some = lambda xs: ("l", list(xs)) if xs else ("s", "nothing")   # noqa: E731
        if kind == "VAL":
            v = self.value(k, x)
            return ("s", "nothing") if v is None and x in self.maybe[k] and not self.maybe[k][x] else e(v)
        if kind in ("TOP", "V2", "V2_IS", "IS_T"):
            chain = self._chain(k, x)
            if kind == "TOP":
                return e(chain[-1]) if chain else None
            if kind == "V2":
                return e(chain[1]) if len(chain) > 1 else None
            return yes(g("A2")[0] in (chain[1:2] if kind == "V2_IS" else chain))
        if kind in ("INV", "INV_N"):
            have = [o for o, v in self.val[k].items() if v == g("A2")[0]]
            return ("s", f"n{len(have)}") if kind == "INV_N" else some(have)
        if kind == "SAME":
            va, vb = self.value(k, x), self.value(k, g("A2")[0])
            return yes(va is not None and va == vb)
        if kind == "WHO":
            return e(self.by[k].get(x))
        if kind == "HOLDER":
            return e(self.carrier.get(g("A2")[0])) or ("s", "nothing")
        if kind == "HELD":
            return yes(self.carrier.get(g("A2")[0]) == x)
        if kind in ("H_IS", "H_LIST", "H_INV"):
            def among(o, y):                                     # is o one of the y: told so, and not told that none of its sets is
                mine = {o} | {z for z, _ in self._sets(o)}
                return y in mine and y != o and not any((z, y) in self.never for z in mine)
            if kind == "H_IS":
                return yes(among(x, g("A2")[0]))
            if kind == "H_LIST":
                return some(sorted({y for y, kk in self._sets(x) if kk == k and among(x, y)}))
            return some([o for o in self.single if among(o, g("A2")[0])])
        if kind == "O_TOP":
            big = self._closure(k)
            if any(o in self.lex.flip for o in a.get("O", ())):
                big = {(q, p) for p, q in big}
            top = {p for p, _ in big} - {q for _, q in big}
            return e(next(iter(top))) if len(top) == 1 else None
        if kind == "L_END":
            d, ed0 = self.lex.direction(g("D")[0]), self._edges(k)
            ends = [o for o in ed0 if all(self._towards(k, o, d, y) for y in ed0 if y != o)]
            return e(ends[0]) if len(ends) == 1 else None
        if kind in ("O_MAX", "O_MIN"):
            big, named = self._closure(k), g("A1")
            for c in named:
                if all(((c, o) if kind == "O_MAX" else (o, c)) in big for o in named if o != c):
                    return e(c)
            return None
        if kind in ("QVAL", "QIS", "QMAX", "QMORE", "QLESS"):
            of = g("A2")[0] if g("A2") else ""
            if kind == "QVAL":
                return ("s", f"n{self.qty.get((x, of), 0)}")
            if kind == "QIS":
                return yes(self.qty.get((x, of), 0) == int(g("N")[0]))
            if kind in ("QMORE", "QLESS"):
                return yes(self.qty.get((x, of), 0) > int(g("N")[0]) if kind == "QMORE" else self.qty.get((x, of), 0) < int(g("N")[0]))
            have = sorted(((n, p) for (p, o), n in self.qty.items() if o == of), reverse=True)
            return e(have[0][1]) if have and (len(have) == 1 or have[0][0] > have[1][0]) else None
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
            return yes(self._towards(k, x, self.lex.direction(g("D")[0]), g("A2")[0]))
        if kind == "L_LIST":
            d, y = self.lex.direction(g("D")[0]), g("A2")[0]
            return some(sorted(o for o in ed if o != y and self._towards(k, o, d, y)))
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
