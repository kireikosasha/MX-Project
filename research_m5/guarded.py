"""Quantities that wait on each other, found together (stages 92-93).

parts.py builds a rule one part at a time, each part moved by plain events; two quantities each of which waits on the
other (a car leaves with a bag only when both are there, a bag is thrown away only when no car waits) cannot be built that
way: neither of the two explains anything before the other is there. Here the whole machine is looked for at once, as a
counter machine (Minsky 1967) written in guarded commands (Dijkstra 1975): a few states of control and a few whole
numbers, all at 0 at the start. A number is a count (it never goes below 0, and is tested for being 0) or a difference
(any whole number, tested for its sign). A symbol reads some of the tests and the state, and by what they say it moves
the control and adds -1, 0 or +1 to every number, or makes one whole move, the loops of Minsky's machines said at once:
pours all of one count into another, takes two counts from each other while both are above 0, or empties a count. What
is shown is read from the state and the tests (a table), and when the answers hold numbers it is words with places for
numbers ('short 3', '2 waiting', 'idle 2 waiting 0'), the words read off the table and each place filled by a number
of the machine or minus it. A count starts at 0, or, when numbers are shown before any symbol, at what they show. A
symbol reads at most two tests and the display at most three: an event's condition is kept short, as a person keeps it.

The machine is found by a solver as the cheapest that answers everything heard (core-guided MaxSAT, Ignatiev, Morgado
and Marques-Silva 2019): its price is that of the numbers and states kept in mind, of the tests read (by a symbol or by
the display), of the rows of the tables that do something, and of the moves they make. Every test a symbol reads is paid
for, so a symbol reads only what the stories show it must, and does where nothing was heard what it does where it was:
that is how the machine answers stories longer than any it heard. The stories enter the formulas lazily: the short ones
first, then, of a story a machine found answers wrongly, its beginning up to the first wrong answer. The learner asks
where machines that answer everything heard part (rivals, as parts.py asks; one story parts the machine from several
rivals one after another), and then long stories as a check.

    python guarded.py FAMILY/family.py [--asks 40] [--seed 0] [--minutes 10]
"""
import argparse, importlib.util, itertools, random, re, threading, time
from pysat.card import CardEnc, EncType
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
import parts

ENGINE = "glucose4"  # (as cells.py: a solve that can be interrupted)
COUNT, DIFF = (0, 1), (-1, 0, 1)  # what a count's test says (0, above 0), and a difference's (below 0, 0, above 0)
SHAPES = sorted([(n, z, q, w) for n in range(5) for z in range(3) for q in (1, 2) for w in (0, 1)
                 if 1 <= n + z <= 4 and q * 2 ** n * 3 ** z <= 48 and (not w or 2 <= n and n + z <= 3 and q * 2 ** n * 3 ** z <= 24)],
                key=lambda s: (3 * s[0] + 4 * s[1] + 3 * (s[2] - 1) + 2 * s[3], s))
# (counts, differences, states, whether whole moves are allowed: see Space)
READ = 2  # the price of a test read (by a symbol or by the display)
ROW = 1  # of every row of a symbol's table that does something (a row: one reading of the tests it reads)
# and 1 for every number moved or state changed in it; 3 for every count and state beyond the first, 4 for a difference


def base(shape):
    n, z, q, w = shape
    return 3 * n + 4 * z + 3 * (q - 1) + 2 * w


def wholes(n, K):
    """The whole moves among n counts (Minsky's loops, said at once): all of count i poured into count j, counts i and
    j taken from each other while both are above 0 (as many as the smaller one), count i emptied. (Pouring from or into
    a difference, or taking one count's worth from another, was tried on the open pool 92: the cheap machines it lets
    in crowd the search, and no rule there needed it alone.)"""
    return [("pour", i, j, 1) for i in range(n) for j in range(n) if i != j] + \
        [("even", i, j) for i in range(n) for j in range(i + 1, n)] + [("empty", i) for i in range(n)]


def moved(w):
    """The numbers a whole move changes."""
    return w[1:3] if w[0] in ("pour", "even") else w[1:2]


def whole(c, w):
    c = list(c)
    if w[0] == "pour":
        c[w[2]] += w[3] * c[w[1]]
        c[w[1]] = 0
    elif w[0] == "even":
        m = min(c[w[1]], c[w[2]])
        c[w[1]] -= m
        c[w[2]] -= m
    else:
        c[w[1]] = 0
    return tuple(c)


MOVE = 2  # the price of a whole move (beside the row it is in)
START = 3  # the price of a count that starts above 0
MOST = (2, 3)  # a symbol reads at most two tests, the display at most three (a person keeps an event's condition short)
SLOT = re.compile(r"-?\d+")


def form(y):
    """What is shown, as words with places for numbers, and the numbers: 'short 3' -> ('short \0', [3])."""
    return SLOT.sub("\0", y), [int(x) for x in SLOT.findall(y)]


NONE = 10 ** 6  # (the least price of a shape that has no machine at all)
ROUND = 60  # seconds a round may take once some machine answers everything heard (the shapes not looked at go on later)
SHORT = 4  # the beginnings of at most so many symbols are in the formulas from the start (the rest when answered wrongly)
PER = 15  # seconds the search of one shape is given in a round (it goes on in the next)
NEED = 14  # sure after so many long stories asked in a row answered rightly (as parts.py: SURE plain, then MORE)
NAMES = "abcd"


def sign(v):
    return (v > 0) - (v < 0)


class Machine:
    """Counts and differences and Q states; eff[a][cell] = (state to go to or None to stay, what is added to every
    number), the cell being the state and what the tests say; the display is out[cell], or number j plus off when
    num = (j, off)."""

    def __init__(self, syms, shape, eff, out, num, reads, oreads, cost):
        self.syms, self.shape = list(syms), shape
        n, z, self.Q, _ = shape
        self.K, self.kinds = n + z, [COUNT] * n + [DIFF] * z
        self.eff, self.out, self.num = eff, out, num
        self.reads, self.oreads, self.cost = reads, oreads, cost
        self.total, self.sure = cost + base(shape), False

    def cell(self, q, c):
        x = q
        for v, k in zip(c, self.kinds):
            x = x * len(k) + k.index(sign(v))
        return x

    def show(self, q, c):
        y = self.out[self.cell(q, c)]
        if self.num is None or "\0" not in y:
            return y
        parts = y.split("\0")
        vals = [str(g * c[j] + o) for j, g, o in self.num[y]]
        return "".join(x + v for x, v in zip(parts, vals + [""]))

    def start(self):
        c = getattr(self, "init", None) or (0,) * self.K
        return ((0, c), self.show(0, c))

    def next(self, state, a):
        (q, c), _ = state
        go, d, w = self.eff[a][self.cell(q, c)]
        q = q if go is None else go
        c = tuple(x + y for x, y in zip(c if w is None else whole(c, w), d))
        if w is not None:  # (a count taken from by more than it holds is left at 0: the solver never chose that)
            c = tuple(max(0, x) if k is COUNT else x for x, k in zip(c, self.kinds))
        y = self.show(q, c)
        return ((q, c), y), y

    def run(self, w):
        st = self.start()
        for t in w:
            st, _ = self.next(st, self.syms.index(t))
        return st[1]

    def test(self, t, v):
        return f"state {v}" if t == 0 else f"{NAMES[t - 1]}{'<0' if v < 0 else '=0' if v == 0 else '>0'}"

    def rows(self, a):
        """The moves of symbol a as the rows its tests give: (the tests, the state to go to, the moves)."""
        T, out = self.reads[a], []
        for vals in itertools.product(*[range(self.Q) if t == 0 else self.kinds[t - 1] for t in T]):
            c = [0] * (self.K + 1)
            for t, v in zip(T, vals):
                c[t] = v
            go, d, w = self.eff[a][self.cell(c[0], c[1:])]
            out.append((" and ".join(self.test(t, v) for t, v in zip(T, vals)), go, d, w))
        return out

    def say(self, go, d, w=None):
        m = ([] if w is None else [(f"pour {NAMES[w[1]]} into {NAMES[w[2]]}" if w[3] > 0 else
                                    f"take all of {NAMES[w[1]]} from {NAMES[w[2]]}") if w[0] == "pour" else
                                   f"take {NAMES[w[1]]} and {NAMES[w[2]]} from each other" if w[0] == "even" else
                                   f"empty {NAMES[w[1]]}"]) + [f"{NAMES[i]}{'+1' if x > 0 else '-1'}" for i, x in enumerate(d) if x] + ([f"to state {go}"] if go is not None else [])
        return ", ".join(m) or "nothing"

    def __str__(self):
        words = []
        for a, s in enumerate(self.syms):
            r = [(t, self.say(go, d, w)) for t, go, d, w in self.rows(a)]
            if len({x for _, x in r}) == 1:
                words.append(f"{s}: {r[0][1]}")
            else:
                words.append(f"{s}: " + "; ".join(f"if {t}: {x}" for t, x in r if x != "nothing"))
        names = ["state"] + list(NAMES)
        shown = f"shown by {', '.join(names[t] for t in self.oreads) or 'nothing'}"
        if self.num:
            shown += "; " + "; ".join(repr(y.replace("\0", "{}")) + " with " + ", ".join(
                f"{'-' if g < 0 else ''}{NAMES[j]}{'+' + str(o) if o else ''}" for j, g, o in v) for y, v in self.num.items())
        n, z, q, _ = self.shape
        kinds = f"{n} count{'s' * (n != 1)}" + (f" ({NAMES[:n]})" if n else "") + \
            (f", {z} difference{'s' * (z != 1)} ({NAMES[n:n + z]})" if z else "") + (f", {q} states" if q > 1 else "")
        init = getattr(self, "init", None)
        starts = "" if not init or not any(init) else " | start " + ", ".join(f"{NAMES[i]}={v}" for i, v in enumerate(init) if v)
        return f"{kinds} (price {self.total}): " + " | ".join(words) + f" | {shown}" + starts


def no(x):
    return (not x) if isinstance(x, bool) else -x


class CNF:
    def __init__(self):
        self.top, self.cl = 0, []

    def var(self):
        self.top += 1
        return self.top

    def add(self, lits):
        out = []
        for x in lits:
            if x is True:
                return
            if x is not False:
                out.append(x)
        self.cl.append(out)

    def one(self, xs):  # exactly one
        self.cl.append(list(xs))
        for x, y in itertools.combinations(xs, 2):
            self.cl.append([-x, -y])


def depths(H):
    d = [0] * len(H.parent)
    for n in range(1, len(H.parent)):
        d[n] = d[H.parent[n]] + 1
    return d


class Space:
    """The machines of one shape and one kind of display, as formulas for a solver kept from round to round: the
    beginnings heard are added to it as they come, and what the solver found (its cores, Ignatiev et al. 2019) stays
    true when more is added, so a search stopped by the time goes on from where it stopped in the next round."""

    def __init__(self, H, shape, numeric, nodes, avoid=None):
        self.H, self.shape, self.numeric = H, shape, numeric
        A = self.A = len(H.syms)
        n_, z_, Q, w_ = shape
        K, kinds = n_ + z_, [COUNT] * n_ + [DIFF] * z_
        self.K, self.Q, self.kinds = K, Q, kinds
        W = self.W = wholes(n_, K) if w_ else []
        C = self.C = list(itertools.product(range(Q), *kinds))
        at = {c: i for i, c in enumerate(C)}
        tests = self.tests = ([0] if Q > 1 else []) + list(range(1, K + 1))
        top = lambda t: Q - 1 if t == 0 else 1
        pairs = {t: [(at[c], at[c[:t] + (c[t] + 1,) + c[t + 1:]]) for c in C if c[t] < top(t)] for t in tests}
        # (numeric: what is shown is words with places for numbers, each place filled by a number of the machine, or
        # minus it, plus what that place showed before any symbol; the words are read off the state and the tests)
        self.labels = sorted(set(form(y)[0] if numeric else y for y in H.shown))
        # (a count may start above 0 when numbers are shown and some is above 0 before any symbol: at most the largest
        # of them; what is shown then is told by what the numbers start with)
        self.off = {y: [0] * y.count("\0") for y in self.labels} if numeric else None
        self.V = min(9, max([0] + [abs(v) for v in form(H.shown[0])[1]])) if numeric else 0
        f = self.f = CNF()
        D = self.D = [[[[f.var() for _ in DIFF] for _ in range(K)] for _ in C] for _ in range(A)]  # (adds -1, 0, +1)
        E = self.E = [[[f.var() for _ in range(Q + 1)] for _ in C] for _ in range(A)] if Q > 1 else None  # (stay, go to q)
        R = self.R = [{t: f.var() for t in tests} for _ in range(A)]
        self.V0 = [[f.var() for _ in range(self.V + 1)] if kinds[i] is COUNT else None for i in range(K)]
        for v0 in self.V0:
            if v0 is not None:
                f.one(v0)
        for a in range(A):
            for ci, c in enumerate(C):
                for i in range(K):
                    f.one(D[a][ci][i])
                    if kinds[i] is COUNT and c[i + 1] == 0:
                        f.add([-D[a][ci][i][0]])  # (a count at 0 is not taken from)
                if Q > 1:
                    f.one(E[a][ci])
        M = self.M = [[[f.var() for _ in range(len(W) + 1)] for _ in C] for _ in range(A)] if W else None  # (none, or w)
        if W:
            for a in range(A):
                for ci, c in enumerate(C):
                    f.one(M[a][ci])
                    for k, w in enumerate(W):
                        for i in moved(w):  # (a number moved as a whole is not moved by one in the same row)
                            f.add([-M[a][ci][k + 1], D[a][ci][i][1]])
        # (a label, or words, not heard yet can be shown by no machine: one heard later starts the space anew)
        self.O = [[f.var() for _ in self.labels] for _ in C]
        self.OR = {t: f.var() for t in tests}
        for ci in range(len(C)):
            f.one(self.O[ci])
        if numeric:  # (N[l][s][j, g]: place s of the words l shows g times number j)
            self.N = [[[f.var() for _ in range(2 * K)] for _ in range(y.count("\0"))] for y in self.labels]
            for x in self.N:
                for v in x:
                    f.one(v)
        for a in range(A):  # (at most MOST[0] tests a symbol, MOST[1] the display)
            enc = CardEnc.atmost(lits=[R[a][t] for t in tests], bound=MOST[0], top_id=f.top, encoding=EncType.seqcounter)
            f.top = max(f.top, enc.nv)
            f.cl += enc.clauses
        enc = CardEnc.atmost(lits=list(self.OR.values()), bound=MOST[1], top_id=f.top, encoding=EncType.seqcounter)
        f.top = max(f.top, enc.nv)
        f.cl += enc.clauses
        if avoid is not None:  # (a rival: some symbol, or the display, reads another test than avoid does)
            f.add([-R[a][t] if t in avoid.reads[a] else R[a][t] for a in range(A) for t in tests] +
                  [-self.OR[t] if t in avoid.oreads else self.OR[t] for t in tests])
        cost = self.cost = []
        for a in range(A):  # a test not read: the move is the same whatever the test says
            for t in tests:
                for x, y in pairs[t]:
                    for i in range(K):
                        for j in range(3):
                            f.add([R[a][t], -D[a][x][i][j], D[a][y][i][j]])
                    if Q > 1:
                        for j in range(Q + 1):
                            f.add([R[a][t], -E[a][x][j], E[a][y][j]])
                    if W:
                        for j in range(len(W) + 1):
                            f.add([R[a][t], -M[a][x][j], M[a][y][j]])
                cost += [R[a][t]] * READ
            for ci, c in enumerate(C):  # a move made is paid for once, in the cell where every test not read is at 0;
                nc = [-R[a][t] for t in tests if c[t] != 0]  # and every row of the table that does something, once
                row = f.var()
                cost += [row] * ROW
                for i in range(K):
                    z = f.var()
                    f.add([z, D[a][ci][i][1]] + nc)
                    f.add([row, D[a][ci][i][1]] + nc)
                    cost.append(z)
                if Q > 1:
                    z = f.var()
                    f.add([z, E[a][ci][0]] + nc)
                    f.add([row, E[a][ci][0]] + nc)
                    cost.append(z)
                if W:
                    z = f.var()
                    f.add([z, M[a][ci][0]] + nc)
                    f.add([row, M[a][ci][0]] + nc)
                    cost += [z] * MOVE
        for t in tests:
            for x, y in pairs[t]:
                for l in range(len(self.labels)):
                    f.add([self.OR[t], -self.O[x][l], self.O[y][l]])
            cost += [self.OR[t]] * READ
        for v0 in self.V0:  # (a count that does not start at 0 is paid for, START by what it starts with)
            if v0 is not None:
                cost += [-v0[0]] * START
        self.G, self.S, self.was, self.dep, self.solver = {}, {}, [False] * K, {0: 0}, None
        self.add(nodes)
        w = WCNF()
        for c in f.cl:
            w.append(c)
        weight = {}
        for x in cost:
            weight[x] = weight.get(x, 0) + 1
        for x, k in weight.items():
            w.append([-x], weight=k)
        self.solver = RC2(w, solver=ENGINE)  # (a variable added later is given one of its own by the solver)
        f.cl = None

    def clause(self, lits):
        out = []
        for x in lits:
            if x is True:
                return
            if x is not False:
                out.append(x)
        if self.solver is None:
            self.f.cl.append(out)
        else:
            self.solver.add_clause(out)

    def hi(self, n, i):  # (the most number i can be at beginning n: one a symbol, from what a count may start with)
        return self.dep[n] + (self.V if self.kinds[i] is COUNT else 0)

    def ge(self, n, i, k):
        d = self.dep[n]
        return True if k <= (0 if self.kinds[i] is COUNT else -d) else False if k > self.hi(n, i) else self.G[n][i][k]

    def pre(self, n, c):  # the negation of "beginning n is in cell c", as a list of literals
        out = [no(self.S[n][c[0]])]
        for i in range(self.K):
            v, ge = c[i + 1], self.ge
            out += [ge(n, i, 0)] if v < 0 else [no(ge(n, i, 0)), ge(n, i, 1)] if v == 0 else [no(ge(n, i, 1))]
        return out

    def add(self, nodes):
        """The beginnings in nodes not in the formulas yet (their beginnings before are in them, or among nodes)."""
        H, f, K, Q, C, ge, add = self.H, self.f, self.K, self.Q, self.C, self.ge, self.clause
        for n in sorted(x for x in nodes if x not in self.S):
            d = self.dep[n] = 0 if n == 0 else self.dep[H.parent[n]] + 1
            self.G[n] = [{k: f.var() for k in range((1 if self.kinds[i] is COUNT else -d + 1), self.hi(n, i) + 1)} for i in range(K)]
            for i in range(K):
                for k in self.G[n][i]:
                    add([no(ge(n, i, k + 1)), ge(n, i, k)])
            if n == 0:
                self.S[n] = [q == 0 for q in range(Q)]
                for i in range(K):  # (what each count starts with)
                    for v in range(self.V + 1) if self.kinds[i] is COUNT else ():
                        for k in self.G[n][i]:
                            add([-self.V0[i][v], ge(n, i, k) if k <= v else no(ge(n, i, k))])
            else:
                self.S[n] = [f.var() for _ in range(Q)] if Q > 1 else [True]
                if Q > 1:
                    add(list(self.S[n]))
                    for x, y in itertools.combinations(self.S[n], 2):
                        add([-x, -y])
                p, a = H.parent[n], H.sym[n]
                X = [[f.var() for _ in DIFF] for _ in range(K)]
                W = self.W
                Y = [f.var() for _ in range(len(W) + 1)] if W else None  # (the whole move made here, if any)
                if W:
                    add(list(Y))
                    for x, y in itertools.combinations(Y, 2):
                        add([-x, -y])
                for i in range(K):
                    by = [Y[k + 1] for k, w in enumerate(W) if i in moved(w)]  # (a number moved as a whole here)
                    for x, y in itertools.combinations(X[i], 2):
                        add([-x, -y])
                    for j, delta in enumerate(DIFF):
                        for k in range(0 if self.kinds[i] is COUNT else -d, self.hi(n, i) + 1):
                            u, v = ge(n, i, k), ge(p, i, k - delta)
                            add(by + [-X[i][j], no(u), v])
                            add(by + [-X[i][j], u, no(v)])
                for k, w in enumerate(W):
                    y, dp = Y[k + 1], self.dep[p]
                    i = w[1]
                    if w[0] != "even":  # (number i is 0 after it is poured or emptied)
                        add([-y, no(ge(n, i, 1))])
                        if self.kinds[i] is DIFF:
                            add([-y, ge(n, i, 0)])
                    if w[0] == "pour":  # (j after = j before + g times i before; i is t when i >= t and not i >= t + 1)
                        j, g = w[2], w[3]
                        lo_i, lo_j = (0 if self.kinds[i] is COUNT else -dp), (1 if self.kinds[j] is COUNT else -d + 1)
                        for t in range(lo_i, self.hi(p, i) + 1):
                            if self.kinds[j] is COUNT:  # (a count is not taken below 0: there must be as much)
                                add([-y, no(ge(p, i, t)), ge(p, i, t + 1), ge(p, j, -g * t)])
                        for kk in range(lo_j, self.hi(n, j) + 1):
                            for t in range(lo_i, self.hi(p, i) + 1):
                                add([-y, no(ge(p, i, t)), ge(p, i, t + 1), no(ge(p, j, kk - g * t)), ge(n, j, kk)])
                                add([-y, no(ge(p, i, t)), ge(p, i, t + 1), ge(p, j, kk - g * t), no(ge(n, j, kk))])
                    elif w[0] == "even":  # (j after = j before - i before, if above 0; and i after = 0 or i - j)
                        j = w[2]
                        for (x, z) in ((i, j), (j, i)):
                            for kk in range(1, self.hi(n, x) + 1):
                                for t in range(0, self.hi(p, z) + 1):
                                    add([-y, no(ge(p, x, kk + t)), ge(p, z, t + 1), ge(n, x, kk)])
                                    add([-y, ge(p, x, kk + t), no(ge(p, z, t)), no(ge(n, x, kk))])
                for ci, c in enumerate(C):
                    pr = self.pre(p, c)
                    if True in pr:
                        continue
                    for i in range(K):
                        for j in range(3):
                            add(pr + [-self.D[a][ci][i][j], X[i][j]])
                    for k in range(len(W) + 1 if W else 0):
                        add(pr + [-self.M[a][ci][k], Y[k]])
                    if Q > 1:
                        add(pr + [-self.E[a][ci][0], self.S[n][c[0]]])
                        for q in range(Q):
                            add(pr + [-self.E[a][ci][q + 1], self.S[n][q]])
                # (one machine of every family of the same: the counts in the order in which they first leave 0, in
                # the order in which the beginnings were added, and so the differences, each leaving 0 upward; numbers
                # swapped or turned over are the same machine)
                now = []
                for i in range(K):
                    up, down, was = ge(n, i, 1), no(ge(n, i, 0)), self.was[i]
                    if self.kinds[i] is DIFF:
                        add([no(down), was])
                    z = f.var()
                    add([-z, was, up, down])
                    for x in (was, up, down):
                        add([z, no(x)])
                    now.append(z)
                for i in range(K - 1):
                    if self.kinds[i] is self.kinds[i + 1]:
                        add([-now[i + 1], now[i]])
                self.was = now
            y = H.shown[n]
            if self.numeric:
                y, nums = form(y)
                l = self.labels.index(y)
                for s, v in enumerate(nums):
                    for j in range(K):
                        for gi, g in enumerate((1, -1)):
                            x = self.N[l][s][2 * j + gi]
                            u = g * (v - self.off[y][s])  # (number j is u here)
                            add([-x, ge(n, j, u)])
                            add([-x, no(ge(n, j, u + 1))])
            else:
                l = self.labels.index(y)
            for ci, c in enumerate(C):
                pr = self.pre(n, c)
                if True not in pr:
                    add(pr + [self.O[ci][l]])

    def solve(self, deadline):
        """The cheapest machine; "none" if there is none; None if time ran out first (the search goes on next time)."""
        if deadline <= time.time():
            return None
        r = self.solver
        tm = threading.Timer(deadline - time.time(), r.interrupt)
        tm.start()
        try:
            model = r.compute(expect_interrupt=True)
            stopped = r.interrupted
        finally:
            tm.cancel()
        if stopped:
            r.clear_interrupt()
            return None
        if model is None:
            return "none"
        best = set(x for x in model if x > 0)
        A, K, Q, C = self.A, self.K, self.Q, self.C
        E, D, R = self.E, self.D, self.R
        price = sum(x in best for x in self.cost)
        W, M = self.W, self.M
        eff = [[(None if Q == 1 or E[a][ci][0] in best else next(q for q in range(Q) if E[a][ci][q + 1] in best),
                 tuple(DIFF[next(j for j in range(3) if D[a][ci][i][j] in best)] for i in range(K)),
                 None if not W or M[a][ci][0] in best else W[next(k for k in range(len(W)) if M[a][ci][k + 1] in best)])
                for ci in range(len(C))]
               for a in range(A)]
        reads = [[t for t in self.tests if R[a][t] in best] for a in range(A)]
        out = [self.labels[next(l for l in range(len(self.labels)) if self.O[ci][l] in best)] for ci in range(len(C))]
        num = None
        if self.numeric:
            num = {y: [(k // 2, (1, -1)[k % 2], self.off[y][s]) for s in range(y.count("\0"))
                       for k in [next(k for k in range(2 * K) if self.N[l][s][k] in best)]] for l, y in enumerate(self.labels)}
        m = Machine(self.H.syms, self.shape, eff, out, num, reads, [t for t in self.tests if self.OR[t] in best], price)
        m.init = tuple(0 if v0 is None else next(v for v in range(self.V + 1) if v0[v] in best) for v0 in self.V0)
        return m

    def delete(self):
        self.solver.delete()


def wrong(m, H):
    """The beginnings heard that m answers wrongly while answering every beginning before them rightly."""
    st, bad, ok = [None] * len(H.parent), [], [True] * len(H.parent)
    st[0] = m.start()
    if st[0][1] != H.shown[0]:
        return [0]
    for n in range(1, len(H.parent)):
        p = H.parent[n]
        if not ok[p]:
            ok[n] = False
            continue
        st[n], y = m.next(st[p], H.sym[n])
        if y != H.shown[n]:
            ok[n] = False
            bad.append(n)
    return bad


def explain(H, deadline, log=lambda *a: None, short=4, per=None):
    """The cheapest machine over all shapes that answers everything heard; None when time runs out first. In the
    formulas are the beginnings of at most short symbols and, of the others, only those some machine found answered
    wrongly, up to its first wrong answer (kept in H.active, for every shape and every later round). Of every shape the
    least price it can still have is kept (more stories never lower it; a search stopped by the time leaves the bound
    its cores prove) with the machine found (the cheapest of its shape while it answers everything heard). The shape
    looked at next is the one that could be the cheapest; one whose search runs out of its time (per seconds, twice
    as many every time it runs out, up to 8 per) is left for
    the next round, where its search goes on (or for the next turn of this one, while nothing is found). Once some
    machine answers everything heard, a round takes at most ROUND seconds."""
    dep = depths(H)
    if not hasattr(H, "active"):
        H.active, H.low, H.kept, H.spaces, H.t0, H.slow = set(), {}, {}, {}, time.time(), {}
    nodes = H.active
    nodes |= {n for n in range(len(H.parent)) if dep[n] <= short}
    wordy = any(SLOT.search(y) for y in H.shown)  # (numbers among the answers: words with places for numbers too)
    keys = [shape + (mode,) for shape in SHAPES for mode in ((True, False) if wordy else (False,))]
    bound = lambda k: base(k[:4]) + H.low.get(k, 0)
    best, later, start = None, set(), time.time()
    while time.time() < deadline and (best is None or time.time() < start + ROUND):
        left = [k for k in keys if k not in later and H.low.get(k, 0) < NONE and (best is None or bound(k) < best.total)]
        if not left and best is None and later:  # (every search ran out of its time: they all go on)
            later = set()
            continue
        if not left:
            break
        key = min(left, key=bound)
        shape, mode = key[:4], key[4]
        say = f"{shape}, {'number' if mode else 'table'}"
        m = H.kept.get(key)
        if m is None or wrong(m, H):
            sp = H.spaces.get(key)
            if sp is not None and not set(form(y)[0] if mode else y for y in H.shown) <= set(sp.labels):
                sp.delete()  # (a label not heard before: the space is made anew)
                sp = None
            if sp is None:
                sp = H.spaces[key] = Space(H, shape, mode, nodes)
            sp.add(nodes)
            slice_ = None if per is None else min(per * 2 ** H.slow.get(key, 0), 8 * per)
            m = sp.solve(deadline if per is None else min(deadline, time.time() + slice_))
            if m is None:
                H.slow[key] = H.slow.get(key, 0) + 1  # (the next time it is given twice as long, up to 8 per)
                H.low[key] = max(H.low.get(key, 0), sp.solver.cost)
                later.add(key)
                log(f"    {say}: out of time, at least {H.low[key]} + {base(shape)} ({time.time() - H.t0:.0f} s)")
                continue
            if m == "none":
                H.low[key] = NONE
                log(f"    {say}: no machine ({time.time() - H.t0:.0f} s)")
                continue
        H.low[key], H.kept[key] = m.cost, m
        bad = wrong(m, H)
        log(f"    {say}: price {m.cost} + {base(shape)}, {len(nodes)} beginnings, {len(bad)} wrong ({time.time() - H.t0:.0f} s)")
        if not bad:
            if best is None or m.total < best.total:
                best = m
            continue
        for n in sorted(bad, key=lambda n: dep[n])[:10]:
            while n not in nodes:
                nodes.add(n)
                n = H.parent[n]
    return best


def parting(m, r, deep=10, most=200000, a0=None, b0=None):
    """The shortest story (of at most deep symbols) after which m and r show different things, from the states a0 and
    b0 (their starts if not given); None if there is none (looked for breadth first over the pairs of their states)."""
    A = len(m.syms)
    a0, b0 = a0 or m.start(), b0 or r.start()
    if a0[1] != b0[1]:
        return []
    layer, seen = [(a0, b0, ())], {(a0[0], b0[0])}
    for _ in range(deep):
        nxt = []
        for x, y, w in layer:
            for a in range(A):
                x2, u = m.next(x, a)
                y2, v = r.next(y, a)
                if u != v:
                    return [m.syms[i] for i in w + (a,)]
                if (x2[0], y2[0]) not in seen and len(seen) < most:
                    seen.add((x2[0], y2[0]))
                    nxt.append((x2, y2, w + (a,)))
        layer = nxt
    return None


def rivals(m, H, deadline, slack=6, per=30):
    """Machines that answer everything heard, of price at most slack more than m: the cheapest of every other shape
    kept by explain, or else the cheapest of m's own shape that reads other tests."""
    out = [r for r in H.kept.values() if r is not m and r.total <= m.total + slack and not wrong(r, H)]
    if out:  # (the cheapest of m's own shape reading other tests is looked for only when there is no other rival)
        return sorted(out, key=lambda r: r.total)
    sp = Space(H, m.shape, m.num is not None, H.active, avoid=m)
    r = sp.solve(min(deadline, time.time() + per))
    sp.delete()
    if r is not None and r != "none" and r.total <= m.total + slack and not wrong(r, H):
        out.append(r)
    return sorted(out, key=lambda r: r.total)


def learn(syms, oracle, data, rng, asks=40, deadline=None, H=None, log=lambda *a: None, per=PER, cap=0, short=SHORT):
    """The stories taught on (and H, the stories another learner heard, if given), then up to asks questions: first
    the story that parts the machine kept from a rival (see rivals), then long stories as a check (plain, with symbols
    repeated, kept away from an answer never left, a rare answer drawn out, as parts.py asks them); where the machine is
    wrong it is looked for again. Sure when no rival is left and NEED checks in a row were answered rightly. With cap,
    it stops when the cheapest machine costs more: that is not a rule of a few quantities, and the questions left are
    left to another learner."""
    deadline = deadline or time.time() + 600
    if H is None:
        H = parts.Heard(syms, oracle([]))
        for w, _ in data:
            H.add(list(w), [oracle(list(w[:i])) for i in range(1, len(w) + 1)])
    m = explain(H, deadline, log, short=short, per=per)
    asked, row = 0, 0
    log(f"  kept {m}")
    same = set()  # (rivals no story of deep symbols parts from the machine kept)
    while m is not None and asked < asks and row < NEED and time.time() < deadline and not (cap and m.total > cap):
        w, kind = None, "a check"
        # (one story parts the machine from several rivals, one after another: from where the story so far has brought
        # the machine and the next rival not yet parted, the shortest going-on after which the two show different things)
        left = [r for r in rivals(m, H, deadline) if (str(m), str(r)) not in same]
        story, parted = [], 0
        while left and len(story) < 40:
            r = left.pop(0)
            sm, sr = m.start(), r.start()
            apart = sm[1] != sr[1]
            for t in story:
                sm, _ = m.next(sm, syms.index(t))
                sr, _ = r.next(sr, syms.index(t))
                apart = apart or sm[1] != sr[1]
            if apart:
                parted += 1
                continue
            v = parting(m, r, a0=sm, b0=sr)
            if v is None:
                if not story:
                    same.add((str(m), str(r)))
                continue
            story += v
            parted += 1
        if story:
            kind = f"{parted} rival{'s' * (parted > 1)}"
            w = story + parts.story(syms, rng, max(0, 24 - len(story)))  # (seen after every symbol, it goes on)
        if w is None:
            kind = ("plain", "again", "plain", "again", "alive", "pumped")[row % 6] if row >= 4 else ("plain", "again")[row % 2]
            w = parts.alive(m, syms, rng, rng.randint(*parts.LONG)) if kind == "alive" else \
                parts.pumped(H, rng, parts.LONG[1]) if kind == "pumped" else None
            if w is None:
                w = parts.story(syms, rng, rng.randint(*parts.LONG), 0.3 if kind == "again" else 0.0)
        asked += 1
        H.add(w, [oracle(w[:i]) for i in range(1, len(w) + 1)])
        if all(m.run(w[:i]) == H.shown[H.index[tuple(w[:i])]] for i in range(1, len(w) + 1)):
            row += kind.startswith(("plain", "again", "alive", "pumped"))
            log(f"  asked {asked} ({kind}): answered rightly")
            continue
        row = 0
        m = explain(H, deadline, log, short=short, per=per)
        log(f"  asked {asked} ({kind}): answered wrongly; kept {m}")
    if m is not None:
        m.sure = row >= NEED
    return m, asked, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--asks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--per", type=float, default=PER, help="seconds for a shape in a round")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--short", type=int, default=SHORT)
    ap.add_argument("--plain", action="store_true", help="no whole moves")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("family", a.family)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    syms, answer = list(mod.ALPHABET), mod.answer
    oracle = lambda w: str(answer(list(w)))
    rng = random.Random(a.seed)
    data = [(w, oracle(w)) for w in (tuple(rng.choice(syms) for _ in range(rng.randint(1, 4))) for _ in range(300))]
    global SHAPES
    if a.plain:
        SHAPES = [s for s in SHAPES if not s[3]]
    t0 = time.time()
    m, asked, H = learn(syms, oracle, data, rng, a.asks, t0 + a.minutes * 60, log=(lambda *x: None) if a.quiet else print, per=a.per, short=a.short)
    if m is None:
        print(f"nothing found; asked {asked}; {time.time() - t0:.0f} s")
        return
    print(f"{'' if m.sure else 'NOT SURE: '}{m}; asked {asked}; {time.time() - t0:.0f} s")
    for L in (20, 100, 500):
        tr = random.Random(1000 + L)
        seqs = [[tr.choice(syms) for _ in range(L)] for _ in range(300)]
        print(L, 100 * sum(m.run(s) == oracle(s) for s in seqs) / 300, "%")


if __name__ == "__main__":
    main()
