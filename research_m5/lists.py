"""A machine of parts said as lists of guarded commands (stage 97).

guarded.py looks for a counter machine as one flat table: a row for every state and every reading of every test, so
its formulas grow as the product of what is kept in mind (states x 2 ** counts x 3 ** differences), though a symbol
reads at most three tests. A person says the same machine in a few lines over a few parts: 'grab: if craving salmon
and salmon above 0, salmon - 1; else if craving tuna and tuna above 0, tuna - 1; else if salmon above 0, salmon - 1;
else if tuna above 0, tuna - 1'. Here a machine is said so. It keeps a few parts: numbers (a count, never below 0 and
tested for 0; or a difference, any whole number, tested for its sign) and switches (a few values, the first at the
start). Every symbol has a short list of guarded commands (Dijkstra 1975), tried in order: the first whose guard holds
is done, and when none holds nothing is done. A guard reads at most LITS tests; a command adds -1, 0 or +1 to every
number, sets every switch, moves it on by one (round: from the last value back to the first) or leaves it, and may make
one whole move of guarded.py. What is shown is read as in guarded.py: the display reads at most three tests and has a
table over what they say (a list of guards could not say seven answers over three parts, as the horse stable shows
them), so its formulas too do not grow with the parts.

The shape is not looked over one by one. Within an envelope (so many numbers, switches of at most so many values) the
solver chooses which numbers are counts, which differences and which are not kept at all, and how many values every
switch has. Every part kept is paid for as in guarded.py (3 a count, 4 a difference, 3 every value of a switch beyond
the first), and so is every command (1), every test it reads (2), every number it moves and every switch it sets (1).
The formulas grow with the parts and the commands, not with their product, and the price is the length of what is
said (minimum description length, Rissanen 1978). The envelopes are disjoint layers (a machine of a layer does not fit
in any layer before it), searched by guarded.explain in the order of their bounds as the shapes are there.

    python lists.py FAMILY/family.py [--asks 40] [--seed 0] [--minutes 10]
"""
import argparse, importlib.util, itertools, random, time
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver
import guarded, parts
from guarded import CNF, ENGINE, Fresh, TooBig, CLAUSES, READ, ROW, START, MOVE, NAMES, no, form, sign, whole, moved

KINDS = ("none", "count", "diff")
VALS = (-1, 0, 1)  # what a number's test says: below 0, 0, above 0
ROWS = 4  # commands a symbol has at most
SEE = 3  # tests the display reads at most (what is shown is a table over what they say)
SEE2 = 4  # in a layer with comparisons (the largest of three piles, or none: three comparisons and one test for 0)
LITS = 3  # tests a guard reads at most
KEEP = (0, 3, 4)  # the price of a number not kept, a count, a difference
VALUE = 3  # the price of every value of a switch beyond the first
WHOLE = 4  # the price of a layer with whole moves (as the second kind in guarded.py)
CMP = 1  # what reading a comparison of two counts costs beyond reading a test of one part
SWITCH = "xy"
# the layers: so many numbers, switches of at most so many values (largest first), whole moves or not, comparisons
# of two counts or not (stage 98: 'more fish on the table than open cans'; a layer with them uses one at least)
LAYERS = [(2, (2,), 0, 0), (3, (2,), 0, 0), (3, (3, 2), 0, 0), (4, (3, 2), 0, 0),
          (2, (2,), 2, 0), (3, (2,), 2, 0), (3, (3, 2), 2, 0),
          (2, (2,), 0, 1), (3, (2,), 0, 1), (3, (3, 2), 0, 1)]


def pairs(N, c):
    """The comparisons a layer may read: every two numbers, the first before the second (both counts)."""
    return list(itertools.combinations(range(N), 2)) if c else []


def within(N, S, M, Z):
    """Whether a machine of layer (N, S) may fit in the layer (M, Z) of fewer parts."""
    return N <= M and len(S) <= len(Z) and all(x <= y for x, y in zip(S, Z))


class ListMachine:
    """Numbers (kinds[i]: 'none', 'count' or 'diff') and switches (sizes[s] values); rows[a] = [(guard, adds, sets,
    whole move)], the guard a list of (test, value); what is shown: oread, the tests the display reads, and otab, the
    label for what they say (a tuple of values; a slot not read says 0),
    numbers filled in as in guarded.Machine."""

    def __init__(self, syms, key, kinds, sizes, rows, oread, otab, num, init, cost):
        self.syms, self.key, self.kinds, self.sizes = list(syms), key, kinds, sizes
        self.rows, self.oread, self.otab, self.num, self.init = rows, oread, otab, num, init
        self.N, self.cost = len(kinds), cost
        self.P = pairs(self.N, key[3])
        self.see = SEE2 if key[3] else SEE
        self.total, self.sure, self.proved = cost + (WHOLE if key[2] else 0), False, True
        self.reads = [sorted({t for g, _, _, _ in rows[a] for t, _ in g}) for a in range(len(self.syms))]
        self.oreads = sorted(oread)
        self.shape = (tuple(kinds), tuple(sizes))

    def signed(self, t):  # (a test of a number or a comparison: it says -1, 0 or 1)
        return t < self.N or t >= self.N + len(self.sizes)

    def value(self, st, t):
        if t >= self.N + len(self.sizes):
            i, j = self.P[t - self.N - len(self.sizes)]
            return sign(st[1][i] - st[1][j])
        return sign(st[1][t]) if t < self.N else st[0][t - self.N]

    def seen(self, st):  # (what the tests the display reads say, as the index of the table)
        v = [(VALS.index(self.value(st, t)) if self.signed(t) else self.value(st, t)) for t in self.oread]
        return tuple(v + [0] * (self.see - len(v)))

    def holds(self, g, st):
        return all(self.value(st, t) == v for t, v in g)

    def start(self):
        st = (tuple(0 for _ in self.sizes), tuple(self.init))
        return (st, self.show(st))

    def next(self, state, a):
        st, _ = state
        for g, d, e, w in self.rows[a]:
            if self.holds(g, st):
                sw, c = st
                if w is not None:
                    c = whole(c, w)
                c = tuple(x + y for x, y in zip(c, d))
                if w is not None:  # (a count taken from by more than it holds is left at 0)
                    c = tuple(max(0, x) if k == "count" else x for x, k in zip(c, self.kinds))
                sw = tuple(v if x is None else (v + 1) % n if x == "next" else x for v, x, n in zip(sw, e, self.sizes))
                st = (sw, c)
                break
        y = self.show(st)
        return (st, y), y

    def show(self, st):
        y = self.otab[self.seen(st)]
        if self.num is None or "\0" not in y:
            return y
        bits = y.split("\0")
        vals = [str(g * st[1][j] + o) for j, g, o in self.num[y]]
        return "".join(x + v for x, v in zip(bits, vals + [""]))

    def run(self, w):
        st = self.start()
        for t in w:
            st, _ = self.next(st, self.syms.index(t))
        return st[1]

    def test(self, t, v):
        if t < self.N:
            return f"{NAMES[t]}{'<0' if v < 0 else '=0' if v == 0 else '>0'}"
        if t >= self.N + len(self.sizes):
            i, j = self.P[t - self.N - len(self.sizes)]
            return f"{NAMES[i]}{'<' if v < 0 else '=' if v == 0 else '>'}{NAMES[j]}"
        return f"{SWITCH[t - self.N]}={v}"

    def say(self, d, e, w):
        m = ([] if w is None else [guarded.Machine.say(None, None, (0,) * self.N, w)]) + \
            [f"{NAMES[i]}{'+1' if x > 0 else '-1'}" for i, x in enumerate(d) if x] + \
            [f"{SWITCH[s]} on" if x == "next" else f"{SWITCH[s]}={x}" for s, x in enumerate(e) if x is not None]
        return ", ".join(m) or "nothing"

    def guard(self, g):
        return " and ".join(self.test(t, v) for t, v in g)

    def __str__(self):
        words = []
        for a, s in enumerate(self.syms):
            r = self.rows[a]
            if not r:
                words.append(f"{s}: nothing")
            elif len(r) == 1 and not r[0][0]:
                words.append(f"{s}: {self.say(*r[0][1:])}")
            else:
                words.append(f"{s}: " + "; else ".join(f"if {self.guard(g)}: {self.say(d, e, w)}" if g else
                                                       self.say(d, e, w) for g, d, e, w in r))
        def fill(y):  # (the words with the numbers in their places named)
            if not self.num or "\0" not in y:
                return repr(y)
            bits = y.split("\0")
            names = ["{" + f"{'-' if g < 0 else ''}{NAMES[j]}{'+' + str(o) if o else ''}" + "}" for j, g, o in self.num[y]]
            return repr("".join(x + v for x, v in zip(bits, names + [""])))
        span = [range(1) if k == "none" else range(4) if k == "count" else range(-3, 4) for k in self.kinds]
        can = {self.seen((sw, c)) for c in itertools.product(*span)  # (the readings that can be: a count is not
               for sw in itertools.product(*[range(k) for k in self.sizes])}  # below 0, a<b and b<c give a<c)
        cells = [(c, l) for c, l in sorted(self.otab.items()) if c in can]
        if len(set(l for _, l in cells)) <= 1:
            shown = "shown: " + fill(cells[0][1] if cells else "")
        else:  # (said as a short list: each time the guard of one or two tests that tells the most readings left)
            lits = [(k, v) for k in range(len(self.oread)) for v in sorted({c[k] for c, _ in cells})]
            conjs = [(x,) for x in lits] + [(x, y) for x, y in itertools.combinations(lits, 2) if x[0] != y[0]]
            left, said = list(cells), []
            while len({l for _, l in left}) > 1:
                best = None
                for cj in conjs:
                    hit = [l for c, l in left if all(c[k] == v for k, v in cj)]
                    if hit and len(set(hit)) == 1 and (best is None or len(hit) > best[0] or
                                                       len(hit) == best[0] and len(cj) < len(best[1])):
                        best = (len(hit), cj, hit[0])
                if best is None:  # (no guard of two tests is enough: the whole reading)
                    c, l = left[0]
                    best = (1, tuple(enumerate(c[:len(self.oread)])), l)
                _, cj, l = best
                said.append((cj, l))
                left = [(c, x) for c, x in left if not all(c[k] == v for k, v in cj)]
            tv = lambda k, v: (self.oread[k], VALS[v] if self.signed(self.oread[k]) else v)
            shown = "shown: " + "; ".join(f"if {self.guard([tv(k, v) for k, v in cj])}: {fill(l)}" for cj, l in said) + \
                f"; else {fill(left[0][1])}"
        n = [NAMES[i] for i, k in enumerate(self.kinds) if k == "count"]
        z = [NAMES[i] for i, k in enumerate(self.kinds) if k == "diff"]
        sw = [f"{SWITCH[s]} of {k}" for s, k in enumerate(self.sizes) if k > 1]
        kinds = ", ".join(([f"{len(n)} count{'s' * (len(n) != 1)} ({''.join(n)})"] if n else []) +
                          ([f"{len(z)} difference{'s' * (len(z) != 1)} ({''.join(z)})"] if z else []) +
                          ([f"switch{'es' * (len(sw) != 1)} {', '.join(sw)}"] if sw else [])) or "nothing kept"
        starts = "" if not any(self.init) else " | start " + ", ".join(
            f"{NAMES[i]}={v}" for i, v in enumerate(self.init) if v)
        return f"{kinds} (price {self.total}): " + " | ".join(words) + f" | {shown}" + starts


class ListSpace(guarded.Space):
    """The machines of one layer and one kind of display, as formulas kept from round to round (the search itself,
    by cores or by guessing, is guarded.Space's)."""

    def __init__(self, H, key, nodes, avoid=None, guess=False, low=0):
        N, S, w, c, numeric = key
        self.H, self.key, self.numeric, self.guess = H, key, numeric, guess
        A = self.A = len(H.syms)
        self.N, self.S, self.K = N, S, N
        P = self.P = pairs(N, c)
        T = self.T = N + len(S) + len(P)
        Q = self.Q = N + len(S)  # (the comparisons are the tests from Q on)
        see = self.see = SEE2 if c else SEE
        self.tv = [list(range(3)) for _ in range(N)] + [list(range(k)) for k in S] + [list(range(3)) for _ in P]
        W = self.W = guarded.wholes(N, N, 2) if w else []
        self.labels = sorted(set(form(y)[0] if numeric else y for y in H.shown))
        self.off = {y: [0] * y.count("\0") for y in self.labels} if numeric else None
        self.V = min(9, max([0] + [abs(v) for v in form(H.shown[0])[1]])) if numeric else 0
        f = self.f = CNF()
        var = f.var
        KD = self.KD = [[var() for _ in KINDS] for _ in range(N)]  # (not kept, a count, a difference)
        for i in range(N):
            f.one(KD[i])
            if i + 1 < N:
                f.add([-KD[i][0], KD[i + 1][0]])  # (the numbers not kept are the last)
        SZ = self.SZ = [[var() for _ in range(k)] for k in S]  # (switch s has k + 1 values)
        ZS = self.ZS = [{k: var() for k in range(2, S[s] + 1)} for s in range(len(S))]  # (it has at least k values)
        for s in range(len(S)):
            f.one(SZ[s])
            for k in ZS[s]:
                f.add([-ZS[s][k]] + SZ[s][k - 1:])
                for x in SZ[s][k - 1:]:
                    f.add([-x, ZS[s][k]])
            if s + 1 < len(S):
                for k in ZS[s + 1]:
                    f.add([-ZS[s + 1][k], ZS[s][k]] if k in ZS[s] else [-ZS[s + 1][k]])
        self.ge_size = lambda s, k: True if k <= 1 else ZS[s][k] if k in ZS[s] else False
        for M_, Z_, w_, c_ in LAYERS:  # (a machine of this layer does not fit in a layer before it)
            if (M_, Z_, w_, c_) == (N, S, w, c):
                break
            if w_ == w and c_ == c and within(M_, Z_, N, S):
                f.add(([-KD[M_][0]] if N > M_ else []) +
                      [self.ge_size(s, Z_[s] + 1) for s in range(min(len(S), len(Z_)))] +
                      [self.ge_size(s, 2) for s in range(len(Z_), len(S))])
        U = self.U = [[var() for _ in range(ROWS)] for _ in range(A)]
        L = self.L = [[[[var() for _ in self.tv[t]] for t in range(T)] for _ in range(ROWS)] for _ in range(A)]
        D = self.D = [[[[var() for _ in VALS] for _ in range(N)] for _ in range(ROWS)] for _ in range(A)]
        E = self.E = [[[[var() for _ in range(k + 2)] for k in S] for _ in range(ROWS)] for _ in range(A)]
        Mv = self.M = [[[var() for _ in range(len(W) + 1)] for _ in range(ROWS)] for _ in range(A)] if W else None
        cost = self.cost = []
        for i in range(N):
            cost += [KD[i][1]] * KEEP[1] + [KD[i][2]] * KEEP[2]
        for s in range(len(S)):
            for k in ZS[s]:
                cost += [ZS[s][k]] * VALUE

        def guard(Lr, used):  # (a guard: at most one value of a test, LITS tests, none of a part not kept)
            for t in range(T):
                for x, y in itertools.combinations(Lr[t], 2):
                    f.add([-x, -y])
                for vi, x in enumerate(Lr[t]):
                    f.add([-x, used])
                    if t < N:
                        f.add([-x, -KD[t][0]])
                    elif t >= Q:  # (a comparison of two counts)
                        f.add([-x, KD[P[t - Q][0]][1]])
                        f.add([-x, KD[P[t - Q][1]][1]])
                    else:
                        f.add([-x, self.ge_size(t - N, max(2, vi + 1))])
            enc = CardEnc.atmost(lits=[x for y in Lr for x in y], bound=LITS, top_id=f.top, encoding=EncType.seqcounter)
            f.top = max(f.top, enc.nv)
            f.cl += enc.clauses
            for t, y in enumerate(Lr):
                cost.extend([x for x in y] * (READ + (CMP if t >= Q else 0)))

        for a in range(A):
            for r in range(ROWS):
                u = U[a][r]
                if r + 1 < ROWS:
                    f.add([-U[a][r + 1], u])
                cost += [u] * ROW
                guard(L[a][r], u)
                for i in range(N):
                    f.one(D[a][r][i])
                    f.add([u, D[a][r][i][1]])
                    f.add([-KD[i][0], D[a][r][i][1]])
                    f.add([-D[a][r][i][0], -KD[i][1], L[a][r][i][2]])  # (a count is taken from only when above 0)
                    cost.append(-D[a][r][i][1])
                for s, k in enumerate(S):
                    e = E[a][r][s]
                    f.one(e)
                    f.add([u, e[0]])
                    f.add([self.ge_size(s, 2), e[0]])
                    for v in range(1, k):
                        f.add([-e[2 + v], self.ge_size(s, v + 1)])
                    cost.append(-e[0])
                if W:
                    m = Mv[a][r]
                    f.one(m)
                    f.add([u, m[0]])
                    cost += [-m[0]] * MOVE
                    for k, x in enumerate(W):
                        for i in moved(x) if x[0] != "copy" else ():  # (a number moved as a whole is not moved by one)
                            f.add([-m[k + 1], D[a][r][i][1]])
                        if x[0] in ("even", "empty"):  # (among counts)
                            for i in moved(x):
                                f.add([-m[k + 1], KD[i][1]])
                        elif x[0] == "pour":  # (from a count into a number kept; of the first kind, among counts)
                            f.add([-m[k + 1], KD[x[1]][1]])
                            f.add([-m[k + 1], -KD[x[2]][0]])
                        else:  # (copy: a count from a count, a difference from a number kept)
                            f.add([-m[k + 1], -KD[x[1]][0]])
                            f.add([-m[k + 1], -KD[x[2]][0]])
                            f.add([-m[k + 1], -KD[x[1]][1], KD[x[2]][1]])
        if W:  # (a layer with whole moves makes one at least)
            f.add([-Mv[a][r][0] for a in range(A) for r in range(ROWS)])
        # (the display reads at most SEE tests, in order, the slots not used last, and shows a label for every
        # reading of them: a table of 3 ** SEE cells whatever the parts)
        SL = self.SL = [[var() for _ in range(T + 1)] for _ in range(see)]  # (slot k reads test t; T: none)
        self.combos = list(itertools.product(range(3), repeat=see))
        TB = self.TB = [[var() for _ in self.labels] for _ in self.combos]
        for k in range(see):
            f.one(SL[k])
            cost.append(-SL[k][T])
            cost.extend([-SL[k][T]] * (READ - 1))
            for t in range(T):
                if t < N:
                    f.add([-SL[k][t], -KD[t][0]])
                elif t >= Q:
                    f.add([-SL[k][t], KD[P[t - Q][0]][1]])
                    f.add([-SL[k][t], KD[P[t - Q][1]][1]])
                    cost.extend([SL[k][t]] * CMP)
                else:
                    f.add([-SL[k][t], self.ge_size(t - N, 2)])
            if k + 1 < see:
                f.add([-SL[k][T], SL[k + 1][T]])
                for t in range(T):  # (read in the order of the tests)
                    f.add([-SL[k + 1][t]] + [SL[k][u] for u in range(t)])
        if P:  # (a layer with comparisons reads one at least, in a guard or on the display)
            f.add([x for a in range(A) for r in range(ROWS) for t in range(Q, T) for x in L[a][r][t]] +
                  [SL[k][t] for k in range(see) for t in range(Q, T)])
        for ci, cc in enumerate(self.combos):
            f.one(TB[ci])
            for k in range(see):
                if cc[k]:  # (a slot not used says 0: the other cells are never seen)
                    f.add([-SL[k][T], TB[ci][0]])
        if numeric:  # (NV[l][s][j, g]: place s of the words l shows g times number j)
            self.NV = [[[var() for _ in range(2 * N)] for _ in range(y.count("\0"))] for y in self.labels]
            for x in self.NV:
                for v in x:
                    f.one(v)
                    for j in range(N):
                        f.add([-v[2 * j], -KD[j][0]])
                        f.add([-v[2 * j + 1], -KD[j][0]])
        self.V0 = [[var() for _ in range(self.V + 1)] for _ in range(N)]
        for i in range(N):
            f.one(self.V0[i])
            f.add([KD[i][1], self.V0[i][0]])  # (only a count may start above 0)
            cost += [-self.V0[i][0]] * START
        if avoid is not None:  # (a rival: some symbol, or the display, reads another test than avoid does)
            R = [[var() for _ in range(T)] for _ in range(A)]
            OR = [var() for _ in range(T)]
            for t in range(T):
                for a in range(A):
                    xs = [x for r in range(ROWS) for x in L[a][r][t]]
                    f.add([-R[a][t]] + xs)
                    for x in xs:
                        f.add([-x, R[a][t]])
                xs = [SL[k][t] for k in range(see)]
                f.add([-OR[t]] + xs)
                for x in xs:
                    f.add([-x, OR[t]])
            f.add([-R[a][t] if t in avoid.reads[a] else R[a][t] for a in range(A) for t in range(T)] +
                  [-OR[t] if t in avoid.oreads else OR[t] for t in range(T)])
        self.SW, self.G, self.dep, self.was, self.solver, self.inc, self.n = {}, {}, {0: 0}, [False] * N, None, None, len(f.cl)
        self.CM = {}  # (CM[n][p]: what comparison p says at beginning n, below / level / above)
        self.add(nodes)
        self.lits = list(cost)
        self.tot, self.low = None, low
        flat = lambda x: [v for y in x for v in (flat(y) if isinstance(y, list) else [y])]
        parts_ = flat(KD) + flat(SZ) + flat(self.V0)
        sym = [flat(U[a]) + flat(L[a]) + flat(D[a]) + flat(E[a]) + (flat(Mv[a]) if W else []) for a in range(A)]
        show = flat(SL) + flat(TB) + (flat(self.NV) if numeric else [])
        num = [[x for a in range(A) for r in range(ROWS) for x in D[a][r][i]] + KD[i] for i in range(N)]
        self.table = sorted(set(flat(sym) + show + parts_))
        self.groups = ([sym[a] + show + parts_ for a in range(A)] +
                       [sym[a] + sym[b] for a, b in itertools.combinations(range(A), 2)] + num + [show + parts_])
        self.turn, self.last = 0, None
        if guess:
            self.solver = Fresh(f.cl)
            self.inc = Solver(name=ENGINE, bootstrap_with=f.cl)
        else:
            wc = WCNF()
            for c in f.cl:
                wc.append(c)
            weight = {}
            for x in cost:
                weight[x] = weight.get(x, 0) + 1
            for x, k in weight.items():
                wc.append([-x], weight=k)
            self.solver = RC2(wc, solver=ENGINE)
        f.cl = None

    def hi(self, n, i):
        return self.dep[n] + self.V

    def ge(self, n, i, k):
        d = self.dep[n]
        return True if k <= -d else False if k > d + self.V else self.G[n][i][k]

    def conj(self, n, t, vi):
        """Test t says its value vi at beginning n: a list of literals that all hold."""
        if t >= self.Q:
            return [self.CM[n][t - self.Q][vi]]
        if t < self.N:
            ge = self.ge
            return [no(ge(n, t, 0))] if vi == 0 else [ge(n, t, 0), no(ge(n, t, 1))] if vi == 1 else [ge(n, t, 1)]
        return [self.SW[n][t - self.N][vi]]

    def size(self, nodes):
        N, T, V, W, out, dep = self.N, self.T, self.V, self.W, 0, dict(self.dep)
        per = sum(len(v) for v in self.tv)
        for n in sorted(x for x in nodes if x not in self.SW):
            d = dep[n] = 0 if n == 0 else dep[self.H.parent[n]] + 1
            out += ROWS * (3 * per + 3 * T + 4) + ROWS * (2 * N + 3 * sum(self.S) + len(W) + 2) + 6 * per + 3 ** self.see
            out += 8 * N * (2 * d + V + 2) + len(self.P) * (6 * (d + V) + 4)
            for w in W:
                out += (2 * (2 * d + V) * (d + V) if w[0] == "pour" else 4 * (d + V) * (d + V) if w[0] == "even" else
                        6 * (2 * d + V + 1) if w[0] == "copy" else 2)
        return out

    def guarded_list(self, n, at, U, L, OUT):
        """The commands of a list tried at beginning n on the tests at beginning at: OUT[r] holds when command r is
        the first whose guard holds; the last literal returned holds when none does."""
        f, add, T = self.f, self.clause, self.T
        none = True
        for r in range(len(U)):
            match = f.var()
            fails = []
            add([-match, U[r]])
            for t in range(T):
                fail = f.var()
                fails.append(fail)
                add([-fail] + L[r][t])
                for vi, x in enumerate(L[r][t]):
                    c = self.conj(at, t, vi)
                    for y in c:
                        add([-match, -x, y])
                    add([-fail, -x] + [no(y) for y in c])
            add([match, -U[r]] + fails)
            fire = OUT[r]
            nxt = f.var()
            # fire = none and match; nxt = none and not match
            add([-fire, match])
            add([-nxt, -match])
            if none is not True:
                add([-fire, none])
                add([-nxt, none])
                add([-none, -match, fire])
                add([-none, match, nxt])
            else:
                add([-match, fire])
                add([match, nxt])
            none = nxt
        return none

    def add(self, nodes):
        if self.n + self.size(nodes) > CLAUSES:
            raise TooBig
        H, f, N, S, ge, add, W = self.H, self.f, self.N, self.S, self.ge, self.clause, self.W
        KD = self.KD
        for n in sorted(x for x in nodes if x not in self.SW):
            d = self.dep[n] = 0 if n == 0 else self.dep[H.parent[n]] + 1
            self.G[n] = [{k: f.var() for k in range(-d + 1, d + self.V + 1)} for _ in range(N)]
            for i in range(N):
                for k in self.G[n][i]:
                    add([no(ge(n, i, k + 1)), ge(n, i, k)])
                add([KD[i][2], ge(n, i, 0)])  # (a count, or a number not kept, is never below 0)
                add([-KD[i][0], no(ge(n, i, 1))])  # (a number not kept stays 0)
            self.CM[n] = []
            for i, j in self.P:  # (i above j: i is at least k and j is not, for some k; the counts are never below 0)
                lt, eq, gt = c3 = [f.var() for _ in range(3)]
                add(list(c3))
                for x, y in itertools.combinations(c3, 2):
                    add([-x, -y])
                for big, small, out_ in ((i, j, gt), (j, i, lt)):
                    zs = []
                    for k in range(1, d + self.V + 1):
                        add([out_, no(ge(n, big, k)), ge(n, small, k)])
                        z = f.var()
                        add([-z, ge(n, big, k)])
                        add([-z, no(ge(n, small, k))])
                        zs.append(z)
                    add([-out_] + zs)
                self.CM[n].append(c3)
            if n == 0:
                self.SW[n] = [[v == 0 for v in range(k)] for k in S]
                for i in range(N):
                    for v in range(self.V + 1):
                        for k in self.G[n][i]:
                            add([-self.V0[i][v], ge(n, i, k) if k <= v else no(ge(n, i, k))])
            else:
                self.SW[n] = [[f.var() for _ in range(k)] for k in S]
                for s, k in enumerate(S):
                    add(list(self.SW[n][s]))
                    for x, y in itertools.combinations(self.SW[n][s], 2):
                        add([-x, -y])
                p, a = H.parent[n], H.sym[n]
                X = [[f.var() for _ in VALS] for _ in range(N)]
                Y = [f.var() for _ in range(len(W) + 1)] if W else None
                if W:
                    add(list(Y))
                    for x, y in itertools.combinations(Y, 2):
                        add([-x, -y])
                for i in range(N):
                    add(list(X[i]))
                    for x, y in itertools.combinations(X[i], 2):
                        add([-x, -y])
                    by = [Y[k + 1] for k, w in enumerate(W) if i in moved(w)]
                    for k, w in enumerate(W):  # (made what number j was, and moved by one)
                        if w[0] == "copy" and w[1] == i:
                            for j, delta in enumerate(VALS):
                                for kk in range(-d, d + self.V + 1):
                                    u, v = ge(n, i, kk), ge(p, w[2], kk - delta)
                                    add([-Y[k + 1], -X[i][j], no(u), v])
                                    add([-Y[k + 1], -X[i][j], u, no(v)])
                    for j, delta in enumerate(VALS):
                        for k in range(-d, d + self.V + 1):
                            u, v = ge(n, i, k), ge(p, i, k - delta)
                            add(by + [-X[i][j], no(u), v])
                            add(by + [-X[i][j], u, no(v)])
                for k, w in enumerate(W):
                    y, dp, i = Y[k + 1], self.dep[p], w[1]
                    if w[0] == "copy":
                        continue
                    if w[0] != "even":  # (number i is 0 after it is poured or emptied)
                        add([-y, no(ge(n, i, 1))])
                        add([-y, ge(n, i, 0)])
                    if w[0] == "pour":  # (j after = j before + g times i before; i is a count)
                        j, g = w[2], w[3]
                        for kk in range(-d + 1, d + self.V + 1):
                            for t in range(0, dp + self.V + 1):
                                add([-y, no(ge(p, i, t)), ge(p, i, t + 1), no(ge(p, j, kk - g * t)), ge(n, j, kk)])
                                add([-y, no(ge(p, i, t)), ge(p, i, t + 1), ge(p, j, kk - g * t), no(ge(n, j, kk))])
                    elif w[0] == "even":  # (j after = j before - i before, if above 0; and i after = 0 or i - j)
                        j = w[2]
                        for (x, z) in ((i, j), (j, i)):
                            for kk in range(1, d + self.V + 1):
                                for t in range(0, dp + self.V + 1):
                                    add([-y, no(ge(p, x, kk + t)), ge(p, z, t + 1), ge(n, x, kk)])
                                    add([-y, ge(p, x, kk + t), no(ge(p, z, t)), no(ge(n, x, kk))])
                fire = [f.var() for _ in range(ROWS)]
                idle = self.guarded_list(n, p, self.U[a], self.L[a], fire)
                for r in range(ROWS):
                    for i in range(N):
                        for j in range(3):
                            add([-fire[r], -self.D[a][r][i][j], X[i][j]])
                    if W:
                        for k in range(len(W) + 1):
                            add([-fire[r], -self.M[a][r][k], Y[k]])
                    for s, kk in enumerate(S):
                        e, now, was = self.E[a][r][s], self.SW[n][s], self.SW[p][s]
                        for v in range(kk):
                            add([-fire[r], -e[0], no(was[v]), now[v]])
                            add([-fire[r], -e[2 + v], now[v]])
                            for size in range(2, kk + 1):
                                if v < size:
                                    add([-fire[r], -e[1], no(was[v]), no(self.ge_size(s, size)),
                                         self.ge_size(s, size + 1), now[(v + 1) % size]])
                for i in range(N):
                    add([no(idle), X[i][1]])
                if W:
                    add([no(idle), Y[0]])
                for s, kk in enumerate(S):
                    for v in range(kk):
                        add([no(idle), no(self.SW[p][s][v]), self.SW[n][s][v]])
                # (numbers in the order in which they first leave 0, in the order the beginnings were added; a
                # difference leaves 0 upward first: numbers swapped or turned over are the same machine)
                now = []
                for i in range(N):
                    up, down, was = ge(n, i, 1), no(ge(n, i, 0)), self.was[i]
                    add([no(down), was])
                    z = f.var()
                    add([-z, was, up, down])
                    for x in (was, up, down):
                        add([z, no(x)])
                    now.append(z)
                for i in range(N - 1):
                    add([-now[i + 1], now[i]])
                self.was = now
            for s, k in enumerate(S):
                for v in range(1, k):
                    add([no(self.SW[n][s][v]), self.ge_size(s, v + 1)])
            y = H.shown[n]
            if self.numeric:
                y, nums = form(y)
                for s, v in enumerate(nums):
                    for j in range(N):
                        for gi, g in enumerate((1, -1)):
                            x = self.NV[self.labels.index(y)][s][2 * j + gi]
                            u = g * (v - self.off[y][s])
                            add([-x, ge(n, j, u)])
                            add([-x, no(ge(n, j, u + 1))])
            l = self.labels.index(y)
            sv = [[f.var() for _ in range(3)] for _ in range(self.see)]  # (what slot k reads says here)
            for k in range(self.see):
                add(list(sv[k]))
                for x, y in itertools.combinations(sv[k], 2):
                    add([-x, -y])
                add([-self.SL[k][self.T], sv[k][0]])
                for t in range(self.T):
                    for vi in range(len(self.tv[t])):
                        add([-self.SL[k][t]] + [no(y) for y in self.conj(n, t, vi)] + [sv[k][vi]])
            for ci, c in enumerate(self.combos):
                add([-sv[k][c[k]] for k in range(self.see)] + [self.TB[ci][l]])

    def read(self, model, price):
        best = set(x for x in model if x > 0)
        one = lambda xs: next(i for i, x in enumerate(xs) if x in best)
        N, S, T = self.N, self.S, self.T
        kinds = [KINDS[one(self.KD[i])] for i in range(N)]
        sizes = [one(self.SZ[s]) + 1 for s in range(len(S))]

        def guard(Lr):
            g = []
            for t in range(T):
                for vi, x in enumerate(Lr[t]):
                    if x in best:
                        g.append((t, VALS[vi] if t < N or t >= self.Q else vi))
            return g
        rows = []
        for a in range(self.A):
            rs = []
            for r in range(ROWS):
                if self.U[a][r] not in best:
                    break
                d = tuple(VALS[one(self.D[a][r][i])] for i in range(N))
                e = []
                for s in range(len(S)):
                    x = one(self.E[a][r][s])
                    e.append(None if x == 0 else "next" if x == 1 else x - 2)
                w = None if not self.W or self.M[a][r][0] in best else self.W[one(self.M[a][r]) - 1]
                rs.append((guard(self.L[a][r]), d, tuple(e), w))
            rows.append(rs)
        oread = [t for k in range(self.see) for t in [one(self.SL[k])] if t < T]
        otab = {c: self.labels[one(self.TB[ci])] for ci, c in enumerate(self.combos)}
        num = None
        if self.numeric:
            num = {y: [(k // 2, (1, -1)[k % 2], self.off[y][s]) for s in range(y.count("\0"))
                       for k in [one(self.NV[l][s])]] for l, y in enumerate(self.labels)}
        init = tuple(one(self.V0[i]) for i in range(N))
        return ListMachine(self.H.syms, self.key, kinds, sizes, rows, oread, otab, num, init, price)


class Lists:
    """The family of machines of parts said as lists of commands, for guarded.explain, rivals and learn."""

    def keys(self, wordy):
        return [layer + (mode,) for layer in LAYERS for mode in ((True, False) if wordy else (False,))]

    def base(self, key):
        return WHOLE if key[2] else 0

    def cells(self, key):  # (the order among equal bounds: the smaller formulas first)
        return 3 * key[0] + sum(key[1]) + (12 if key[2] else 0) + 3 * len(pairs(key[0], key[3]))

    def second(self, key):
        return bool(key[2])

    def guess(self, key, rival=False):  # (searched by guessing from the start: cores raise the bound of a layer slowly)
        return True

    def numeric(self, key):
        return key[4]

    def say(self, key):
        return (f"parts {key[0]}, {key[1]}{', whole' if key[2] else ''}{', compared' if key[3] else ''}, "
                f"{'number' if key[4] else 'table'}")

    def key(self, m):
        return m.key

    def space(self, H, key, nodes, guess=False, low=0, avoid=None):
        return ListSpace(H, key, nodes, avoid=avoid, guess=guess, low=low)


LISTS = Lists()


def learn(syms, oracle, data, rng, asks=40, deadline=None, H=None, log=lambda *a: None, per=guarded.PER, cap=0,
          short=guarded.SHORT):
    return guarded.learn(syms, oracle, data, rng, asks, deadline, H, log, per, cap, short, family=LISTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--asks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--per", type=float, default=guarded.PER, help="seconds for a layer in a round")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--short", type=int, default=guarded.SHORT)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("family", a.family)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    syms, answer = list(mod.ALPHABET), mod.answer
    oracle = lambda w: str(answer(list(w)))
    rng = random.Random(a.seed)
    data = [(w, oracle(w)) for w in (tuple(rng.choice(syms) for _ in range(rng.randint(1, 4))) for _ in range(300))]
    t0 = time.time()
    m, asked, H = learn(syms, oracle, data, rng, a.asks, t0 + a.minutes * 60, log=(lambda *x: None) if a.quiet else print,
                        per=a.per, short=a.short)
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
