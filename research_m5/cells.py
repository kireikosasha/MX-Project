"""cells.py - the machine of a new family found by a solver, not written by hand.

One learner for every family. A family is only its symbols and an oracle that gives answers; nothing in the code knows
any family. A machine is drawn from a small basis (Krohn and Rhodes 1965: every finite machine is a cascade of groups
and resets): a finite control (k states, a table of moves), and beside it at most two cells, each a whole number
(every symbol adds -B..B, read clipped to -C..C) or a circle (a number modulo m). The offset a symbol gives a cell may
depend on the state of the control (a heading and a position), and the second cell's offset may depend on the value of
the first circle (a carry: the day of the week and the weeks gone by). The answer is a table from what is seen at the
end: the state and the cells, or the cells only. One more form: the states are the answers themselves (a table of
relations, Tarski 1941).

Learning:
  1. Shapes are tried in the order of their length in bits; for each, a SAT solver looks for a machine that answers
     every story taught on rightly (exact identification of an automaton: Biermann and Feldman 1972, Heule and Verwer
     2010; states numbered by a walk in breadth, Ulyantsev, Zakirzyanov and Shalyto 2015; whole numbers in the order
     encoding, Tamura et al. 2009). The first machine found is kept. A shape is given up only when the solver proves
     that no machine of it fits; one it could not settle comes back with more time.
  2. Asking (Angluin 1987): the solver looks for a second machine, of a shape at most 8 bits longer, that also answers
     every story taught on rightly but some other story otherwise; up to three such stories are asked (the shortest,
     the longest, one between). When there is no second machine, twelve longer stories are asked as a check (repeated
     pieces, as in the pumping lemma; plain ones; and ones on which the kept machine gives each of its answers, the
     rarest first). A wrong answer goes back to 1."""
import argparse, itertools, math, random, sys, threading, time
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver

CONFLICTS = 20000  # conflicts of the solver per second of a limit given in seconds (about glucose's rate here)


# ---------------------------------------------------------------- families: symbols and an oracle, nothing else
def turtle(s):
    h = x = y = 0
    for c in s:
        if c == "left": h = (h + 1) % 4
        elif c == "right": h = (h - 1) % 4
        elif c == "step":
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][h]; x += dx; y += dy
    return "xy" + "-0+"[(x > 0) - (x < 0) + 1] + "-0+"[(y > 0) - (y < 0) + 1]


def odometer(s):
    day = weeks = 0
    for c in s:
        if c == "next":
            day += 1
            if day == 7: day, weeks = 0, weeks + 1
        elif c == "back":
            day -= 1
            if day < 0: day, weeks = 6, weeks - 1
    return "d%d w%s" % (day, "-0+"[(weeks > 0) - (weeks < 0) + 1])


def weekday(s):
    return "d%d" % (sum(1 if c == "next" else -1 if c == "back" else 0 for c in s) % 7)


def shells(s):
    p = [0, 1, 2]
    for c in s:
        if c != "noise":
            i, j = int(c[1]), int(c[2]); p[i], p[j] = p[j], p[i]
    return "o%d" % p[0]


def lies(s):
    t = True
    for c in s:
        if c == "says-lies": t = not t
    return "yes" if t else "no"


def latest(s):
    v = "none"
    for c in s:
        if c.startswith("set"): v = c[3:]
    return v


def majority(s):
    d = sum(1 if c == "a" else -1 if c == "b" else 0 for c in s)
    return "-0+"[(d > 0) - (d < 0) + 1]


def firstlast(s):
    return "same" if s[0] == s[-1] else "other"


def parity(s):
    return "odd" if s.count("one") % 2 else "even"


def navigate(s):
    """Steps forward, back, left and right of the walker, turns; the answer: back where it began or not."""
    h = x = y = 0
    for c in s:
        if c == "turn-left": h = (h + 1) % 4
        elif c == "turn-around": h = (h + 2) % 4
        elif c in ("forward", "backward"):
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][h]
            k = 1 if c == "forward" else -1
            x += k * dx; y += k * dy
    return "home" if x == 0 and y == 0 else "away"


FAMILIES = {
    "turtle": (["left", "right", "step", "noise"], turtle),
    "odometer": (["next", "back", "noise"], odometer),
    "weekday": (["next", "back", "noise"], weekday),
    "shells": (["s01", "s12", "s02", "noise"], shells),
    "lies": (["says-truth", "says-lies"], lies),
    "latest": (["setA", "setB", "setC", "noise"], latest),
    "majority": (["a", "b", "noise"], majority),
    "firstlast": (["x", "y", "z"], firstlast),
    "parity": (["one", "zero"], parity),
    "navigate": (["forward", "backward", "turn-left", "turn-around"], navigate),
}


# ---------------------------------------------------------------- shapes and their length in bits
# A cell beside the finite control is ("Z", B, C): a whole number, offsets -B..B, read clipped to -C..C; or ("M", m):
# a number kept modulo m (a circle: days of the week, a heading), every symbol turns it by 0..m-1, read as it is.
# Or ("L", n, B): a level 0..n-1 that begins at a value found with the machine; every symbol moves it by -B..B, held
# at the ends ("never above boil, never below cold"), or sets it to a value; read as it is.
SET = 100  # an operation of a level >= SET sets it to (operation - SET)


def values(cell):
    return range(-cell[2], cell[2] + 1) if cell[0] == "Z" else range(cell[1])


def step(c, z, o):
    """The value of cell c after operation o from value z."""
    if c[0] == "Z":
        return z + o
    if c[0] == "M":
        return (z + o) % c[1]
    return o - SET if o >= SET else max(0, min(c[1] - 1, z + o))


def bits_of(k, cells, nsym, nlab, by_state=True, chain=False):
    b = nsym * k * math.log2(k) if k > 1 else 0.0
    for j, c in enumerate(cells):
        each = math.log2(2 * c[1] + 1) if c[0] == "Z" else math.log2(c[1]) if c[0] == "M" else math.log2(2 * c[2] + 1 + c[1])
        b += nsym * k * each * (cells[0][1] if chain and j else 1) + (math.log2(c[1]) if c[0] == "L" else 0)
    reads = k if by_state else 1
    for c in cells:
        reads *= len(values(c))
    return b + reads * math.log2(max(nlab, 2))


def shapes(nsym, nlab, most_k=12):
    Z = [("Z", B, C) for B in (1, 2) for C in (1, 2)]
    M = [("M", m) for m in range(2, 13)]
    L = [("L", n, B) for n in range(3, 9) for B in (1, 2)]
    out = [(k, ()) for k in range(1, most_k + 1)]
    out += [(k, (c,)) for k in range(1, 7) for c in Z + M + L]
    two = [(z, z) for z in Z] + [(z, m) for z in Z for m in M if m[1] <= 7] + \
          [(m1, m2) for i, m1 in enumerate(M) for m2 in M[i:] if m2[1] <= 7] + \
          [(l, l) for l in L if l[1] <= 6] + [(l, m) for l in L for m in M if m[1] <= 4]
    out += [(k, cs) for k in range(1, 5) for cs in two]
    # the answer is read from the state and the numbers, or (a cascade read at its last cell) from the numbers only
    every = [(bits_of(k, cs, nsym, nlab), k, cs, True, False) for k, cs in out] + \
            [(bits_of(k, cs, nsym, nlab, False), k, cs, False, False) for k, cs in out if k > 1 and cs]
    # a cascade of cells: the second number is moved by the value of a circle before it (a carry)
    chained = [(m, x) for m in M if m[1] <= 7 for x in Z + [n for n in M if n[1] <= 7]] + \
              [(l, x) for l in L for x in Z]  # (a level moving a number: rain counted as snow when it is cold)
    every += [(bits_of(k, cs, nsym, nlab, by, True), k, cs, by, True) for k in range(1, 5) for cs in chained
              for by in (True, False) if by or k > 1]
    # the states are the answers themselves, and one more to begin in: a table of relations (Tarski 1941), where
    # "what is X to Z" follows from "what is X to Y" and the next link
    k = nlab + 1
    every.append((nsym * k * math.log2(k), k, (), "named", False))
    return sorted(every)


def name(shape):
    _, k, cells, by_state, chain = shape
    if by_state == "named":
        return f"{k - 1} states named by the answers and one to begin in"
    words = [f"{k} states"]
    for c in cells:
        words.append(f"a whole number (offsets -{c[1]}..{c[1]}, read clipped to -{c[2]}..{c[2]})" if c[0] == "Z"
                     else f"a circle of {c[1]}" if c[0] == "M" else f"a level of {c[1]} (steps -{c[2]}..{c[2]}, held at the ends, or set)")
    return " + ".join(words) + (", the second moved by the value of the first" if chain else "") + \
        ("" if by_state or not cells else ", the answer read from the numbers only")


# ---------------------------------------------------------------- the machine, as found
class Machine:
    lex = None
    sure = None  # True: it passed the last check; False: the questions ran out before it did; None: no check

    def __init__(self, shape, syms, labels, tau, dl, lam):
        self.shape, self.k, self.cells, self.syms, self.labels = shape, shape[1], shape[2], syms, labels
        self.by_state, self.chain = shape[3], shape[4]
        self.start = self.k - 1 if self.by_state == "named" else 0
        self.tau, self.dl, self.lam = tau, dl, lam
        self.zstart = [0] * len(self.cells)

    def run(self, seq):
        if self.lex is not None:
            if any(a not in self.lex for a in seq):
                return None
            seq = [self.lex[a] for a in seq]
        q, z = self.start, list(self.zstart)
        for a in seq:
            w = z[0] if self.chain else 0
            for j, c in enumerate(self.cells):
                z[j] = step(c, z[j], self.dl[a, q, j, w if j else 0])
            q = self.tau[a, q]
        r = tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(self.cells, z))
        return self.labels[self.lam[(q if self.by_state else 0,) + r]]

    def __str__(self):
        return name(self.shape)

    def table(self):
        lines = []
        for a in self.syms:
            moves = " ".join(f"{q}>{self.tau[a, q]}" for q in range(self.k))
            offs = " ".join("(" + ",".join(str(self.dl[a, q, j, 0]) for j in range(len(self.cells))) + ")" for q in range(self.k))
            lines.append(f"    {a:12s} states {moves}" + (f"   numbers by state {offs}" if self.cells else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------- the solver: plain SAT (Heule and Verwer 2010)
class Encoding:
    """Boolean variables: the move of every (symbol, state), the offset of every (cell, symbol, state), the answer of
    every read; and for every node of the tree of stories (stories share their beginnings) its state and the value of
    every cell. Only what a node must be is implied; a machine read off the model is checked again in plain Python."""

    def __init__(self, shape, syms, labels, lexicon=0):
        self.shape, self.syms, self.labels = shape, syms, labels
        self.raw = None
        if lexicon:  # the symbols met are read into a few symbols of the machine; the reading is found with it
            self.raw, self.syms = list(syms), [f"<{i}>" for i in range(lexicon)]
            syms = self.syms
        _, self.k, self.cells, self.by_state, self.chain = shape
        self.ids, self.clauses = IDPool(), []
        self.reads = list(itertools.product(range(self.k) if self.by_state else [0], *[values(c) for c in self.cells]))
        k = self.k
        if self.raw is not None:
            m = len(self.syms)
            for i, r in enumerate(self.raw):
                self.exactly([self.v("w", r, s) for s in range(m)])
                for s in range(m):  # u: symbol s of the machine is used by one of the first i + 1 symbols met
                    u, w = self.v("u", i, s), self.v("w", r, s)
                    self.clauses.append([-w, u])
                    if i:
                        self.clauses += [[-self.v("u", i - 1, s), u], [-u, self.v("u", i - 1, s), w]]
                    else:
                        self.clauses.append([-u, w])
                    if s:  # a symbol of the machine is first used after the one before it
                        self.clauses.append([-w, self.v("u", i - 1, s - 1)] if i else [-w])
        for a in syms:
            for q in range(k):
                self.exactly([self.v("t", a, q, r) for r in range(k)])
                for j, c in enumerate(self.cells):
                    for w in self.keys(j):
                        self.exactly([self.v("o", j, a, q, w, o) for o in self.offsets(c)])
        for r in self.reads:
            self.exactly([self.v("l", r, y) for y in range(len(labels))])
        if self.by_state == "named":  # the states are told apart by their names: nothing to break
            for q in range(len(labels)):
                self.clauses.append([self.v("l", (q,), q)])
        else:
            self.symmetry()
        self.node = {(): 0}
        self.depth = {(): 0}
        self.ndepth = {0: 0}
        self.clauses.append([self.v("x", 0, self.k - 1 if self.by_state == "named" else 0)])
        for j, c in enumerate(self.cells):
            if c[0] == "M":
                self.clauses.append([self.v("z", 0, j, 0)])
            elif c[0] == "L":  # where a level begins is found with the machine
                self.exactly([self.v("z", 0, j, val) for val in range(c[1])])

    def v(self, *key):
        return self.ids.id(key)

    def symmetry(self):
        """One machine, one numbering. States are numbered in the order a walk in breadth from state 0 meets them,
        symbols tried in their order (Ulyantsev, Zakirzyanov and Shalyto 2015). A whole number may be turned over
        (x -> -x) and a circle turned by any of its units (x -> u x): the first offset is kept non-negative, or a
        divisor of the circle; two cells of one kind are kept in order by it."""
        k, S, v, C = self.k, self.syms, self.v, self.clauses
        for j in range(1, k):
            for i in range(j):
                tt = v("tt", i, j)  # some symbol leads from i to j
                C.append([-tt] + [v("t", a, i, j) for a in S])
                for a in S:
                    C.append([tt, -v("t", a, i, j)])
                p = v("p", j, i)  # i is the first state with a move to j
                C.append([-p, tt])
                for m in range(i):
                    C.append([-p, -v("tt", m, j)])
                C.append([p, -tt] + [v("tt", m, j) for m in range(i)])
                for n, a in enumerate(S):  # a is the first symbol that leads from i to j
                    mm = v("mm", a, i, j)
                    C.append([-mm, v("t", a, i, j)])
                    for b in S[:n]:
                        C.append([-mm, -v("t", b, i, j)])
                    C.append([mm, -v("t", a, i, j)] + [v("t", b, i, j) for b in S[:n]])
            C.append([v("p", j, i) for i in range(j)])
        for j in range(1, k - 1):
            for i in range(j):
                for m in range(i):
                    C.append([-v("p", j, i), -v("p", j + 1, m)])
                for n, a in enumerate(S):
                    for b in S[:n]:
                        C.append([-v("p", j, i), -v("p", j + 1, i), -v("mm", a, i, j), -v("mm", b, i, j + 1)])
        a0 = S[0]
        for j, c in enumerate(self.cells):
            if c[0] == "Z":
                for o in range(-c[1], 0):
                    C.append([-v("o", j, a0, 0, 0, o)])
            elif c[0] == "M":
                for o in range(1, c[1]):
                    if c[1] % o:
                        C.append([-v("o", j, a0, 0, 0, o)])
        for j in range(1, len(self.cells)):
            if self.cells[j] == self.cells[j - 1]:
                for o1 in self.offsets(self.cells[j - 1]):
                    for o2 in self.offsets(self.cells[j]):
                        if o2 < o1:
                            C.append([-v("o", j - 1, a0, 0, 0, o1), -v("o", j, a0, 0, 0, o2)])

    def keys(self, j):
        """The values of the first cell an offset of cell j depends on (only one key unless chained)."""
        return range(self.cells[0][1]) if self.chain and j else range(1)

    @staticmethod
    def offsets(c):
        if c[0] == "L":
            return list(range(-c[2], c[2] + 1)) + [SET + v for v in range(c[1])]
        return range(-c[1], c[1] + 1) if c[0] == "Z" else range(c[1])

    def span(self, c, depth):
        return range(-c[1] * depth, c[1] * depth + 1) if c[0] == "Z" else range(c[1])

    def exactly(self, lits):
        self.clauses.append(lits)
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                self.clauses.append([-lits[i], -lits[j]])

    def ge(self, n, j, c):
        """A whole number in the order encoding (Tamura et al. 2009): the literal 'the value at node n is at least c',
        or True / False where the depth of the node already decides it."""
        D = self.cells[j][1] * self.ndepth[n]
        if c <= -D:
            return True
        if c > D:
            return False
        return self.v("g", n, j, c)

    def add(self, lits):
        """A clause whose literals may be True or False."""
        if any(l is True for l in lits):
            return
        self.clauses.append([l for l in lits if l is not False])

    def at(self, seq):
        if seq in self.node:
            return self.node[seq]
        n0 = self.at(seq[:-1])
        a, n, dep = seq[-1], len(self.node), self.depth[seq[:-1]]
        self.node[seq], self.depth[seq], self.ndepth[n] = n, dep + 1, dep + 1
        neg = lambda l: (not l) if isinstance(l, bool) else -l
        read = [(a, [])] if self.raw is None else [(s, [-self.v("w", a, i)]) for i, s in enumerate(self.syms)]
        for q in range(self.k):
            x = self.v("x", n0, q)
            for a, guard in read:
                for r in range(self.k):
                    self.clauses.append([-x, -self.v("t", a, q, r), self.v("x", n, r)] + guard)
            for j, c in enumerate(self.cells):
                for w in self.keys(j):
                  for a, guard in read:
                    key = ([-self.v("z", n0, 0, w)] if self.chain and j else []) + guard
                    for o in self.offsets(c):
                        off = self.v("o", j, a, q, w, o)
                        if c[0] == "Z":
                            D = c[1] * (dep + 1)
                            for b in range(-D + 1, D + 1):
                                self.add([-x, -off, neg(self.ge(n0, j, b - o)), self.ge(n, j, b)] + key)
                                self.add([-x, -off, self.ge(n0, j, b - o), neg(self.ge(n, j, b))] + key)
                        else:
                            for val in range(c[1]):
                                self.clauses.append([-x, -off, -self.v("z", n0, j, val), self.v("z", n, j, step(c, val, o))] + key)
        for j, c in enumerate(self.cells):
            if c[0] == "Z":
                D = c[1] * (dep + 1)
                for b in range(-D + 1, D):
                    self.add([neg(self.ge(n, j, b + 1)), self.ge(n, j, b)])
        return n

    def read(self, seq):
        """For every read: the literals whose falsity puts a node at that read (through clipped values), or None
        where the read cannot be reached from this node."""
        n = self.at(tuple(seq))
        neg = lambda l: (not l) if isinstance(l, bool) else -l
        lits = {}
        for r in self.reads:
            out = [-self.v("x", n, r[0])] if self.by_state else []
            for j, c in enumerate(self.cells):
                v = r[1 + j]
                if c[0] in "ML":
                    out.append(-self.v("z", n, j, v))
                    continue
                conj = [neg(self.ge(n, j, -c[2] + 1))] if v == -c[2] else [self.ge(n, j, c[2])] if v == c[2] else \
                    [self.ge(n, j, v), neg(self.ge(n, j, v + 1))]
                for l in conj:
                    if l is False:
                        out = None
                        break
                    if l is not True:
                        out.append(-l)
                if out is None:
                    break
            lits[r] = out
        return n, lits

    def says(self, seq, y, excused=None):
        """The story is answered y; or, given a literal, it may be excused."""
        _, lits = self.read(seq)
        for r, ls in lits.items():
            if ls is not None:
                self.clauses.append(ls + [self.v("l", r, y)] + ([excused] if excused else []))

    def free(self, seq):
        """An answer variable for a story whose answer is not given."""
        n, lits = self.read(seq)
        ans = [self.v("a", n, y) for y in range(len(self.labels))]
        for i in range(len(ans)):
            for j in range(i + 1, len(ans)):
                self.clauses.append([-ans[i], -ans[j]])
        for r, ls in lits.items():
            if ls is None:
                continue
            for y in range(len(self.labels)):
                self.clauses.append(ls + [-self.v("l", r, y), ans[y]])
        return ans

    def solve(self, seconds):
        """The solver's limit is a number of conflicts (CONFLICTS a second of the old limit), not time: the same work on
        any machine, so the same result. A wall clock far beyond it only guards against a hang."""
        with Solver(name="glucose4", bootstrap_with=self.clauses) as s:
            s.conf_budget(max(1000, int(seconds * CONFLICTS)))
            timer = threading.Timer(20 * seconds + 60, s.interrupt)
            timer.start()
            ok = s.solve_limited(expect_interrupt=True)
            timer.cancel()
            self.used = s.accum_stats().get("conflicts", 0)
            if ok is None:
                return None, "unknown"
            if not ok:
                return None, "unsat"
            model = set(l for l in s.get_model() if l > 0)
        on = lambda *key: self.ids.obj2id.get(key) in model
        tau = {(a, q): next(r for r in range(self.k) if on("t", a, q, r)) for a in self.syms for q in range(self.k)}
        dl = {(a, q, j, w): next(o for o in self.offsets(c) if on("o", j, a, q, w, o))
              for a in self.syms for q in range(self.k) for j, c in enumerate(self.cells) for w in self.keys(j)}
        lam = {r: next(y for y in range(len(self.labels)) if on("l", r, y)) for r in self.reads}
        m = Machine(self.shape, self.syms, self.labels, tau, dl, lam)
        m.zstart = [next(v for v in range(c[1]) if on("z", 0, j, v)) if c[0] == "L" else 0 for j, c in enumerate(self.cells)]
        if self.raw is not None:
            m.lex = {r: self.syms[next(i for i in range(len(self.syms)) if on("w", r, i))] for r in self.raw}
        return m, "sat"


def fit(shape, syms, labels, data, seconds, other=None, pool=(), excuse=0.0, lexicon=0, weights=None, most=None):
    """A machine of this shape that answers every story of data rightly (and, given another machine and a pool of
    stories, answers some story of the pool otherwise than it). Solved first on a part of the stories; the stories the
    machine found answers wrongly are added and it is solved again (a 'no' on a part is a 'no' on all).
    excuse (a share of the stories) or most (a number): so many stories may be answered otherwise than the set; with
    weights (story -> how many times it is told), a story counts as many times as it is told."""
    lab = {l: i for i, l in enumerate(labels)}
    rng = random.Random(len(data))
    if most is None:  # stories the machine may answer otherwise than the set (its errors, or the set's)
        most = int(excuse * (sum(weights.get(s, 1) for s, _ in data) if weights else len(data)))
    part = list(data) if most else rng.sample(data, min(40, len(data)))
    left = seconds  # (in seconds of the old limit, spent by the conflicts each solving used)
    while True:
        e = Encoding(shape, syms, labels, lexicon)
        if most:
            ex = [e.v("excused", i) for i in range(len(part))]
            for (seq, y), x in zip(part, ex):
                e.says(seq, lab[y], x)
            lits = [x for (seq, _), x in zip(part, ex) for _ in range(weights.get(seq, 1) if weights else 1)]
            e.clauses += CardEnc.atmost(lits=lits, bound=most, vpool=e.ids, encoding=EncType.seqcounter).clauses
        else:
            for seq, y in part:
                e.says(seq, lab[y])
        if other is not None:
            e.clauses.append([-e.free(w)[lab[other.run(w)]] for w in pool])
        m, how = e.solve(max(0.05, left))
        left -= e.used / CONFLICTS
        if m is None:
            return None, how
        wrong = [(s, y) for s, y in data if m.run(s) != y]
        if sum(weights.get(s, 1) if weights else 1 for s, _ in wrong) <= most:
            return m, "sat"
        if left <= 0:
            return None, "unknown"
        part += rng.sample(wrong, min(20, len(wrong)))


def fit_greedy(shape, syms, labels, data, seconds, lexicon=0, start=200, batch=10, log=print):
    """For stories some of which no machine can explain (errors of the set, or of the way found between the two
    asked about): a machine is found for a part; of the stories it answers wrongly, a few are added; if no machine
    explains them all, they are tried one by one, and a story that cannot be added is set aside. Returns the machine
    and the stories set aside."""
    rng = random.Random(len(data))
    part = rng.sample(data, min(start, len(data)))
    aside = []
    m, how = fit(shape, syms, labels, part, seconds, lexicon=lexicon)
    if m is None:
        return None, how
    while True:
        wrong = [x for x in data if x not in aside and m.run(x[0]) != x[1]]
        log(f"  {len(part)} stories explained, {len(aside)} set aside, {len(wrong)} still answered wrongly")
        if not wrong:
            return m, aside
        more = rng.sample(wrong, min(batch, len(wrong)))
        got, how = fit(shape, syms, labels, part + more, seconds, lexicon=lexicon)
        if got is not None:
            m, part = got, part + more
            continue
        for x in more:  # one by one
            got, how = fit(shape, syms, labels, part + [x], seconds, lexicon=lexicon)
            if got is not None:
                m, part = got, part + [x]
            else:
                aside.append(x)


def rival(shape, syms, labels, data, machine, pool, seconds):
    """A machine of this shape that answers every story taught on rightly and some story of the pool otherwise; the
    stories on which the two part, at most three: the shortest, the longest and one between."""
    other, how = fit(shape, syms, labels, data, seconds, machine, pool)
    if other is None:
        return how  # "unsat": there is no such machine; "unknown": the solver could not tell
    differs = sorted({w for w in pool if other.run(w) != machine.run(w)}, key=lambda w: (len(w), w))
    if not differs:
        return "unsat"
    return [differs[0]] + ([differs[-1]] if len(differs) > 1 else []) + ([differs[len(differs) // 2]] if len(differs) > 2 else [])


def pumped(syms, rng, lo, hi, n):
    """Stories made by repeating a short piece (the pumping lemma): where a counter and a finite imitation of it part."""
    out = set()
    while len(out) < n:
        u = tuple(rng.choice(syms) for _ in range(rng.randint(1, 3)))
        v = tuple(rng.choice(syms) for _ in range(rng.randint(0, 2)))
        reps = rng.randint(max(1, lo // len(u)), max(1, hi // len(u)))
        out.add(v + u * reps)
    return sorted(out)  # (a set's order changes from one run of Python to the next: sorted, it does not)


def _call(job):
    f, args = job
    return f(*args)


def side(f, jobs, workers):
    """Run f on every argument tuple, several processes at once; the results in the order of the jobs."""
    if workers <= 1 or len(jobs) == 1:
        return [f(*j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as p:
        return p.map(_call, [(f, j) for j in jobs])


def learn(syms, oracle, data, rng, asks=40, seconds=60, longest=4, quiet=False, most_k=12, slack=8, pool=None,
          workers=4, shorten=False, reserve=0):
    labels = sorted({y for _, y in data})
    asked, dead, fixed, tries = 0, set(), pool, {}
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    machine = shape = None
    while True:
        new = sorted(set(labels) | {y for _, y in data})
        if new != labels:
            labels, dead, tries, machine = new, set(), {}, None
        every = shapes(len(syms), len(labels), most_k)
        if machine is not None and all(machine.run(s) == y for s, y in data):
            pass  # the kept machine answered what was asked rightly: it stays
        else:
            machine = shape = None
            t0 = time.time()
            for cap in (2, 5):  # a shape the solver gave up on comes back with twice the work; if nothing at all is
                live = [sh for sh in every if sh[1:] not in dead and tries.get(sh[1:], 0) <= cap]   # found, up to 32 times
                for i in range(0, len(live), workers):  # the shapes in order of their length, several at once
                    part = live[i:i + workers]
                    got = side(fit, [(sh, syms, labels, data, seconds * 2 ** tries.get(sh[1:], 0)) for sh in part], workers)
                    for sh, (m, how) in zip(part, got):
                        if m:
                            machine, shape = m, sh
                            break
                        if how == "unsat":  # only a proved 'no' is final
                            dead.add(sh[1:])
                        else:
                            tries[sh[1:]] = tries.get(sh[1:], 0) + 1
                            log(f"  {name(sh)}: the solver gave up after {int(seconds * 2 ** (tries[sh[1:]] - 1) * CONFLICTS)} conflicts")
                    if machine:
                        log(f"  kept {machine} ({shape[0]:.0f} bits, {time.time() - t0:.1f} s); {len(data)} stories, {asked} asked")
                        break
                if machine:
                    break
        if machine is None:
            return None, asked
        if asked >= asks:  # the questions are spent and the last check was not passed: not sure
            machine.sure = False
            return machine, asked
        # a machine of this shape, or of one at most `slack` bits longer, that answers every story taught on rightly
        # and some other story otherwise: ask that story
        seen = {s for s, _ in data}
        if fixed is None:
            ask_from = sorted({tuple(rng.choice(syms) for _ in range(rng.randint(1, 2 * longest))) for _ in range(100)})
            ask_from += sorted({tuple(rng.choice(syms) for _ in range(rng.randint(2 * longest, 4 * longest))) for _ in range(30)})
            ask_from += pumped(syms, rng, 2 * longest, 4 * longest, 40)
        else:
            ask_from = rng.sample(fixed, min(150, len(fixed)))
        ask_from = [w for w in ask_from if w not in seen]
        t0, w = time.time(), None
        near = [sh for sh in every if shape[0] <= sh[0] <= shape[0] + slack and sh[1:] not in dead]
        if reserve and asks - asked <= reserve:  # the last questions are kept for the check
            near = []
        unsettled = []
        for more in (1, 4):  # a shape on which the solver could not tell is searched again with four times the work
            for i in range(0, len(near), workers):
                got = side(rival, [(sh, syms, labels, data, machine, ask_from, seconds * more) for sh in near[i:i + workers]], workers)
                hit = [(sh, x) for sh, x in zip(near[i:i + workers], got) if isinstance(x, list)]
                unsettled += [sh for sh, x in zip(near[i:i + workers], got) if x == "unknown"]
                if hit:
                    sh, w = hit[0]
                    break
            if w is not None or not unsettled:
                break
            near, unsettled = unsettled, []
        if w is None and unsettled:
            log(f"  of {len(unsettled)} shapes the solver could not tell whether a second machine is there")
        if w is not None:
            told = [(x, oracle(list(x))) for x in w[:max(1, asks - asked)]]
            asked += len(told)
            wrong = sum(machine.run(x) != y for x, y in told)
            log(f"  {name(sh)} answers otherwise on {len(w)} stories ({time.time() - t0:.1f} s): asked "
                f"{' | '.join(' '.join(x) + ' = ' + str(y) for x, y in told)}; the kept machine was wrong on {wrong}")
            data = data + told
            continue
        if fixed is not None:  # nothing longer can be asked
            return machine, asked
        # the check by trying: a few longer stories, pumped and plain
        tried = [tuple(rng.choice(syms) for _ in range(rng.randint(3 * longest, 6 * longest))) for _ in range(12)] + \
            pumped(syms, rng, 4 * longest, 8 * longest, 3)
        # and stories on which the kept machine gives every answer it can give, the rarest first: where an answer is
        # rare among long stories, plain trying would almost never ask about it
        long = [tuple(rng.choice(syms) for _ in range(rng.randint(3 * longest, 6 * longest))) for _ in range(400)]
        long += pumped(syms, rng, 3 * longest, 6 * longest, 200)
        by = {}
        for x in long:
            by.setdefault(machine.run(x), []).append(x)
        tried += [by[y][0] for y in sorted(by, key=lambda y: len(by[y]))[:6]]
        tried = tried[:asks - asked]  # never more than the questions allowed
        if not tried:
            return machine, asked
        told = [(w, oracle(list(w))) for w in tried]
        asked += len(tried)
        wrong = [(w, y) for w, y in told if machine.run(w) != y]
        if not wrong:
            log(f"  no other machine within {slack} bits; {len(tried)} longer stories asked, all answered rightly ({time.time() - t0:.1f} s)")
            machine.sure = True
            return machine, asked
        log(f"  no other machine within {slack} bits; of {len(tried)} longer stories asked, {len(wrong)} answered wrongly")
        if not shorten:
            data = data + told
            continue
        # a long story answered wrongly is not put into the stories taught on as it is (every formula after would
        # carry it): the beginning where the machine first goes wrong is found by halving (Rivest and Schapire 1993)
        known = dict(data)
        for w, y in sorted(wrong, key=lambda x: len(x[0]))[:2]:
            lo, hi = 0, len(w)
            known[w] = y
            while hi - lo > 1 and asked < asks:
                mid = (lo + hi) // 2
                if w[:mid] not in known:
                    known[w[:mid]] = oracle(list(w[:mid]))
                    asked += 1
                if machine.run(w[:mid]) != known[w[:mid]]:
                    hi = mid
                else:
                    lo = mid
            data = data + [(w[:hi], known[w[:hi]])]
            log(f"  of a story of {len(w)} the machine goes wrong at its first {hi}")


def clutrr_chains(path):
    """CLUTRR given as a graph: the chain of links from the first person of the question to the second. A link is
    the word of the set (the second person is the first one's ...); a link walked against its direction gets ^-1 and
    the sex of the person it reaches. Nothing else is said of the words."""
    import ast, csv
    from collections import defaultdict
    out = []
    for r in csv.DictReader(open(path)):
        se, ty, q = (ast.literal_eval(r[x]) for x in ("story_edges", "edge_types", "query_edge"))
        sex = [x.split(":")[1] for x in r["genders"].split(",")]
        adj = defaultdict(list)
        for (x, y), w in zip(se, ty):
            adj[x].append((y, w)); adj[y].append((x, w + "^-1"))
        prev, todo = {q[0]: None}, [q[0]]
        while todo:
            x = todo.pop(0)
            for y, w in adj[x]:
                if y not in prev:
                    prev[y] = (x, w); todo.append(y)
        if q[1] not in prev:
            continue
        seq, x = [], q[1]
        while prev[x]:
            px, w = prev[x]
            seq.append(w + (":" + sex[x] if w.endswith("^-1") else "")); x = px
        out.append((tuple(reversed(seq)), r["target"], len(ty)))
    return out


def clutrr(a):
    import glob
    from collections import Counter, defaultdict
    train = clutrr_chains(glob.glob(a.clutrr + "/*_train.csv")[0])
    said = defaultdict(Counter)
    for s, y, _ in train:
        said[s][y] += 1
    every = sorted((s, c.most_common(1)[0][0]) for s, c in said.items())  # the answer most often given
    syms = sorted({x for s, _ in every for x in s})
    rng = random.Random(a.seed)
    data = rng.sample(every, min(a.teach, len(every)))
    known = dict(every)
    oracle = lambda w: known.get(tuple(w))
    t0 = time.time()
    answer = None
    if a.named:
        # the form given (the states are the answers), all chains. Stories no machine of the form explains (the set
        # tells some chains with different answers): the fewest, each counted as often as it is told.
        labels = sorted({y for _, y in every})
        sh = [x for x in shapes(len(syms), len(labels)) if x[3] == "named"][0]
        weights = {s: sum(c.values()) for s, c in said.items()}
        data = list(every)
        if a.hint:  # a link whose word is a word of the answers: standing alone, it answers itself (stage 42's hint)
            data += [((x,), x) for x in syms if x in labels]
        print(f"CLUTRR (graph): {len(train)} chains of 2-3 links, {len(every)} different; {len(syms)} kinds of link; "
              f"the form given: {name(sh)}; taught on all {len(every)}" + (f", and {len(data) - len(every)} links "
              f"that are words of the answers, alone" if a.hint else ""), flush=True)
        lo, hi, m = 0, 0, None
        while m is None:  # doubled until a machine is found, then halved between
            m, how = fit(sh, syms, labels, data, a.seconds * 30, weights=weights, most=hi)
            if m is None:
                lo, hi = hi + 1, max(1, 2 * hi)
        while lo < hi:
            mid = (lo + hi) // 2
            got, how = fit(sh, syms, labels, data, a.seconds * 30, weights=weights, most=mid)
            if got is not None:
                hi, m = mid, got
            else:
                lo = mid + 1
        if m is None or hi != lo:
            m, how = fit(sh, syms, labels, data, a.seconds * 30, weights=weights, most=lo)
        if m is None:
            print("  nothing found"); return
        print(f"  the fewest stories answered otherwise than the set, counted with repeats: {lo} of {sum(weights.values())}")
        for s, y in data:
            if m.run(s) != y:
                print(f"    {' '.join(s):45s} the set: {y:15s} the machine: {m.run(s)}  (told {weights.get(s, 1)} times)")
        if a.chart:
            # the table read by a chart: every bracketing of the chain, each join only through a move the stories
            # taught on went through; the answer most bracketings give (stage 38 read its hand-written table so)
            seen = set()
            for s, y in data:
                q = m.start
                for x in s:
                    seen.add((x, q))
                    q = m.tau[x, q]
            word = lambda q: m.labels[q] if q < len(m.labels) else None

            def answer(s):
                n, val = len(s), {}
                for i in range(n):
                    val[i, i + 1] = Counter([m.tau[s[i], m.start]] if (s[i], m.start) in seen else [])
                for w in range(2, n + 1):
                    for i in range(n - w + 1):
                        c = Counter()
                        for j in range(i + 1, i + w):
                            for p, cp in val[i, j].items():
                                for q, cq in val[j, i + w].items():
                                    if word(q) in syms and (word(q), p) in seen:
                                        c[m.tau[word(q), p]] += cp * cq
                        val[i, i + w] = c
                return word(val[0, n].most_common(1)[0][0]) if val[0, n] else m.run(s)
    else:
        print(f"CLUTRR (graph): {len(train)} chains of 2-3 links, {len(every)} different; {len(syms)} kinds of link; "
              f"taught on {len(data)}, may ask {a.asks} more of them", flush=True)
        m, asked = learn(syms, oracle, data, rng, a.asks, a.seconds, 3, pool=[s for s, _ in every], most_k=a.most_k,
                         workers=a.workers)
        if m is None:
            print("  nothing found"); return
        print(f"  kept {m}; asked {asked} ({time.time() - t0:.0f} s)")
    allr = alln = 0
    for f in sorted(glob.glob(a.clutrr + "/*_test.csv"), key=lambda f: tuple(int(x) for x in f.split("/")[-1].split("_")[0].split("."))):
        test = clutrr_chains(f)
        right = sum(all(x in syms for x in s) and (answer or m.run)(s) == y for s, y, _ in test)
        allr += right; alln += len(test)
        print(f"  {f.split('/')[-1]:14s} {len(test):4d} chains, links {test[0][2]}: {100 * right / len(test):5.1f} %", flush=True)
    print(f"  all tests: {100 * allr / alln:.1f} %")
    if a.table:
        print(m.table())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--teach", type=int, default=300, help="stories taught on")
    ap.add_argument("--longest", type=int, default=4, help="the longest story taught on")
    ap.add_argument("--asks", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=10, help="the solver's limit for one shape")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--clutrr", help="a folder of CLUTRR given as a graph (story_edges, edge_types)")
    ap.add_argument("--most-k", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--named", action="store_true", help="CLUTRR: give the form whose states are the answers")
    ap.add_argument("--hint", action="store_true", help="CLUTRR --named: a link that is a word of the answers answers itself, alone")
    ap.add_argument("--chart", action="store_true", help="CLUTRR --named: the table read by a chart, through moves seen in teaching")
    a = ap.parse_args()
    if a.clutrr:
        return clutrr(a)
    for name in a.families:
        syms, oracle = FAMILIES[name]
        rng = random.Random(a.seed)
        data = []
        for _ in range(a.teach):
            w = tuple(rng.choice(syms) for _ in range(rng.randint(1, a.longest)))
            data.append((w, oracle(list(w))))
        print(f"{name}: {len(syms)} symbols, {a.teach} stories of 1-{a.longest} symbols", flush=True)
        t0 = time.time()
        m, asked = learn(syms, oracle, data, rng, a.asks, a.seconds, a.longest, workers=a.workers)
        if m is None:
            print(f"  nothing found up to the largest shape ({time.time() - t0:.0f} s)\n", flush=True)
            continue
        test = {}
        for L in (20, 100, 500):
            tr = random.Random(1000 + L)
            seqs = [[tr.choice(syms) for _ in range(L)] for _ in range(300)]
            test[L] = 100 * sum(m.run(s) == oracle(s) for s in seqs) / len(seqs)
        print(f"  {name}: {m}; asked {asked}; exact on length 20 / 100 / 500: "
              f"{test[20]:.1f} / {test[100]:.1f} / {test[500]:.1f} %  ({time.time() - t0:.0f} s)", flush=True)
        if a.table:
            print(m.table())
        print(flush=True)


if __name__ == "__main__":
    main()
