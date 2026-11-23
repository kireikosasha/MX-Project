"""cells.py - the machine of a new family found by a solver, not written by hand.

One learner for every family. A family is only its symbols and an oracle that gives answers; nothing in the code knows
any family. A machine is drawn from a small basis (Krohn and Rhodes 1965: every finite machine is a cascade of groups
and resets): a finite control (k states, a table of moves), and beside it at most two cells, each a whole number
(every symbol adds -B..B, read clipped to -C..C) or a circle (a number modulo m). The offset a symbol gives a cell may
depend on the state of the control (a heading and a position), and the second cell's offset may depend on the value of
the first circle (a carry: the day of the week and the weeks gone by). The answer is a table from what is seen at the
end: the state and the cells, or the cells only. One more form: the states are the answers themselves (a table of
relations, Tarski 1941). With dep, a shape is a body (the states and the cells) and a ceiling of its price: which moves
depend on the state, or on what the first cell reads, and which parts the answer reads, is found with the machine and
priced by the rule (see ceilings).

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
import argparse, functools, itertools, math, random, sys, threading, time
from pysat.card import CardEnc, EncType, ITotalizer
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
MIRROR = True  # a level turned end for end is the same machine: only one of the two is searched
INCREMENTAL = True  # fit keeps one encoding and one solver while it adds stories
ENGINE = "glucose4"  # the solver (any of pysat's that takes a limit of conflicts)
ZLONG = 0  # (warm) a shape with a whole number is first told only the stories of at most so many symbols (0: all):
# the formula of a whole number grows with the square of a story's length; a longer story is told when the machine
# found answers it wrongly, the shortest first
LONG = 32  # (far) a story heard that is longer is not put into the formulas: the machine is only checked on it
INERT = True  # (anchor) of the machines that explain the stories, the one where something happens in the fewest places
ALIKE = False  # (mealy) of the machines that explain the stories, the one where the fewest moves with one answer lead apart
MEALY_K = 20  # (mealy) a plain control of at most so many states
HIDDEN = 4  # (anchor) the answers as states, each with at most so many hidden values
PART = 40  # fit begins with so many of the stories, and adds those its machine answers wrongly
WORK = [0]  # the conflicts spent so far, in this process (what the patience of a learner is counted in)
DEPTHS = (4, 7, 12)  # (warm) a second machine is looked for on stories of at most so many symbols first, then of `build`
DOUBT = True  # a shape on which the solver could not tell whether a second machine is there leaves the machine unsure


def values(cell):
    return range(-cell[2], cell[2] + 1) if cell[0] == "Z" else range(cell[1])


def step(c, z, o):
    """The value of cell c after operation o from value z."""
    if c[0] == "Z":
        return z + o
    if c[0] == "M":
        return (z + o) % c[1]
    return o - SET if o >= SET else max(0, min(c[1] - 1, z + o))


def each(c):
    """Bits of one offset of a cell."""
    return math.log2(2 * c[1] + 1) if c[0] == "Z" else math.log2(c[1]) if c[0] == "M" else math.log2(2 * c[2] + 1 + c[1])


def bits_of(k, cells, nsym, nlab, by_state=True, chain=False):
    if isinstance(chain, tuple) and chain[0] == "a":  # a ceiling: every machine of the body priced at most so many bits
        return chain[1]
    if isinstance(chain, tuple) and chain[0] == "f":  # the control is f switches (k = 2^f states): every symbol keeps,
        f = chain[1]                                    # clears, sets or flips each; the cells move by the whole state
        b = nsym * f * 2 + sum(nsym * k * each(c) + (math.log2(c[1]) if c[0] == "L" else 0) for c in cells)
    elif isinstance(chain, tuple) and chain[0] == "s":  # a test of one cell: G symbols have a second row of moves, taken
        G = chain[1]                                    # when the cell the solver picks reads at least a value it picks
        one = (math.log2(k) if k > 1 else 0.0) + sum(each(c) for c in cells)
        tests = sum(len(values(c)) - 1 for c in cells)
        b = nsym * k * one + G * (k * one + math.log2(tests)) + math.log2(math.comb(nsym, G)) + \
            sum(math.log2(c[1]) for c in cells if c[0] == "L")
    elif isinstance(chain, tuple) and chain[0] == "e":  # a move by default for every (symbol, state), and E exceptions:
        E, R = chain[1], math.prod(len(values(c)) for c in cells)  # a (symbol, state, read) and the move it makes there
        one = (math.log2(k) if k > 1 else 0.0) + sum(each(c) for c in cells)
        b = nsym * k * one + math.log2(math.comb(nsym * k * R, E)) + E * one + \
            sum(math.log2(c[1]) for c in cells if c[0] == "L")
    elif isinstance(chain, tuple):  # guarded: G symbols move by the read of the first cell, the others as without it
        G, R = chain[1], math.prod(len(values(c)) for c in cells)
        one = (math.log2(k) if k > 1 else 0.0) + sum(each(c) for c in cells)
        b = (nsym - G) * k * one + G * R * k * one + math.log2(math.comb(nsym, G)) + \
            sum(math.log2(c[1]) for c in cells if c[0] == "L")
    else:
        b = nsym * k * math.log2(k) if k > 1 else 0.0
        for j, c in enumerate(cells):
            b += nsym * k * each(c) * (cells[0][1] if chain and j else 1) + (math.log2(c[1]) if c[0] == "L" else 0)
    if by_state in ("number", "lastnum"):  # the answer is the first (or the last) number itself: no table of answers
        return b + 1
    if by_state == "last":  # the answer is read from the last cell alone: a table over its values
        return b + len(values(cells[-1])) * math.log2(max(nlab, 2))
    reads = k if by_state else 1
    for c in cells:
        reads *= len(values(c))
    if by_state == "mealy":  # the answer is of the move: a table over what is read and the symbol
        reads *= nsym
    return b + reads * math.log2(max(nlab, 2))


# ---------------------------------------------------------------- shapes priced by what every move depends on (dep)
# A body is k states and its cells; only the first cell is read. For every symbol and every part it moves (the control,
# each cell) the machine says whether that move depends on the state, on what the first cell reads, or on both; a move
# that depends on neither is one row (the control keeps its state or goes to one state). The answer says which parts
# it reads. The price is the length of that description (Rissanen 1978): the rows, which symbols have which kind of
# row, and the table of answers over the parts it reads. A shape is a body and a ceiling: every machine of the body
# priced at most so many bits, the dependences found by the solver with the machine.
STEP = 16  # the ceilings of a body: every so many bits
TOP = 400  # and none above this
MAXREAD = 3  # with two cells, at most so many symbols move one part by what the first cell reads
PAIRK = 2  # with two cells, at most so many states


def parts(k, cells):
    """The parts a symbol moves: the control (with more than one state), and every cell."""
    return (["c"] if k > 1 else []) + list(range(len(cells)))


def rowcosts(k, cells, p):
    """Bits of the moves of one (symbol, part): alike everywhere, a row by the state, a row by what the first cell
    reads, a row by both."""
    R0 = len(values(cells[0])) if cells else 1
    if p == "c":
        lk, l1 = math.log2(k), math.log2(k + 1)
        return l1, k * lk, R0 * l1, k * R0 * lk
    e = each(cells[p])
    return e, k * e, R0 * e, k * R0 * e


def gmax(n, c):
    """Which of n symbols, at most c of them: the largest log2 C(n, x) for x up to c (so the price never falls as c
    grows)."""
    return max(math.log2(math.comb(n, x)) for x in range(c + 1))


def part_price(k, cells, p, nsym, cS, cR, cW):
    """One part, when cS symbols move it by the state, cR by the read and cW of them by both."""
    z, S, R, W = rowcosts(k, cells, p)
    return nsym * z + cS * (S - z) + cR * (R - z) + cW * (W - S - R + z) + gmax(nsym, cS) + gmax(nsym, cR) + gmax(nsym, cW)


def answer_price(k, cells, nlab, A, numeric=False):
    """The table of answers over the parts in A, and which parts those are; a number told as it is: one bit."""
    if numeric:
        return 1.0
    n = (k if "c" in A else 1) * math.prod(len(values(c)) if j in A else 1 for j, c in enumerate(cells))
    return n * math.log2(max(nlab, 2)) + len(parts(k, cells))


def price(k, cells, nsym, nlab, counts, A, numeric=False):
    return sum(math.log2(c[1]) for c in cells if c[0] == "L") + answer_price(k, cells, nlab, A, numeric) + \
        sum(part_price(k, cells, p, nsym, *counts[p]) for p in parts(k, cells))


def vectors(k, cells, nsym):
    """The counts (by the state, by the read, by both) one part may have."""
    most = min(nsym, MAXREAD) if len(cells) > 1 else nsym
    return [(cS, cR, cW) for cS in (range(nsym + 1) if k > 1 else [0]) for cR in (range(most + 1) if cells else [0])
            for cW in range(min(cS, cR) + 1)]


def patterns(k, cells, by):
    """The sets of parts the answer may read: a number told is its own cell; with two cells the second is read."""
    P = parts(k, cells)
    if by in ("number", "lastnum"):
        return [frozenset([len(cells) - 1 if by == "lastnum" else 0])]
    out = [frozenset(x for x, on in zip(P, bits) if on) for bits in itertools.product((0, 1), repeat=len(P))]
    return [A for A in out if len(cells) < 2 or 1 in A]


def useful(k, cells, counts, A):
    """Every part matters (as Encoding.dependencies demands): the control is read by the answer or moves a cell by the
    state; the first cell is read by the answer or read by a move of another part; the second is read by the answer."""
    if k > 1 and "c" not in A and not any(counts[j][0] for j in range(len(cells))):
        return False
    if cells and 0 not in A and not any(counts[p][1] for p in parts(k, cells) if p != 0):
        return False
    return len(cells) < 2 or 1 in A


@functools.lru_cache(maxsize=None)
def lowest(k, cells, nsym, nlab, by=True):
    """The price of the cheapest machine of the body in which every part matters (the price only grows with the
    counts, so counts of 0 and 1 are enough to find it)."""
    P, numeric = parts(k, cells), by in ("number", "lastnum")
    small = [v for v in vectors(k, cells, nsym) if max(v) <= 1]
    pp = {p: {v: part_price(k, cells, p, nsym, *v) for v in small} for p in P}
    base = sum(math.log2(c[1]) for c in cells if c[0] == "L")
    best = math.inf
    for A in patterns(k, cells, by):
        ab = base + answer_price(k, cells, nlab, A, numeric)
        for vec in itertools.product(small, repeat=len(P)):
            counts = dict(zip(P, vec))
            if useful(k, cells, counts, A):
                best = min(best, ab + sum(pp[p][v] for p, v in counts.items()))
    return best


@functools.lru_cache(maxsize=None)
def highest(k, cells, nsym, nlab, by=True):
    """The price of the dearest profile the body allows."""
    P, numeric = parts(k, cells), by in ("number", "lastnum")
    cS = nsym if k > 1 else 0
    cR = (min(nsym, MAXREAD) if len(cells) > 1 else nsym) if cells else 0
    return price(k, cells, nsym, nlab, {p: (cS, cR, min(cS, cR)) for p in P}, frozenset(P), numeric)


@functools.lru_cache(maxsize=256)  # (one fit asks for the same ceiling in every round)
def forbidden(k, cells, nsym, nlab, b, numeric=False, need=frozenset()):
    """The price is monotone in the counts of every part and in the parts the answer reads, so the profiles priced at
    most b are a down-set; it is cut by one clause per minimal profile above b (the minimal elements of the up-set
    left). Returns those profiles: (the parts the answer reads, a count vector per part)."""
    import numpy as np
    P = parts(k, cells)
    base = sum(math.log2(c[1]) for c in cells if c[0] == "L")
    apats = [frozenset()] if numeric else [A for A in (frozenset(x for x, on in zip(P, bits) if on)
                                                       for bits in itertools.product((0, 1), repeat=len(P))) if need <= A]
    if not P:
        return [(A, ()) for A in apats if base + answer_price(k, cells, nlab, A, numeric) > b + 1e-9]
    vs = vectors(k, cells, nsym)
    idx = {v: i for i, v in enumerate(vs)}
    along = lambda i: [-1 if j == i else 1 for j in range(len(P))]
    pp = [np.array([part_price(k, cells, p, nsym, *v) for v in vs]) for p in P]
    tot = sum(np.reshape(pp[i], along(i)) for i in range(len(P)))
    down = [np.array([idx.get(tuple(x - (i == d) for i, x in enumerate(v)), -1) for v in vs]) for d in range(3)]
    bad = {A: base + answer_price(k, cells, nlab, A, numeric) + tot > b + 1e-9 for A in apats}
    out = []
    for A in apats:
        m = bad[A].copy()
        for i in range(len(P)):  # every profile one step lower (and still a profile) is allowed
            for n in down:
                m &= ~(np.take(bad[A], np.where(n >= 0, n, 0), axis=i) & np.reshape(n >= 0, along(i)))
        for x in A - need:  # and so is the answer reading one part less
            m &= ~bad[A - {x}]
        out += [(A, tuple(vs[i] for i in ix)) for ix in zip(*np.nonzero(m))]
    return out


def bodies(most_k=12, numeric=False):
    """No cell (up to most_k states); one cell (up to 6 states); two cells (up to PAIRK states), the first a small one
    read by the moves, the second read by the answer; a number told as it is."""
    Z = [("Z", B, C) for B in (1, 2) for C in (1, 2)]
    M = [("M", m) for m in range(2, 13)]
    L = [("L", n, B) for n in range(3, 9) for B in (1, 2)]
    first = [("Z", 1, 1)] + [("L", n, 1) for n in range(3, 9)] + [("M", m) for m in range(2, 5)]
    second = [("Z", 1, 1), ("Z", 2, 1)] + [("M", m) for m in range(2, 8)] + [("M", 12)] + [("L", n, 1) for n in range(3, 7)]
    out = [(k, (), True) for k in range(1, most_k + 1)]
    out += [(k, (c,), True) for k in range(1, 7) for c in Z + M + L]
    out += [(k, (c0, c1), True) for k in range(1, PAIRK + 1) for c0 in first for c1 in second]
    if numeric:
        out += [(k, (c,), "number") for k in range(1, 7) for c in Z + L]
        out += [(k, (c0, c1), "lastnum") for k in range(1, PAIRK + 1) for c0 in first for c1 in second if c1[0] in "ZL"]
    return out


def ceiling(sh):
    return isinstance(sh[4], tuple) and sh[4][0] == "a"


def ceilings(nsym, nlab, most_k=12, numeric=False, flags=False):
    """Every body at the ceilings from its cheapest machine up to its dearest (at most TOP bits), every STEP bits; and
    the switches and the named form as in shapes(). In the order of the ceilings, the cheapest bodies first."""
    every = []
    for k, cs, by in bodies(most_k, numeric):
        lo, hi = lowest(k, cs, nsym, nlab, by), highest(k, cs, nsym, nlab, by)
        b = STEP * max(1, math.ceil(lo / STEP))
        while b <= TOP and b < hi + STEP:
            every.append((b, k, cs, by, ("a", b)))
            b += STEP
    if flags:
        Z = [("Z", B, C) for B in (1, 2) for C in (1, 2)]
        M = [("M", m) for m in range(2, 13)]
        L = [("L", n, B) for n in range(3, 9) for B in (1, 2)]
        cand = [(2 ** f, ()) for f in (2, 3, 4)] + [(2 ** f, (c,)) for f in (2, 3) for c in Z + L + M if c[0] != "M" or c[1] <= 4]
        every += [(bits_of(k, cs, nsym, nlab, by, ("f", k.bit_length() - 1)), k, cs, by, ("f", k.bit_length() - 1))
                  for k, cs in cand for by in (True, False) if by or cs]
    k = nlab + 1
    every.append((nsym * k * math.log2(k), k, (), "named", False))
    first = lambda s: lowest(s[1], s[2], nsym, nlab, s[3]) if ceiling(s) else s[0]
    return sorted(every, key=lambda s: (s[0], first(s), s[1], s[2], str(s[3]), str(s[4])))


def rank(shape):
    """How many memories beyond the plain machine a shape has (see tier): a second cell that moves apart from the first
    (two cells in a cascade, the second moved by the first, are one mechanism); moves priced by the rule."""
    _, k, cells, by, chain = shape
    return (1 if len(cells) == 2 and chain is not True else 0) + \
        (2 if isinstance(chain, tuple) and chain[0] in ("e", "a") else 0)


def shapes(nsym, nlab, most_k=12, guards=False, numeric=False, flags=False, cascades=False, rules=False, dep=False,
           tier=0, anchor=0, mealy=False):
    """tier (bits): the price of a shape is its length and so many bits for every step of its rank: the plain shapes are
    searched far ahead of the others, which come in only when no plain machine near their length explains the stories
    (with rules, the moves priced by the rule are then a third tier, beside the tables, not in their place).
    anchor (bits): the machines whose states are what is seen, the answers, each with a few hidden values (1, 2, ...
    HIDDEN) for what is not seen, are all given this one price: they come right after the shapes at most so long, the
    fewest hidden values first. Their table is long, but it is read off the answers, not searched for: a watcher of a
    display learns its states from the display, and looks for a hidden memory only where the same key at the same
    display does two things."""
    key = (nsym, nlab, most_k, guards, numeric, flags, cascades, rules, dep, tier, anchor, mealy, ALIKE)
    if key not in SHAPES:
        SHAPES.clear()  # (one list is kept: a learner asks for the same one again and again)
        SHAPES[key] = shapes_(nsym, nlab, most_k, guards, numeric, flags, cascades, rules, dep, tier, anchor, mealy)
    return SHAPES[key]


SHAPES = {}


def shapes_(nsym, nlab, most_k, guards, numeric, flags, cascades, rules, dep, tier, anchor=0, mealy=False):
    if dep:  # the bodies at their ceilings take the place of every other shape but the switches and the named form
        return ceilings(nsym, nlab, most_k, numeric, flags or guards == "flags")
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
    if guards:  # moves that read a number (Minsky 1967: a counter tested for zero): G of the symbols move the control
        # and the cells by the state and by what the cells read (a whole number clipped, a level, a circle)
        G1 = [c for c in Z if c[1] == 1] + [c for c in L if c[2] == 1] + [m for m in M if m[1] <= 4]
        G2 = [(c0, c1) for c0 in G1 for c1 in G1 + [z for z in Z if z[1] == 2]]
        cand = [(k, (c,)) for k in range(1, 7) for c in G1 + [z for z in Z if z[1] == 2]] + \
               [(k, cs) for k in range(1, 4) for cs in G2]
        if rules:  # (priced by the rule, not the table: a move by default, and the few places it is otherwise)
            every += [(bits_of(k, cs, nsym, nlab, by, ("e", E)), k, cs, by, ("e", E)) for k, cs in cand
                      for E in range(1, 9) for by in (True, False) if by or k > 1]
        if tier or not rules:
            every += [(bits_of(k, cs, nsym, nlab, by, ("g", G)), k, cs, by, ("g", G)) for k, cs in cand
                      for G in range(1, nsym + 1) for by in (True, False) if by or k > 1]
        if guards == "tests":  # (and a test of one cell, the cheaper form of the same)
            every += [(bits_of(k, cs, nsym, nlab, by, ("s", G)), k, cs, by, ("s", G)) for k, cs in cand
                      for G in range(1, nsym + 1) for by in (True, False) if by or k > 1]
    if guards == "flags" or flags:  # a control made of switches (flip-flops, the reset part of Krohn and Rhodes):
        cand = [(2 ** f, ()) for f in (2, 3, 4)] + [(2 ** f, (c,)) for f in (2, 3) for c in Z + L + M if c[0] != "M" or c[1] <= 4]
        every += [(bits_of(k, cs, nsym, nlab, by, ("f", k.bit_length() - 1)), k, cs, by, ("f", k.bit_length() - 1))
                  for k, cs in cand for by in (True, False) if by or cs]
    if numeric:  # the answers are numbers: the answer may be the first cell's number itself (a count told as it is)
        firsts = [(k, cs, ch) for _, k, cs, by, ch in every if by is True and cs and cs[0][0] in "ZL"]
        every += [(bits_of(k, cs, nsym, nlab, "number", ch), k, cs, "number", ch) for k, cs, ch in firsts]
    if cascades:  # a cascade (Krohn and Rhodes 1965): a level or a circle drives a second one by its value, and the
        # answer is read from the second alone (the part one watches), or is its number
        lv, ci = [("L", n, 1) for n in range(3, 7)], [("M", m) for m in range(2, 7)]
        every += [(bits_of(k, cs, nsym, nlab, "last", True), k, cs, "last", True) for k in (1, 2)
                  for cs in itertools.product(lv + ci, lv + ci)]
        if numeric:
            every += [(bits_of(k, cs, nsym, nlab, "lastnum", True), k, cs, "lastnum", True) for k in (1, 2)
                      for cs in itertools.product(lv + ci, lv + [("Z", 1, 1)])]
    # the states are the answers themselves, and one more to begin in: a table of relations (Tarski 1941), where
    # "what is X to Z" follows from "what is X to Y" and the next link
    if mealy:  # the answer is of the last move, not of the place arrived at (Mealy 1955): what a device replies to a
        # request depends on the request. A plain control, or one with a level or a circle
        small = [("L", n, 1) for n in range(3, 7)] + [("M", m) for m in range(2, 7)]
        of = [(bits_of(k, (), nsym, nlab, "mealy"), k, (), "mealy", False) for k in range(1, MEALY_K + 1)] + \
             [(bits_of(k, (c,), nsym, nlab, "mealy"), k, (c,), "mealy", False) for k in range(1, 5) for c in small]
        # (with anchor they come with the machines whose states are the answers, the shorter table first: their
        # states are not given by the answers, but every move is told apart by its own answer, and the search is quick)
        # (with ALIKE the plain tables come before those with a cell: the same answer is taken for the same place)
        every += [(anchor + 1 + 1e-4 * sh[0] + (0.5 if ALIKE and sh[2] else 0),) + sh[1:] for sh in of] \
            if anchor and not numeric else of
    k = nlab + 1
    if anchor and not numeric:  # (numbers are counted, not named) the answers alone; then with a hidden cell that a
        # few symbols read, the shorter table first; then
        every.append((anchor + (1 + 1e-4 * nsym * k * math.log2(k) if mealy else 0), k, (), "named", ("h", 1)))  # with hidden
        # values that have no order (with mealy, the answers alone come in their place by the length of their table)
        hid = [("L", n, 1) for n in range(3, 7)] + [("M", m) for m in range(2, 5)] + [("L", n, 2) for n in range(3, 6)]
        every += [(anchor + 1 + 1e-4 * bits_of(k, (c,), nsym, nlab, True, ("g", G)), k, (c,), "named", ("g", G))
                  for c in hid for G in range(1, min(nsym, 3) + 1)]
        every += [(anchor + 2 + 0.01 * h, nlab * h + 1, (), "named", ("h", h)) for h in range(2, HIDDEN + 1)]
    else:
        every.append((nsym * k * math.log2(k), k, (), "named", False))
    if tier:
        every = [(sh[0] + tier * rank(sh),) + sh[1:] for sh in every]
    return sorted(every, key=lambda s: (s[0], s[1], s[2], str(s[3]), str(s[4])))  # (the old order where the old shapes tie)


def name(shape):
    _, k, cells, by_state, chain = shape
    if by_state == "named" and isinstance(chain, tuple) and chain[0] == "h" and chain[1] > 1:
        return f"the answers as states, each with {chain[1]} hidden values ({k - 1} states), and one to begin in"
    if by_state == "named" and not cells:
        return f"{k - 1} states named by the answers and one to begin in"
    words = [f"{k - 1} states named by the answers and one to begin in" if by_state == "named" else f"{k} states"]
    for c in cells:
        words.append(f"a whole number (offsets -{c[1]}..{c[1]}, read clipped to -{c[2]}..{c[2]})" if c[0] == "Z"
                     else f"a circle of {c[1]}" if c[0] == "M" else f"a level of {c[1]} (steps -{c[2]}..{c[2]}, held at the ends, or set)")
    if isinstance(chain, tuple) and chain[0] == "f":
        words[0] = f"{chain[1]} switches"
    elif isinstance(chain, tuple) and chain[0] == "a":
        words[-1] += f", every move by the state{' or what the first number reads' if cells else ''} only where it " \
                     f"must (at most {chain[1]:.0f} bits)"
    elif isinstance(chain, tuple) and chain[0] == "s":
        words[-1] += f", {chain[1]} of the symbols moving otherwise when a number reads at least a value"
    elif isinstance(chain, tuple) and chain[0] == "e":
        words[-1] += f", moving alike whatever the numbers read but in {chain[1]} places"
    elif isinstance(chain, tuple):
        words[-1] += f", {chain[1]} of the symbols moving by what the first number reads"
    if by_state == "named":
        return " + ".join(words)
    if by_state == "mealy":
        return " + ".join(words) + ", the answer read from the state" + (" and the number" if cells else "") + \
            " before the last symbol and that symbol"
    return " + ".join(words) + (", the second moved by the value of the first" if chain is True else "") + \
        ("" if by_state or not cells else ", the answer read from the numbers only") + \
        (", the answer is the first number itself" if by_state == "number" else "") + \
        (", the answer read from the last number only" if by_state == "last" else "") + \
        (", the answer is the last number itself" if by_state == "lastnum" else "")


# ---------------------------------------------------------------- the machine, as found
class Machine:
    lex = None
    sure = None  # True: it passed the last check; False: the questions ran out before it did; None: no check
    bits = None  # (a ceiling shape) the price of the machine itself: what its moves and its answer really depend on
    mealy = False  # the answer is of the last move (see shapes)

    def __init__(self, shape, syms, labels, tau, dl, lam):
        self.shape, self.k, self.cells, self.syms, self.labels = shape, shape[1], shape[2], syms, labels
        self.by_state, self.chain = shape[3], shape[4]
        self.guard = isinstance(self.chain, tuple) and self.chain[0] in ("g", "e", "a")
        self.dep = isinstance(self.chain, tuple) and self.chain[0] == "a"  # (only the first cell is read)
        self.stump = isinstance(self.chain, tuple) and self.chain[0] == "s"
        self.test = {}  # (stump) symbol -> (cell, value): the second row when the cell reads at least the value
        self.start = self.k - 1 if self.by_state == "named" else 0
        self.tau, self.dl, self.lam = tau, dl, lam
        self.mealy = self.by_state == "mealy"  # (the answer of the last move is kept as one more value after the cells)
        self.zstart = [0] * (len(self.cells) + self.mealy)

    def run(self, seq):
        if self.lex is not None:
            if any(a not in self.lex for a in seq):
                return None
            seq = [self.lex[a] for a in seq]
        q, z = self.start, list(self.zstart)
        for a in seq:
            q, z = self.move(q, z, a)
        return self.out(q, z)

    def out(self, q, z):
        """The answer read at a state and values of the cells."""
        if self.by_state == "number":
            return str(z[0])
        if self.by_state == "lastnum":
            return str(z[-1])
        if self.mealy:
            return self.labels[z[-1]]
        r = tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(self.cells, z))
        return self.labels[self.lam[(q if self.by_state else 0,) + r]]

    def move(self, q, z, a):
        """One symbol: the next state and values of the cells."""
        if self.mealy:  # (and the answer of this move: read before it, from the state, the cells and the symbol)
            z = list(z[:len(self.cells)])
            r = (q,) + tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(self.cells, z)) + (a,)
            q2, z2 = self.moved(q, z, a)
            return q2, z2 + [self.lam[r]]
        return self.moved(q, z, a)

    def moved(self, q, z, a):
        z = list(z)
        if self.stump:  # one cell is read (clipped) before the move and compared with a value
            j, v = self.test[a]
            c = self.cells[j]
            b = int((max(-c[2], min(c[2], z[j])) if c[0] == "Z" else z[j]) >= v)
            for i, c in enumerate(self.cells):
                z[i] = step(c, z[i], self.dl[a, q, i, b])
            return self.tau[(a, b), q], z
        if self.guard:  # the cells are read (clipped) before the move, and the move depends on what they read
            read = self.cells[:1] if self.dep else self.cells  # (a ceiling shape reads its first cell only)
            g = tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(read, z))
            for j, c in enumerate(self.cells):
                z[j] = step(c, z[j], self.dl[a, q, j, g])
            return self.tau[(a, g), q], z
        w = z[0] if self.chain is True else 0
        for j, c in enumerate(self.cells):
            z[j] = step(c, z[j], self.dl[a, q, j, w if j else 0])
        return self.tau[a, q], z

    def __str__(self):
        return name(self.shape) + (f"; {self.bits:.1f} bits" if self.bits is not None else "")

    def table(self):
        lines = []
        if self.stump:
            return "\n".join(f"    {a:12s} {('cell %d >= %d' % self.test[a]) if b else 'else':12s}: states " +
                             " ".join(f"{q}>{self.tau[(a, b), q]}" for q in range(self.k)) + "   numbers by state " +
                             " ".join("(" + ",".join(str(self.dl[a, q, j, b]) for j in range(len(self.cells))) + ")" for q in range(self.k))
                             for a in self.syms for b in (1, 0))
        if self.guard:
            return "\n".join(f"    {a:12s} read {g}: states " + " ".join(f"{q}>{self.tau[(a, g), q]}" for q in range(self.k)) +
                             "   numbers by state " + " ".join("(" + ",".join(str(self.dl[a, q, j, g]) for j in range(len(self.cells))) + ")"
                                                           for q in range(self.k))
                             for a in self.syms for g in (([(v,) for v in values(self.cells[0])] if self.cells else [()])
                                                          if self.dep else itertools.product(*[values(c) for c in self.cells])))
        for a in self.syms:
            moves = " ".join(f"{q}>{self.tau[a, q]}" for q in range(self.k))
            offs = " ".join("(" + ",".join(str(self.dl[a, q, j, 0]) for j in range(len(self.cells))) + ")" for q in range(self.k))
            lines.append(f"    {a:12s} states {moves}" + (f"   numbers by state {offs}" if self.cells else ""))
        return "\n".join(lines)


class Summed:
    """A number shown, told as a sum: the machine inside answers what the last symbol added to the number (the answers
    of a watcher of the display, seen as changes), and the number is where it began and everything added since. The
    count itself is then no part of the machine: only what decides the additions is."""

    def __init__(self, machine, begin):
        self.machine, self.begin = machine, begin

    sure = property(lambda self: self.machine.sure)

    def run(self, seq):
        m = self.machine
        if m.lex is not None:
            if any(a not in m.lex for a in seq):
                return None
            seq = [m.lex[a] for a in seq]
        q, z, n = m.start, list(m.zstart), self.begin
        for a in seq:
            q, z = m.move(q, z, a)
            n += int(m.out(q, z))
        return str(n)

    def __str__(self):
        return f"a number that begins at {self.begin}, every symbol adding to it what this machine answers: {self.machine}"


def added(oracle):
    """The answers of a number shown, as what the last symbol added to it ('+1', '+0', '-2': names, not numbers)."""
    return lambda w: "%+d" % (int(oracle(list(w))) - int(oracle(list(w[:-1]))))


def profile(m):
    """(A machine of a ceiling shape) its exact price: per part, how many symbols move it by the state, by what the
    first cell reads, and by both; and the parts its answer really reads. Returns the price, the counts, the parts."""
    k, cs, nsym, nlab = m.k, m.cells, len(m.syms), len(m.labels)
    G = [(v,) for v in values(cs[0])] if cs else [()]
    P = parts(k, cs)
    counts = {}
    for p in P:
        cS = cR = cW = 0
        for a in m.syms:
            if p == "c":  # not by the state: in every read the control keeps its state, or goes to one state
                rows = {g: tuple(m.tau[(a, g), q] for q in range(k)) for g in G}
                byS = any(not (all(r[q] == q for q in range(k)) or len(set(r)) == 1) for r in rows.values())
                byR = len(set(rows.values())) > 1
            else:
                rows = {(q, g): m.dl[a, q, p, g] for q in range(k) for g in G}
                byS = any(len({rows[q, g] for q in range(k)}) > 1 for g in G)
                byR = any(len({rows[q, g] for g in G}) > 1 for q in range(k))
            cS, cR, cW = cS + byS, cR + byR, cW + (byS and byR)
        counts[p] = (cS, cR, cW)
    numeric = m.by_state in ("number", "lastnum")
    A = set()
    for p in ([] if numeric else P):
        i, first = (0, 0) if p == "c" else (1 + p, values(cs[p])[0])
        if any(m.lam[r] != m.lam[r[:i] + (first,) + r[i + 1:]] for r in m.lam):
            A.add(p)
    return price(k, cs, nsym, nlab, counts, A, numeric), counts, A


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
        self.dep = isinstance(self.chain, tuple) and self.chain[0] == "a"  # (a ceiling: see dependencies)
        self.guard = isinstance(self.chain, tuple) and self.chain[0] in ("g", "e", "a")
        self.stump = isinstance(self.chain, tuple) and self.chain[0] == "s"
        self.tests = [(j, v) for j, c in enumerate(self.cells) for v in list(values(c))[1:]] if self.stump else []
        self.ids, self.clauses = IDPool(), []
        self.sent = 0  # (the clauses already given to a live solver)
        self.warm = False  # (kept from one search to the next: see Warm)
        self.tot = self.model = None  # (see inertia)
        self.nc = len(self.cells) - 1 if self.by_state == "lastnum" else 0  # (the cell whose number is the answer)
        self.reads = [] if self.by_state in ("number", "lastnum") else \
            list(itertools.product(range(self.k) if self.by_state else [0], *[values(c) for c in self.cells])) \
            if self.by_state != "mealy" else \
            list(itertools.product(range(self.k), *[values(c) for c in self.cells], syms))
        self.gvals = ([(v,) for v in values(self.cells[0])] if self.cells else [()]) if self.dep else \
            list(itertools.product(*[values(c) for c in self.cells])) if self.guard else []
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
        # the symbols the control moves by: with guards, a symbol and what the first cell reads
        self.csyms = [(a, g) for a in syms for g in self.gvals] if self.guard else \
            [(a, b) for a in syms for b in (0, 1)] if self.stump else list(syms)
        for a in self.csyms:
            for q in range(k):
                self.exactly([self.v("t", a, q, r) for r in range(k)])
        for a in syms:
            for q in range(k):
                for j, c in enumerate(self.cells):
                    for w in self.keys(j):
                        self.exactly([self.v("o", j, a, q, w, o) for o in self.offsets(c)])
        if self.stump:  # every symbol picks one test; a symbol whose second row is not used moves alike either way
            for a in syms:
                self.exactly([self.v("tc", a, i) for i in range(len(self.tests))])
        if self.guard and self.chain[0] == "e":  # a move by default for every (symbol, state); at most E (symbol, state,
            for a in syms:                        # read) move otherwise (the frame by default: Reiter 1980)
                for q in range(k):
                    self.exactly([self.v("dt", a, q, r) for r in range(k)])
                    for j, c in enumerate(self.cells):
                        self.exactly([self.v("do", a, q, j, o) for o in self.offsets(c)])
                    for g in self.gvals:
                        x = self.v("ex", a, q, g)
                        for r in range(k):
                            self.clauses.append([x, -self.v("t", (a, g), q, r), self.v("dt", a, q, r)])
                        for j, c in enumerate(self.cells):
                            for o in self.offsets(c):
                                self.clauses.append([x, -self.v("o", j, a, q, g, o), self.v("do", a, q, j, o)])
            exs = [self.v("ex", a, q, g) for a in syms for q in range(k) for g in self.gvals]
            if self.chain[1] < len(exs):
                self.clauses += CardEnc.atmost(lits=exs, bound=self.chain[1], vpool=self.ids,
                                               encoding=EncType.seqcounter).clauses
        elif self.dep:
            self.dependencies()
        elif self.guard or self.stump:  # a symbol that does not read the cell moves alike whatever it reads; at most G do
            g0, gs = (self.gvals[0], self.gvals[1:]) if self.guard else (0, [1])
            for a in syms:
                gd = self.v("gd", a)
                for g in gs:
                    for q in range(k):
                        for r in range(k):
                            self.clauses.append([gd, -self.v("t", (a, g), q, r), self.v("t", (a, g0), q, r)])
                        for j, c in enumerate(self.cells):
                            for o in self.offsets(c):
                                self.clauses.append([gd, -self.v("o", j, a, q, g, o), self.v("o", j, a, q, g0, o)])
            if self.chain[1] < len(syms):
                self.clauses += CardEnc.atmost(lits=[self.v("gd", a) for a in syms], bound=self.chain[1], vpool=self.ids,
                                               encoding=EncType.seqcounter).clauses
        for r in self.reads:
            self.exactly([self.v("l", r, y) for y in range(len(labels))])
        if self.by_state == "last":  # read from the last cell alone: reads that agree there give one answer
            for r in self.reads:
                r0 = (0,) + tuple(values(c)[0] for c in self.cells[:-1]) + (r[-1],)
                if r != r0:
                    for y in range(len(labels)):
                        self.clauses += [[-self.v("l", r, y), self.v("l", r0, y)], [self.v("l", r, y), -self.v("l", r0, y)]]
        self.hidden = self.chain[1] if self.by_state == "named" and isinstance(self.chain, tuple) and self.chain[0] == "h" else 1
        self.first = set()  # (named, with hidden values: the answers a story said has ended at)
        if self.by_state == "named":  # the states are told apart by their names: nothing to break
            for r in self.reads:  # (with hidden values: state q is answer q // hidden, hidden value q % hidden)
                if r[0] < self.k - 1:
                    self.clauses.append([self.v("l", r, r[0] // self.hidden)])
        else:
            if isinstance(self.chain, tuple) and self.chain[0] == "f":
                self.switches(self.chain[1])  # (states are named by their switches: no numbering to break)
            else:
                self.symmetry()
        if MIRROR:
            self.mirror()
        self.node = {(): 0}
        self.memo = {}  # (the symbol variables of a story the solver writes; the answer variables of a node)
        self.depth = {(): 0}
        self.ndepth = {0: 0}
        self.clauses.append([self.v("x", 0, self.k - 1 if self.by_state == "named" else 0)])
        for j, c in enumerate(self.cells):
            if c[0] == "M":
                self.clauses.append([self.v("z", 0, j, 0)])
            elif c[0] == "L":  # where a level begins is found with the machine
                self.exactly([self.v("z", 0, j, val) for val in range(c[1])])
        if self.stump or self.dep:
            self.onevalue(0)

    def v(self, *key):
        return self.ids.id(key)

    def inertia(self):
        """(the answers as states) A counter over what a rule told by a person would have to say: where a symbol changes
        the state (at a state and, for a symbol that reads the cell, at what it reads); how a symbol moves the hidden
        cell (one move for all the states, if it is not 'leave it'); and at which states it moves it otherwise. 'Otherwise
        nothing happens' is not said (the frame by default: McCarthy and Hayes 1969). Of the machines that explain the
        stories, the one that needs the least said is kept."""
        if self.tot is None:
            lits = []
            def counted(lit, a, g, *key):  # (a symbol that does not read the cell moves alike at every read: once)
                if not self.guard or g == self.gvals[0]:
                    return lit
                more = self.v("more", a, g, *key)
                self.clauses.append([-lit, -self.v("gd", a), more])
                return more
            for a in self.syms:
                for g in (self.gvals if self.guard else [0]):
                    for j, c in enumerate(self.cells):
                        self.exactly([self.v("dflt", j, a, g, o) for o in self.offsets(c)])
                        lits.append(counted(-self.v("dflt", j, a, g, 0), a, g, "d", j))
                    for q in range(self.k - 1):
                        lits.append(counted(-self.v("t", (a, g) if self.guard else a, q, q), a, g, "t", q))
                        for j, c in enumerate(self.cells):
                            other = self.v("other", j, a, q, g)
                            for o in self.offsets(c):
                                self.clauses.append([-self.v("o", j, a, q, g, o), self.v("dflt", j, a, g, o), other])
                            lits.append(counted(other, a, g, "o", j, q))
            self.count = lits
            self.tot = ITotalizer(lits=lits, ubound=min(len(lits), 120), top_id=self.ids.top)
            self.ids.occupy(self.ids.top + 1, self.tot.top_id)
            self.clauses += self.tot.cnf.clauses
        return self.tot

    def alike(self):
        """(the answer of the last move, no cells) A counter over the moves that do not lead where the moves with the same
        answer lead: every answer has one place its moves lead to by default, and a move that leads elsewhere is
        counted. Of the machines that explain the stories, the one with the fewest such moves is kept: the same reply is
        taken for the same place until a story shows two places (a closed connection is one place, however it came to
        be closed). It is the watcher's 'the state is what is seen' for a machine whose answers are of its moves."""
        if self.tot is None:
            lits = []
            for y in range(len(self.labels)):
                self.exactly([self.v("to", y, d) for d in range(self.k)])
            for q in range(self.k):
                for a in self.syms:
                    lits.append(self.v("else", q, a))
                    for y in range(len(self.labels)):
                        for d in range(self.k):
                            self.clauses.append([-self.v("l", (q, a), y), -self.v("t", a, q, d), self.v("to", y, d), lits[-1]])
            self.count = lits
            self.tot = ITotalizer(lits=lits, ubound=min(len(lits), 120), top_id=self.ids.top)
            self.ids.occupy(self.ids.top + 1, self.tot.top_id)
            self.clauses += self.tot.cnf.clauses
        return self.tot

    def happens(self):
        """In how many places of the table something happens, in the last machine found."""
        return sum((l in self.model) if l > 0 else (-l not in self.model) for l in self.count)

    def switches(self, f):
        """The move of every (symbol, state) is the one its operation on every switch gives: keep, clear, set, flip."""
        ops = [lambda b: b, lambda b: 0, lambda b: 1, lambda b: 1 - b]
        for a in self.csyms:
            for i in range(f):
                self.exactly([self.v("op", a, i, o) for o in range(4)])
            for q in range(self.k):
                for r in range(self.k):
                    for i in range(f):
                        for o in range(4):
                            if ops[o]((q >> i) & 1) != (r >> i) & 1:
                                self.clauses.append([-self.v("t", a, q, r), -self.v("op", a, i, o)])
        a0 = self.csyms[0]  # the switches may be numbered in any order: the first symbol's operations do not go down
        for i in range(f - 1 if MIRROR else 0):
            for o1 in range(4):
                for o2 in range(o1):
                    self.clauses.append([-self.v("op", a0, i, o1), -self.v("op", a0, i + 1, o2)])

    def mirror(self):
        """A level may be turned end for end (v -> n-1-v, a step o -> -o, 'set v' -> 'set n-1-v'): it begins in its
        lower half. Not where the number is the answer itself, nor with a test 'reads at least v'."""
        if self.stump:
            return
        for j, c in enumerate(self.cells):
            if c[0] == "L" and not (j == self.nc and self.by_state in ("number", "lastnum")):
                for val in range(c[1]):
                    if val > c[1] - 1 - val:
                        self.clauses.append([-self.v("z", 0, j, val)])

    def symmetry(self):
        """One machine, one numbering. States are numbered in the order a walk in breadth from state 0 meets them,
        symbols tried in their order (Ulyantsev, Zakirzyanov and Shalyto 2015). A whole number may be turned over
        (x -> -x) and a circle turned by any of its units (x -> u x): the first offset is kept non-negative, or a
        divisor of the circle; two cells of one kind are kept in order by it."""
        k, S, v, C = self.k, self.csyms, self.v, self.clauses
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
        a0 = self.syms[0]
        w0 = tuple(0 for _ in self.cells) if self.guard else 0  # (a key that turning a number over or round keeps)
        if self.dep:
            w0 = (0,) if self.cells else ()  # (a ceiling shape reads its first cell only)
        for j, c in enumerate(self.cells if not self.stump else []):  # (a test 'at least v' is not kept by turning)
            if c[0] == "Z" and not (j == self.nc and self.by_state in ("number", "lastnum")):  # (the number told is not turned over)
                for o in range(-c[1], 0):
                    C.append([-v("o", j, a0, 0, w0, o)])
            elif c[0] == "M":
                for o in range(1, c[1]):
                    if c[1] % o:
                        C.append([-v("o", j, a0, 0, w0, o)])
        for j in range(1, len(self.cells)):
            if self.cells[j] == self.cells[j - 1] and not self.chain:  # (a chained or read first cell is not like the second)
                for o1 in self.offsets(self.cells[j - 1]):
                    for o2 in self.offsets(self.cells[j]):
                        if o2 < o1:
                            C.append([-v("o", j - 1, a0, 0, 0, o1), -v("o", j, a0, 0, 0, o2)])

    def keys(self, j):
        """The values of the first cell an offset of cell j depends on (only one key unless chained)."""
        if self.guard:
            return self.gvals
        if self.stump:
            return (0, 1)
        return range(self.cells[0][1]) if self.chain is True and j else range(1)

    def outcome(self, n, a):
        """(stump) The literal 'the test symbol a picked holds at node n', defined once per node and symbol."""
        if ("out", n, a) in self.memo:
            return self.memo[("out", n, a)]
        out = self.v("out", n, a)
        self.memo[("out", n, a)] = out
        for i, (j, v) in enumerate(self.tests):
            c = self.cells[j]
            if c[0] == "Z":
                ge = self.ge(n, j, v)
            else:  # a level or a circle: one value at a node (at most one is true: see onevalue), 'at least v' by a variable
                ge = self.v("geq", n, j, v)
                if ("geq", n, j, v) not in self.memo:
                    self.memo[("geq", n, j, v)] = True
                    self.clauses.append([-ge] + [self.v("z", n, j, u) for u in range(v, c[1])])
                    for u in range(v, c[1]):
                        self.clauses.append([-self.v("z", n, j, u), ge])
            neg = (not ge) if isinstance(ge, bool) else -ge
            self.add([-self.v("tc", a, i), neg, out])
            self.add([-self.v("tc", a, i), ge, -out])
        return out

    def onevalue(self, n):
        """(stump, ceiling) At most one value of every level or circle at node n: a test reads it both ways."""
        for j, c in enumerate(self.cells):
            if c[0] in "ML":
                for u in range(c[1]):
                    for w in range(u + 1, c[1]):
                        self.clauses.append([-self.v("z", n, j, u), -self.v("z", n, j, w)])

    @staticmethod
    def selfread(c, g, u, up):
        """(ceiling) The whole number read is the first cell, read g before the move; its clause from 'at least u'
        (up) or from 'below u' (not up) is needed only where the read leaves u open. A read inside -C..C is the value
        itself (only u = g, or u = g + 1, are needed); C is any value from C up, -C any value from -C down. The chain of
        the order encoding at the next node implies the rest."""
        if -c[2] < g < c[2]:
            return u == (g if up else g + 1)
        if g == c[2]:
            return u >= (c[2] if up else c[2] + 1)
        return u <= (-c[2] if up else -c[2] + 1)

    def dependencies(self):
        """(ceiling) Per symbol a and part p: dS, its move of p depends on the state; dR, on what the first cell reads;
        dW, on both. Without dS the rows of p are alike in every state (the control keeps its state or goes to one
        state); without dR they are alike for every read. Per part dA: the answer reads it. Per (part, kind) a unary
        counter of the indicators (the sequential counter, Sinz 2005), and the price of the counts kept at most the
        ceiling by one clause per minimal profile above it (forbidden)."""
        k, cs, syms, V, C = self.k, self.cells, self.syms, self.v, self.clauses
        nsym, nlab, b = len(syms), len(self.labels), self.chain[1]
        P, G = parts(k, cs), self.gvals
        g0 = G[0]
        ind = {}
        for a in syms:
            for p in P:
                S = V("dS", a, p) if k > 1 else None
                R = V("dR", a, p) if cs else None
                W = V("dW", a, p) if k > 1 and cs else None
                if W is not None:
                    C.extend([[-W, S], [-W, R], [-S, -R, W]])
                ind[a, p] = (S, R, W)
                for g in G:
                    if p == "c":
                        K = V("keep", a, g)
                        for q in range(k):
                            C.append([S, -K, V("t", (a, g), q, q)])
                            for r in range(k):
                                if q:
                                    C.append([S, K, -V("t", (a, g), q, r), V("t", (a, g), 0, r)])
                                if R is not None and g != g0:
                                    C.append([R, -V("t", (a, g), q, r), V("t", (a, g0), q, r)])
                        continue
                    for q in range(k):
                        for o in self.offsets(cs[p]):
                            if S is not None and q:
                                C.append([S, -V("o", p, a, q, g, o), V("o", p, a, 0, g, o)])
                            if R is not None and g != g0:
                                C.append([R, -V("o", p, a, q, g, o), V("o", p, a, q, g0, o)])
        numeric = self.by_state in ("number", "lastnum")
        A = {p: V("dA", p) for p in P}
        if numeric:  # a number told as it is reads its own cell and nothing else
            C.extend([[A[p]] if p == self.nc else [-A[p]] for p in P])
        for r in self.reads:  # without dA[p] the answer is alike for every value of p
            for p in P:
                i, first = (0, 0) if p == "c" else (1 + p, values(cs[p])[0])
                if r[i] != first:
                    r0 = r[:i] + (first,) + r[i + 1:]
                    for y in range(nlab):
                        C.append([A[p], -V("l", r, y), V("l", r0, y)])
        # every part matters (else the machine is one of a smaller body): the control is read by the answer or moves
        # a cell; the first cell is read by the answer or by a move of another part; a second cell is read by the answer
        if k > 1:
            C.append([A["c"]] + [ind[a, j][0] for a in syms for j in range(len(cs))])
        if cs:
            C.append([A[0]] + [ind[a, p][1] for a in syms for p in P if p != 0])
        if len(cs) == 2:
            C.append([A[1]])
        cnt = {}
        for p in P:
            for kind in range(3):
                lits = [ind[a, p][kind] for a in syms if ind[a, p][kind] is not None]
                prev = {}
                for i, x in enumerate(lits):  # cur[j]: at least j of the first i + 1 indicators hold
                    cur = {j: V("cnt", p, kind, i, j) for j in range(1, i + 2)}
                    for j, u in cur.items():
                        if j in prev:
                            C.append([-prev[j], u])
                        C.append([-x, u] if j == 1 else [-x, -prev[j - 1], u])
                    prev = cur
                cnt[p, kind] = prev
                if kind == 1 and len(cs) > 1 and MAXREAD + 1 in prev:
                    C.append([-prev[MAXREAD + 1]])
        need = frozenset([1]) if len(cs) > 1 and not numeric else frozenset()  # (a second cell is always read)
        for Aset, vec in forbidden(k, cs, nsym, nlab, b, numeric, need):
            cl = [-A[p] for p in Aset] + [-cnt[p, kind][v[kind]] for p, v in zip(P, vec) for kind in range(3) if v[kind]]
            if not cl:  # (the ceiling is below every machine of the body)
                cl = [V("over")]
                C.append([-V("over")])
            C.append(cl)

    def notat(self, n, g):
        """Literals (some may be True / False) one of which holds unless the cells at node n read g."""
        out = []
        neg = lambda l: (not l) if isinstance(l, bool) else -l
        for j, (c, gj) in enumerate(zip(self.cells, g)):
            if c[0] in "ML":
                out.append(-self.v("z", n, j, gj))
            elif gj == -c[2]:
                out.append(self.ge(n, j, gj + 1))
            elif gj == c[2]:
                out.append(neg(self.ge(n, j, gj)))
            else:
                out += [neg(self.ge(n, j, gj)), self.ge(n, j, gj + 1)]
        return out

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
        if self.stump or self.dep:
            self.onevalue(n)
        neg = lambda l: (not l) if isinstance(l, bool) else -l
        if isinstance(a, tuple) and a[:1] == ("?",):  # a symbol of a story the solver writes itself
            if ("fx", a[1]) not in self.memo:
                self.memo[("fx", a[1])] = True
                self.exactly([self.v("f", a[1], s) for s in self.syms])
            read = [(s, [-self.v("f", a[1], s)]) for s in self.syms]
        else:
            read = [(a, [])] if self.raw is None else [(s, [-self.v("w", a, i)]) for i, s in enumerate(self.syms)]
        for q in range(self.k):
            x = self.v("x", n0, q)
            for a, guard in read:
                for r in range(self.k):
                    if self.stump:
                        out = self.outcome(n0, a)
                        self.add([-x, -self.v("t", (a, 1), q, r), self.v("x", n, r), -out] + guard)
                        self.add([-x, -self.v("t", (a, 0), q, r), self.v("x", n, r), out] + guard)
                    elif self.guard:
                        for g in self.gvals:
                            self.add([-x, -self.v("t", (a, g), q, r), self.v("x", n, r)] + guard + self.notat(n0, g))
                    else:
                        self.clauses.append([-x, -self.v("t", a, q, r), self.v("x", n, r)] + guard)
            for j, c in enumerate(self.cells):
                for w in self.keys(j):
                  for a, guard in read:
                    key = (self.notat(n0, w) if self.guard else ([-self.outcome(n0, a)] if w else [self.outcome(n0, a)])
                           if self.stump else [-self.v("z", n0, 0, w)] if self.chain is True and j else []) + guard
                    own = self.dep and j == 0  # (the read cell's own move: only what its read leaves open)
                    for o in self.offsets(c):
                        off = self.v("o", j, a, q, w, o)
                        if c[0] == "Z":
                            D = c[1] * (dep + 1)
                            for b in range(-D + 1, D + 1):
                                if not own or self.selfread(c, w[0], b - o, True):
                                    self.add([-x, -off, neg(self.ge(n0, j, b - o)), self.ge(n, j, b)] + key)
                                if not own or self.selfread(c, w[0], b - o, False):
                                    self.add([-x, -off, self.ge(n0, j, b - o), neg(self.ge(n, j, b))] + key)
                        else:
                            for val in range(c[1]):
                                if own and w != (val,):  # (at most one value at a node: the read is the value)
                                    continue
                                self.add([-x, -off, -self.v("z", n0, j, val), self.v("z", n, j, step(c, val, o))] + key)
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
        end, last = n, None
        if self.by_state == "mealy":  # (read at the node before the last symbol, and by that symbol)
            last, n = seq[-1], self.at(tuple(seq[:-1]))
        for r in self.reads:
            if last is not None and not (isinstance(last, tuple) and last[:1] == ("?",)) and r[-1] != last:
                lits[r] = None
                continue
            out = [-self.v("x", n, r[0])] if self.by_state else []
            if last is not None and isinstance(last, tuple) and last[:1] == ("?",):  # (a symbol the solver writes)
                out.append(-self.v("f", last[1], r[-1]))
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
        return end, lits

    def isnot(self, n, v):
        """Literals one of which holds unless the first cell at node n is v (True / False where depth decides)."""
        j = self.nc
        c = self.cells[j]
        if c[0] == "L":
            return [-self.v("z", n, j, v)] if 0 <= v < c[1] else [True]
        neg = lambda l: (not l) if isinstance(l, bool) else -l
        return [neg(self.ge(n, j, v)), self.ge(n, j, v + 1)]

    def says(self, seq, y, excused=None):
        """The story is answered y; or, given a literal, it may be excused."""
        if self.by_state in ("number", "lastnum"):  # the first (last) number is the answer: it is y
            n = self.at(tuple(seq))
            ex = [excused] if excused else []
            c, v = self.cells[self.nc], int(self.labels[y])
            if c[0] == "L" and 0 <= v < c[1]:  # a level: its value is v and no other (only the true value is implied)
                for u in range(c[1]):
                    if u != v:
                        self.clauses.append([-self.v("z", n, self.nc, u)] + ex)
            for l in self.isnot(n, v):  # every literal of 'it is not y' is false
                if l is True:  # the number cannot be y here
                    self.clauses += [ex + [self.v("false")], [-self.v("false")]]
                elif l is not False:
                    self.clauses.append([-l] + ex)
            return
        n, lits = self.read(seq)
        for r, ls in lits.items():
            if ls is not None:
                self.clauses.append(ls + [self.v("l", r, y)] + ([excused] if excused else []))
        if self.hidden > 1 and not excused and y not in self.first:  # the hidden values of an answer may be numbered
            self.first.add(y)                                        # in any order: the first story said that ends at
            self.clauses.append([self.v("x", n, y * self.hidden)])   # an answer is at its hidden value 0

    def differs(self, seq, y):
        """A literal true only if the story is not answered y."""
        if self.by_state in ("number", "lastnum"):
            n = self.at(tuple(seq))
            d = self.v("differs", n, y)
            if not self.warm or ("dif", n, y) not in self.memo:  # (kept: said once)
                self.memo[("dif", n, y)] = True
                self.add([-d] + self.isnot(n, int(y)))
            return d
        if y in self.labels:
            n = self.at(tuple(seq))
            if ("ans", n) not in self.memo:
                self.memo[("ans", n)] = self.free(seq)
            return -self.memo[("ans", n)][self.labels.index(y)]
        if not self.warm or "true" not in self.memo:
            self.memo["true"] = True
            self.clauses.append([self.v("true")])
        return self.v("true")

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

    def solve(self, seconds, live=None, assume=()):
        """The solver's limit is a number of conflicts (CONFLICTS a second of the old limit), not time: the same work on
        any machine, so the same result. A wall clock far beyond it only guards against a hang.
        live: a solver kept between calls (INCREMENTAL): only the clauses added since the last call are given to it, and
        what it learned stays (Een and Sorensson 2003). assume: literals held true for this call only."""
        if live is None:
            with Solver(name=ENGINE, bootstrap_with=self.clauses) as s:
                return self.solved(s, seconds, 0)
        live.append_formula(self.clauses[self.sent:])
        self.sent = len(self.clauses)
        if self.warm:  # (the solver has them: the list is not kept twice)
            self.clauses, self.sent = [], 0
        return self.solved(live, seconds, live.accum_stats().get("conflicts", 0), assume)

    def solved(self, s, seconds, before, assume=()):
        if True:
            s.conf_budget(max(1000, int(seconds * CONFLICTS)))
            timer = threading.Timer(20 * seconds + 60, s.interrupt) if ENGINE.startswith("glucose") else None
            if timer:
                timer.start()
            ok = s.solve_limited(assumptions=list(assume), expect_interrupt=True)
            if timer:
                timer.cancel()
                s.clear_interrupt()
            self.used = s.accum_stats().get("conflicts", 0) - before
            WORK[0] += self.used
            if ok is None:
                return None, "unknown"
            if not ok:
                return None, "unsat"
            model = self.model = set(l for l in s.get_model() if l > 0)
        on = lambda *key: self.ids.obj2id.get(key) in model
        tau = {(a, q): next(r for r in range(self.k) if on("t", a, q, r)) for a in self.csyms for q in range(self.k)}
        dl = {(a, q, j, w): next(o for o in self.offsets(c) if on("o", j, a, q, w, o))
              for a in self.syms for q in range(self.k) for j, c in enumerate(self.cells) for w in self.keys(j)}
        lam = {r: next(y for y in range(len(self.labels)) if on("l", r, y)) for r in self.reads}
        m = Machine(self.shape, self.syms, self.labels, tau, dl, lam)
        m.zstart = [next(v for v in range(c[1]) if on("z", 0, j, v)) if c[0] == "L" else 0 for j, c in enumerate(self.cells)]
        if m.mealy:
            m.zstart.append(0)
        if self.raw is not None:
            m.lex = {r: self.syms[next(i for i in range(len(self.syms)) if on("w", r, i))] for r in self.raw}
        if self.stump:
            m.test = {a: self.tests[next(i for i in range(len(self.tests)) if on("tc", a, i))] for a in self.syms}
        if self.dep:  # priced by what its tables really depend on: never above the ceiling
            m.bits = profile(m)[0]
            assert m.bits <= self.chain[1] + 1e-6, (self.shape, m.bits)
        return m, "sat"


def story_differs(e, other, syms, D):
    """Clauses: the machine of the encoding answers otherwise than `other` at the end of some beginning of a story of at
    most D symbols that the solver writes itself (its symbols are variables). The other machine's run along that story is
    given by the layers of its configurations reachable in t symbols."""
    chain = tuple(("?", t) for t in range(1, D + 1))
    c0 = (other.start, tuple(other.zstart))
    layer, y = {c0}, {}
    some = []
    e.clauses.append([e.v("y", 0, c0)])
    for t in range(1, D + 1):
        nxt = set()
        for c in layer:
            for s in syms:
                q2, z2 = other.move(c[0], list(c[1]), s)
                c2 = (q2, tuple(z2))
                nxt.add(c2)
                e.clauses.append([-e.v("y", t - 1, c), -e.v("f", t, s), e.v("y", t, c2)])
        layer = nxt
        e.at(chain[:t])
        d = e.v("d", t)
        some.append(d)
        for c in layer:
            e.add([-d, -e.v("y", t, c), e.differs(chain[:t], other.out(c[0], list(c[1])))])
    e.clauses.append(some)


def block(e, other, syms, upto, D):
    """(warm) What story_differs says, for a solver kept between searches. Its clauses hold only under a literal (the
    solver is asked with it assumed), so the block of a machine no longer kept is switched off by one clause; and it is
    written depth by depth as asked: 'they part after t symbols' has a literal for every t, and a search on short
    stories assumes the later ones false (most second machines part from the first on a short story). The run of the
    machine looked for along the story the solver writes is the same for every other machine: it is written once.
    Returns the literal and the list of the literals of the depths."""
    b = e.block
    if b is None or b["other"] is not other or len(b["d"]) != D:
        if b is not None:
            e.clauses.append([-b["act"]])
        act = e.v("act", e.blocks)
        e.blocks += 1
        c0 = (other.start, tuple(other.zstart))
        b = e.block = {"other": other, "act": act, "layer": {c0}, "t": 0, "d": [e.v("d", act, t) for t in range(1, D + 1)]}
        e.clauses.append([-act, e.v("y", act, 0, c0)])
        e.clauses.append([-act] + b["d"])
    act = b["act"]
    chain = tuple(("?", t) for t in range(1, D + 1))
    while b["t"] < upto:
        t = b["t"] + 1
        nxt = set()
        for c in b["layer"]:
            for s in syms:
                q2, z2 = other.move(c[0], list(c[1]), s)
                c2 = (q2, tuple(z2))
                nxt.add(c2)
                e.clauses.append([-e.v("y", act, t - 1, c), -e.v("f", t, s), e.v("y", act, t, c2)])
        b["layer"], b["t"] = nxt, t
        e.at(chain[:t])
        for c in nxt:
            e.add([-b["d"][t - 1], -e.v("y", act, t, c), e.differs(chain[:t], other.out(c[0], list(c[1])))])
    return act, b["d"]


class Warm:
    """The encodings and the solvers of the shapes, kept from one search to the next. The stories of a learner only
    grow, so what a solver learned of a shape stays true (Een and Sorensson 2003): a search the solver gave up on goes
    on from where it stopped, a new story is a few clauses more, and a second machine is looked for under an assumed
    literal, in the same solver."""
    MOST = 120  # so many shapes at most (the one used longest ago goes first)
    ROOM = 2500  # megabytes: when the process holds more, the half of the shapes used longest ago goes
    # (sealed_run.py --room)

    def __init__(self):
        self.kept, self.dead, self.mark = {}, set(), 0

    @staticmethod
    def held():
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * 4096 / 1e6
        except (OSError, ValueError, IndexError):
            return 0

    def of(self, shape, syms, labels):
        key = shape[1:]
        if key in self.kept:
            self.kept[key] = self.kept.pop(key)
        else:
            e = Encoding(shape, syms, labels)
            e.warm, e.block, e.blocks, e.said, e.saidset = True, None, 0, [], set()
            self.kept[key] = (e, Solver(name=ENGINE))
            while len(self.kept) > self.MOST:
                self.drop(next(iter(self.kept)))
        if len(self.kept) > 2 and self.held() > max(self.ROOM, 1.15 * self.mark):  # (built again when asked for)
            for old in list(self.kept)[:max(1, len(self.kept) // 2)]:
                if old != key:
                    self.drop(old)
            self.mark = self.held()
        return self.kept[key]

    def drop(self, key):
        if key in self.kept:
            self.kept.pop(key)[1].delete()

    def clear(self):
        for key in list(self.kept):
            self.drop(key)
        self.dead = set()

    def __del__(self):
        self.clear()


def settle(m, data):
    """(the answers as states) A move no story taught on has made leaves the state as it is: most keys do nothing at
    most displays. The machine explains the stories as before; what it was free to say elsewhere is said the plain way."""
    made = set()
    read = lambda z: tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(m.cells, z))
    reads = {}  # the symbols that move by what the cell reads (the others move alike at every read: one place each)
    if m.guard:
        gs = sorted({key[0][1] for key in m.tau})
        for a in m.syms:
            reads[a] = any(m.tau[(a, g), q] != m.tau[(a, gs[0]), q] or
                           any(m.dl[a, q, j, g] != m.dl[a, q, j, gs[0]] for j in range(len(m.cells)))
                           for g in gs for q in range(m.k))
    for s, _ in data:
        q, z = m.start, list(m.zstart)
        for a in s:
            made.add((a, q, read(z) if reads.get(a) else ()))
            q, z = m.move(q, z, a)
    for key in list(m.tau):  # (a symbol, or a symbol and what the cell reads; the state)
        a, g = key[0] if m.guard else (key[0], ())
        q = key[1]
        if (a, q, g if reads.get(a) else ()) not in made and q != m.start:
            m.tau[key] = q
            for j in range(len(m.cells)):
                m.dl[a, q, j, g if m.guard else 0] = 0
    return m


def W_ok(warm, workers):
    return bool(warm) and workers <= 1


def separating(m1, m2, syms, D):
    """The shortest story (of at most D symbols) on which two machines answer otherwise; a walk in breadth over the
    pairs of their configurations."""
    start = ((m1.start, tuple(m1.zstart)), (m2.start, tuple(m2.zstart)))
    frontier, seen = [((), start)], set()  # (the pair begun in is not seen yet: no story has ended at it)
    for _ in range(D):
        nxt = []
        for w, (c1, c2) in frontier:
            for s in syms:
                q1, z1 = m1.move(c1[0], list(c1[1]), s)
                q2, z2 = m2.move(c2[0], list(c2[1]), s)
                p = ((q1, tuple(z1)), (q2, tuple(z2)))
                if p in seen:
                    continue
                seen.add(p)
                if m1.out(q1, z1) != m2.out(q2, z2):
                    return w + (s,)
                nxt.append((w + (s,), p))
        frontier = nxt[:20000]
    return None


def fit(shape, syms, labels, data, seconds, other=None, pool=(), excuse=0.0, lexicon=0, weights=None, most=None, build=0,
        warm=None):
    """A machine of this shape that answers every story of data rightly (and, given another machine and a pool of
    stories, answers some story of the pool otherwise than it). Solved first on a part of the stories; the stories the
    machine found answers wrongly are added and it is solved again (a 'no' on a part is a 'no' on all).
    excuse (a share of the stories) or most (a number): so many stories may be answered otherwise than the set; with
    weights (story -> how many times it is told), a story counts as many times as it is told."""
    lab = {l: i for i, l in enumerate(labels)}
    rng = random.Random(len(data))
    if most is None:  # stories the machine may answer otherwise than the set (its errors, or the set's)
        most = int(excuse * (sum(weights.get(s, 1) for s, _ in data) if weights else len(data)))
    part = list(data) if most else rng.sample(data, min(PART, len(data)))
    left = seconds  # (in seconds of the old limit, spent by the conflicts each solving used)
    if warm is not None and not most and not lexicon and (other is None or build):
        e, live = warm.of(shape, syms, labels)
        def say(stories):
            for seq, y in stories:
                e.says(seq, lab[y])
            e.said += stories
            e.saidset.update(stories)
        heavy = ZLONG and any(c[0] == "Z" for c in shape[2])
        if not e.said:
            say([x for x in part if len(x[0]) <= ZLONG] if heavy else part)
        how = "unsat"
        for upto in ([None] if other is None else [d for d in DEPTHS if d < build] + [build]):
            assume = []
            if other is not None:
                act, ds = block(e, other, syms, upto, build)
                assume = [act] + [-d for d in ds[upto:]]
            while True:
                m, how = e.solve(max(0.05, left), live, assume)
                left -= e.used / CONFLICTS
                if m is None:
                    break
                wrong = [(s, y) for s, y in data if m.run(s) != y]
                same = ALIKE and e.by_state == "mealy" and not e.cells
                if not wrong and other is None and (same or INERT and e.by_state == "named" and isinstance(e.chain, tuple)):
                    tot = e.alike() if same else e.inertia()  # the machine where something happens in the fewest places
                    while left > 0:
                        b = min(e.happens(), len(tot.rhs)) - 1
                        if b < 0:
                            break
                        m2, _ = e.solve(max(0.05, left), live, [-tot.rhs[b]])
                        left -= e.used / CONFLICTS
                        if m2 is None or any(m2.run(s) != y for s, y in data):
                            break
                        m = m2
                if not wrong:
                    return m, "sat"
                new = [x for x in wrong if x not in e.saidset]
                if left <= 0 or not new:
                    return None, "unknown"
                say(sorted(new, key=lambda x: (len(x[0]), x[0]))[:5] if heavy else rng.sample(new, min(20, len(new))))
            if how != "unsat":
                return None, how
        if other is not None:  # no second machine: is it that the shape explains the stories no more?
            live.conf_budget(200)
            if live.solve_limited(expect_interrupt=True) is False:
                warm.dead.add(shape[1:])
        if shape[1:] in warm.dead or other is None:
            warm.drop(shape[1:])
        return None, "unsat"
    if INCREMENTAL and not most:  # one encoding and one solver; each round adds only the stories added to the part
        e = Encoding(shape, syms, labels, lexicon)
        if other is not None and build:  # the solver writes the story the two machines part on
            story_differs(e, other, syms, build)
        elif other is not None:
            e.clauses.append([e.differs(w, other.run(w)) for w in pool])
        said = 0
        with Solver(name=ENGINE) as live:
            while True:
                for seq, y in part[said:]:
                    e.says(seq, lab[y])
                said = len(part)
                m, how = e.solve(max(0.05, left), live)
                left -= e.used / CONFLICTS
                if m is None:
                    return None, how
                wrong = [(s, y) for s, y in data if m.run(s) != y]
                if not wrong:
                    return m, "sat"
                if left <= 0:
                    return None, "unknown"
                if all(x in part for x in wrong):
                    return None, "unknown"
                part += rng.sample(wrong, min(20, len(wrong)))
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
        if other is not None and build:  # the solver writes the story the two machines part on
            story_differs(e, other, syms, build)
        elif other is not None:
            e.clauses.append([e.differs(w, other.run(w)) for w in pool])
        m, how = e.solve(max(0.05, left))
        left -= e.used / CONFLICTS
        if m is None:
            return None, how
        wrong = [(s, y) for s, y in data if m.run(s) != y]
        if sum(weights.get(s, 1) if weights else 1 for s, _ in wrong) <= most:
            return m, "sat"
        if left <= 0:
            return None, "unknown"
        if all(x in part for x in wrong):  # the solver's machine is wrong on stories it was given: nothing to add
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


def rival(shape, syms, labels, data, machine, pool, seconds, build=0, keep=False, warm=None):
    """A machine of this shape that answers every story taught on rightly and some story of the pool otherwise; the
    stories on which the two part, at most three: the shortest, the longest and one between."""
    other, how = fit(shape, syms, labels, data, seconds, machine, pool, build=build, warm=warm)
    if other is None:
        return how  # "unsat": there is no such machine; "unknown": the solver could not tell
    if build:  # the shortest story the two part on
        w = separating(machine, other, syms, 2 * build)
        return "unsat" if w is None else ([w], other) if keep else [w]
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
          workers=4, shorten=False, reserve=0, explore=False, frugal=False, guards=False, build=0, plan=False,
          committee=0, flags=False, cascades=False, rules=False, deadline=None, keep=False, dep=False, watch=False,
          warm=False, patience=0, tier=0, tour_=0, anchor=0, far=0, deep=0, mealy=False, reach=False, pairs=False,
          rare=False, known=(), lazy=False):
    """frugal: a question is asked only while it can still tell something: of the stories on which a second machine
    answers otherwise, one (its answer refutes one of the two); of the check, one by one up to the first answered wrongly
    (the counterexample of an equivalence query, Angluin 1987).
    dep: the bodies at their ceilings (see ceilings) in place of the other shapes; the kept machine is priced by itself,
    and the second machine is looked for in every body whose cheapest machine is at most slack bits above it.
    warm: the solvers of the shapes are kept from one search to the next (see Warm); one process only.
    With build, a shape proved to hold no second machine for the kept machine is not searched again while that machine
    is kept (more stories leave it so); and when the kept machine is refuted by a story of at most build symbols, every
    such shape is proved empty (a machine of it that explained the stories would have parted from the kept one there).
    patience (seconds of the solver's limit; one process only): when the search for a second machine has taken so long
    without one, one story of the check is asked meanwhile, and the search goes on.
    tour_ (watched): a story asked goes on, up to so many symbols, through the moves of the kept machine that no story
    heard has made (see tour): the story that parts two machines, the story to a read not seen, the story asked meanwhile.
    far (watched): eight of the twelve plain stories of the check are of up to so many symbols (the others as before). Seen
    after every symbol, a long story costs one question and checks the machine as far as it will be used; a check on
    stories of a dozen symbols passes machines that go wrong later. A story heard that is longer than LONG is not put
    into the formulas (they would grow with its length): the machine found is checked on it, and only the shortest
    beginning it answers wrongly is taught on.
    reach (watched): the plain stories of the check begin at the places of the kept machine (see reaches).
    pairs (watched): before the check, while questions beyond the reserve are left, stories are asked through the pairs
    of moves of the kept machine that no story heard has made (see untrodden); and a machine is 'sure' only when none
    is left.
    rare (watched): before a second machine is looked for, every symbol is tried after every beginning that ends in an
    answer barely seen (at most half the questions; not when the answers are numbers); and of the pairs
    of moves not made, first the ones whose first move leads to the place the fewest stories heard have been in (see
    untrodden).
    deep (warm, with build): the last search for a second machine, the one 'sure' rests on, writes stories of up to so
    many symbols, not build: two machines may agree on every story of a dozen symbols and part later.
    known (watched): stories already asked and paid for by another learner; they are heard again for nothing.
    lazy (with known and far): what was heard of the known stories is not put into the formulas (they would grow by
    every beginning of every story): the machine found is checked on them, and only the shortest beginning it answers
    wrongly is taught on, as with a long story."""
    tour_ = tour_ if watch else 0
    far = far if watch else 0
    deep = deep if W_ok(warm, workers) and build else 0
    def missed():  # (far) the shortest long story heard that the machine answers wrongly, if any, as a list
        have = {x for x, _ in data}
        return sorted(((x, y) for x, y in heard.items() if (len(x) > LONG or x in idle) and x not in have and machine.run(x) != y),
                      key=lambda t: (len(t[0]), t[0]))[:1]
    # where a second machine is looked for: the shapes at most slack bits longer; of the shapes with the answers as
    # states, only the kept machine's own (one with more hidden values always holds another machine)
    around = lambda: [sh for sh in every if shape[0] <= sh[0] <= shape[0] + slack and sh[1:] not in dead
                      and (not anchor or sh[3] not in ("named", "mealy") or sh == shape)]
    bad = lambda x: machine.run(x) != heard[tuple(x)] if not watch else \
        any(machine.run(x[:i]) != heard[tuple(x[:i])] for i in range(1, len(x) + 1))
    settled, settled_for, hurried = set(), None, 0
    W = Warm() if warm and workers <= 1 else None
    heard, idle = {}, set()  # (idle: beginnings heard that are kept out of the formulas until the machine misses one)
    if watch:  # the answer is seen after every symbol, as a display is read while the keys are pressed: a story asked
        ask = oracle  # (one question) tells the answers of all its beginnings; so do the stories taught on
        def oracle(w):
            for i in range(1, len(w) + 1):
                if tuple(w[:i]) not in heard:
                    heard[tuple(w[:i])] = ask(list(w[:i]))
            return heard[tuple(w)]
        for w, y in data:
            oracle(list(w))
        before = set(heard)
        for w in known:
            oracle(list(w))
        if lazy and far:
            idle = set(heard) - before
    labels = sorted({y for _, y in data})
    committee = committee if build else 0  # (a committee needs the machines a story is written against)
    asked, dead, fixed, tries, novel = 0, set(), pool, {}, 0
    under = {}  # (dep) body -> the highest ceiling proved empty: every lower ceiling of the body is empty too
    gone = lambda sh: sh[1:] in dead or (ceiling(sh) and sh[0] <= under.get(sh[1:4], -1))
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    machine = shape = last = None
    late_ = lambda: deadline is not None and time.time() > deadline  # (past the time given: the last machine is told)
    while True:
        if watch:  # (every beginning seen so far is a story taught on)
            have = {x for x, _ in data}
            data = data + sorted((x, y) for x, y in heard.items() if x not in have and x not in idle and (not far or len(x) <= LONG))
            if far and machine is not None:
                data = data + missed()
        new = sorted(set(labels) | {y for _, y in data})
        if new != labels:  # a shape proved impossible stays so: a machine with more answers that explains more
            if not labels:  # stories gives, with each new answer sent to an old one, one that explains these (with no
                dead, tries = set(), {}  # old answer, nothing was proved)
                under.clear()
            if W:  # (an encoding is of its answers: begun again; what was proved stays in dead)
                W.clear()
            labels, machine = new, None
        numeric = (guards or dep) and all(y.lstrip("-").isdigit() and str(int(y)) == y for y in labels)  # (as written)
        every = shapes(len(syms), len(labels), most_k, guards, numeric, flags, cascades, rules, dep, tier, anchor,
                       mealy)
        if machine is not None and all(machine.run(s) == y for s, y in data):
            pass  # the kept machine answered what was asked rightly: it stays
        else:
            if machine is not None and settled_for is machine and settled and \
                    any(len(s) <= build and machine.run(s) != y for s, y in data):
                log(f"  {len(settled - dead)} shapes held no second machine: they are empty now")
                dead |= settled
                if W:
                    for key in settled:
                        W.drop(key)
            again = shape if W and shape is not None and shape in every and not gone(shape) else None
            machine = shape = None
            t0 = time.time()
            for cap in (2, 5):  # a shape the solver gave up on comes back with twice the work; if nothing at all is
                live = [sh for sh in every if not gone(sh) and tries.get(sh[1:], 0) <= cap]   # found, up to 32 times
                if again is not None and cap == 2:  # (warm) the shape of the machine just refuted first: every cheaper
                    live = [again] + [sh for sh in live if sh != again]  # one is empty or was given up on
                for i in range(0, len(live), workers):  # the shapes in order of their length, several at once
                    if late_():
                        break
                    part = live[i:i + workers]
                    got = side(fit, [(sh, syms, labels, data, seconds * 2 ** tries.get(sh[1:], 0), None, (), 0.0, 0, None,
                                      None, 0, W) for sh in part], workers)
                    for sh, (m, how) in zip(part, got):
                        if m:
                            machine, shape = m, sh
                            if anchor and sh[3] == "named":
                                settle(machine, data)
                            break
                        if how == "unsat":  # only a proved 'no' is final
                            dead.add(sh[1:])
                            if ceiling(sh):
                                under[sh[1:4]] = max(sh[0], under.get(sh[1:4], -1))
                        else:
                            tries[sh[1:]] = tries.get(sh[1:], 0) + 1
                            log(f"  {name(sh)}: the solver gave up after {int(seconds * 2 ** (tries[sh[1:]] - 1) * CONFLICTS)} conflicts")
                    if machine:
                        log(f"  kept {machine} ({(shape[0] if machine.bits is None else machine.bits):.0f} bits, "
                            f"{time.time() - t0:.1f} s); {len(data)} stories, {asked} asked")
                        break
                if machine or late_():
                    break
        if far and machine is not None and missed() and not late_():  # (it explains the stories taught on, not a long
            data = data + missed()                                    # one heard: that one is taught on, and again)
            machine = None
            continue
        if machine is None:  # nothing explains every story (or the time is up): the last machine kept, if any, not sure
            if last is not None:
                last.sure = False
            return last, asked
        last = machine
        if asked >= asks:  # the questions are spent and the last check was not passed: not sure
            machine.sure = False
            return machine, asked
        if rare and watch and fixed is None and not numeric and asks - asked > reserve and novel < asks // 2 \
                and not late_():
            # before any second machine is looked for: what has barely been seen is looked at first. An answer seen
            # at the end of fewer beginnings than twice the symbols there are is new; after a beginning that ends in a
            # new answer every symbol is tried, the rarest answer and the shortest beginning first. No machine is
            # needed for it: where nearly every move falls into one place and stays (a closed connection), the kept
            # machine is wrong about the few other places for a long time, and a story chosen by chance or by a second
            # machine falls into that place too. At most half the questions go here (numbers are not new in this
            # sense: they are counted)
            seen_y, gone_on, whole = {}, {}, {}
            for x, y in heard.items():
                seen_y[y] = seen_y.get(y, 0) + 1
                gone_on[x[:-1]] = gone_on.get(x[:-1], 0) + 1  # (how many symbols have been tried after a beginning)
            for x, y in heard.items():  # (the moves from one beginning that give one answer lead to one place until
                if gone_on.get(x, 0) == len(syms):  # shown otherwise: one of them is gone on from with every symbol
                    whole[x[:-1], y] = whole.get((x[:-1], y), 0) + 1  # before a second is begun)
            new_x = [(whole.get((x[:-1], y), 0), seen_y[y], -gone_on.get(x, 0), len(x), x + (b,))
                     for x, y in heard.items() if seen_y[y] < 2 * len(syms) and len(x) < 24
                     and whole.get((x[:-1], y), 0) < 2  # (and a second, lest the first was not like the others)
                     for b in syms if x + (b,) not in heard]
            if new_x:
                x = min(new_x)[4]
                oracle(list(x))
                asked, novel = asked + 1, novel + 1
                log(f"  after an answer barely seen: asked {' '.join(x)} = {heard[x]}; "
                    f"the kept machine was {'wrong' if bad(x) else 'right'}")
                continue
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
        t0, w, doubt = time.time(), None, False
        near = window(every, shape, machine, slack, gone, len(syms), len(labels)) if dep else around()
        owed = len([w for w in unseen(machine, syms, data) if w not in seen]) if plan and explore else 0
        late = bool(reserve) and asks - asked <= reserve + owed  # the last questions are kept for the check (and, with
        if late:                                                 # plan, for the reads no story taught on ends at, asked
            near = []                                            # before it); the second machine is looked for after it
        if settled_for is not machine:
            settled, settled_for, hurried = set(), machine, 0
        near = [sh for sh in near if sh[1:] not in settled]
        unsettled, crowd, hurry, work0 = [], [], False, WORK[0]
        for more in (1, 4):  # a shape on which the solver could not tell is searched again with four times the work
            for i in range(0, len(near), workers):
                if late_():
                    doubt = True
                    break
                if patience and workers <= 1 and WORK[0] - work0 > patience * CONFLICTS:
                    hurry = True
                    break
                got = side(rival, [(sh, syms, labels, data, machine, ask_from, seconds * more, build, bool(committee), W)
                                   for sh in near[i:i + workers]], workers)
                if W and W.dead:
                    dead |= W.dead
                hit = [(sh, x) for sh, x in zip(near[i:i + workers], got) if isinstance(x, (list, tuple))]
                if build and not committee:
                    settled |= {sh[1:] for sh, x in zip(near[i:i + workers], got) if x == "unsat"}
                unsettled += [sh for sh, x in zip(near[i:i + workers], got) if x == "unknown"]
                if committee:  # several second machines, of several shapes, the cheapest first
                    crowd += hit
                    if len(crowd) >= committee:
                        break
                elif hit:
                    sh, w = hit[0]
                    break
            if committee and crowd:
                # query by committee (Seung, Opper and Sompolinsky 1992): of the stories that part any two of the machines,
                # the one on which their answers are split most evenly; a single answer then refutes the most of them
                crowd = crowd[:committee]
                ms = [machine] + [x[1] for _, x in crowd]
                cands = [x[0][0] for _, x in crowd]
                for p in range(1, len(ms)):
                    for q in range(p + 1, len(ms)):
                        s = separating(ms[p], ms[q], syms, 2 * build)
                        if s is not None:
                            cands.append(s)
                def split(s):
                    cnt = {}
                    for m in ms:
                        y = m.run(s)
                        cnt[y] = cnt.get(y, 0) + 1
                    return -sum(c / len(ms) * math.log2(c / len(ms)) for c in cnt.values())
                best = max(sorted(set(cands), key=lambda s: (len(s), s)), key=split)
                sh, w = crowd[0][0], [best]
                log(f"  a committee of {len(ms)} machines; the story that splits them most: {' '.join(best)}")
                break
            if w is not None or not unsettled or hurry:
                break
            near, unsettled = unsettled, []
        if w is None and unsettled and not hurry:
            log(f"  of {len(unsettled)} shapes the solver could not tell whether a second machine is there")
            doubt = DOUBT
        if w is not None:
            w = w[:1 if frugal else max(1, asks - asked)]
            if tour_:
                w = [tour(machine, syms, heard, x, max(tour_, len(x)), rng=rng) for x in w]
            told = [(x, oracle(list(x))) for x in w]
            asked += len(told)
            wrong = sum(bad(x) for x, y in told) if watch else sum(machine.run(x) != y for x, y in told)
            log(f"  {name(sh)} answers otherwise on {len(w)} stories ({time.time() - t0:.1f} s): asked "
                f"{' | '.join(' '.join(x) + ' = ' + str(y) for x, y in told)}; the kept machine was wrong on {wrong}")
            data = data + told
            continue
        if fixed is not None:  # nothing longer can be asked
            return machine, asked
        if pairs and watch and not late:  # (the pairs of moves not made yet, while there are questions for them)
            x = untrodden(machine, syms, heard, max(tour_, 24), rare=rare)
            if x is not None and x not in heard:
                told = [(x, oracle(list(x)))]
                asked += 1
                log(f"  pairs of moves no story has made: asked {' '.join(x)} = {told[0][1]}; "
                    f"the kept machine was {'wrong' if bad(x) else 'right'}")
                data = data + told
                continue
        if explore and not machine.mealy:  # first the reads the machine can reach that no story taught on ends at: their answers are free
            free = [w for w in unseen(machine, syms, data) if w not in seen][:max(0, asks - asked)]  # (out of the reserve too)
            if free and tour_:  # one story to the first of them and on through the moves not made; then the rest anew
                x = tour(machine, syms, heard, free[0], max(tour_, len(free[0])), rng=rng)
                told = [(x, oracle(list(x)))]
                asked += 1
                log(f"  {len(free)} reads no story taught on ends at: asked {' '.join(x)} = {told[0][1]}; "
                    f"the kept machine was {'wrong' if bad(x) else 'right'}")
                data = data + told
                continue
            if free:
                told = [(w, oracle(list(w))) for w in free]
                asked += len(told)
                wrong = sum(machine.run(w) != y for w, y in told)
                log(f"  {len(free)} reads no story taught on ends at: asked; the kept machine was wrong on {wrong}")
                data = data + told
                if wrong:
                    continue
        # the check by trying: a few longer stories, pumped and plain
        begun = reaches(machine, syms, heard) if reach and watch else [()]  # (the plain stories begin at the places
        begun = begun or [()]                                             # of the machine, the least visited first)
        tried = [begun[i % len(begun)] +
                 tuple(rng.choice(syms) for _ in range(max(4, (rng.randint(far // 2, far) if far and i % 3 else
                                                              rng.randint(3 * longest, 6 * longest)) -
                                                       len(begun[i % len(begun)])))) for i in range(12)] + \
            pumped(syms, rng, 4 * longest, 8 * longest, 3)
        # and stories on which the kept machine gives every answer it can give, the rarest first: where an answer is
        # rare among long stories, plain trying would almost never ask about it
        long = [tuple(rng.choice(syms) for _ in range(rng.randint(3 * longest, 6 * longest))) for _ in range(400)]
        long += pumped(syms, rng, 3 * longest, 6 * longest, 200)
        by = {}
        for x in long:
            by.setdefault(machine.run(x), []).append(x)
        tried += [by[y][0] for y in sorted(by, key=lambda y: len(by[y]))[:6]]
        full = len(tried)
        if hurry:  # (the search for a second machine is long: one story of the check meanwhile, each time another)
            tried = (tried[15:] + tried[12:15] + tried[:12])[hurried % full:][:1]
            hurried += 1
            x = tour(machine, syms, heard, (), tour_) if tour_ else ()
            if len(x) >= 2 and x not in heard:  # (first the moves of the machine no story has made)
                tried = [x]
        tried = tried[:asks - asked]  # never more than the questions allowed
        if not tried:
            machine.sure = False
            return machine, asked
        if frugal and not hurry:  # the stories that cover the answers first, then the pumped, then the plain; up to the
            tried = tried[15:] + tried[12:15] + tried[:12]  # first wrong
            told = []
            for w in tried:
                told.append((w, oracle(list(w))))
                if machine.run(w) != told[-1][1] or \
                        (watch and any(machine.run(w[:i]) != heard[tuple(w[:i])] for i in range(1, len(w)))):
                    break
        else:
            told = [(w, oracle(list(w))) for w in tried]
        asked += len(told)
        wrong = [(w, y) for w, y in told if machine.run(w) != y]
        if watch and not wrong:  # (watched: a beginning answered wrongly is a story answered wrongly)
            wrong = sorted(((x, y) for x, y in heard.items() if machine.run(x) != y), key=lambda t: len(t[0]))[:2]
        if not wrong and hurry:
            log(f"  the search for a second machine is long: a story of the check asked meanwhile, answered rightly")
            data = data + [x for x in told if x[0] not in seen]
            continue
        if not wrong:
            if late and len(told) >= full:  # the second machine was not looked for on these stories (the questions were
                # kept for the check): it is looked for now, on them and the check, and nothing more is asked; if there
                # is one, or the solver cannot tell, the machine kept is not sure
                more = [x for x in data + told if x[1] in labels  # (a number answered rightly may be a new answer)
                        and (not far or len(x[0]) <= LONG)]
                near = window(every, shape, machine, slack, gone, len(syms), len(labels)) if dep else around()
                if not deep:  # (no second machine there: more stories leave it so; with deep, none on a short story)
                    near = [sh for sh in near if sh[1:] not in settled]
                for i in range(0, len(near), workers):
                    if late_():
                        doubt = True
                        break
                    got = side(rival, [(sh, syms, labels, more, machine, ask_from, seconds, max(build, deep), False, W)
                                       for sh in near[i:i + workers]], workers)
                    if any(isinstance(x, (list, tuple)) or x == "unknown" for x in got):
                        doubt = True
                        break
            log(f"  {'a second machine is not ruled out' if doubt else f'no other machine within {slack} bits'}; "
                f"{len(told)} longer stories asked, all answered rightly ({time.time() - t0:.1f} s)")
            machine.sure = len(told) >= full and not doubt  # a check cut short by the limit of questions is not passed
            if pairs and watch and machine.sure and untrodden(machine, syms, heard, 24) is not None:
                log("  pairs of moves are left that no story has made: not sure")
                machine.sure = False
            return machine, asked
        log(f"  no other machine within {slack} bits; of {len(told)} longer stories asked, {len(wrong)} answered wrongly")
        if not shorten:
            data = data + told
            continue
        # a long story answered wrongly is not put into the stories taught on as it is (every formula after would
        # carry it): the beginning where the machine first goes wrong is found by halving (Rivest and Schapire 1993)
        known = dict(data)
        known.update(heard)  # (watched: the beginnings are seen already, halving asks nothing)
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
        if keep:  # no answer paid for is thrown away: the check stories answered rightly, the long story answered
            have = {x for x, _ in data}  # wrongly, and every beginning asked while halving it
            more = [(x, y) for x, y in told if x not in have] + [(x, y) for x, y in known.items() if x not in have]
            if far:
                more = [t for t in more if len(t[0]) <= LONG]
            data = data + list(dict(more).items())


def window(every, shape, machine, slack, gone, nsym, nlab):
    """(dep) Where a second machine is looked for: every live body whose cheapest machine is at most slack bits above
    the kept machine's own price, at that ceiling (a ceiling holds every cheaper machine of its body), and the other
    shapes as without dep; the cheapest first."""
    lo = shape[0] if machine.bits is None else machine.bits  # (not top - slack: that may come back a bit above lo)
    top = lo + slack
    out = [sh for sh in every if not ceiling(sh) and lo <= sh[0] <= top and not gone(sh)]
    for k, cs, by in dict.fromkeys(sh[1:4] for sh in every if ceiling(sh)):
        sh = (top, k, cs, by, ("a", top))
        if lowest(k, cs, nsym, nlab, by) <= top and not gone(sh):
            out.append(sh)
    return sorted(out, key=lambda s: (lowest(s[1], s[2], nsym, nlab, s[3]) if ceiling(s) else s[0], s[1], s[2], str(s[3])))


def unseen(machine, syms, data, most=12, depth=12):
    """Reads (the state and the values the answer is read from) the machine can reach but no story taught on ends at;
    for each, the shortest story that leads there (a walk in breadth over the machine). Their answers are free in the
    machine found, so they are asked (the observation table of L* made closed)."""
    def at(seq):
        if machine.lex is not None:
            seq = [machine.lex[a] for a in seq]
        q, z = machine.start, list(machine.zstart)
        for a in seq:
            q, z = machine.move(q, z, a)
        return q, tuple(z)

    def read(conf):
        q, z = conf
        r = (q if machine.by_state else 0,) + tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v
                                                    for c, v in zip(machine.cells, z))
        return (0, r[-1]) if machine.by_state == "last" else r
    seen = {read(at(s)) for s, _ in data}
    found, frontier, visited = [], [((), at(()))], {at(())}
    for _ in range(depth):
        nxt = []
        for seq, conf in frontier:
            for a in syms:
                s2 = seq + (a,)
                c2 = at(s2)
                if c2 in visited:
                    continue
                visited.add(c2)
                r = read(c2)
                if r not in seen:
                    seen.add(r)
                    found.append(s2)
                    if len(found) >= most:
                        return found
                nxt.append((s2, c2))
        frontier = nxt[:5000]
    return found


def reaches(machine, syms, stories, most=64, depth=8):
    """The shortest story to every place the machine can be in (its state and what its cells read), those where the
    fewest stories heard have ended first. A check that begins at every place of the machine tests each of them; a
    story begun at random soon sits where most stories end (a closed connection, an alarm that stays on)."""
    def view(q, z):
        return (q,) + tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(machine.cells, z))
    ends = {}
    for s in stories:
        q, z = machine.start, list(machine.zstart)
        for a in s:
            q, z = machine.move(q, z, a)
        ends[view(q, z)] = ends.get(view(q, z), 0) + 1
    c0 = (machine.start, tuple(machine.zstart))
    to, frontier, met = {}, [((), c0)], {c0}
    for _ in range(depth):
        nxt = []
        for p, (q, z) in frontier:
            for a in syms:
                q2, z2 = machine.move(q, list(z), a)
                c2 = (q2, tuple(z2))
                if c2 in met:
                    continue
                met.add(c2)
                to.setdefault(view(q2, z2), p + (a,))
                nxt.append((p + (a,), c2))
        frontier = nxt[:2000]
        if len(to) >= most:
            break
    return [p for v, p in sorted(to.items(), key=lambda t: (ends.get(t[0], 0), len(t[1]), t[1]))][:most]


def untrodden(machine, syms, stories, most=24, depth=8, rare=False):
    """(watched) A story through pairs of moves, one after the other, that no story heard has made: from a place of the
    machine (its state and what its cells read), a symbol and then another. A machine one state short of the true one
    sends some move to the wrong place, and it shows in the answer to the very next symbol there: a machine whose every
    pair of moves has been heard is not such a machine, unless the next answer does not tell the places apart (the
    test for a hidden state more: Chow 1978, Vasilevskii 1973, with the answer after every symbol as the suffix).
    Returns the story, or None when every pair within reach has been heard.
    rare: of the pairs not made, first the one whose first move leads to the place where the fewest stories heard
    have been (then the nearest). Where most moves fall into one place and stay (a closed connection), the pairs
    through it are most of the pairs and tell the least: the places seldom seen are where a state more can hide."""
    def view(q, z):
        return (q,) + tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(machine.cells, z))
    done, been = set(), {}
    for s in stories:
        q, z = machine.start, list(machine.zstart)
        v0 = a0 = None
        for a in s:
            v = view(q, z)
            if v0 is not None:
                done.add((v0, a0, a))
            v0, a0 = v, a
            q, z = machine.move(q, z, a)
        been[view(q, z)] = been.get(view(q, z), 0) + 1
    story, q, z = [], machine.start, list(machine.zstart)
    while len(story) + 2 <= most:
        frontier, met, path, best = [((), q, tuple(z))], {(q, tuple(z))}, None, None
        for _ in range(depth):
            nxt = []
            for p, q1, z1 in frontier:
                for a in syms:
                    lack = [b for b in syms if (view(q1, z1), a, b) not in done]
                    q2, z2 = machine.move(q1, list(z1), a)
                    if lack and rare:
                        key = (been.get(view(q2, z2), 0), len(p))
                        if best is None or key < best[0]:
                            best = (key, p + (a, lack[0]))
                    elif lack:
                        path = p + (a, lack[0])
                        break
                    if (q2, tuple(z2)) not in met:
                        met.add((q2, tuple(z2)))
                        nxt.append((p + (a,), q2, tuple(z2)))
                if path:
                    break
            if path or not nxt:
                break
            frontier = nxt[:2000]
        if rare and best is not None:
            path = best[1]
        if path is None:
            break
        v0 = a0 = None
        for a in path[:most - len(story)]:
            v = view(q, z)
            if v0 is not None:
                done.add((v0, a0, a))
            v0, a0 = v, a
            q, z = machine.move(q, z, a)
            story.append(a)
        # (the last move of the piece is paired with the first of the next: gone on from here)
    return tuple(story) if story else None


def tour(machine, syms, stories, start=(), most=24, depth=6, rng=None):
    """(watched) A story that goes on from `start` through the moves of the machine that no story heard has made yet, the
    nearest first (a move: what the machine sees, its state and what its cells read, and a symbol). The answer is seen
    after every symbol, so one question checks every move on the way (a transition tour: Naito and Tsunoyama 1981).
    rng: when no such move is near, the story goes on by chance up to its length (a story of the check, for nothing)."""
    def view(q, z):
        return (q,) + tuple(max(-c[2], min(c[2], v)) if c[0] == "Z" else v for c, v in zip(machine.cells, z))
    done = set()
    for s in stories:
        q, z = machine.start, list(machine.zstart)
        for a in s:
            done.add((view(q, z), a))
            q, z = machine.move(q, z, a)
    story = list(start)
    q, z = machine.start, list(machine.zstart)
    for a in story:
        done.add((view(q, z), a))
        q, z = machine.move(q, z, a)
    while len(story) < most:
        frontier, met, path = [((), q, tuple(z))], {(q, tuple(z))}, None
        for _ in range(depth):
            nxt = []
            for p, q1, z1 in frontier:
                for a in syms:
                    if (view(q1, z1), a) not in done:
                        path = p + (a,)
                        break
                    q2, z2 = machine.move(q1, list(z1), a)
                    if (q2, tuple(z2)) not in met:
                        met.add((q2, tuple(z2)))
                        nxt.append((p + (a,), q2, tuple(z2)))
                if path:
                    break
            if path or not nxt:
                break
            frontier = nxt[:2000]
        if path is None and rng is None:
            break
        if path is None:
            path = (rng.choice(syms),)
        for a in path[:most - len(story)]:
            done.add((view(q, z), a))
            q, z = machine.move(q, z, a)
            story.append(a)
    return tuple(story)


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
