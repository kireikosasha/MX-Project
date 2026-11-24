"""A display watched while keys are pressed, explained by a few small quantities.

cells.py looks for a machine as a table (states, cells and their moves found together by a solver). A rule written by a
person is more often a few quantities, each moved by two or three events and by nothing else: guests come and go, wood
is stacked and burnt, a night is cold when there are guests and no wood. Here such a rule is found the way a person
finds it:

  - a quantity (a part) is a counter that does not go below zero, a level with a top, or a circle;
  - it is moved by at most three events: a symbol, or a symbol together with what the display was seen to do at it (the
    number went up by one; the label changed). One of its moves may be made only when another part reads something
    (a guard); or one of its moves may add all that another part holds (what is on the bench goes to the shelf);
  - a row of flags, one for each value of a circle or level (a seat for every place the wheel can stop at): an event
    raises, lowers or turns over the flag where that part points, and the flag where it points is what is read;
  - a stack, the last put on top: a symbol puts someone of its kind on top, another takes the top one off; who is on
    top, who is under it, and whether one or more are there is what is read (the next rung of Chomsky's ladder after
    the counters: brackets of several kinds, undo);
  - a line, first come first served: a symbol brings someone of its kind to the back, another sends away the first
    (or the last), another empties the line or sends away every one of a kind; who is first, who is last, how many
    there are (of a kind) is what is read;
  - what the display does at a symbol is a table over what is seen before the symbol (the label; whether the number is
    zero) and what one or two parts read (zero or not, at the top or not, the value itself; or how one part stands to
    another: below it, level with it, above it); or the number shown goes up, or down, by what a part holds;
  - nothing is searched for by a solver. Every part of a small description is tried, run over every story heard, and
    kept if it makes the display a function of what is read; of the explanations found, the shortest is told;
  - a number shown is taken as a sum: what is explained is what each symbol adds to it. (When that explains nothing, the
    numbers are taken as names.)

Stories are then asked, one question each, seen after every symbol; where the machine is wrong the parts are looked for
again. The machine is 'sure' when it has answered the last SURE stories asked, the long ones among them.

    python parts.py FAMILY.py [--asks 40]
"""
import argparse, importlib.util, itertools, math, random, time
import numpy as np

ACTS = [("add", 1), ("add", -1), ("set", 0), ("set", "top"), ("add", 2), ("add", -2), ("add", 3), ("add", -3)]
GROW = 150  # so many of the parts that tell the most of a symbol are given one more slot, or made to wait on a guard
READS = ["zero", "full", "top", "ge2", "ge3"]
KINDS = [("counter", 0)] + [("level", n) for n in range(2, 8)] + [("circle", n) for n in range(2, 8)] + [("circle", 12)]
FIRSTS = 8  # with so many of the parts that tell the most of a symbol alone, a second part is looked for
GUARDS = 12  # so many plain parts, the ones that settle the most steps of a symbol by themselves, are tried as guards
GIVERS = 40  # so many plain counters that some slot empties are tried as what another part takes at that slot
BIG = 120  # (a counter is kept below this: values are held in a byte)
ROWS = 7  # a row of flags is one flag for each value of a circle or level of at most so many values
FLAG_ACTS = [("flag", 1), ("flag", 0), ("flag", -1)]  # raise, lower, turn over the flag where the address points
SWITCHES = 8  # so many plain parts of two values are tried as what turns the way an address moves
QLEN = 5  # a line holds at most so many, of at most three kinds (each kind is brought by one symbol)
SDEPTH = 30  # (in the search) a stack holds at most so many; the stack of the machine found holds any number
STACK_READS = ["top", "under", "topn"]  # who is on top; who is under it; who is on top and whether one or more are there
LINE_READS = ["front", "back", "len", "count1", "count2"]
OTHERS = 3  # so many other explanations of a symbol, the next cheapest, are kept as rivals of the one told
SHARED = 64  # so many of them are weighed again with the parts other symbols read already paid for
SURE = 10  # the machine is sure when no rival is left and it answered the last so many stories asked, long ones
LONG = (60, 120)  # the last stories asked are of so many symbols
EARLY = (5, 8)  # (give_up) the learner gives up when its first 5 questions, or 7 of its first 8, all refuted its machine


class Heard:
    """The stories heard, as a tree of their beginnings: for every beginning the symbol it ends in, the beginning before
    it, and what was shown after it."""

    def __init__(self, syms, begin):
        self.syms, self.begin = list(syms), begin
        self.numeric = begin.lstrip("-").isdigit()
        self.index = {(): 0}
        self.parent, self.sym, self.shown = [0], [-1], [begin]

    def add(self, story, shows):
        """A story and what was shown after each of its symbols."""
        for i in range(1, len(story) + 1):
            p = tuple(story[:i])
            if p not in self.index:
                self.index[p] = len(self.parent)
                self.parent.append(self.index[p[:-1]])
                self.sym.append(self.syms.index(p[-1]))
                self.shown.append(shows[i - 1])
                self.numeric = self.numeric and shows[i - 1].lstrip("-").isdigit()


class View:
    """What the display does at every beginning heard, and what is seen before it.
    sum (a number shown): the symbol's effect is what it added; before it the number is seen as 0, 1 or more.
    names: the effect is the new label, or -1 when the label stays; before it the label is seen."""

    def __init__(self, H, summed):
        self.H, self.summed = H, summed
        self.par, self.sym = np.array(H.parent), np.array(H.sym)
        par = self.par
        if summed:
            self.names = None
            val = np.array([int(s) for s in H.shown])
            self.eff = val - val[par]
            self.ctx = np.clip(val[par], 0, 2)
            seen = self.eff
        else:
            self.names = sorted(set(H.shown))
            val = np.array([self.names.index(s) for s in H.shown])
            self.eff = np.where(val == val[par], -1, val)
            self.ctx = val[par]
            seen = (val != val[par]).astype(int)
        # the slots a part may be moved by: every symbol; and, for a symbol the display does more than one thing at,
        # the symbol together with what was seen (numbers: how much was added; names: whether the label changed)
        S = len(H.syms)
        kinds = {a: sorted({int(e) for a2, e in zip(self.sym[1:], seen[1:]) if a2 == a}) for a in range(S)}
        self.slots = [(a, None) for a in range(S)] + [(a, e) for a in range(S) if len(kinds[a]) > 1 for e in kinds[a]]
        self.slot = {k: i for i, k in enumerate(self.slots)}
        self.sslot = np.array([-1] + [int(a) for a in self.sym[1:]])
        self.eslot = np.array([-1] + [self.slot.get((int(a), int(e)), -1) for a, e in zip(self.sym[1:], seen[1:])])

    def seen_of(self, eff, before, after):
        return eff if self.summed else int(after != before)


def acts_of(kind, n):
    """The actions a part of this kind may be moved by."""
    out = []
    for act in ACTS:
        if kind == "counter" and act == ("set", "top"):
            continue
        if kind == "circle" and (act == ("set", "top") and n > 2 or act == ("add", -1) and n == 2 or
                                 act[0] == "add" and abs(act[1]) > 1 and n <= 3):
            continue  # (a circle of two is a switch: turned over, or set either way)
        if kind == "level" and act[0] == "add" and abs(act[1]) >= n:
            continue
        out.append(act)
    return out


def clash(touched, V):
    """Slots that do not go together: one symbol twice, unless as two of the things seen at it (at one reading of the
    guard) or as one slot at the two readings of the guard."""
    W, slots = len(V.slots), V.slots
    for s1, s2 in itertools.combinations(touched, 2):
        b1, b2 = s1 % W, s2 % W
        if slots[b1][0] != slots[b2][0]:
            continue
        if (slots[b1][1] is None) != (slots[b2][1] is None) or (b1 == b2 and (s1 < W or s2 < W)) or \
                (b1 != b2 and s1 // W != s2 // W):
            return True
    return False


def idle(kind, touched, chosen, V):
    """A part that says nothing new: it holds one value, nothing ever raises it, or it is a plainer part said the long
    way (the two things seen at a symbol, or the two readings of the guard, with one move)."""
    W, slots = len(V.slots), V.slots
    if all(a[0] == "set" for a in chosen) and len({a[1] for a in chosen}) < 2:
        return True
    if kind != "circle" and not any(a[0] == "set" and a[1] == "top" or a[0] in ("add", "take") and a[1] > 0 for a in chosen):
        return True
    for (s1, a1), (s2, a2) in itertools.combinations(zip(touched, chosen), 2):
        if a1 == a2 and slots[s1 % W][0] == slots[s2 % W][0]:
            if s1 % W == s2 % W or sum(sl[0] == slots[s1 % W][0] and sl[1] is not None for sl in slots) == 2:
                return True
    return False


def forms(kind, n, touched, chosen):
    """The part, and (a level moved by more than one) the part that does not make a move that does not fit."""
    big = any(a[0] == "add" and abs(a[1]) > 1 for a in chosen)
    return [(kind, n, reject, tuple(touched), tuple(chosen), None) for reject in ((False, True) if big and kind == "level" else (False,))]


def templates(V, most=2):
    """Every part of a small description: (kind, size, a move that does not fit is not made, the slots that move it,
    their actions, the guard or None)."""
    out = []
    for kind, n in KINDS:
        acts = acts_of(kind, n)
        for r in range(1, most + 1):
            for touched in itertools.combinations(range(len(V.slots)), r):
                if clash(touched, V):
                    continue
                for chosen in itertools.product(acts, repeat=r):
                    if not idle(kind, touched, chosen, V):
                        out += forms(kind, n, touched, chosen)
    return out


def grown(ts, V):
    """The parts with one more slot each (every slot that goes with theirs, every action)."""
    out, seen = [], set()
    for kind, n, _, touched, chosen, _ in ts:
        for s in range(len(V.slots)):
            if s in touched or clash(touched + (s,), V):
                continue
            for act in acts_of(kind, n):
                order = sorted(zip(touched + (s,), chosen + (act,)))
                t2, c2 = tuple(x for x, _ in order), tuple(x for _, x in order)
                if (kind, n, t2, c2) not in seen and not idle(kind, t2, c2, V):
                    seen.add((kind, n, t2, c2))
                    out += forms(kind, n, t2, c2)
    return out


def guarded(ts, V):
    """The parts (without a guard) made to wait on one: one of the slots only at one reading of the guard; one of the
    slots at the two readings with two moves; or one more slot, at one reading. The slot s whatever the guard reads is
    s; only when it reads 0, W + s; only when it reads 1, 2 W + s."""
    W, out, seen = len(V.slots), [], set()

    def add(kind, n, touched, chosen):
        order = sorted(zip(touched, chosen))
        t2, c2 = tuple(x for x, _ in order), tuple(x for _, x in order)
        if (kind, n, t2, c2) not in seen and not clash(t2, V) and not idle(kind, t2, c2, V):
            seen.add((kind, n, t2, c2))
            out.extend(forms(kind, n, t2, c2))
    for kind, n, _, touched, chosen, _ in ts:
        acts = acts_of(kind, n)
        for i, s in enumerate(touched):
            rest_t, rest_c = touched[:i] + touched[i + 1:], chosen[:i] + chosen[i + 1:]
            for b in (0, 1):
                add(kind, n, rest_t + (W * (1 + b) + s,), rest_c + (chosen[i],))
                for act in acts:
                    if act != chosen[i]:
                        add(kind, n, rest_t + (W * (1 + b) + s, W * (2 - b) + s), rest_c + (chosen[i], act))
        if len(touched) <= 2:
            for s in range(W):
                if s not in touched:
                    for b in (0, 1):
                        for act in acts:
                            add(kind, n, touched + (W * (1 + b) + s,), chosen + (act,))
    return out


def price(t, W):
    """The length of a part's description, in bits: its kind (a level and a circle say their size), and for every slot
    which it is and what it does; a guard, and which reading of it each slot is for."""
    kind, n, reject, touched, chosen, guard = t
    return math.log2(3) + (0 if kind == "counter" else math.log2(7)) + (1 if reject else 0) + \
        len(touched) * (math.log2(W) + math.log2(len(ACTS))) + \
        (0 if guard is None else 2 if guard[1] in ("gives", "address") else 2 + len(touched))


def run_all(ts, V, g=None, gv=None):
    """The value of every part at every beginning heard. g (what a guard reads, 0 or 1, at every beginning): the parts
    are guarded ones. gv (what a giver holds at every beginning): the parts take it at one of their slots."""
    W, par = len(V.slots), V.par
    C, width = len(ts), W if g is None else 3 * W
    add, setv = np.zeros((C, width + 1), np.int16), -np.ones((C, width + 1), np.int16)  # (the last column: no slot)
    take = np.zeros((C, width + 1), np.int16)
    kind, size, rej = np.zeros(C, np.int8), np.zeros(C, np.int16), np.zeros(C, bool)
    for i, (k, n, reject, touched, chosen, _) in enumerate(ts):
        kind[i], size[i], rej[i] = ("counter", "level", "circle").index(k), n, reject
        for s, (what, v) in zip(touched, chosen):
            if what == "add":
                add[i, s] = v
            elif what == "take":
                take[i, s] = v
            else:
                setv[i, s] = n - 1 if v == "top" else v
    top = np.where(kind == 0, BIG, size - 1).astype(np.int16)
    mod = np.maximum(size, 1)
    out = np.zeros((C, len(par)), np.int8)
    for i in range(1, len(par)):
        v = out[:, par[i]].astype(np.int16)
        hit = [V.sslot[i], V.eslot[i]]  # (the plainest slot first: a later one's 'set' is the one made)
        if g is not None:
            b = W * (1 + int(g[par[i]]))
            hit += [b + V.sslot[i], b + V.eslot[i] if V.eslot[i] >= 0 else -1]
        a, s = 0, -1
        for h in hit:
            if h >= 0:
                a = a + add[:, h]
                if gv is not None:
                    a = a + take[:, h] * int(gv[par[i]])
                s = np.where(setv[:, h] >= 0, setv[:, h], s)
        w = v + a
        nv = np.where(kind == 2, np.mod(w, mod), np.where(rej & ((w < 0) | (w > top)), v, np.clip(w, 0, top)))
        out[:, i] = np.where(s >= 0, s, nv)
    return out


def read_of(v, t, how):
    """What is read of one part at value(s) v."""
    if how == "zero":
        return (v >= 1) * 1
    if how in ("full", "value+", "value-"):
        return v
    if how == "top":
        return (v == t[1] - 1) * 1
    return (v >= int(how[2])) * 1


def valued(how):
    return isinstance(how, str) and how.startswith("value")


def reading_at(vals, parts, j, how):
    """What is read at the values of the parts: part j read in the way `how`; or, for ("cmp", k), how part j stands to
    part k (0 below it, 1 level with it, 2 above it)."""
    if isinstance(how, tuple):
        return (vals[j] > vals[how[1]]) - (vals[j] < vals[how[1]]) + 1
    if how == "here":  # (a row of flags: the flag where its address points)
        return (int(vals[j]) >> int(vals[parts.index(parts[j][5][0])])) & 1
    if how in LINE_READS:
        return int(LINE_READ[how][int(vals[j])])
    if how in STACK_READS:
        return stack_read(int(vals[j]), how)
    return int(read_of(vals[j], parts[j], how))


def reading(Vs, sizes, counters, how):
    """What is read of many parts, and for which of them the reading is of this kind."""
    if how == "zero":
        return (Vs >= 1).astype(np.int8), np.ones(len(sizes), bool)
    if how in ("value+", "value-"):
        return Vs, np.ones(len(sizes), bool)
    if how == "full":
        return Vs, ~counters & (sizes > 2) & (sizes <= 12)
    if how == "top":
        return (Vs == (sizes - 1)[:, None]).astype(np.int8), ~counters & (sizes > 2)
    c = int(how[2])
    return (Vs >= c).astype(np.int8), counters | (sizes > c + 1)


def counted(F, ctx, y):
    """For every row of F and every reading (ctx, value of F) met: how many of the steps of each answer are read so."""
    Y = np.stack([(y == c) for c in sorted(set(y.tolist()))], 1).astype(np.float32)
    for c in sorted(set(ctx.tolist())):
        cols = ctx == c
        Fc, Yc = F[:, cols], Y[cols]
        for v in range(int(Fc.max()) + 1 if Fc.size else 0):
            yield (Fc == v).astype(np.float32) @ Yc


def untold(F, ctx, y):
    """For every row of F: how many pairs of steps are read alike, by (ctx, F), and have different y (0: y is a function
    of what is read); and how many different readings are met. A count of pairs, not of errors: a reading that sets
    apart some steps of one answer tells something even while the commonest answer stays the commonest everywhere."""
    out, met = np.zeros(F.shape[0], np.int64), np.zeros(F.shape[0], np.int64)
    for counts in counted(F, ctx, y):
        n = counts.sum(1)
        out += ((n ** 2 - (counts ** 2).sum(1)) / 2).astype(np.int64)
        met += n > 0
    return out, met


def settled(F, ctx, y):
    """For every row of F: how many of the steps are read as no step with another y is: the steps the reading settles
    by itself (the part of a rule that holds 'only when' settles all the steps where it does not hold)."""
    out = np.zeros(F.shape[0], np.int64)
    for counts in counted(F, ctx, y):
        out += np.where(counts.max(1) == counts.sum(1), counts.sum(1), 0).astype(np.int64)
    return out


class Pool:
    """Parts with their values at every beginning heard; parts with the same values are one part (the cheapest)."""

    def __init__(self, N):
        self.ts, self.cost, self.V, self.seen = [], [], [np.zeros((0, N), np.int8)], set()

    def take(self, ts, cost, V):
        keep = []
        for i in np.argsort(np.array(cost), kind="stable"):
            h = V[i].tobytes()
            if h not in self.seen:
                self.seen.add(h)
                keep.append(i)
        self.ts += [ts[i] for i in keep]
        self.cost += [cost[i] for i in keep]
        self.V.append(V[keep])

    def done(self):
        self.V = [np.concatenate(self.V)]
        self.costs = np.array(self.cost)
        self.sizes = np.array([t[1] for t in self.ts], np.int16)
        self.counters = np.array([t[0] == "counter" for t in self.ts])
        return self

    def values(self):
        return self.V[0]


def pool_of(ts, V, W, g=None, gv=None, chunk=40000):
    pool = Pool(len(V.par))
    for lo in range(0, len(ts), chunk):
        part = ts[lo:lo + chunk]
        pool.take(part, [price(t, W) for t in part], run_all(part, V, g, gv))
    return pool.done()


class Told:
    """What the parts of a pool read before each step of one symbol: for every kind of reading the different readings
    met, each with the cheapest part that reads so."""

    def __init__(self, pool, steps, par, y, summed=False):
        self.pool, self.Vs, self.y, self.summed = pool, pool.values()[:, par[steps]], y, summed
        order = np.argsort(pool.costs, kind="stable")
        mix = np.random.RandomState(0).randint(1, 2 ** 40, size=len(steps)).astype(np.int64)
        self.feats = {}
        for how in READS:
            F, ok = reading(self.Vs, pool.sizes, pool.counters, how)
            idx = order[ok[order]]
            if len(idx):
                Fo = F[idx]
                _, first = np.unique(Fo.astype(np.int64) @ mix, return_index=True)  # (the same readings, the same sum)
                first.sort()
                self.feats[how] = (np.ascontiguousarray(Fo[first]), idx[first])
        self.nlab = math.log2(max(2, len(set(y.tolist()))))

    def feat(self, i, how):
        return reading(self.Vs[i:i + 1], self.pool.sizes[i:i + 1], self.pool.counters[i:i + 1], how)[0][0].astype(np.int64)

    def rows(self, c):  # (the table: a row for every reading met)
        return len(np.unique(c)) * self.nlab

    def tellers(self, c):
        """Every (part, reading) that with c tells more of y than c alone: (how much is left untold, price, this, part,
        reading). What is left untold is the pairs of steps read alike with different y, as a share of what readings
        as many, dealt by chance, would leave: a fine reading sets steps apart by chance alone."""
        found = []
        base, groups = untold(np.zeros((1, len(self.y)), np.int8), c, self.y)
        for how, (Fu, rep) in self.feats.items():
            left, met = untold(Fu, c, self.y)
            for j in np.nonzero(left < base[0])[0]:
                found.append((float(left[j]) * met[j] / (base[0] * groups[0]), float(self.pool.costs[rep[j]]) + (0.5 if how != "zero" else 0),
                              self, int(rep[j]), how))
        if self.summed and groups[0] == 1 and len(self.pool.ts):  # the number goes up (down) by what the part holds
            for how, sign in (("value+", 1), ("value-", -1)):
                same = np.nonzero((self.Vs == sign * self.y[None, :]).all(1))[0]
                if len(same):
                    i = same[np.argmin(self.pool.costs[same])]
                    found.append((0.0, float(self.pool.costs[i]), self, int(i), how))
        return found


def compared(T, vf, first, c):
    """The parts of a pool that, by how they stand to a given part (below it, level with it, above it), make y a
    function with c: tellers as of Told.tellers, each with its reading at the steps. Counters and levels only."""
    idx = np.nonzero(np.array([t[0] != "circle" for t in T.pool.ts]))[0]
    if not len(idx):
        return []
    idx = idx[np.argsort(T.pool.costs[idx], kind="stable")]
    D = (np.sign(T.Vs[idx].astype(np.int16) - vf[None, :].astype(np.int16)) + 1).astype(np.int8)
    mix = np.random.RandomState(1).randint(1, 2 ** 40, size=D.shape[1]).astype(np.int64)
    _, keep = np.unique(D.astype(np.int64) @ mix, return_index=True)
    keep.sort()
    left, _ = untold(np.ascontiguousarray(D[keep]), c, T.y)
    return [(0.0, float(T.pool.costs[idx[k]]) + 0.5, T, int(idx[k]), ("cmp", first), D[k].astype(np.int64))
            for k, l in zip(keep, left) if l == 0]


def exacts(c, tellers, most=1):
    """Of the tellers that with c make y a function, the cheapest ones, table and all: [(price, part, reading)]."""
    ex = [(s[1] + (1 if valued(s[4]) else s[2].rows(c * 16 + (s[5] if len(s) > 5 else s[2].feat(s[3], s[4])))), n)
          for n, s in enumerate(tellers) if s[0] == 0]
    return [(pr, tellers[n][2].pool.ts[tellers[n][3]], tellers[n][4]) for pr, n in sorted(ex)[:most]]


def exact(c, tellers):
    ex = exacts(c, tellers)
    return ex[0] if ex else None


def ranked(tellers):
    return sorted(tellers, key=lambda s: (s[0], s[1], s[3]))


def cheap(t):
    """A part one would name first: two slots at most, moved by one or set."""
    return len(t[3]) <= 2 and all(a[0] == "set" or abs(a[1]) == 1 for a in t[4]) and t[5] is None


def addresses(V, base):
    """What a row of flags may be addressed by: the plain circles and levels of 3 to ROWS values; and such a circle
    moved one way or the other by what a plain part of two values reads (a direction turned by a symbol). Returns the
    parts and their values at every beginning."""
    W, S = len(V.slots), len(V.H.syms)
    order = sorted(range(len(base.ts)), key=lambda i: price(base.ts[i], W))
    # (moved by symbols alone, not by what the display did at them: where a wheel stands is not seen on the display)
    plain = [i for i in order if base.ts[i][0] in ("circle", "level") and 3 <= base.ts[i][1] <= ROWS and cheap(base.ts[i])
             and all(sl < S for sl in base.ts[i][3])]
    out = [(base.ts[i], base.values()[i].astype(np.int16)) for i in plain]
    switches = [i for i in order if base.ts[i][1] == 2 and cheap(base.ts[i]) and all(sl < S for sl in base.ts[i][3])][:SWITCHES]
    one = [base.ts[i] for i in plain if base.ts[i][0] == "circle" and len(base.ts[i][3]) == 1 and abs(base.ts[i][4][0][1]) == 1
           and base.ts[i][4][0][0] == "add"]
    for gi in switches:
        gt = base.ts[gi]
        g = np.asarray(read_of(base.values()[gi].astype(np.int16), gt, "zero"))
        ts = []
        for kind, n, _, (s0,), ((_, x),), _ in one:
            for b in (0, 1):  # (+x while the switch reads b, -x while it reads the other)
                order2 = sorted([(W * (1 + b) + s0, ("add", x)), (W * (2 - b) + s0, ("add", -x))])
                ts.append((kind, n, False, tuple(p for p, _ in order2), tuple(a for _, a in order2), (gt, "zero")))
        ts = list(dict.fromkeys(ts))
        if ts:
            vals = run_all(ts, V, g)
            out += [(t, v.astype(np.int16)) for t, v in zip(ts, vals)]
    seen, kept = set(), []
    for t, v in out:  # (the same values: one address, the cheapest)
        if v.tobytes() not in seen:
            seen.add(v.tobytes())
            kept.append((t, v))
    return kept


def row_forms(V):
    """The rows of a small description: one or two symbols, each raising, lowering or turning over the flag."""
    out = []
    for r in (1, 2):
        for touched in itertools.combinations(range(len(V.H.syms)), r):
            if clash(touched, V):
                continue
            for chosen in itertools.product(FLAG_ACTS, repeat=r):
                if r == 2 and chosen[0] == chosen[1] and V.slots[touched[0]][0] == V.slots[touched[1]][0]:
                    continue
                if any(a == ("flag", 1) or a == ("flag", -1) for a in chosen):
                    out.append((touched, chosen))
    return out


def run_rows(rows, V, addr):
    """The flags (a bit for every value of the address) of many rows at every beginning heard; addr: for every row,
    the values of its address at every beginning (the address is read before the symbol, as every reading is)."""
    W, par = len(V.slots), V.par
    C = len(rows)
    act = np.zeros((C, W + 1), np.int8)  # (0 nothing, 1 raise, 2 lower, 3 turn over; the last column: no slot)
    for i, (touched, chosen) in enumerate(rows):
        for sl, (_, x) in zip(touched, chosen):
            act[i, sl] = 1 if x == 1 else 2 if x == 0 else 3
    out = np.zeros((C, len(par)), np.int32)
    for i in range(1, len(par)):
        v = out[:, par[i]]
        bit = np.left_shift(1, addr[:, par[i]].astype(np.int32))
        a = act[:, V.sslot[i]] if V.sslot[i] >= 0 else np.zeros(C, np.int8)
        if V.eslot[i] >= 0:
            a = np.where(act[:, V.eslot[i]] > 0, act[:, V.eslot[i]], a)
        out[:, i] = np.where(a == 1, v | bit, np.where(a == 2, v & ~bit, np.where(a == 3, v ^ bit, v)))
    return out


def flag_rows(V, adds, steps, c, y, late=lambda: False, most=1 + 3, chunk=20000):
    """The rows of flags (with their addresses) that, read where the address points, make y a function with c:
    [(price, part, "here")], the cheapest first."""
    W, par = len(V.slots), V.par
    forms_ = row_forms(V)
    nlab = math.log2(max(2, len(set(y.tolist()))))
    found = []
    pairs = [(k, f) for k in range(len(adds)) for f in range(len(forms_))]
    for lo in range(0, len(pairs), chunk):
        if late():
            break
        part = pairs[lo:lo + chunk]
        addr = np.stack([adds[k][1] for k, _ in part])
        flags = run_rows([forms_[f] for _, f in part], V, addr)
        F = ((flags[:, par[steps]] >> addr[:, par[steps]]) & 1).astype(np.int8)
        left, _ = untold(F, c, y)
        for j in np.nonzero(left == 0)[0]:
            k, f = part[j]
            at = adds[k][0]
            touched, chosen = forms_[f]
            t = ("row", at[1], False, touched, chosen, (at, "address"))
            pr = price(t, W) + price(at, W) + (price(at[5][0], W) if at[5] is not None else 0) + \
                len(np.unique(c * 16 + F[j])) * nlab + 0.5
            found.append((pr, t, "here"))
    return sorted(found, key=lambda x: x[0])[:most]


def _lines():
    """Every line of at most QLEN, of kinds 1 to 3, as a number: a leading 1 and then a base-4 digit for everyone in it,
    the first come first (0 is the empty line too). The tables: the line after every operation, and what is read."""
    M = 4 ** (QLEN + 1)
    code = lambda t: int("1" + "".join(str(k) for k in t), 4)
    every = [()]
    for n in range(1, QLEN + 1):
        every += [t + (k,) for t in every if len(t) == n - 1 for k in (1, 2, 3)]
    names = ["none"] + [f"push{k}" for k in (1, 2, 3)] + [f"move{k}" for k in (1, 2, 3)] + ["popf", "popb", "clear"] + \
            [f"drop{k}" for k in (1, 2, 3)]
    ops = np.tile(np.arange(M, dtype=np.int16), (len(names), 1))
    reads = {h: np.zeros(M, np.int16) for h in LINE_READS}
    for t in every:
        v = code(t)
        after = {"none": t, "popf": t[1:], "popb": t[:-1], "clear": ()}
        for k in (1, 2, 3):
            after[f"push{k}"] = t + (k,) if len(t) < QLEN else t
            rest = tuple(x for x in t if x != k)
            after[f"move{k}"] = rest + (k,)
            after[f"drop{k}"] = rest
        for i, nm in enumerate(names):
            ops[i, v] = code(after[nm])
        reads["front"][v], reads["back"][v], reads["len"][v] = (t[0] if t else 0), (t[-1] if t else 0), len(t)
        reads["count1"][v], reads["count2"][v] = t.count(1), t.count(2)
    ops[:, 0] = ops[:, 1]
    for h in LINE_READS:
        reads[h][0] = reads[h][1]
    return {nm: i for i, nm in enumerate(names)}, ops, reads


LINE_OP, LINE_OPS, LINE_READ = _lines()


def line_forms(V):
    """The lines of a small description: one or two slots that bring someone (each its own kind), one that sends away
    the first or the last, and maybe one more that empties the line or sends away every one of a kind."""
    W, out = len(V.slots), []
    for p in (1, 2):
        for pushers in itertools.combinations(range(W), p):
            if clash(pushers, V):
                continue
            for mode in ("push", "move"):
                for r in range(W):
                    if r in pushers or clash(pushers + (r,), V):
                        continue
                    for pop in ("popf", "popb"):
                        base = [(sl, (mode, k + 1)) for k, sl in enumerate(pushers)] + [(r, (pop, 0))]
                        extras = [None] + [(e, x) for e in range(W) if e not in pushers and e != r
                                           and not clash(pushers + (r, e), V)
                                           for x in [("clear", 0)] + [("drop", k + 1) for k in range(p)]]
                        for ex in extras:
                            items = sorted(base + ([ex] if ex else []))
                            out.append(("line", QLEN, False, tuple(sl for sl, _ in items), tuple(a for _, a in items), None))
    return out


def run_lines(ts, V):
    """The line of many line parts at every beginning heard (as numbers: see _lines)."""
    W, par = len(V.slots), V.par
    opid = np.zeros((len(ts), W + 1), np.int16)
    for i, t in enumerate(ts):
        for sl, (what, k) in zip(t[3], t[4]):
            opid[i, sl] = LINE_OP[what + (str(k) if what in ("push", "move", "drop") else "")]
    out = np.zeros((len(ts), len(par)), np.int16)
    rows = np.arange(len(ts))
    for i in range(1, len(par)):
        v = LINE_OPS[opid[:, V.sslot[i]], out[:, par[i]]] if V.sslot[i] >= 0 else out[:, par[i]]
        if V.eslot[i] >= 0:
            v = LINE_OPS[opid[:, V.eslot[i]], v]
        out[:, i] = v
    return out


def line_parts(V, steps, c, y, cache, late=lambda: False, most=1 + 3):
    """The lines that, by who is first or last or how many there are, make y a function with c:
    [(price, part, reading)], the cheapest first. cache: the lines and their values, made once for every search."""
    W, par = len(V.slots), V.par
    if "ts" not in cache:
        cache["ts"] = line_forms(V)
        cache["V"] = run_lines(cache["ts"], V) if cache["ts"] else np.zeros((0, len(par)), np.int16)
    ts, L = cache["ts"], cache["V"]
    nlab = math.log2(max(2, len(set(y.tolist()))))
    found = []
    at = L[:, par[steps]]
    for how in LINE_READS:
        if late():
            break
        F = LINE_READ[how][at].astype(np.int8)
        left, _ = untold(F, c, y)
        for j in np.nonzero(left == 0)[0]:
            found.append((price(ts[j], W) + len(np.unique(c * 16 + F[j])) * nlab + 0.5, ts[j], how))
    return sorted(found, key=lambda x: x[0])[:most]


def stack_read(v, how):
    """What is read of a stack held as a number: a leading 1, then a base-4 digit (its kind) for every one in it, the
    top last (0 is the empty stack too)."""
    top = v % 4 if v > 1 else 0
    if how == "top":
        return top
    if how == "under":
        return (v // 4) % 4 if v // 4 > 1 else 0
    return top * 3 + (0 if v <= 1 else 1 if v < 16 else 2)


def stack_forms(V):
    """The stacks of a small description: one to three symbols that put someone of their kind on top, one to three that
    take the top one off, and maybe one that empties the stack."""
    S, out = len(V.H.syms), []
    for p in (1, 2, 3):
        for pushers in itertools.combinations(range(S), p):
            rest = [x for x in range(S) if x not in pushers]
            for q in (1, 2, 3):
                for poppers in itertools.combinations(rest, q):
                    others = [x for x in rest if x not in poppers]
                    for ex in [None] + others:
                        items = sorted([(sl, ("push", k + 1)) for k, sl in enumerate(pushers)] + [(sl, ("pop", 0)) for sl in poppers] +
                                       ([(ex, ("clear", 0))] if ex is not None else []))
                        out.append(("stack", SDEPTH, False, tuple(sl for sl, _ in items), tuple(a for _, a in items), None))
    return out


def run_stacks(ts, V):
    """The stack of many stack parts at every beginning heard, for the search: the top SDEPTH of it as base-4 digits
    (the top last) and how deep it is (deeper than that, the bottom is forgotten)."""
    par = V.par
    S = len(V.H.syms)
    op = np.zeros((len(ts), S + 1), np.int64)  # (1-3: put on top that kind; 4: take off; 5: empty)
    for i, t in enumerate(ts):
        for sl, (what, k) in zip(t[3], t[4]):
            op[i, sl] = k if what == "push" else 4 if what == "pop" else 5
    v, d = np.zeros((len(ts), len(par)), np.int64), np.zeros((len(ts), len(par)), np.int32)
    keep = 4 ** SDEPTH
    for i in range(1, len(par)):
        v0, d0 = v[:, par[i]], d[:, par[i]]
        o = op[:, V.sslot[i]] if V.sslot[i] >= 0 else np.zeros(len(ts), np.int64)
        push, pop, clear = (o >= 1) & (o <= 3), (o == 4) & (d0 > 0), o == 5
        v[:, i] = np.where(push, (v0 * 4 + o) % keep, np.where(pop, v0 // 4, np.where(clear, 0, v0)))
        d[:, i] = np.where(push, d0 + 1, np.where(pop, d0 - 1, np.where(clear, 0, d0)))
    return v, d


def run_stack_exact(t, V):
    """One stack part at every beginning heard, exactly (the machine's numbers: see stack_read)."""
    par, sym = V.par, V.sym
    op = {sl: act for sl, act in zip(t[3], t[4])}
    out = [1] * len(par)
    for i in range(1, len(par)):
        v = out[par[i]]
        what, k = op.get(int(sym[i]), (None, 0))
        out[i] = v * 4 + k if what == "push" else (v // 4 if v > 1 else v) if what == "pop" else 1 if what == "clear" else v
    return np.array(out, dtype=object)


def stack_parts(V, steps, c, y, cache, late=lambda: False, most=1 + 3):
    """The stacks that, by who is on top (or under it), make y a function with c: [(price, part, reading)]."""
    W, par = len(V.slots), V.par
    if "ts" not in cache:
        cache["ts"] = stack_forms(V)
        cache["V"] = run_stacks(cache["ts"], V)
    ts, (St, Sd) = cache["ts"], cache["V"]
    nlab = math.log2(max(2, len(set(y.tolist()))))
    at, deep = St[:, par[steps]], Sd[:, par[steps]]
    top = np.where(deep > 0, at % 4, 0)
    feats = {"top": top, "under": np.where(deep > 1, (at // 4) % 4, 0),
             "topn": top * 3 + np.minimum(deep, 2)}
    found = []
    for how in STACK_READS:
        if late():
            break
        F = feats[how].astype(np.int8)
        left, _ = untold(F, c, y)
        for j in np.nonzero(left == 0)[0]:
            found.append((price(ts[j], W) + len(np.unique(c * 16 + F[j])) * nlab + 0.5, ts[j], how))
    found.sort(key=lambda x: x[0])
    # (and the cheapest of those every symbol has a part in: on short stories a stack moved by fewer symbols explains
    # one symbol as well, but only one that all of them move can be the one stack all the symbols read)
    whole = [f for f in found if len(f[1][3]) == len(V.H.syms)][:64]
    return found[:most] + [f for f in whole if f not in found[:most]]


class Parts:
    """The machine: the parts, and for every symbol which of them are read (and how) with the table of what the display
    does."""
    sure = None
    others = {}

    def __init__(self, V, parts, rules, bits):
        self.H, self.summed, self.names, self.slots, self.slot = V.H, V.summed, V.names, V.slots, V.slot
        self.parts, self.rules, self.bits = parts, rules, bits
        self.guard = [None if t[5] is None else parts.index(t[5][0]) for t in parts]
        self.missed = 0  # (how many steps of the stories run met a reading its table has no row for)

    def move(self, vals, a, seen):
        W = len(self.slots)
        base = [a] + ([self.slot[a, seen]] if (a, seen) in self.slot else [])
        out = []
        for j, (v, (kind, n, reject, touched, chosen, guard)) in enumerate(zip(vals, self.parts)):
            if kind == "stack":
                if a in touched:
                    what, k = chosen[touched.index(a)]
                    v = max(int(v), 1) * 4 + k if what == "push" else (v // 4 if v > 1 else v) if what == "pop" else 1
                out.append(v)
                continue
            if kind == "line":
                for sl in base:
                    if sl in touched:
                        what, k = chosen[touched.index(sl)]
                        v = int(LINE_OPS[LINE_OP[what + (str(k) if what in ("push", "move", "drop") else "")], v])
                out.append(v)
                continue
            if kind == "row":  # (the flag where the address points, as it was before this symbol)
                acts = [chosen[touched.index(sl)] for sl in base if sl in touched]
                if acts:
                    bit, x = 1 << int(vals[self.guard[j]]), acts[-1][1]
                    v = v | bit if x == 1 else v & ~bit if x == 0 else v ^ bit
                out.append(v)
                continue
            hit = list(base)
            if guard is not None and guard[1] != "gives":
                b = W * (1 + int(read_of(vals[self.guard[j]], guard[0], guard[1])))
                hit += [b + s for s in base]
            acts = [chosen[touched.index(s)] for s in hit if s in touched]
            if acts:
                sets = [x for what, x in acts if what == "set"]
                if sets:
                    v = n - 1 if sets[-1] == "top" else sets[-1]
                else:
                    w = v + sum(x * vals[self.guard[j]] if what == "take" else x for what, x in acts)
                    if kind == "circle":
                        v = w % n
                    else:
                        top = BIG if kind == "counter" else n - 1
                        v = v if reject and not 0 <= w <= top else max(0, min(top, w))
            out.append(v)
        return out

    def start(self):
        """The state before any symbol: the values of the parts and what is shown."""
        H = self.H
        shown = int(H.begin) if self.summed else self.names.index(H.begin) if H.begin in self.names else None
        return (0,) * len(self.parts), shown

    def next(self, state, a):
        """The state after symbol a (its number), and whether a reading was met that the table has no row for."""
        vals, shown = state
        ctxs, specs, table, default = self.rules[a]
        ctx = (max(0, min(shown, 2)) if self.summed else shown) if ctxs else 0
        key = (ctx,) + tuple(reading_at(vals, self.parts, j, how) for j, how in specs)
        lacks = False
        if specs and valued(specs[0][1]):  # (the number goes up or down by what the part holds)
            eff = key[1] if specs[0][1] == "value+" else -key[1]
        elif key in table:
            eff = table[key]
        else:  # (a reading never met: the commonest thing; for names, the label stays)
            lacks, eff = bool(specs or ctxs), default
        new = shown + eff if self.summed else shown if eff < 0 else eff
        return (tuple(self.move(vals, a, eff if self.summed else int(new != shown))), new), lacks

    def run(self, seq):
        state = self.start()
        if state[1] is None:
            return None
        for sym in seq:
            state, lacks = self.next(state, self.H.syms.index(sym))
            self.missed += lacks
        return str(state[1]) if self.summed else self.names[state[1]]

    def said(self, s):
        W = len(self.slots)
        a, e = self.slots[s % W]
        return self.H.syms[a] + ("" if e is None else " (and the number goes %s)" % ("up by %d" % e if e > 0 else "down by %d" % -e if e else "nowhere")
                                 if self.summed else " (and the label changes)" if e else " (and the label stays)")

    def effect(self, v):
        return "%+d" % v if self.summed else "the same" if v < 0 else self.names[v]

    def reads(self, j, how, v):
        """In words: part j read in the way `how` gives v."""
        if isinstance(how, tuple):
            return f"part {j} is {('below', 'level with', 'above')[v]} part {how[1]}"
        if how == "here":
            return f"the flag of part {j} where part {self.guard[j]} points is {'up' if v else 'down'}"
        if how in ("top", "under"):
            return f"{'on top of' if how == 'top' else 'under the top of'} stack {j} is {self.kind_name(j, v) if v else 'nobody'}"
        if how == "topn":
            return f"on top of stack {j} is {self.kind_name(j, v // 3) if v // 3 else 'nobody'}" + ("" if v % 3 == 0 else ", the only one" if v % 3 == 1 else ", with more under it")
        if how in ("top", "under"):
            return f"{'on top of' if how == 'top' else 'under the top of'} stack {j} is {self.kind_name(j, v) if v else 'nobody'}"
        if how == "topn":
            return f"on top of stack {j} is {self.kind_name(j, v // 3) if v // 3 else 'nobody'}" + ("" if v % 3 == 0 else ", the only one" if v % 3 == 1 else ", with more under it")
        if how in ("front", "back"):
            return f"the {'first' if how == 'front' else 'last'} in line {j} is {self.kind_name(j, v) if v else 'nobody (it is empty)'}"
        if how == "len":
            return f"line {j} holds {v}"
        if how in ("count1", "count2"):
            return f"line {j} holds {v} of {self.kind_name(j, int(how[-1]))}"
        if how == "zero":
            return f"part {j} is {'above zero' if v else 'zero'}"
        if how == "top":
            return f"part {j} is {'at' if v else 'below'} its top"
        if how in ("ge2", "ge3"):
            return f"part {j} is {how[2] + ' or more' if v else 'less than ' + how[2]}"
        return f"part {j} is {v}"

    def kind_name(self, j, k):
        """The kind k of line j, named by the symbol that brings it."""
        t = self.parts[j]
        for sl, (what, x) in zip(t[3], t[4]):
            if what in ("push", "move") and x == k:
                return "a '" + self.H.syms[self.slots[sl % len(self.slots)][0] if t[0] != "stack" else sl] + "'"
        return f"kind {k}"

    def seen(self, c):
        if self.summed:
            return "the number shown is " + ("0", "1", "2 or more")[c]
        return "the label shown is " + self.names[c]

    def __str__(self):
        words = "the number shown as a sum; " if self.summed else ""
        if not self.parts:
            return words + ("no part" if all(not ctxs for ctxs, _, _, _ in self.rules) else
                            f"no part (the display by what it shows and the symbol); {self.bits:.0f} bits")
        return words + f"{len(self.parts)} part{'s' if len(self.parts) != 1 else ''}: " + \
            "; ".join((f"a row of {t[1]} flags" if t[0] == "row" else f"a line of at most {t[1]}" if t[0] == "line" else "a stack" if t[0] == "stack" else f"a {t[0]}{'' if t[0] == 'counter' else ' of %d' % t[1]}") +
                      ("" if t[5] is None or t[0] == "row" else " that takes what another holds" if t[5][1] == "gives" else " under a guard")
                      for t in self.parts) + f"; {self.bits:.0f} bits"

    def code(self):
        """The rule as a program: ALPHABET and answer(seq), which runs as the machine does (the parts are p0, p1, ...;
        every reading is taken before the symbol moves anything)."""
        H, W, S = self.H, len(self.slots), len(self.H.syms)
        q = repr
        begin = int(H.begin) if self.summed else H.begin
        kinds = {}  # (line, kind) -> the symbol that brings it
        for j, t in enumerate(self.parts):
            if t[0] in ("line", "stack"):
                for sl, (what, k) in zip(t[3], t[4]):
                    if what in ("push", "move"):
                        kinds[j, k] = H.syms[self.slots[sl % W][0] if t[0] == "line" else sl]
        lines = [f"ALPHABET = {q(list(H.syms))}", "", "", "def answer(seq):",
                 f"    shown = {q(begin)}  # what the display shows"]
        for j, t in enumerate(self.parts):
            kind, n = t[0], t[1]
            what = {"counter": "a counter, not below zero", "level": f"a level from 0 to {n - 1}", "circle": f"a circle of {n}",
                    "row": f"a row of {n} flags, one for each value of p{self.guard[j]}", "line": f"a line of at most {n}, first come first served",
                    "stack": "a stack, the last put on top"}[kind]
            lines.append(f"    p{j} = {'[]' if kind in ('line', 'stack') else '0'}  # {what}")
        def read(j, how):
            t = self.parts[j]
            if isinstance(how, tuple):
                return f"(p{j} > p{how[1]}) - (p{j} < p{how[1]}) + 1"
            if how == "here":
                return f"(p{j} >> p{self.guard[j]}) & 1"
            if how in STACK_READS:  # (the stack as a list, the top last)
                ks = {name: k for (jj, k), name in kinds.items() if jj == j}
                top = f"({q(ks)}[p{j}[-1]] if p{j} else 0)"
                if how == "top":
                    return top
                if how == "under":
                    return f"({q(ks)}[p{j}[-2]] if len(p{j}) > 1 else 0)"
                return f"({top} * 3 + min(len(p{j}), 2))"
            if how in ("front", "back"):
                ks = {name: k for (jj, k), name in kinds.items() if jj == j}
                at = "0" if how == "front" else "-1"
                return f"({q(ks)}[p{j}[{at}]] if p{j} else 0)"
            if how == "len":
                return f"len(p{j})"
            if how in ("count1", "count2"):
                return f"p{j}.count({q(kinds.get((j, int(how[-1])), ''))})"
            if how == "zero":
                return f"int(p{j} >= 1)"
            if how in ("full", "value+", "value-"):
                return f"p{j}"
            if how == "top":
                return f"int(p{j} == {t[1] - 1})"
            return f"int(p{j} >= {how[2]})"
        lines.append("    for tok in seq:")
        for a, (ctxs, specs, table, default) in enumerate(self.rules):
            lines.append(f"        {'if' if a == 0 else 'elif'} tok == {q(H.syms[a])}:")
            ctx = ("min(max(shown, 0), 2)" if self.summed else "shown") if ctxs else "0"
            if specs and valued(specs[0][1]):
                lines.append(f"            effect = {'' if specs[0][1] == 'value+' else '-'}{read(*specs[0])}")
            elif not specs and not ctxs:
                lines.append(f"            effect = {q(default if self.summed else (None if default < 0 else self.names[default]))}")
            else:
                key = "(" + ", ".join([ctx] + [read(j, how) for j, how in specs]) + ",)"
                tab = {(k if not ctxs or self.summed else (self.names[k[0]],) + k[1:]) if ctxs else (0,) + k[1:]:
                       (v if self.summed else None if v < 0 else self.names[v]) for k, v in table.items()}
                common = max(set(tab.values()), key=lambda v: (list(tab.values()).count(v), str(v)))
                rare = {k: v for k, v in tab.items() if v != common}
                dflt = q(default if self.summed else (None if default < 0 else self.names[default]))
                lines.append(f"            effect = {q(rare)}.get({key}, {q(common)})" if len(rare) < len(tab) else
                             f"            effect = {q(tab)}.get({key}, {dflt})")
            if self.summed:
                lines.append("            new = shown + effect")
                lines.append("            seen = effect")
            else:
                lines.append("            new = shown if effect is None else effect")
                lines.append("            seen = int(new != shown)")
            # the moves of the parts at this symbol, from the values before it
            for j, (kind, n, reject, touched, chosen, guard) in enumerate(self.parts):
                if kind == "stack":
                    if a in touched:
                        what, k = chosen[touched.index(a)]
                        lines.append(f"            n{j} = " + (f"p{j} + [{q(kinds.get((j, k), ''))}]" if what == "push" else
                                                         f"p{j}[:-1]" if what == "pop" else "[]"))
                    continue
                mine = [(sl, act) for sl, act in zip(touched, chosen) if self.slots[sl % W][0] == a]
                if not mine:
                    continue
                conds = []
                for sl, act in mine:
                    e, b = self.slots[sl % W][1], sl // W - 1
                    c = [] if e is None else [f"seen == {q(e) if self.summed else int(e)}"]
                    if b >= 0 and kind not in ("row", "line"):
                        c.append(f"{read(self.guard[j], guard[1])} == {b}")
                    conds.append((" and ".join(c), act))
                if kind == "row":
                    for c, (_, x) in conds:
                        op = {1: "|", 0: "& ~", -1: "^"}[x]
                        lines.append(f"            {'if ' + c + ': ' if c else ''}n{j} = p{j} {op} (1 << p{self.guard[j]})")
                    continue
                if kind == "line":
                    for c, (what, k) in conds:
                        nm = q(kinds.get((j, k), ""))
                        expr = {"push": f"p{j} + [{nm}] if len(p{j}) < {n} else p{j}", "move": f"[x for x in p{j} if x != {nm}] + [{nm}]",
                                "drop": f"[x for x in p{j} if x != {nm}]", "popf": f"p{j}[1:]", "popb": f"p{j}[:-1]", "clear": "[]"}[what]
                        lines.append(f"            {'if ' + c + ': ' if c else ''}n{j} = {expr}")
                    continue
                top = "10 ** 9" if kind == "counter" else str(n - 1)
                if len(conds) == 1 and not conds[0][0] and conds[0][1][0] in ("add", "set"):  # (one plain move: said plainly)
                    what, x = conds[0][1]
                    w = f"p{j} - {-x}" if what == "add" and x < 0 else f"p{j} + {x}"
                    lines.append(f"            n{j} = " + (str(n - 1 if x == "top" else x) if what == "set" else
                                 f"({w}) % {n}" if kind == "circle" else
                                 f"{w} if 0 <= {w} <= {top} else p{j}" if reject else
                                 f"max(0, {w})" if kind == "counter" and x < 0 else f"{w}" if kind == "counter" else
                                 f"min({top}, {w})" if x > 0 else f"max(0, {w})"))
                    continue
                lines.append(f"            d{j}, s{j} = 0, None")
                for c, (what, x) in conds:
                    st = f"s{j} = {n - 1 if x == 'top' else x}" if what == "set" else \
                        f"d{j} += {x} * p{self.guard[j]}" if what == "take" else f"d{j} += {x}"
                    lines.append(f"            {'if ' + c + ': ' if c else ''}{st}")
                step = f"(p{j} + d{j}) % {n}" if kind == "circle" else \
                    (f"p{j} + d{j} if 0 <= p{j} + d{j} <= {top} else p{j}" if reject else f"max(0, min({top}, p{j} + d{j}))")
                lines.append(f"            n{j} = s{j} if s{j} is not None else {step}")
            moved = [j for j, t in enumerate(self.parts) if (a in t[3] if t[0] == "stack" else any(self.slots[sl % W][0] == a for sl in t[3]))]
            for j in moved:
                lines.append(f"            p{j} = n{j}")
            lines.append("            shown = new")
        lines.append("    return str(shown)")
        return "\n".join(lines) + "\n"

    def table(self):
        """The rule in words: the parts and what moves them; then, for every symbol, what the display does."""
        H, W, lines = self.H, len(self.slots), []
        for j, (kind, n, reject, touched, chosen, guard) in enumerate(self.parts):
            if kind == "stack":
                lines.append(f"    part {j}, a stack, the last put on top. " + "; ".join(
                    self.H.syms[sl] + ": " + (f"{self.kind_name(j, x)} goes on top" if what == "push" else
                                              "the top one is taken off" if what == "pop" else "the stack is emptied")
                    for sl, (what, x) in zip(touched, chosen)))
                continue
            if kind == "line":
                words = {"popf": "the first in line leaves", "popb": "the last in line leaves", "clear": "the line is emptied"}
                lines.append(f"    part {j}, a line of at most {n}, first come first served. " + "; ".join(
                    self.said(sl) + ": " + (f"{self.kind_name(j, x)} joins the back of the line (when there is room)" if what == "push" else
                                            f"{self.kind_name(j, x)} goes to the back of the line (not twice)" if what == "move" else
                                            f"every {self.kind_name(j, x)} leaves the line" if what == "drop" else words[what])
                    for sl, (what, x) in zip(touched, chosen)))
                continue
            if kind == "row":
                lines.append(f"    part {j}, a row of {n} flags, one for each value of part {self.guard[j]}. " +
                             "; ".join(self.said(sl) + ": " + ("raise", "lower")[1 - x] if x in (0, 1) else self.said(sl) + ": turn over"
                                       for sl, (_, x) in zip(touched, chosen)) + f" the flag where part {self.guard[j]} points")
                continue
            moves = "; ".join(self.said(s) + ("" if s < W else ", only while " + self.reads(self.guard[j], guard[1], s // W - 1)) +
                              (f": {'%+d' % x}" if what == "add" else f": plus all that part {self.guard[j]} holds" if what == "take"
                               else f": to {n - 1 if x == 'top' else x}")
                              for s, (what, x) in zip(touched, chosen))
            lines.append(f"    part {j}, a {kind}{'' if kind == 'counter' else ' of %d (0 to %d)' % (n, n - 1) if kind == 'level' else ' of %d' % n}"
                         f"{' (a move that does not fit is not made)' if reject else ''}. {moves}")
        for a, (ctxs, specs, table, default) in enumerate(self.rules):
            if not specs and not ctxs:
                lines.append(f"    {H.syms[a]}: {'the display stays' if (default == 0 if self.summed else default < 0) else 'always ' + self.effect(default)}")
            elif specs[:1] and valued(specs[0][1]):
                lines.append(f"    {H.syms[a]}: the number goes {'up' if specs[0][1] == 'value+' else 'down'} by what part {specs[0][0]} holds")
            else:  # (the commonest thing the display does, and the readings at which it does another)
                common = max(set(table.values()), key=lambda v: (sum(x == v for x in table.values()), v == (0 if self.summed else -1)))
                rows = [" and ".join(([self.seen(k[0])] if ctxs else []) + [self.reads(j, how, v) for (j, how), v in zip(specs, k[1:])]) +
                        " -> " + self.effect(v) for k, v in sorted(table.items()) if v != common]
                lines.append(f"    {H.syms[a]}: " + "; ".join(rows[:12]) + (" ..." if len(rows) > 12 else "") +
                             f"; otherwise {'the display stays' if (common == 0 if self.summed else common < 0) else self.effect(common)}")
        return "\n".join(lines)


def explain(V, log=print, before=None, late=lambda: False):
    """The cheapest parts that make what the display does at every symbol a function of what is read. Returns the
    machine and whether every symbol is explained.
    For a symbol the display does more than one thing at: a part of one or two slots that tells it all; else the parts
    that tell the most are given a third slot; else two parts; else a part that waits on a guard."""
    H, par, sym, eff, ctx = V.H, V.par, V.sym, V.eff, V.ctx
    W, N = len(V.slots), len(par)
    t0 = time.time()
    base = pool_of(templates(V), V, W)
    log(f"  {len(base.ts)} parts of one or two slots, different on the {N} beginnings heard ({time.time() - t0:.1f} s)")
    found, others, adds, lines, every, stacks = {}, {}, None, {}, {}, {}  # symbol -> (price, with what is seen, [(part, reading)]); the next
    # cheapest ones; the addresses of rows of flags and the lines, made when first needed
    was = [] if before is None else [(t[5][0][:5], t[5][1]) for t in before.parts if t[5] is not None and t[5][1] not in ("gives", "address")]
    for a in range(len(H.syms)):
        steps = np.nonzero(sym == a)[0]
        y = eff[steps]
        if len(set(y.tolist())) <= 1:
            found[a] = (0.0, False, [])
            continue
        if late():
            break
        T0 = Told(base, steps, par, y, V.summed)
        best, cands = [None], []

        def better(cand):
            cands.append(cand)
            best[0] = cand if best[0] is None or cand[0] < best[0][0] else best[0]
        kept, allc = {}, {}
        for use_ctx in (False, True):
            if use_ctx and not (ctx[steps] != ctx[steps][0]).any():
                continue
            c = ctx[steps] if use_ctx else np.zeros(len(steps), int)
            extra = 2.0 if use_ctx else 0.0
            allc[use_ctx] = (c, extra)
            if untold(np.zeros((1, len(y)), np.int8), c, y)[0][0] == 0:
                better((extra + T0.rows(c), use_ctx, []))
                continue
            tel = ranked(T0.tellers(c))
            one = exact(c, tel)
            if not one:  # the parts that tell the most, with a third slot
                more = pool_of(grown([base.ts[s[3]] for s in tel[:GROW]], V), V, W)
                T3 = Told(more, steps, par, y, V.summed)
                tel = ranked(tel + T3.tellers(c))
                one = exact(c, tel)
            if one:
                for pr, t, how in exacts(c, tel, 1 + OTHERS):
                    better((pr + extra, use_ctx, [(t, how)]))
            else:
                kept[use_ctx] = (c, extra, tel, T3)
        plains = {}
        for use_ctx, (c, extra, tel, T3) in kept.items():  # the plain parts, first those that settle the most steps
            plain = [s for s in tel if cheap(s[2].pool.ts[s[3]]) and s[4] in ("zero", "top")]  # of the symbol by themselves
            if plain:
                sure = settled(np.stack([s[2].feat(s[3], s[4]) for s in plain]).astype(np.int8), c, y)
                plain = [s for _, _, s in sorted(zip(-sure, range(len(plain)), plain), key=lambda x: (x[0], x[2][1], x[1]))]
            plains[use_ctx] = plain
        if best[0] is None:  # two parts: with each of the few that tell the most, and of the plain ones, the second that
            for use_ctx, (c, extra, tel, T3) in kept.items():  # tells the rest
                firsts = tel[:FIRSTS] + [s for s in plains[use_ctx][:GUARDS] if s not in tel[:FIRSTS]]
                for imp, pr, T, i, how in firsts:
                    if late():
                        break
                    c2 = c * 16 + T.feat(i, how)
                    tel2 = ranked(T0.tellers(c2) + T3.tellers(c2))
                    two = exact(c2, tel2)
                    if not two:  # (beside the first, other parts tell the most: they too are given a third slot)
                        ts3 = grown([s[2].pool.ts[s[3]] for s in tel2[:GROW // 3] if len(s[2].pool.ts[s[3]][3]) < 3], V)
                        two = exact(c2, Told(pool_of(ts3, V, W), steps, par, y, V.summed).tellers(c2)) if ts3 else None
                    if two:
                        better((pr + two[0] + extra + 1, use_ctx, [(T.pool.ts[i], how), two[1:]]))
                    elif T.pool.ts[i][0] != "circle":  # or a part that, by how it stands to the first (below it, level
                        for Tx in (T0, T3):             # with it, above it), tells it all: more of these than of those
                            for pr2, t2, how2 in exacts(c, compared(Tx, T.Vs[i], T.pool.ts[i], c), 1):
                                better((pr + pr2 + extra + 1, use_ctx, [(t2, how2)]))
        if best[0] is None:  # a part some move of which waits on what a plain part reads (a guard)
            for use_ctx, (c, extra, tel, T3) in kept.items():
                plain = plains[use_ctx]
                old = [s for s in plain if (s[2].pool.ts[s[3]][:5], s[4]) in was]  # (the guards of the machine before)
                for n, (imp, pr, T, i, how) in enumerate(old + [s for s in plain[:GUARDS] if s not in old]):
                    if late() or best[0] is not None and old and n >= len(old):  # (the old guards did: no other is tried)
                        break
                    gt = T.pool.ts[i]
                    g = np.asarray(read_of(T.pool.values()[i].astype(np.int16), gt, how))
                    c2 = c * 16 + T.feat(i, how)
                    near = ranked(T0.tellers(c2) + T3.tellers(c2))[:GROW // 2]  # (what tells the most beside the guard)
                    ts2 = [t[:5] + ((gt, how),) for t in guarded(list(dict.fromkeys(s[2].pool.ts[s[3]] for s in near)), V)]
                    if not ts2:
                        continue
                    Tg = Told(pool_of(ts2, V, W, g), steps, par, y, V.summed)
                    two = exact(c2, Tg.tellers(c2))
                    if two:  # (read with its guard)
                        better((pr + two[0] + extra + 1, use_ctx, [(gt, how), two[1:]]))
                    alone = exact(c, Tg.tellers(c))
                    if alone:
                        better((pr + alone[0] + extra, use_ctx, [alone[1:]]))
        if best[0] is None:  # a part that at some slot takes all a plain counter holds: at the slot that empties the counter
            givers = [t for t in sorted(base.ts, key=lambda t: price(t, W))
                      if cheap(t) and t[0] == "counter" and ("set", 0) in t[4] and ("add", 1) in t[4]][:GIVERS]
            for use_ctx, (c, extra, tel, T3) in kept.items():
                # (it is also moved as the parts that tell the most are, and by one slot more)
                moves = list(dict.fromkeys(m for s in tel[:GROW // 2] for m in zip(s[2].pool.ts[s[3]][3], s[2].pool.ts[s[3]][4])))
                for gt in givers:
                    if late():
                        break
                    slot = gt[3][gt[4].index(("set", 0))]
                    seeds = []
                    for kind, n in KINDS:
                        if kind == "circle":
                            continue
                        seeds += forms(kind, n, (slot,), (("take", 1),))
                        for s2, a2 in moves:
                            if s2 != slot and a2 in acts_of(kind, n) and not clash((slot, s2), V):
                                order = sorted([(slot, ("take", 1)), (s2, a2)])
                                seeds += forms(kind, n, tuple(x for x, _ in order), tuple(x for _, x in order))
                    ts5 = [t[:5] + ((gt, "gives"),) for t in dict.fromkeys(seeds + grown([t for t in seeds if len(t[3]) == 2], V))]
                    Tv = Told(pool_of(ts5, V, W, None, base.values()[base.ts.index(gt)]), steps, par, y, V.summed)
                    one = exact(c, Tv.tellers(c))
                    if one:
                        better((one[0] + extra, use_ctx, [one[1:]]))
        # a row of flags, read where its address points: also when what was found is longer than the shortest row can
        # be (a row and its address, each of one slot, and a table of two rows): the found one may be a long way round
        least = 2 * (math.log2(3) + math.log2(7) + math.log2(W) + math.log2(len(ACTS))) + 2 + 2 * math.log2(max(2, len(set(y.tolist()))))
        if (best[0] is None or best[0][0] > least) and not late():
            if adds is None:
                adds = addresses(V, base)
                log(f"  {len(adds)} addresses for rows of flags ({time.time() - t0:.1f} s)")
            for use_ctx, (c, extra) in allc.items():  # (with what is seen or without, whatever was found before)
                for pr, t, how in flag_rows(V, adds, steps, c, y, late):
                    better((pr + extra, use_ctx, [(t, how)]))
        if (best[0] is None or best[0][0] > least) and not late():  # a line, first come first served
            for use_ctx, (c, extra) in allc.items():  # (with what is seen or without, whatever was found before)
                for pr, t, how in line_parts(V, steps, c, y, lines, late):
                    better((pr + extra, use_ctx, [(t, how)]))
        if (best[0] is None or best[0][0] > least) and not late():  # a stack, the last put on top
            for use_ctx, (c, extra) in allc.items():  # (with what is seen or without, whatever was found before)
                for pr, t, how in stack_parts(V, steps, c, y, stacks, late):
                    better((pr + extra, use_ctx, [(t, how)]))
        if best[0] is not None:
            found[a] = best[0]
            rest = sorted((c for c in cands if c[1:] != best[0][1:]), key=lambda c: c[0])
            others[a] = list({str(c[1:]): c for c in reversed(rest[:OTHERS])}.values())
            every[a] = [best[0]] + list({str(c[1:]): c for c in reversed(rest[:SHARED])}.values())
        log(f"  {H.syms[a]}: {'not explained' if best[0] is None else '%.0f bits' % best[0][0]} ({time.time() - t0:.1f} s)")
    # a part that another symbol reads is in the machine already: it is not paid for twice (Rissanen: the description
    # of the whole machine is what is short, not of each symbol on its own). The explanations are chosen together: the
    # length of all the parts read, each once, and of every symbol's table; a move puts one symbol on another of its
    # explanations, or every symbol that can read some part on its explanation that reads it
    own = lambda cand: cand[0] - sum(price(p, W) for p in chain(cand[2]))
    def total(pick):
        used = set()
        for cand in pick.values():
            used |= chain(cand[2])
        return sum(own(c) for c in pick.values()) + sum(price(p, W) for p in used)
    best_total = total(found)
    for _ in range(4):
        moves = []
        for a in every:
            for cand in every[a]:
                if cand[1:] != found[a][1:]:
                    moves.append({**found, a: cand})
        shared = {}
        for a in every:
            for cand in every[a]:
                for p in chain(cand[2]):
                    shared.setdefault(p, {}).setdefault(a, []).append(cand)
        for p, by in shared.items():
            if len(by) > 1:
                moves.append({**found, **{a: min(cs, key=own) for a, cs in by.items()}})
        better_ = min(moves, key=total, default=None)
        if better_ is None or total(better_) >= best_total - 1e-9:
            break
        for a in found:
            if better_[a][1:] != found[a][1:]:
                others[a] = [found[a]] + [c for c in others.get(a, []) if c[1:] != better_[a][1:]][:OTHERS - 1]
        found, best_total = dict(better_), total(better_)
    m, ok = machine(V, found)
    m.others, m.every = others, every
    return m, len(found) == len(H.syms)


def chain(specs):
    """The parts an explanation reads, and the parts those wait on, take from or are addressed by."""
    out = set()
    todo = [t for t, _ in specs] + [how[1] for _, how in specs if isinstance(how, tuple)]
    while todo:
        t = todo.pop()
        if t not in out:
            out.add(t)
            if t[5] is not None:
                todo.append(t[5][0])
    return out


def machine(V, found):
    """The machine of the parts found: for every symbol (price, with what is seen, [(part, reading)]). Its tables are
    read off the beginnings heard; the second thing returned says whether they could be (no reading met with two
    different effects)."""
    H, par, sym, eff, ctx = V.H, V.par, V.sym, V.eff, V.ctx
    W = len(V.slots)
    chosen = []

    def index(t):
        if t[5] is not None:
            index(t[5][0])
        if t not in chosen:
            chosen.append(t)
        return chosen.index(t)
    for a in sorted(found):
        for t, how in found[a][2]:
            index(t)
            if isinstance(how, tuple):
                index(how[1])
    vals = {}  # the values of the chosen parts at every beginning (a guarded part after its guard)
    for j, t in enumerate(chosen):
        if t[0] == "line":
            vals[j] = run_lines([t], V)[0]
        elif t[0] == "stack":
            vals[j] = run_stack_exact(t, V)
        elif t[5] is None:
            vals[j] = run_all([t], V)[0]
        elif t[5][1] == "gives":
            vals[j] = run_all([t], V, None, vals[chosen.index(t[5][0])])[0]
        elif t[5][1] == "address":
            vals[j] = run_rows([(t[3], t[4])], V, vals[chosen.index(t[5][0])][None, :].astype(np.int32))[0]
        else:
            vals[j] = run_all([t], V, np.asarray(read_of(vals[chosen.index(t[5][0])].astype(np.int16), t[5][0], t[5][1])))[0]
    rules, ok = [], True
    for a in range(len(H.syms)):
        steps = np.nonzero(sym == a)[0]
        y = eff[steps]
        default = (int(np.bincount(y - y.min()).argmax() + y.min()) if len(y) else 0) if V.summed else -1
        if a not in found:
            rules.append((False, [], {}, default))
            continue
        pr, use_ctx, specs = found[a]
        local = [(chosen.index(t), (how[0], chosen.index(how[1])) if isinstance(how, tuple) else how) for t, how in specs]
        table = {}
        for s in steps:
            at = {j: int(vals[j][par[s]]) for j in vals}
            key = (int(ctx[s]) if use_ctx else 0,) + tuple(reading_at(at, chosen, j, how) for j, how in local)
            if specs and valued(specs[0][1]):
                ok = ok and int(eff[s]) == (key[1] if specs[0][1] == "value+" else -key[1])
            else:
                ok = ok and table.get(key, int(eff[s])) == int(eff[s])
                table[key] = int(eff[s])
        if not specs and not use_ctx and len(y):
            default = int(y[0])
        rules.append((use_ctx, local, table, default))
    bits = sum(price(t, W) for t in chosen) + sum(len(r[2]) * math.log2(max(2, len(set(eff.tolist())))) for r in rules)
    m = Parts(V, chosen, rules, bits)
    m.V, m.found = V, dict(found)
    return m, ok


def alike(t, kind, n):
    """Part t as another kind or size (None where its moves do not fit that kind)."""
    if (kind, n) == (t[0], t[1]) or any(x not in acts_of(kind, n) for x in t[4] if x[0] != "take"):
        return None
    return (kind, n, t[2] and kind == "level") + t[3:]


def rivals(m):
    """Other machines that explain every beginning heard: the kept one with one of its parts of another kind or size
    (a counter for a level, a level with another top: the stories heard may never have reached the top); and the kept
    one with one symbol explained by the next cheapest parts found for it."""
    out = []
    for t in m.parts:
        for kind, n in KINDS:
            t2 = alike(t, kind, n)
            if t2 is None:
                continue
            swap = lambda x: t2 if x == t else x if x[5] is None or x[5][0] != t else x[:5] + ((t2, x[5][1]),)
            found = {a: (pr, c, [(swap(x), (how[0], swap(how[1])) if isinstance(how, tuple) else how) for x, how in specs])
                     for a, (pr, c, specs) in m.found.items()}
            if any(how in ("top", "full") and x[0] == "counter" or how == "full" and x[1] > 12
                   for _, _, specs in found.values() for x, how in specs):
                continue
            r, ok = machine(m.V, found)
            if ok:
                out.append(r)
    for a, cands in m.others.items():
        for cand in cands:
            r, ok = machine(m.V, {**m.found, a: cand})
            if ok:
                out.append(r)
    return out


def parting(m1, m2, depth=24, most=60000):
    """The shortest story (of at most depth symbols) after which two machines show different things: a walk in breadth
    over the pairs of their states (two dozen symbols: a circle of twelve and a counter part after thirteen)."""
    syms = m1.H.syms
    s0 = (m1.start(), m2.start())
    frontier, seen = [((), s0)], {s0}
    for _ in range(depth):
        nxt = []
        for w, (s1, s2) in frontier:
            for a in range(len(syms)):
                t1, t2 = m1.next(s1, a)[0], m2.next(s2, a)[0]
                if t1[1] != t2[1]:
                    return [syms[x] for x in w + (a,)]
                if (t1, t2) not in seen:
                    seen.add((t1, t2))
                    nxt.append((w + (a,), (t1, t2)))
        frontier = nxt
        if len(seen) > most or not frontier:
            break
    return None


def lacking(m, depth=12, most=60000):
    """The shortest story at whose last symbol the machine meets a reading its table has no row for."""
    syms = m.H.syms
    s0 = m.start()
    frontier, seen = [((), s0)], {s0}
    for _ in range(depth):
        nxt = []
        for w, st in frontier:
            for a in range(len(syms)):
                t, lacks = m.next(st, a)
                if lacks:
                    return [syms[x] for x in w + (a,)]
                if t not in seen:
                    seen.add(t)
                    nxt.append((w + (a,), t))
        frontier = nxt
        if len(seen) > most or not frontier:
            break
    return None


def explained(H, log=print, before=None, late=lambda: False):
    """A number shown as a sum; when that does not explain every symbol and the numbers met are few, as names."""
    views = [True, False] if H.numeric else [False]
    if before is not None and before.summed in views:
        views = [before.summed] + [v for v in views if v != before.summed]
    kept = None
    for summed in views:
        if not summed and H.numeric and len(set(H.shown)) > 13:
            continue
        m, whole = explain(View(H, summed), log, before if before is not None and before.summed == summed else None, late)
        if whole:
            return m, True
        kept = kept or (m, False)
    return kept


def story(syms, rng, n, again=0.3):
    """A story of n symbols in which a symbol is often repeated: tops and bottoms are reached."""
    w, last = [], None
    for _ in range(n):
        last = last if last is not None and rng.random() < again else rng.choice(syms)
        w.append(last)
    return w


def learn(syms, oracle, data, rng, asks=40, quiet=False, deadline=None, give_up=None):
    """The stories taught on (with the answers of all their beginnings, as a watcher sees them) and up to asks stories
    asked. A story is asked where it can tell something: the story on which a rival machine (see rivals) shows another
    thing than the kept one; the story that meets a reading the kept machine's tables have no row for; and, when there
    is neither, long stories as a check. Where the kept machine is wrong the parts are looked for again. It is sure
    when no rival is left, no row is lacking and the last SURE long stories were answered rightly.
    Returns the machine, the number of questions asked, the stories heard, and whether every symbol was explained.
    give_up: the learner stops when some symbol is not explained; when every one of its first EARLY[0] questions, or
    all but one of its first EARLY[1], refuted the machine it kept (no few parts explain this rule: every story asked
    shows it); or when so many questions are asked and the machine has not yet answered SURE // 2 stories in a row.
    Another learner may then take the rest of the questions."""
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    late = lambda: deadline is not None and time.time() > deadline
    H = Heard(syms, oracle([]))
    hear = lambda w: H.add(w, [oracle(list(w[:i])) for i in range(1, len(w) + 1)])
    for w, _ in data:
        hear(list(w))
    asked, row, run, best_run, wrongs, same = 0, 0, 0, 0, 0, set()
    m, whole = explained(H, log, None, late)
    H.trace = [("", whole, m.bits)]  # what every question was for, whether it was answered rightly, the bits of the machine kept
    log(f"  kept {m}\n{m.table()}")
    while asked < asks and not late():
        if give_up is not None and (not whole or asked == EARLY[0] and wrongs == asked or
                                    asked == EARLY[1] and wrongs >= asked - 1 or asked >= give_up and best_run < SURE // 2):
            break
        w, why = None, "a check"
        if whole:
            for r in rivals(m):
                key = str(sorted(r.found.items()))
                if key not in same:
                    w = parting(m, r)
                    if w is not None:
                        why = f"a rival ({r})"
                        break
                    same.add(key)  # (no story of two dozen symbols parts them: taken for the same machine)
            if w is None:
                w, why = lacking(m), "a reading not met"
        if w is None:
            if whole and row >= SURE:
                break
            # (every other long story with no symbol repeated on purpose: as the machine will be used)
            w, why = story(syms, rng, rng.randint(*LONG) if whole else 24, 0.3 if row % 2 else 0.0), "a check"
        elif len(w) < 24:  # (seen after every symbol, a longer story costs the same: it goes on)
            w = w + story(syms, rng, 24 - len(w))
        asked += 1
        hear(w)
        right = all(m.run(w[:i]) == oracle(w[:i]) for i in range(1, len(w) + 1))
        V = View(H, m.summed)
        if right and V.slots == m.V.slots:  # the same machine, its tables read off what is heard now
            others = m.others
            m, _ = machine(V, m.found)
            m.others = others
            run, row = run + 1, row + (why == "a check")
            best_run = max(best_run, run)
            H.trace.append((why[:8], True, m.bits))
            log(f"  asked {asked} ({why}): answered rightly")
            continue
        run, row, wrongs = 0, 0, wrongs + 1
        m, whole = explained(H, lambda *a: None, m, late)
        H.trace.append((why[:8], False if whole else None, m.bits))
        log(f"  asked {asked} ({why}): {'answered wrongly' if not right else 'a thing not seen before'}; kept {m}\n{m.table()}")
    m.sure = bool(whole and row >= SURE)
    return m, asked, H, whole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--asks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--give-up", type=int, default=None, help="stop when a symbol is not explained, or after so many questions without a run of right answers")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--code", action="store_true", help="print the rule found as a program (and check it against the machine)")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("family", a.family)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    syms, answer = list(mod.ALPHABET), mod.answer
    oracle = lambda w: str(answer(list(w)))
    rng = random.Random(a.seed)
    data = [(w, oracle(w)) for w in (tuple(rng.choice(syms) for _ in range(rng.randint(1, 4))) for _ in range(300))]
    t0 = time.time()
    m, asked, H, whole = learn(syms, oracle, data, rng, a.asks, a.quiet, None, a.give_up)
    print(m.table())
    print(f"{'' if m.sure else 'NOT SURE: '}{m}; asked {asked}; {time.time() - t0:.0f} s; every symbol explained: {whole}")
    if a.code:  # (the rule as a program, checked against the machine on stories of every length)
        src = m.code()
        print(src)
        mine = {}
        exec(src, mine)
        tr = random.Random(7)
        same = all(mine["answer"](w) == m.run(w) for w in ([tr.choice(syms) for _ in range(tr.randint(0, 200))] for _ in range(300)))
        print("the program answers as the machine:", same)
    # (the questions in order: + answered rightly, - wrongly, ? and some symbol was then left unexplained; the bits kept)
    print("trace:", " ".join(("+" if ok else "?" if ok is None else "-") + f"{bits:.0f}" for _, ok, bits in H.trace[1:]))
    for L in (20, 100, 500):
        tr = random.Random(1000 + L)
        seqs = [[tr.choice(syms) for _ in range(L)] for _ in range(300)]
        print(L, 100 * sum(m.run(s) == oracle(s) for s in seqs) / 300, "%")


if __name__ == "__main__":
    main()
