"""The machine (register shape, lexicon of 9 symbols) that explains the MOST teaching stories, found by a MaxSAT
solver: hard clauses = cells.Encoding (machine + reading of every symbol met into one of 9 machine symbols + the tree
of stories), and for every different teaching story a selector 'excused_i' put into its answer clauses; the soft unit
clause [-excused_i] (weight 1) asks the story to be explained. RC2 (core-guided) or LSU (model-improving) from pysat.

Optional, named general steps (all off by default):
  --inverse   general prior: a link walked against the order its sentence names the two ("sign -1" of steps_of)
              takes away what it adds when walked with it (the offsets are negated). The reading is then of views,
              not of (view, sign) pairs. (The hand register of one.py uses the same prior.)
  --cover     the sample of stories is widened so every symbol met in teaching occurs at least once: for each symbol
              not yet in the sample, the shortest story (fewest links) containing it is added.
  --complete  after the solver: the machine (offsets, answer table) fixed, every symbol of the teaching stories that
              the solver did not read gets the machine symbol under which the most teaching stories are answered
              rightly (passes until nothing changes). A count over the stories, no solver.
Protocol: teaching stories only; the test (1000 per k, random.Random(1)) is read only for the final measurement.
"""
import argparse, json, os, pickle, random, sys, threading, time
from collections import Counter, defaultdict
sys.path.insert(0, '../..')
sys.argv, _argv = ['x'], sys.argv
import cells
sys.argv = _argv
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.examples.lsu import LSU

HERE = os.path.dirname(os.path.abspath(__file__))
PLAIN = os.environ.get("RC2_PLAIN") == "1"  # RC2 with its default settings (no core exhaustion/minimisation/trimming)
SHAPE = (0, 1, (("Z", 1, 1), ("Z", 1, 1)), True, False)  # the StepGame register shape (fixed, said so)


def key_sign(a, inverse):
    """(raw symbol of the reading, sign the offsets are applied with)"""
    if inverse and isinstance(a, tuple) and len(a) == 2 and a[1] in (1, -1):
        return a[0], a[1]
    return a, 1


class Enc(cells.Encoding):
    """cells.Encoding; with inverse=True a step (view, -1) applies the offsets of view negated."""

    def __init__(self, shape, syms, labels, lexicon, inverse=False):
        self.inverse = inverse
        raw = sorted({key_sign(a, inverse)[0] for a in syms}, key=repr)
        super().__init__(shape, raw, labels, lexicon)

    def at(self, seq):
        if not self.inverse:
            return super().at(seq)
        if seq in self.node:
            return self.node[seq]
        n0 = self.at(seq[:-1])
        a, n, dep = seq[-1], len(self.node), self.depth[seq[:-1]]
        r_a, sg = key_sign(a, True)
        self.node[seq], self.depth[seq], self.ndepth[n] = n, dep + 1, dep + 1
        neg = lambda l: (not l) if isinstance(l, bool) else -l
        read = [(r_a, [])] if self.raw is None else [(s, [-self.v("w", r_a, i)]) for i, s in enumerate(self.syms)]
        for q in range(self.k):
            x = self.v("x", n0, q)
            for s, guard in read:
                for r in range(self.k):
                    self.clauses.append([-x, -self.v("t", s, q, r), self.v("x", n, r)] + guard)
            for j, c in enumerate(self.cells):
                for w in self.keys(j):
                    for s, guard in read:
                        keyl = ([-self.v("z", n0, 0, w)] if self.chain and j else []) + guard
                        for o in self.offsets(c):
                            off = self.v("o", j, s, q, w, o)
                            oo = sg * o
                            if c[0] == "Z":
                                D = c[1] * (dep + 1)
                                for b in range(-D + 1, D + 1):
                                    self.add([-x, -off, neg(self.ge(n0, j, b - oo)), self.ge(n, j, b)] + keyl)
                                    self.add([-x, -off, self.ge(n0, j, b - oo), neg(self.ge(n, j, b))] + keyl)
                            else:
                                for val in range(c[1]):
                                    self.clauses.append([-x, -off, -self.v("z", n0, j, val), self.v("z", n, j, (val + oo) % c[1])] + keyl)
        for j, c in enumerate(self.cells):
            if c[0] == "Z":
                D = c[1] * (dep + 1)
                for b in range(-D + 1, D):
                    self.add([neg(self.ge(n, j, b + 1)), self.ge(n, j, b)])
        return n

    def machine(self, model, syms):
        """the machine read off a model; with inverse, every machine symbol s gets a twin s~ with negated offsets,
        and (view, -1) is read into the twin, so cells.Machine.run works unchanged"""
        model = set(l for l in model if l > 0)
        on = lambda *key: self.ids.obj2id.get(key) in model
        tau = {(a, q): next(r for r in range(self.k) if on("t", a, q, r)) for a in self.syms for q in range(self.k)}
        dl = {(a, q, j, w): next(o for o in self.offsets(c) if on("o", j, a, q, w, o))
              for a in self.syms for q in range(self.k) for j, c in enumerate(self.cells) for w in self.keys(j)}
        lam = {r: next(y for y in range(len(self.labels)) if on("l", r, y)) for r in self.reads}
        if self.raw is None:  # direct (k = 1): a symbol is read into its offset vector; the machine symbols are the vectors
            import itertools
            vec = lambda t: "(" + ",".join(map(str, t)) + ")"
            vs = list(itertools.product(*[self.offsets(c) for c in self.cells]))
            rawlex = {a: vec(tuple(dl[a, 0, j, 0] for j in range(len(self.cells)))) for a in self.syms}
            tau = {(vec(t), 0): 0 for t in vs}
            dl = {(vec(t), 0, j, 0): t[j] for t in vs for j in range(len(self.cells))}
            return build(self.shape, [vec(t) for t in vs], self.labels, tau, dl, lam, rawlex, syms, self.inverse)
        rawlex = {r: self.syms[next(i for i in range(len(self.syms)) if on("w", r, i))] for r in self.raw}
        return build(self.shape, self.syms, self.labels, tau, dl, lam, rawlex, syms, self.inverse)


def build(shape, msyms, labels, tau, dl, lam, rawlex, syms, inverse):
    if inverse:
        twins = [s + "~" for s in msyms]
        tau = dict(tau) | {(s + "~", q): tau[s, q] for s in msyms for q in range(shape[1])}
        dl = dict(dl) | {(k[0] + "~",) + k[1:]: -o for k, o in dl.items()}
        msyms = list(msyms) + twins
    m = cells.Machine(shape, list(msyms), labels, tau, dl, lam)
    m.lex = {}
    for a in syms:
        r, sg = key_sign(a, inverse)
        if r in rawlex:
            m.lex[a] = rawlex[r] + ("~" if sg < 0 else "")
    m.rawlex = rawlex
    return m


def solve_maxsat(e, n_soft, sels, seconds, how, log):
    w = WCNF()
    for c in e.clauses:
        w.append(c)
    for s in sels:
        w.append([-s], weight=1)
    log(f"  MaxSAT: {n_soft} soft, {len(e.clauses)} hard clauses, {e.ids.top} variables")
    t0 = time.time()
    if how == "rc2":
        s = RC2(w, solver="g4", adapt=False, exhaust=True, minz=True, trim=5) if not PLAIN else RC2(w, solver="g4")
        timer = threading.Timer(seconds, s.interrupt)
        timer.start()
        model = s.compute(expect_interrupt=True)
        timer.cancel()
        status = "optimum" if model is not None else "unknown"
        cost = s.cost if model is not None else None
        s.delete()
    else:
        s = LSU(w, solver="g4", expect_interrupt=True)
        timer = threading.Timer(seconds, s.interrupt)
        timer.start()
        ok = s.solve()
        timer.cancel()
        model = s.get_model() if s.model is not None else None
        status = "optimum" if s.found_optimum() else ("best found" if model is not None else "unknown")
        cost = s.cost if model is not None else None
        s.delete()
    return model, status, cost, time.time() - t0


def complete(m, every, inverse, log, passes=6, polish=False):
    """machine fixed; every symbol of the teaching stories not yet read gets the machine symbol under which the most
    teaching stories are answered rightly (only stories whose other symbols are all read count)."""
    raws = defaultdict(list)  # raw symbol -> indices of stories containing it
    for i, (s, y, _) in enumerate(every):
        for a in set(s):
            raws[key_sign(a, inverse)[0]].append(i)
    choices = [x for x in m.syms if not x.endswith("~")]
    all_syms = {a for s, _, _ in every for a in s}
    by_raw = defaultdict(list)
    for a in all_syms:
        by_raw[key_sign(a, inverse)[0]].append(a)
    added = 0
    for p in range(passes):
        todo = [r for r in raws if polish or r not in m.rawlex]
        changed = 0
        for r in sorted(todo, key=lambda r: -len(raws[r])):
            old = m.rawlex.get(r)
            best, bestn = old, (sum(m.run(every[i][0]) == every[i][1] for i in raws[r]) if old else 0)
            for c in choices:
                for a in by_raw[r]:
                    sg = key_sign(a, inverse)[1]
                    m.lex[a] = c + ("~" if sg < 0 else "")
                n = sum(m.run(every[i][0]) == every[i][1] for i in raws[r])
                if n > bestn:
                    best, bestn = c, n
            for a in by_raw[r]:
                m.lex.pop(a, None)
            if best is not None:
                changed += best != old
                m.rawlex[r] = best
                for a in by_raw[r]:
                    m.lex[a] = best + ("~" if key_sign(a, inverse)[1] < 0 else "")
        added += changed
        log(f"  {'polish' if polish else 'completion'} pass {p}: {changed} symbols read anew; {len([r for r in raws if r not in m.rawlex])} still unread")
        if not changed:
            break
    return added


def explained(m, data):
    return sum(m.run(s) == y for s, y, *_ in data) / len(data)


def measure(m, test):
    acc, none = [], []
    for k in range(1, 11):
        st = test[f"k={k}"]
        outs = [m.run(s) if s is not None else None for s, _ in st]
        acc.append(round(100 * sum(o == y for o, (_, y) in zip(outs, st)) / len(st), 1))
        none.append(round(100 * sum(o is None for o in outs) / len(st), 1))
    return acc, none


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="different teaching stories (random.Random(seed) sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--solver", default="rc2", choices=["rc2", "lsu"])
    ap.add_argument("--lexicon", type=int, default=9)
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--direct", action="store_true", help="k = 1 only: lexicon 9 over two cells of offsets -1..1 is the "
                    "same as reading every symbol straight into its offset vector; encoded so (no lexicon variables)")
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--complete", action="store_true")
    ap.add_argument("--polish", action="store_true", help="after completion: every symbol re-read the same way (count over all teaching stories, machine fixed)")
    ap.add_argument("--test", action="store_true", help="final measurement on the test")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    name = a.name or f"n{a.n}_{a.solver}" + ("_dir" if a.direct else "") + ("_inv" if a.inverse else "") + ("_cov" if a.cover else "") + ("_cpl" if a.complete else "") + ("_pol" if a.polish else "")
    logf = open(os.path.join(HERE, name + ".log"), "w")
    T0 = time.time()

    def log(x):
        line = f"[{time.time() - T0:6.1f} s] {x}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    d = pickle.load(open(os.path.join(HERE, "data.pkl"), "rb"))
    every = d["every"]
    labels = sorted({y for _, y, _ in every})
    rng = random.Random(a.seed)
    data = rng.sample(every, min(a.n, len(every)))
    if a.cover:
        have = {key_sign(x, a.inverse)[0] for s, _, _ in data for x in s}
        chosen = {s for s, _, _ in data}
        for s, y, c in sorted(every, key=lambda t: (len(t[0]), -t[2], repr(t[0]))):
            new = {key_sign(x, a.inverse)[0] for x in s} - have
            if new and s not in chosen:
                data.append((s, y, c)); chosen.add(s); have |= new
        log(f"cover: {len(data) - a.n} stories added so every symbol is met")
    syms = sorted({x for s, _, _ in data for x in s}, key=repr)
    lab = {l: i for i, l in enumerate(labels)}
    log(f"{name}: {len(data)} different teaching stories, {len(syms)} symbols, shape fixed: {cells.name(SHAPE)}; lexicon {a.lexicon}; inverse prior {a.inverse}")
    t0 = time.time()
    e = Enc(SHAPE, syms, labels, 0 if a.direct else a.lexicon, a.inverse)
    sels = []
    for i, (s, y, _) in enumerate(data):
        x = e.v("excused", i)
        e.says(s, lab[y], x)
        sels.append(x)
    tb = time.time() - t0
    log(f"  built in {tb:.1f} s")
    model, status, cost, ts = solve_maxsat(e, len(data), sels, a.seconds, a.solver, log)
    log(f"  solver: {status}, cost {cost} (stories not explained), {ts:.1f} s")
    res = dict(name=name, n=len(data), n_sample=a.n, symbols=len(syms), build_s=round(tb, 1), solve_s=round(ts, 1),
               status=status, cost=cost, inverse=a.inverse, cover=a.cover, complete=a.complete, polish=a.polish, solver=a.solver, direct=a.direct)
    if model is None:
        res["wall_s"] = round(time.time() - T0, 1)
        json.dump(res, open(os.path.join(HERE, name + ".json"), "w"), indent=1)
        return
    m = e.machine(model, {x for s, _, _ in every for x in s})
    res["sample_explained"] = round(100 * explained(m, data), 2)
    res["set_aside"] = sum(m.run(s) != y for s, y, _ in data)
    res["teach_explained_before_completion"] = round(100 * explained(m, every), 2)
    res["teach_unread_before_completion"] = round(100 * sum(m.run(s) is None for s, _, _ in every) / len(every), 2)
    log(f"  machine explains {res['sample_explained']}% of its sample ({res['set_aside']} set aside), "
        f"{res['teach_explained_before_completion']}% of all {len(every)} different teaching stories "
        f"({res['teach_unread_before_completion']}% have a symbol not read)")
    offs = {s: tuple(m.dl[s, 0, j, 0] for j in range(2)) for s in m.syms}
    log(f"  machine symbols -> offsets: {offs}")
    log(f"  answer table: {{{', '.join(f'{r[1:]}: {labels[y]}' for r, y in m.lam.items())}}}")
    log(f"  reading: {dict(Counter(m.rawlex.values()))}")
    if a.complete:
        tc = time.time()
        res["completed_symbols"] = complete(m, every, a.inverse, log)
        if a.polish:
            res["polished_symbols"] = complete(m, every, a.inverse, log, polish=True)
        res["complete_s"] = round(time.time() - tc, 1)
        res["teach_explained"] = round(100 * explained(m, every), 2)
        res["teach_unread"] = round(100 * sum(m.run(s) is None for s, _, _ in every) / len(every), 2)
        log(f"  after completion: {res['teach_explained']}% of the different teaching stories explained "
            f"({res['teach_unread']}% unread), {res['complete_s']} s")
    else:
        res["teach_explained"] = res["teach_explained_before_completion"]
    pickle.dump(dict(shape=SHAPE, msyms=m.syms, labels=labels, tau=m.tau, dl=m.dl, lam=m.lam, lex=m.lex),
                open(os.path.join(HERE, name + ".machine.pkl"), "wb"))
    res["wall_s_before_test"] = round(time.time() - T0, 1)
    if a.test:
        acc, none = measure(m, d["test"])
        res["test_pct_by_k"], res["no_answer_pct_by_k"] = acc, none
        log(f"  TEST accuracy by k=1..10: {acc}")
        log(f"  TEST no answer by k=1..10: {none}")
    res["wall_s"] = round(time.time() - T0, 1)
    log(f"  wall {res['wall_s']} s")
    json.dump(res, open(os.path.join(HERE, name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
