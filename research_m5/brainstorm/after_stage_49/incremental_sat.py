#!/usr/bin/env python3
"""inc.py - the lexicon of the solver's machine found by ONE incremental SAT solver (Een and Sorensson 2003).

The encoding is cells.Encoding with lexicon=m (the base: the reading w(raw, s) of every symbol met into one of m symbols
of the machine, value precedence, the machine's moves, offsets and answer table, symmetry breaking), changed in three
ways:
  1. every symbol met gets derived variables for what it does (T: its move, O: its offsets) = what the machine symbol
     it is read into does; a node of the tree of stories then uses T/O of its own symbol, so its clauses are not
     repeated once for every machine symbol (m times fewer clauses per node);
  2. clauses are streamed into one solver that is kept for the whole run (learnt clauses kept across calls);
  3. the answer clause of story i carries its own selector: (... or -sel_i); a story is taught by assuming sel_i.
Growing: the machine of the last model is run on the stories of the pool; of those it answers wrongly the first B (in
the order of the pool) are assumed with all the stories already taught; if that is unsatisfiable they are tried one
by one, and a story that cannot be added is set aside for good (unit clause -sel_i).
The order of the pool: stories told more often first (a general prior: a story told many times is more likely told
rightly), then shorter ones. Nothing of StepGame is used beyond the symbols one.steps_of gives."""
import argparse, json, pickle, random, sys, threading, time
from collections import Counter

ARGS = sys.argv[1:]
sys.argv = ['x']
sys.path.insert(0, '../..')
import cells
from pysat.solvers import Solver

HERE = '.'
REGISTER = (0, 1, (("Z", 1, 1), ("Z", 1, 1)), True, False)


class IncEncoding(cells.Encoding):
    def __init__(self, shape, raw, labels, m, solver="glucose4"):
        super().__init__(shape, raw, labels, m)
        self.linked = set()
        self.s = Solver(name=solver)
        self.sel = {}
        self.flush()

    def flush(self):
        if self.clauses:
            self.s.append_formula(self.clauses)
            self.clauses.clear()

    def link(self, a):
        """T(a,q,r) <-> t(s,q,r) and O(a,j,q,w,o) <-> o(j,s,q,w,o) for the machine symbol s that a is read into."""
        self.linked.add(a)
        k, C, v = self.k, self.clauses, self.v
        for i, s in enumerate(self.syms):
            w = v("w", a, i)
            for q in range(k):
                for r in range(k):
                    T, t = v("t", a, q, r), v("t", s, q, r)
                    C += [[-w, -t, T], [-w, t, -T]]
                for j, c in enumerate(self.cells):
                    for wk in self.keys(j):
                        for o in self.offsets(c):
                            O, oo = v("o", j, a, q, wk, o), v("o", j, s, q, wk, o)
                            C += [[-w, -oo, O], [-w, oo, -O]]
        for q in range(k):  # (implied, but helps propagation)
            self.exactly([v("t", a, q, r) for r in range(k)])
            for j, c in enumerate(self.cells):
                for wk in self.keys(j):
                    self.exactly([v("o", j, a, q, wk, o) for o in self.offsets(c)])

    def at(self, seq):
        if seq in self.node:
            return self.node[seq]
        for a in seq:
            if a not in self.linked:
                self.link(a)
        raw, self.raw = self.raw, None  # the node is built as for raw symbols: on T/O of its own symbol
        try:
            return cells.Encoding.at(self, seq)
        finally:
            self.raw = raw

    def teach(self, i, seq, y):
        if i not in self.sel:
            self.sel[i] = self.v("sel", i)
            self.says(seq, y, -self.sel[i])
        return self.sel[i]

    def solve(self, assumptions, seconds):
        self.flush()
        timer = threading.Timer(seconds, self.s.interrupt)
        timer.start()
        ok = self.s.solve_limited(assumptions=assumptions, expect_interrupt=True)
        timer.cancel()
        if ok is None:
            self.s.clear_interrupt()
        return ok

    def machine(self, read_syms):
        model = set(l for l in self.s.get_model() if l > 0)
        on = lambda *key: self.ids.obj2id.get(key) in model
        tau = {(a, q): next(r for r in range(self.k) if on("t", a, q, r)) for a in self.syms for q in range(self.k)}
        dl = {(a, q, j, w): next(o for o in self.offsets(c) if on("o", j, a, q, w, o))
              for a in self.syms for q in range(self.k) for j, c in enumerate(self.cells) for w in self.keys(j)}
        lam = {r: next(y for y in range(len(self.labels)) if on("l", r, y)) for r in self.reads}
        m = cells.Machine(self.shape, self.syms, self.labels, tau, dl, lam)
        m.lex = {r: self.syms[next(i for i in range(len(self.syms)) if on("w", r, i))] for r in read_syms}
        return m


def grow(pool, labels, shape=REGISTER, m=9, batch=20, seconds=1500, step_seconds=60, solver="glucose4", log=print,
         init_acc=(), init_aside=()):
    """pool: list of (seq, answer) in the order to teach; init_acc / init_aside: indices of the pool already taught /
    set aside (e.g. by MaxSAT on a part). -> (machine, taught indices, set-aside indices, stats)"""
    lab = {l: i for i, l in enumerate(labels)}
    raw = sorted({x for s, _ in pool for x in s}, key=repr)
    t0 = time.time()
    e = IncEncoding(shape, raw, labels, m, solver)
    acc, aside, S, timeouts, calls = list(init_acc), set(init_aside), [], 0, Counter()
    taught_syms = set()
    for i in acc:
        S.append(e.teach(i, pool[i][0], lab[pool[i][1]]))
        taught_syms.update(pool[i][0])
    for i in aside:
        e.s.add_clause([-e.teach(i, pool[i][0], lab[pool[i][1]])])
    assert e.solve(S, step_seconds)
    mach = e.machine(taught_syms)
    rounds = 0
    while True:
        accset = set(acc)
        mach.lex = {r: mach.lex.get(r) for r in taught_syms}  # (read only the symbols of the stories taught)
        wrong = [i for i, (s, y) in enumerate(pool) if i not in accset and i not in aside and
                 (any(x not in mach.lex for x in s) or mach.run(s) != y)]
        rounds += 1
        if rounds % 10 == 1 or not wrong:
            log(f"[{time.time() - t0:6.1f} s] {len(acc)} taught, {len(aside)} set aside, {len(wrong)} of the pool "
                f"answered wrongly or not read; solver calls {dict(calls)}, {e.s.nof_vars()} vars {e.s.nof_clauses()} clauses")
        if not wrong or time.time() - t0 > seconds:
            break
        part = wrong[:batch]
        sels = [e.teach(i, pool[i][0], lab[pool[i][1]]) for i in part]
        ok = e.solve(S + sels, step_seconds)
        calls["batch-" + str(ok)] += 1
        if ok:
            acc += part
            S += sels
            for i in part:
                taught_syms.update(pool[i][0])
            mach = e.machine(taught_syms)
            continue
        core = set(e.s.get_core() or []) if ok is False else None
        inn = [i for i in part if core is None or e.sel[i] in core]
        if len(inn) == 1:  # the core names one story of the batch: it alone cannot go with the stories taught
            aside.add(inn[0])
            e.s.add_clause([-e.sel[inn[0]]])
            continue  # the rest of the batch comes back (still wrong) in the next round
        last_ok = False
        for i in part:  # one by one
            ok = e.solve(S + [e.sel[i]], step_seconds)
            calls["one-" + str(ok)] += 1
            if ok:
                acc.append(i)
                S.append(e.sel[i])
                taught_syms.update(pool[i][0])
                last_ok = True
            else:
                timeouts += ok is None
                aside.add(i)
                e.s.add_clause([-e.sel[i]])
                last_ok = False
        if not last_ok:
            assert e.solve(S, step_seconds)
        mach = e.machine(taught_syms)
    stats = {"seconds": time.time() - t0, "calls": dict(calls), "timeouts": timeouts, "rounds": rounds}
    return mach, acc, sorted(aside), stats


def load():
    d = pickle.load(open(HERE + '/data.pkl', 'rb'))
    return d['every'], d['test']


def make_pool(every, n, seed=0):
    """n different teaching stories: first a cover (every symbol met in teaching is in some story of the pool: for the
    rarest symbol first, the story told most often that has it), then drawn at random; ordered by how often told, then
    by length."""
    rng = random.Random(seed)
    order = sorted(range(len(every)), key=lambda i: (-every[i][2], len(every[i][0]), i))
    has = {}
    for i in order:
        for x in every[i][0]:
            has.setdefault(x, []).append(i)
    chosen, met = [], set()
    for x in sorted(has, key=lambda x: (len(has[x]), repr(x))):
        if x not in met:
            i = has[x][0]
            chosen.append(i)
            met.update(every[i][0])
    rest = [i for i in range(len(every)) if i not in set(chosen)]
    chosen += rng.sample(rest, max(0, n - len(chosen)))
    chosen = sorted(set(chosen), key=lambda i: (-every[i][2], len(every[i][0]), i))
    return [(every[i][0], every[i][1]) for i in chosen], len(met)


def measure(mach, every, test):
    right = sum(mach.run(s) == y for s, y, _ in every)
    wright = sum(c for s, y, c in every if mach.run(s) == y)
    out = {"explains_different": 100 * right / len(every), "explains_weighted": 100 * wright / sum(c for *_, c in every),
           "test": {}, "no_answer": {}}
    for k in range(1, 11):
        ss = test[f"k={k}"]
        got = [mach.run(s) if s is not None else None for s, _ in ss]
        out["test"][k] = 100 * sum(g == y for g, (_, y) in zip(got, ss)) / len(ss)
        out["no_answer"][k] = 100 * sum(g is None for g in got) / len(ss)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="different teaching stories in the pool")
    ap.add_argument("--m", type=int, default=9, help="symbols of the machine (lexicon)")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=1200)
    ap.add_argument("--solver", default="glucose4")
    ap.add_argument("--out", default="run")
    ap.add_argument("--extend", action="store_true", help="then go on over all the other different teaching stories")
    a = ap.parse_args(ARGS)
    t0 = time.time()
    every, test = load()
    pool, nsym = make_pool(every, a.n)
    labels = sorted({y for _, y, _ in every})
    logf = open(f"{HERE}/{a.out}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"pool: {len(pool)} different teaching stories of {len(every)}, {nsym} symbols met; shape fixed: {cells.name(REGISTER)}; lexicon {a.m}")
    mach, acc, aside, st = grow(pool, labels, REGISTER, a.m, a.batch, a.seconds, solver=a.solver, log=log)
    if a.extend:  # from the machine of the pool on: every other different teaching story, told more often first
        log(f"pool done: {len(acc)} taught, {len(aside)} set aside; going on over all {len(every)} different stories")
        seen = set(s for s, _ in pool)
        rest = sorted([x for x in every if x[0] not in seen], key=lambda x: (-x[2], len(x[0]), repr(x[0])))
        pool = pool + [(s, y) for s, y, _ in rest]
        mach, acc, aside, st2 = grow(pool, labels, REGISTER, a.m, a.batch, a.seconds, solver=a.solver, log=log,
                                     init_acc=acc, init_aside=aside)
        st = {"pool_part": st, "extension": st2, "seconds": st["seconds"] + st2["seconds"]}
    r = measure(mach, every, test)
    r.update(pool=len(pool), taught=len(acc), aside=len(aside), stats=st, wall=time.time() - t0,
             symbols_read=len(mach.lex), machine_symbols_used=len(set(mach.lex.values())))
    log(f"taught {len(acc)}, set aside {len(aside)} of {len(pool)}; {st}")
    log(f"explains {r['explains_different']:.2f}% of the {len(every)} different teaching stories "
        f"({r['explains_weighted']:.2f}% of the teaching stories counted with repeats)")
    log("test: " + "  ".join(f"k={k}: {r['test'][k]:.1f}" for k in range(1, 11)))
    log("no answer: " + "  ".join(f"k={k}: {r['no_answer'][k]:.1f}" for k in range(1, 11)))
    log(f"wall {r['wall']:.0f} s; {r['symbols_read']} symbols read into {r['machine_symbols_used']} machine symbols")
    offs = {s: tuple(mach.dl[s, 0, j, 0] for j in range(2)) for s in mach.syms}
    log("machine symbols -> offsets: " + str(offs) + "; answer table: " +
        str({rd: labels[y] for rd, y in mach.lam.items()}))
    json.dump(r, open(f"{HERE}/{a.out}.json", "w"), indent=1, default=str)
    pickle.dump({"lex": mach.lex, "dl": mach.dl, "tau": mach.tau, "lam": mach.lam, "labels": labels, "aside": [pool[i] for i in aside]},
                open(f"{HERE}/{a.out}.machine.pkl", "wb"))


if __name__ == "__main__":
    main()
